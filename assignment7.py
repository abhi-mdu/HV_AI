"""
Deep MLP Model on MNIST Dataset with Keras Tuner Hyperparameter Optimization

This module implements a comprehensive deep Multi-Layer Perceptron (MLP) for MNIST digit classification
with systematic hyperparameter tuning using Keras Tuner. The implementation includes comparison with
baseline models and detailed performance analysis.

Based on Chapter 10: Introduction to Artificial Neural Networks with Keras

Author: Abhishek Kumar, Aniruddha Biswas, Dewan Niaz Morshed
Institution: University West, Trollhattan, Sweden
Course: Masters in AI and Automation
Date: November 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
import time
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class DeepMLPMNISTAnalysis:
    """
    Comprehensive Deep MLP analysis for MNIST dataset with Keras Tuner optimization
    and comparison with baseline models.
    """
    
    def __init__(self, validation_split=0.1, random_state=42):
        """
        Initialize the Deep MLP MNIST analysis.
        
        Parameters:
        -----------
        validation_split : float
            Proportion of training data for validation (default: 0.1)
        random_state : int
            Random seed for reproducibility (default: 42)
        """
        self.validation_split = validation_split
        self.random_state = random_state
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.tuner = None
        self.best_model = None
        self.baseline_models = {}
        self.results = {}
        
    def load_and_prepare_data(self):
        """
        Load and preprocess MNIST dataset.
        """
        print("Loading MNIST dataset...")
        (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()
        
        # Normalize pixel values to [0, 1]
        X_train_full = X_train_full.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        # Reshape to flatten images
        X_train_full = X_train_full.reshape(X_train_full.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        
        # Split training data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, 
            test_size=self.validation_split, 
            random_state=self.random_state,
            stratify=y_train_full
        )
        
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        print(f"Training set: {X_train.shape}, Labels: {y_train.shape}")
        print(f"Validation set: {X_val.shape}, Labels: {y_val.shape}")
        print(f"Test set: {X_test.shape}, Labels: {y_test.shape}")
        print(f"Input shape: {X_train.shape[1]} features")
        print(f"Number of classes: {len(np.unique(y_train))}")
        
        # Display class distribution
        train_dist = pd.Series(y_train).value_counts().sort_index()
        print(f"Training class distribution:\\n{train_dist}")
        
    def build_model(self, hp):
        """
        Build MLP model with hyperparameters for Keras Tuner.
        
        Parameters:
        -----------
        hp : HyperParameters
            Keras Tuner hyperparameter object
            
        Returns:
        --------
        model : keras.Model
            Compiled Keras model
        """
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Dense(
            units=hp.Int('input_units', min_value=128, max_value=512, step=64),
            activation='relu',
            input_shape=(784,)
        ))
        
        # Hidden layers - tunable number and units
        num_layers = hp.Int('num_layers', min_value=1, max_value=4)
        for i in range(num_layers):
            model.add(layers.Dense(
                units=hp.Int(f'units_{i}', min_value=64, max_value=512, step=64),
                activation='relu'
            ))
            
            # Optional dropout for regularization
            if hp.Boolean('dropout'):
                model.add(layers.Dropout(
                    rate=hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
                ))
        
        # Output layer
        model.add(layers.Dense(10, activation='softmax'))
        
        # Compile model with tunable learning rate
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def hyperparameter_tuning(self, max_trials=20, epochs=10):
        """
        Perform hyperparameter tuning using Keras Tuner Random Search.
        
        Parameters:
        -----------
        max_trials : int
            Maximum number of trials for hyperparameter search (default: 20)
        epochs : int
            Number of epochs per trial (default: 10)
        """
        print("\\n=== Starting Hyperparameter Tuning ===")
        
        # Initialize Random Search tuner
        self.tuner = kt.RandomSearch(
            self.build_model,
            objective='val_accuracy',
            max_trials=max_trials,
            directory='mlp_tuning',
            project_name='mnist_mlp_optimization',
            overwrite=True
        )
        
        # Display search space
        print("\\nHyperparameter Search Space:")
        print("- Input Layer Units: 128-512 (step 64)")
        print("- Number of Hidden Layers: 1-4")
        print("- Hidden Layer Units: 64-512 (step 64)")
        print("- Dropout: True/False")
        print("- Dropout Rate: 0.1-0.5 (step 0.1)")
        print("- Learning Rate: 1e-4 to 1e-2 (log scale)")
        print(f"- Max Trials: {max_trials}")
        print(f"- Epochs per Trial: {epochs}")
        
        # Early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        # Start tuning
        start_time = time.time()
        print(f"\\nStarting Random Search with {max_trials} trials...")
        
        self.tuner.search(
            self.X_train, self.y_train,
            epochs=epochs,
            validation_data=(self.X_val, self.y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        tuning_time = time.time() - start_time
        print(f"\\nHyperparameter tuning completed in {tuning_time:.2f} seconds")
        
        # Get best hyperparameters
        best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        print("\\n=== Best Hyperparameters ===")
        print(f"Input Units: {best_hps.get('input_units')}")
        print(f"Number of Hidden Layers: {best_hps.get('num_layers')}")
        
        for i in range(best_hps.get('num_layers')):
            print(f"Hidden Layer {i+1} Units: {best_hps.get(f'units_{i}')}")
        
        print(f"Dropout: {best_hps.get('dropout')}")
        if best_hps.get('dropout'):
            print(f"Dropout Rate: {best_hps.get('dropout_rate')}")
        print(f"Learning Rate: {best_hps.get('learning_rate'):.6f}")
        
        return best_hps, tuning_time
        
    def train_best_model(self, epochs=50):
        """
        Train the best model found by hyperparameter tuning.
        
        Parameters:
        -----------
        epochs : int
            Number of training epochs (default: 50)
        """
        print("\\n=== Training Best Model ===")
        
        # Get the best model
        self.best_model = self.tuner.get_best_models(num_models=1)[0]
        
        # Callbacks for training
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]
        
        # Train the model
        start_time = time.time()
        history = self.best_model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            validation_data=(self.X_val, self.y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"\\nModel training completed in {training_time:.2f} seconds")
        
        return history, training_time
        
    def train_baseline_models(self):
        """
        Train baseline models for comparison.
        """
        print("\\n=== Training Baseline Models ===")
        
        # Simple MLP (2 hidden layers, fixed architecture)
        print("Training Simple MLP...")
        simple_mlp = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(784,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        
        simple_mlp.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        start_time = time.time()
        simple_history = simple_mlp.fit(
            self.X_train, self.y_train,
            epochs=20,
            validation_data=(self.X_val, self.y_val),
            verbose=0
        )
        simple_time = time.time() - start_time
        
        self.baseline_models['Simple MLP'] = {
            'model': simple_mlp,
            'history': simple_history,
            'training_time': simple_time
        }
        
        # Deep MLP (4 hidden layers, larger)
        print("Training Deep MLP...")
        deep_mlp = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(784,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        
        deep_mlp.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        start_time = time.time()
        deep_history = deep_mlp.fit(
            self.X_train, self.y_train,
            epochs=20,
            validation_data=(self.X_val, self.y_val),
            verbose=0
        )
        deep_time = time.time() - start_time
        
        self.baseline_models['Deep MLP'] = {
            'model': deep_mlp,
            'history': deep_history,
            'training_time': deep_time
        }
        
        print("Baseline models training completed.")
        
    def evaluate_all_models(self):
        """
        Evaluate all models on test set and compile results.
        """
        print("\\n=== Model Evaluation ===")
        
        # Evaluate tuned model
        print("Evaluating Tuned MLP...")
        tuned_loss, tuned_accuracy = self.best_model.evaluate(self.X_test, self.y_test, verbose=0)
        tuned_predictions = self.best_model.predict(self.X_test, verbose=0)
        tuned_pred_classes = np.argmax(tuned_predictions, axis=1)
        
        self.results['Tuned MLP'] = {
            'test_loss': tuned_loss,
            'test_accuracy': tuned_accuracy,
            'predictions': tuned_pred_classes
        }
        
        # Evaluate baseline models
        for name, model_info in self.baseline_models.items():
            print(f"Evaluating {name}...")
            model = model_info['model']
            loss, accuracy = model.evaluate(self.X_test, self.y_test, verbose=0)
            predictions = model.predict(self.X_test, verbose=0)
            pred_classes = np.argmax(predictions, axis=1)
            
            self.results[name] = {
                'test_loss': loss,
                'test_accuracy': accuracy,
                'predictions': pred_classes,
                'training_time': model_info['training_time']
            }
        
        # Create results summary
        results_df = pd.DataFrame({
            name: {
                'Test Accuracy': info['test_accuracy'],
                'Test Loss': info['test_loss'],
                'Training Time (s)': info.get('training_time', 'N/A')
            } for name, info in self.results.items()
        }).T
        
        print("\\n=== Results Summary ===")
        print(results_df.round(4))
        
        return results_df
        
    def create_visualizations(self):
        """
        Create comprehensive visualizations for analysis.
        """
        print("\\n=== Creating Visualizations ===")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Model Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Deep MLP MNIST Analysis: Performance and Training Dynamics', 
                     fontsize=16, fontweight='bold')
        
        # Test Accuracy Comparison
        ax1 = axes[0, 0]
        accuracies = [self.results[name]['test_accuracy'] for name in self.results.keys()]
        model_names = list(self.results.keys())
        
        bars = ax1.bar(model_names, accuracies, color=['red', 'blue', 'green'])
        ax1.set_title('Test Accuracy Comparison', fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0.95, 1.0)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Training Time Comparison
        ax2 = axes[0, 1]
        training_times = []
        time_labels = []
        
        for name in self.results.keys():
            if 'training_time' in self.results[name] and self.results[name]['training_time'] != 'N/A':
                training_times.append(self.results[name]['training_time'])
                time_labels.append(name)
        
        if training_times:
            bars2 = ax2.bar(time_labels, training_times, color='orange')
            ax2.set_title('Training Time Comparison', fontweight='bold')
            ax2.set_ylabel('Training Time (seconds)')
            ax2.grid(axis='y', alpha=0.3)
            
            for bar, time_val in zip(bars2, training_times):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(training_times)*0.01,
                        f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # Confusion Matrix for Tuned Model
        ax3 = axes[1, 0]
        cm = confusion_matrix(self.y_test, self.results['Tuned MLP']['predictions'])
        im = ax3.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax3.set_title('Tuned MLP Confusion Matrix', fontweight='bold')
        ax3.set_xlabel('Predicted Label')
        ax3.set_ylabel('True Label')
        
        # Add colorbar
        plt.colorbar(im, ax=ax3)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax3.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=8)
        
        # Hyperparameter Tuning Results
        ax4 = axes[1, 1]
        if hasattr(self, 'tuner') and self.tuner is not None:
            # Get trial results
            trials = self.tuner.oracle.get_best_trials(num_trials=10)
            trial_scores = [trial.score for trial in trials]
            trial_numbers = list(range(1, len(trial_scores) + 1))
            
            ax4.plot(trial_numbers, trial_scores, 'o-', color='purple', linewidth=2, markersize=6)
            ax4.set_title('Hyperparameter Tuning Progress', fontweight='bold')
            ax4.set_xlabel('Trial Number')
            ax4.set_ylabel('Validation Accuracy')
            ax4.grid(alpha=0.3)
            
            # Highlight best trial
            best_idx = np.argmax(trial_scores)
            ax4.plot(trial_numbers[best_idx], trial_scores[best_idx], 
                    'r*', markersize=15, label=f'Best: {trial_scores[best_idx]:.4f}')
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig('deep_mlp_mnist_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Training History Visualization
        if hasattr(self, 'best_model_history'):
            plt.figure(figsize=(12, 5))
            
            # Training and validation accuracy
            plt.subplot(1, 2, 1)
            plt.plot(self.best_model_history.history['accuracy'], label='Training Accuracy', color='blue')
            plt.plot(self.best_model_history.history['val_accuracy'], label='Validation Accuracy', color='red')
            plt.title('Model Accuracy During Training', fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(alpha=0.3)
            
            # Training and validation loss
            plt.subplot(1, 2, 2)
            plt.plot(self.best_model_history.history['loss'], label='Training Loss', color='blue')
            plt.plot(self.best_model_history.history['val_loss'], label='Validation Loss', color='red')
            plt.title('Model Loss During Training', fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        print("Visualizations saved successfully!")
        
    def analyze_convergence(self):
        """
        Analyze convergence rate and training dynamics.
        """
        print("\\n=== Convergence Analysis ===")
        
        if hasattr(self, 'best_model_history'):
            history = self.best_model_history.history
            
            # Find convergence epoch (when validation loss stops improving significantly)
            val_losses = history['val_loss']
            convergence_epoch = len(val_losses)
            
            for i in range(5, len(val_losses)):
                if all(val_losses[i] >= val_losses[i-j] - 0.001 for j in range(1, 6)):
                    convergence_epoch = i
                    break
            
            print(f"Model converged around epoch {convergence_epoch}")
            print(f"Final training accuracy: {history['accuracy'][-1]:.4f}")
            print(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
            print(f"Best validation accuracy: {max(history['val_accuracy']):.4f}")
            
            # Calculate convergence rate (accuracy improvement per epoch in first 10 epochs)
            if len(history['val_accuracy']) >= 10:
                early_acc = history['val_accuracy'][:10]
                convergence_rate = (early_acc[-1] - early_acc[0]) / 9
                print(f"Initial convergence rate: {convergence_rate:.4f} accuracy/epoch")
        
    def run_complete_analysis(self, max_trials=20, tuning_epochs=10, training_epochs=50):
        """
        Run the complete Deep MLP analysis pipeline.
        
        Parameters:
        -----------
        max_trials : int
            Maximum trials for hyperparameter tuning (default: 20)
        tuning_epochs : int
            Epochs per trial during tuning (default: 10)
        training_epochs : int
            Epochs for final model training (default: 50)
        """
        print("="*60)
        print("DEEP MLP MODEL ON MNIST DATASET")
        print("Hyperparameter Tuning with Keras Tuner")
        print("="*60)
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Hyperparameter tuning
        best_hps, tuning_time = self.hyperparameter_tuning(max_trials, tuning_epochs)
        
        # Train best model
        self.best_model_history, best_training_time = self.train_best_model(training_epochs)
        
        # Train baseline models
        self.train_baseline_models()
        
        # Evaluate all models
        results_df = self.evaluate_all_models()
        
        # Analyze convergence
        self.analyze_convergence()
        
        # Create visualizations
        self.create_visualizations()
        
        # Final summary
        print(f"\\n{'='*60}")
        print("FINAL ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        tuned_accuracy = self.results['Tuned MLP']['test_accuracy']
        print(f"Tuned MLP Test Accuracy: {tuned_accuracy:.4f}")
        
        if 'Simple MLP' in self.results:
            simple_accuracy = self.results['Simple MLP']['test_accuracy']
            improvement = tuned_accuracy - simple_accuracy
            print(f"Simple MLP Test Accuracy: {simple_accuracy:.4f}")
            print(f"Improvement over Simple MLP: {improvement:.4f} ({improvement*100:.2f}%)")
        
        print(f"\\nHyperparameter Tuning Time: {tuning_time:.2f} seconds")
        print(f"Best Model Training Time: {best_training_time:.2f} seconds")
        
        print(f"\\n{'='*60}")
        print("Analysis completed successfully!")
        print("Generated files:")
        print("- deep_mlp_mnist_analysis.png")
        print("- training_history.png")
        print(f"{'='*60}")
        
        return results_df, best_hps

def main():
    """
    Main function to run the complete analysis.
    """
    # Create analysis instance
    analysis = DeepMLPMNISTAnalysis(validation_split=0.1, random_state=42)
    
    # Run complete analysis with reasonable parameters for demonstration
    results, best_hyperparameters = analysis.run_complete_analysis(
        max_trials=15,  # Reduced for faster execution
        tuning_epochs=8,  # Reduced for faster execution
        training_epochs=30  # Reduced for faster execution
    )
    
    return analysis, results, best_hyperparameters

if __name__ == "__main__":
    analysis, results, best_hps = main()
