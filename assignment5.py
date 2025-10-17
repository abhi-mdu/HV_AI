"""
Building an SVM Classifier for MNIST with Hyperparameter Tuning and Comparative Analysis
Author: Abhishek,Anirudh,Niyaz
Date: October 2025

This script implements SVM classifiers with different kernels and compares their performance
with KNN, SGD, and Random Forest classifiers on the MNIST dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, classification_report, confusion_matrix)
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

class MNISTClassifierComparison:
    """
    A comprehensive class for comparing SVM with other classifiers on MNIST dataset.
    """
    
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = {}
        self.best_models = {}
        
    def load_and_prepare_data(self, sample_size=10000):
        """
        Load and prepare the MNIST dataset with optional sampling for faster computation.
        
        Args:
            sample_size (int): Number of samples to use for training (default: 10000)
        """
        print("Loading MNIST dataset...")
        
        # Load MNIST dataset
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist["data"], mnist["target"]
        y = y.astype(np.uint8)
        
        print(f"Dataset shape: {X.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Sample data for faster computation
        if sample_size < len(X_train):
            indices = np.random.choice(len(X_train), sample_size, replace=False)
            X_train = X_train[indices]
            y_train = y_train[indices]
            print(f"Sampled training data to {sample_size} samples")
        
        # Normalize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        print("Data preparation completed!")
        
    def train_svm_with_grid_search(self):
        """
        Train SVM classifiers with different kernels and perform hyperparameter tuning.
        """
        print("\n" + "="*60)
        print("TRAINING SVM CLASSIFIERS WITH HYPERPARAMETER TUNING")
        print("="*60)
        
        svm_results = {}
        
        # Linear SVM
        print("\n1. Training Linear SVM...")
        linear_params = {
            'C': [0.1, 1, 10, 100]
        }
        
        linear_svm = SVC(kernel='linear', random_state=42)
        start_time = time.time()
        linear_grid = GridSearchCV(linear_svm, linear_params, cv=3, 
                                 scoring='accuracy', n_jobs=-1, verbose=1)
        linear_grid.fit(self.X_train, self.y_train)
        linear_time = time.time() - start_time
        
        svm_results['Linear'] = {
            'model': linear_grid.best_estimator_,
            'best_params': linear_grid.best_params_,
            'cv_score': linear_grid.best_score_,
            'training_time': linear_time
        }
        
        # Polynomial SVM
        print("\n2. Training Polynomial SVM...")
        poly_params = {
            'C': [1, 10, 100],
            'degree': [2, 3, 4],
            'coef0': [0, 1, 10]
        }
        
        poly_svm = SVC(kernel='poly', random_state=42)
        start_time = time.time()
        poly_grid = GridSearchCV(poly_svm, poly_params, cv=3, 
                               scoring='accuracy', n_jobs=-1, verbose=1)
        poly_grid.fit(self.X_train, self.y_train)
        poly_time = time.time() - start_time
        
        svm_results['Polynomial'] = {
            'model': poly_grid.best_estimator_,
            'best_params': poly_grid.best_params_,
            'cv_score': poly_grid.best_score_,
            'training_time': poly_time
        }
        
        # RBF SVM
        print("\n3. Training RBF SVM...")
        rbf_params = {
            'C': [1, 10, 100, 1000],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        }
        
        rbf_svm = SVC(kernel='rbf', random_state=42)
        start_time = time.time()
        rbf_grid = GridSearchCV(rbf_svm, rbf_params, cv=3, 
                              scoring='accuracy', n_jobs=-1, verbose=1)
        rbf_grid.fit(self.X_train, self.y_train)
        rbf_time = time.time() - start_time
        
        svm_results['RBF'] = {
            'model': rbf_grid.best_estimator_,
            'best_params': rbf_grid.best_params_,
            'cv_score': rbf_grid.best_score_,
            'training_time': rbf_time
        }
        
        # Store results and find best SVM
        self.svm_results = svm_results
        best_kernel = max(svm_results.keys(), key=lambda k: svm_results[k]['cv_score'])
        self.best_svm = svm_results[best_kernel]['model']
        self.best_svm_kernel = best_kernel
        
        print(f"\nBest SVM kernel: {best_kernel}")
        print(f"Best parameters: {svm_results[best_kernel]['best_params']}")
        print(f"Best CV score: {svm_results[best_kernel]['cv_score']:.4f}")
        
        return svm_results
    
    def train_baseline_classifiers(self):
        """
        Train baseline classifiers (KNN, SGD, Random Forest) for comparison.
        """
        print("\n" + "="*60)
        print("TRAINING BASELINE CLASSIFIERS")
        print("="*60)
        
        baseline_results = {}
        
        # KNN Classifier
        print("\n1. Training K-Nearest Neighbors...")
        knn_params = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        }
        
        knn = KNeighborsClassifier()
        start_time = time.time()
        knn_grid = GridSearchCV(knn, knn_params, cv=3, 
                              scoring='accuracy', n_jobs=-1, verbose=1)
        knn_grid.fit(self.X_train, self.y_train)
        knn_time = time.time() - start_time
        
        baseline_results['KNN'] = {
            'model': knn_grid.best_estimator_,
            'best_params': knn_grid.best_params_,
            'cv_score': knn_grid.best_score_,
            'training_time': knn_time
        }
        
        # SGD Classifier
        print("\n2. Training Stochastic Gradient Descent...")
        sgd_params = {
            'alpha': [0.0001, 0.001, 0.01],
            'loss': ['hinge', 'log_loss'],
            'max_iter': [1000, 2000]
        }
        
        sgd = SGDClassifier(random_state=42)
        start_time = time.time()
        sgd_grid = GridSearchCV(sgd, sgd_params, cv=3, 
                              scoring='accuracy', n_jobs=-1, verbose=1)
        sgd_grid.fit(self.X_train, self.y_train)
        sgd_time = time.time() - start_time
        
        baseline_results['SGD'] = {
            'model': sgd_grid.best_estimator_,
            'best_params': sgd_grid.best_params_,
            'cv_score': sgd_grid.best_score_,
            'training_time': sgd_time
        }
        
        # Random Forest Classifier
        print("\n3. Training Random Forest...")
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        
        rf = RandomForestClassifier(random_state=42)
        start_time = time.time()
        rf_grid = GridSearchCV(rf, rf_params, cv=3, 
                             scoring='accuracy', n_jobs=-1, verbose=1)
        rf_grid.fit(self.X_train, self.y_train)
        rf_time = time.time() - start_time
        
        baseline_results['Random Forest'] = {
            'model': rf_grid.best_estimator_,
            'best_params': rf_grid.best_params_,
            'cv_score': rf_grid.best_score_,
            'training_time': rf_time
        }
        
        self.baseline_results = baseline_results
        return baseline_results
    
    def evaluate_all_classifiers(self):
        """
        Evaluate all trained classifiers on the test set.
        """
        print("\n" + "="*60)
        print("EVALUATING ALL CLASSIFIERS")
        print("="*60)
        
        all_results = {}
        
        # Evaluate SVM classifiers
        for kernel, svm_data in self.svm_results.items():
            model = svm_data['model']
            
            start_time = time.time()
            y_pred = model.predict(self.X_test)
            prediction_time = time.time() - start_time
            
            all_results[f'SVM_{kernel}'] = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred, average='weighted'),
                'recall': recall_score(self.y_test, y_pred, average='weighted'),
                'f1_score': f1_score(self.y_test, y_pred, average='weighted'),
                'training_time': svm_data['training_time'],
                'prediction_time': prediction_time,
                'best_params': svm_data['best_params']
            }
        
        # Evaluate baseline classifiers
        for name, classifier_data in self.baseline_results.items():
            model = classifier_data['model']
            
            start_time = time.time()
            y_pred = model.predict(self.X_test)
            prediction_time = time.time() - start_time
            
            all_results[name] = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred, average='weighted'),
                'recall': recall_score(self.y_test, y_pred, average='weighted'),
                'f1_score': f1_score(self.y_test, y_pred, average='weighted'),
                'training_time': classifier_data['training_time'],
                'prediction_time': prediction_time,
                'best_params': classifier_data['best_params']
            }
        
        self.results = all_results
        return all_results
    
    def print_results_summary(self):
        """
        Print a comprehensive summary of all classifier results.
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE RESULTS SUMMARY")
        print("="*80)
        
        # Create results DataFrame
        df_data = []
        for classifier, metrics in self.results.items():
            df_data.append({
                'Classifier': classifier,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'Training Time (s)': metrics['training_time'],
                'Prediction Time (s)': metrics['prediction_time']
            })
        
        df = pd.DataFrame(df_data)
        df = df.sort_values('Accuracy', ascending=False)
        
        print("\nPerformance Metrics:")
        print(df.to_string(index=False, float_format='%.4f'))
        
        # Best parameters summary
        print("\n" + "="*80)
        print("BEST HYPERPARAMETERS")
        print("="*80)
        for classifier, metrics in self.results.items():
            print(f"\n{classifier}:")
            print(f"  Best Parameters: {metrics['best_params']}")
        
        return df
    
    def create_visualizations(self):
        """
        Create comprehensive visualizations for the analysis.
        """
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # Extract data for plotting
        classifiers = list(self.results.keys())
        accuracies = [self.results[c]['accuracy'] for c in classifiers]
        precisions = [self.results[c]['precision'] for c in classifiers]
        recalls = [self.results[c]['recall'] for c in classifiers]
        f1_scores = [self.results[c]['f1_score'] for c in classifiers]
        training_times = [self.results[c]['training_time'] for c in classifiers]
        
        # 1. Accuracy Comparison
        plt.subplot(2, 3, 1)
        bars = plt.bar(classifiers, accuracies, color=sns.color_palette("husl", len(classifiers)))
        plt.title('Accuracy Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0.8, 1.0)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. All Metrics Comparison
        plt.subplot(2, 3, 2)
        x = np.arange(len(classifiers))
        width = 0.2
        
        plt.bar(x - 1.5*width, accuracies, width, label='Accuracy', alpha=0.8)
        plt.bar(x - 0.5*width, precisions, width, label='Precision', alpha=0.8)
        plt.bar(x + 0.5*width, recalls, width, label='Recall', alpha=0.8)
        plt.bar(x + 1.5*width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        plt.title('All Performance Metrics Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Score')
        plt.xticks(x, classifiers, rotation=45, ha='right')
        plt.legend()
        plt.ylim(0.8, 1.0)
        
        # 3. Training Time Comparison
        plt.subplot(2, 3, 3)
        bars = plt.bar(classifiers, training_times, color=sns.color_palette("viridis", len(classifiers)))
        plt.title('Training Time Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Training Time (seconds)')
        plt.xticks(rotation=45, ha='right')
        plt.yscale('log')  # Use log scale for better visualization
        
        # Add value labels on bars
        for bar, time_val in zip(bars, training_times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                    f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # 4. Performance vs Training Time Scatter Plot
        plt.subplot(2, 3, 4)
        colors = sns.color_palette("husl", len(classifiers))
        for i, (classifier, color) in enumerate(zip(classifiers, colors)):
            plt.scatter(training_times[i], accuracies[i], 
                       color=color, s=100, alpha=0.7, label=classifier)
        
        plt.title('Accuracy vs Training Time', fontsize=14, fontweight='bold')
        plt.xlabel('Training Time (seconds)')
        plt.ylabel('Accuracy')
        plt.xscale('log')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 5. SVM Kernels Detailed Comparison
        plt.subplot(2, 3, 5)
        svm_classifiers = [c for c in classifiers if c.startswith('SVM_')]
        svm_accuracies = [self.results[c]['accuracy'] for c in svm_classifiers]
        svm_kernels = [c.replace('SVM_', '') for c in svm_classifiers]
        
        bars = plt.bar(svm_kernels, svm_accuracies, color=['red', 'green', 'blue'])
        plt.title('SVM Kernels Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy')
        plt.ylim(0.8, 1.0)
        
        for bar, acc in zip(bars, svm_accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 6. Overall Rankings
        plt.subplot(2, 3, 6)
        # Create a ranking based on multiple criteria
        rankings = []
        for classifier in classifiers:
            score = (self.results[classifier]['accuracy'] * 0.4 + 
                    self.results[classifier]['f1_score'] * 0.4 + 
                    (1/np.log10(max(1, self.results[classifier]['training_time']))) * 0.2)
            rankings.append(score)
        
        sorted_indices = np.argsort(rankings)[::-1]
        sorted_classifiers = [classifiers[i] for i in sorted_indices]
        sorted_rankings = [rankings[i] for i in sorted_indices]
        
        bars = plt.barh(sorted_classifiers, sorted_rankings, color=sns.color_palette("coolwarm", len(classifiers)))
        plt.title('Overall Performance Ranking', fontsize=14, fontweight='bold')
        plt.xlabel('Composite Score (Accuracy + F1 + Time Efficiency)')
        
        plt.tight_layout()
        plt.savefig('mnist_classifier_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved as 'mnist_classifier_comparison.png'")
        
        return fig
    
    def run_complete_analysis(self, sample_size=10000):
        """
        Run the complete analysis pipeline.
        
        Args:
            sample_size (int): Number of training samples to use
        """
        print("STARTING COMPREHENSIVE MNIST CLASSIFIER ANALYSIS")
        print("="*80)
        
        # Load and prepare data
        self.load_and_prepare_data(sample_size)
        
        # Train all classifiers
        self.train_svm_with_grid_search()
        self.train_baseline_classifiers()
        
        # Evaluate all classifiers
        self.evaluate_all_classifiers()
        
        # Print results and create visualizations
        results_df = self.print_results_summary()
        self.create_visualizations()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        
        return results_df


def main():
    """
    Main function to run the complete analysis.
    """
    # Create classifier comparison instance
    classifier_comparison = MNISTClassifierComparison()
    
    # Run complete analysis with a manageable sample size
    # Adjust sample_size based on your computational resources
    results_df = classifier_comparison.run_complete_analysis(sample_size=8000)
    
    # Save results to CSV for further analysis
    results_df.to_csv('mnist_classifier_results.csv', index=False)
    print("\nResults saved to 'mnist_classifier_results.csv'")
    
    return classifier_comparison, results_df


if __name__ == "__main__":
    classifier_comparison, results_df = main()
