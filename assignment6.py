#!/usr/bin/env python3
"""
Assignment 6: Unsupervised Learning Using K-Means Clustering on California Housing Data

This script implements K-Means clustering analysis on the California Housing dataset,
focusing on longitude, latitude, and median_income features for housing market segmentation.

Authors: Abhishek Kumar, Aniruddha Biswas, Dewan Niaz Morshed
University West, Sweden
Date: October 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CaliforniaHousingClusterAnalysis:
    """
    Comprehensive clustering analysis for California Housing dataset
    """
    
    def __init__(self):
        self.housing_data = None
        self.features = None
        self.scaled_features = None
        self.scaler = StandardScaler()
        self.clustering_results = {}
        self.silhouette_scores = {}
        
    def load_and_prepare_data(self):
        """Load and prepare California Housing data for clustering"""
        print("🏠 Loading California Housing Dataset...")
        
        # Load the dataset
        housing = fetch_california_housing()
        
        # Create DataFrame
        self.housing_data = pd.DataFrame(housing.data, columns=housing.feature_names)
        self.housing_data['target'] = housing.target
        
        # Select features for clustering: longitude, latitude, median_income
        feature_columns = ['Longitude', 'Latitude', 'MedInc']
        self.features = self.housing_data[feature_columns].copy()
        
        # Scale the features
        self.scaled_features = self.scaler.fit_transform(self.features)
        
        print(f"📊 Dataset shape: {self.housing_data.shape}")
        print(f"🎯 Selected features for clustering: {feature_columns}")
        print(f"📈 Feature statistics:")
        print(self.features.describe())
        
        return self.features
    
    def find_optimal_clusters_silhouette(self, k_range=(2, 15)):
        """Find optimal number of clusters using silhouette score"""
        print(f"\n🔍 Finding optimal clusters using silhouette score (k={k_range[0]} to {k_range[1]})...")
        
        silhouette_scores = []
        k_values = range(k_range[0], k_range[1] + 1)
        
        for k in k_values:
            print(f"   Testing k={k}...", end=' ')
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.scaled_features)
            silhouette_avg = silhouette_score(self.scaled_features, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            self.silhouette_scores[k] = silhouette_avg
            print(f"Silhouette Score: {silhouette_avg:.4f}")
        
        # Find optimal k
        optimal_k = k_values[np.argmax(silhouette_scores)]
        max_score = max(silhouette_scores)
        
        print(f"\n✅ Optimal number of clusters: {optimal_k} (Silhouette Score: {max_score:.4f})")
        
        return optimal_k, silhouette_scores, k_values
    
    def perform_kmeans_clustering(self, k_values=[3, 4, 5, 6, 7]):
        """Perform K-Means clustering with different k values"""
        print(f"\n🤖 Performing K-Means clustering for k values: {k_values}")
        
        for k in k_values:
            print(f"   Clustering with k={k}...")
            
            # Perform K-Means clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.scaled_features)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(self.scaled_features, cluster_labels)
            
            # Store results
            self.clustering_results[f'kmeans_k{k}'] = {
                'model': kmeans,
                'labels': cluster_labels,
                'silhouette_score': silhouette_avg,
                'cluster_centers': kmeans.cluster_centers_,
                'inertia': kmeans.inertia_
            }
            
            print(f"      Silhouette Score: {silhouette_avg:.4f}")
            print(f"      Inertia: {kmeans.inertia_:.2f}")
            
            # Add cluster labels to features for analysis
            self.features[f'cluster_k{k}'] = cluster_labels
    
    def perform_dbscan_clustering(self, eps_values=[0.3, 0.5, 0.7], min_samples=10):
        """Perform DBSCAN clustering for comparison"""
        print(f"\n🔄 Performing DBSCAN clustering...")
        print(f"   Testing eps values: {eps_values}, min_samples: {min_samples}")
        
        best_eps = None
        best_score = -1
        
        for eps in eps_values:
            print(f"   DBSCAN with eps={eps}...")
            
            # Perform DBSCAN clustering
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(self.scaled_features)
            
            # Check if we have valid clusters (more than one cluster, not all noise)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            
            if n_clusters > 1:
                silhouette_avg = silhouette_score(self.scaled_features, cluster_labels)
                
                self.clustering_results[f'dbscan_eps{eps}'] = {
                    'model': dbscan,
                    'labels': cluster_labels,
                    'silhouette_score': silhouette_avg,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise
                }
                
                print(f"      Clusters: {n_clusters}, Noise points: {n_noise}")
                print(f"      Silhouette Score: {silhouette_avg:.4f}")
                
                if silhouette_avg > best_score:
                    best_score = silhouette_avg
                    best_eps = eps
                    
                # Add cluster labels to features for analysis
                self.features[f'dbscan_eps{eps}'] = cluster_labels
            else:
                print(f"      Invalid clustering (clusters: {n_clusters}, noise: {n_noise})")
        
        if best_eps:
            print(f"\n✅ Best DBSCAN configuration: eps={best_eps} (Silhouette Score: {best_score:.4f})")
    
    def analyze_cluster_characteristics(self, cluster_method='kmeans_k5'):
        """Analyze characteristics of clusters"""
        print(f"\n📊 Analyzing cluster characteristics for {cluster_method}...")
        
        if cluster_method not in self.clustering_results:
            print(f"❌ Clustering method {cluster_method} not found!")
            return
        
        labels = self.clustering_results[cluster_method]['labels']
        
        # Add cluster labels to housing data for analysis
        analysis_data = self.housing_data.copy()
        analysis_data['cluster'] = labels
        
        # Calculate cluster statistics
        cluster_stats = analysis_data.groupby('cluster').agg({
            'Longitude': ['mean', 'std', 'count'],
            'Latitude': ['mean', 'std'],
            'MedInc': ['mean', 'std'],
            'target': ['mean', 'std']  # Median house value
        }).round(4)
        
        print("🎯 Cluster Statistics:")
        print(cluster_stats)
        
        # Calculate cluster sizes
        cluster_sizes = analysis_data['cluster'].value_counts().sort_index()
        print(f"\n📈 Cluster Sizes:")
        for cluster_id, size in cluster_sizes.items():
            percentage = (size / len(analysis_data)) * 100
            print(f"   Cluster {cluster_id}: {size} points ({percentage:.1f}%)")
        
        return cluster_stats, cluster_sizes
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print(f"\n🎨 Creating visualizations...")
        
        # 1. Silhouette Score Analysis
        plt.figure(figsize=(15, 12))
        
        # Plot 1: Silhouette scores for different k values
        plt.subplot(2, 3, 1)
        k_values = list(self.silhouette_scores.keys())
        scores = list(self.silhouette_scores.values())
        plt.plot(k_values, scores, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score vs Number of Clusters')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: K-Means clustering results (k=5)
        plt.subplot(2, 3, 2)
        if 'kmeans_k5' in self.clustering_results:
            labels = self.clustering_results['kmeans_k5']['labels']
            scatter = plt.scatter(self.features['Longitude'], self.features['Latitude'], 
                                c=labels, cmap='viridis', alpha=0.6, s=10)
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title('K-Means Clustering (k=5)\nGeographical Distribution')
            plt.colorbar(scatter)
        
        # Plot 3: Median Income distribution by clusters
        plt.subplot(2, 3, 3)
        if 'kmeans_k5' in self.clustering_results:
            labels = self.clustering_results['kmeans_k5']['labels']
            for i in range(5):
                cluster_data = self.features[labels == i]['MedInc']
                plt.hist(cluster_data, alpha=0.7, label=f'Cluster {i}', bins=20)
            plt.xlabel('Median Income')
            plt.ylabel('Frequency')
            plt.title('Median Income Distribution by Cluster')
            plt.legend()
        
        # Plot 4: DBSCAN results
        plt.subplot(2, 3, 4)
        if 'dbscan_eps0.5' in self.clustering_results:
            labels = self.clustering_results['dbscan_eps0.5']['labels']
            scatter = plt.scatter(self.features['Longitude'], self.features['Latitude'], 
                                c=labels, cmap='viridis', alpha=0.6, s=10)
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title('DBSCAN Clustering (eps=0.5)\nGeographical Distribution')
            plt.colorbar(scatter)
        
        # Plot 5: Cluster comparison (K-Means vs DBSCAN)
        plt.subplot(2, 3, 5)
        methods = []
        scores = []
        for method, results in self.clustering_results.items():
            methods.append(method)
            scores.append(results['silhouette_score'])
        
        plt.bar(range(len(methods)), scores, alpha=0.7)
        plt.xlabel('Clustering Method')
        plt.ylabel('Silhouette Score')
        plt.title('Clustering Method Comparison')
        plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
        
        # Plot 6: 3D scatter plot (if possible)
        plt.subplot(2, 3, 6)
        if 'kmeans_k5' in self.clustering_results:
            labels = self.clustering_results['kmeans_k5']['labels']
            scatter = plt.scatter(self.features['MedInc'], self.features['Latitude'], 
                                c=labels, cmap='viridis', alpha=0.6, s=10)
            plt.xlabel('Median Income')
            plt.ylabel('Latitude')
            plt.title('Income vs Latitude\nby Cluster')
            plt.colorbar(scatter)
        
        plt.tight_layout()
        plt.savefig('assignment6_clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create individual plots for better quality
        self._create_individual_plots()
    
    def _create_individual_plots(self):
        """Create individual high-quality plots"""
        
        # 1. Silhouette Analysis
        plt.figure(figsize=(10, 6))
        k_values = list(self.silhouette_scores.keys())
        scores = list(self.silhouette_scores.values())
        plt.plot(k_values, scores, 'bo-', linewidth=3, markersize=10)
        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel('Silhouette Score', fontsize=12)
        plt.title('Silhouette Score Analysis for K-Means Clustering', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        optimal_k = k_values[np.argmax(scores)]
        plt.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
        plt.legend()
        plt.savefig('1_silhouette_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Geographical Clustering
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # K-Means geographical plot
        if 'kmeans_k5' in self.clustering_results:
            labels = self.clustering_results['kmeans_k5']['labels']
            scatter1 = axes[0].scatter(self.features['Longitude'], self.features['Latitude'], 
                                     c=labels, cmap='viridis', alpha=0.6, s=15)
            axes[0].set_xlabel('Longitude', fontsize=12)
            axes[0].set_ylabel('Latitude', fontsize=12)
            axes[0].set_title('K-Means Clustering (k=5)\nGeographical Distribution', fontsize=14, fontweight='bold')
            plt.colorbar(scatter1, ax=axes[0])
        
        # DBSCAN geographical plot
        if 'dbscan_eps0.5' in self.clustering_results:
            labels = self.clustering_results['dbscan_eps0.5']['labels']
            scatter2 = axes[1].scatter(self.features['Longitude'], self.features['Latitude'], 
                                     c=labels, cmap='viridis', alpha=0.6, s=15)
            axes[1].set_xlabel('Longitude', fontsize=12)
            axes[1].set_ylabel('Latitude', fontsize=12)
            axes[1].set_title('DBSCAN Clustering (eps=0.5)\nGeographical Distribution', fontsize=14, fontweight='bold')
            plt.colorbar(scatter2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig('2_geographical_clustering.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Feature Distribution Analysis
        plt.figure(figsize=(15, 5))
        
        if 'kmeans_k5' in self.clustering_results:
            labels = self.clustering_results['kmeans_k5']['labels']
            
            # Median Income by cluster
            plt.subplot(1, 3, 1)
            for i in range(5):
                cluster_data = self.features[labels == i]['MedInc']
                plt.hist(cluster_data, alpha=0.7, label=f'Cluster {i}', bins=25)
            plt.xlabel('Median Income (10k USD)', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title('Median Income Distribution\nby Cluster', fontsize=14, fontweight='bold')
            plt.legend()
            
            # Longitude distribution
            plt.subplot(1, 3, 2)
            for i in range(5):
                cluster_data = self.features[labels == i]['Longitude']
                plt.hist(cluster_data, alpha=0.7, label=f'Cluster {i}', bins=25)
            plt.xlabel('Longitude', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title('Longitude Distribution\nby Cluster', fontsize=14, fontweight='bold')
            plt.legend()
            
            # Latitude distribution
            plt.subplot(1, 3, 3)
            for i in range(5):
                cluster_data = self.features[labels == i]['Latitude']
                plt.hist(cluster_data, alpha=0.7, label=f'Cluster {i}', bins=25)
            plt.xlabel('Latitude', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title('Latitude Distribution\nby Cluster', fontsize=14, fontweight='bold')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('3_feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Method Comparison
        plt.figure(figsize=(12, 6))
        
        methods = []
        scores = []
        colors = []
        
        for method, results in self.clustering_results.items():
            methods.append(method.replace('_', '\n'))
            scores.append(results['silhouette_score'])
            if 'kmeans' in method:
                colors.append('skyblue')
            else:
                colors.append('lightcoral')
        
        bars = plt.bar(range(len(methods)), scores, color=colors, alpha=0.8, edgecolor='black')
        plt.xlabel('Clustering Method', fontsize=12)
        plt.ylabel('Silhouette Score', fontsize=12)
        plt.title('Clustering Algorithm Performance Comparison', fontsize=14, fontweight='bold')
        plt.xticks(range(len(methods)), methods, fontsize=10)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='skyblue', label='K-Means'),
                          Patch(facecolor='lightcoral', label='DBSCAN')]
        plt.legend(handles=legend_elements)
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('4_method_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self):
        """Generate summary report of clustering analysis"""
        print(f"\n📋 CLUSTERING ANALYSIS SUMMARY REPORT")
        print("=" * 60)
        
        print(f"\n📊 Dataset Information:")
        print(f"   • Total samples: {len(self.housing_data):,}")
        print(f"   • Features used: Longitude, Latitude, Median Income")
        print(f"   • Feature scaling: StandardScaler applied")
        
        print(f"\n🎯 Optimal Cluster Analysis:")
        if self.silhouette_scores:
            best_k = max(self.silhouette_scores.keys(), key=lambda k: self.silhouette_scores[k])
            best_score = self.silhouette_scores[best_k]
            print(f"   • Optimal K-Means clusters: {best_k}")
            print(f"   • Best silhouette score: {best_score:.4f}")
        
        print(f"\n🤖 K-Means Results:")
        for method, results in self.clustering_results.items():
            if 'kmeans' in method:
                k = method.split('k')[1]
                print(f"   • k={k}: Silhouette Score = {results['silhouette_score']:.4f}, "
                      f"Inertia = {results['inertia']:.2f}")
        
        print(f"\n🔄 DBSCAN Results:")
        for method, results in self.clustering_results.items():
            if 'dbscan' in method:
                eps = method.split('eps')[1]
                print(f"   • eps={eps}: Silhouette Score = {results['silhouette_score']:.4f}, "
                      f"Clusters = {results['n_clusters']}, Noise = {results['n_noise']}")
        
        print(f"\n🏆 Best Performing Algorithm:")
        if self.clustering_results:
            best_method = max(self.clustering_results.keys(), 
                            key=lambda k: self.clustering_results[k]['silhouette_score'])
            best_result = self.clustering_results[best_method]
            print(f"   • Method: {best_method}")
            print(f"   • Silhouette Score: {best_result['silhouette_score']:.4f}")
        
        print(f"\n📈 Key Insights:")
        print(f"   • Geographical clustering reveals distinct housing market regions")
        print(f"   • Median income shows clear segmentation across clusters")
        print(f"   • K-Means provides interpretable cluster boundaries")
        print(f"   • DBSCAN identifies outliers and noise in housing data")
        
        print("\n" + "=" * 60)


def main():
    """Main execution function"""
    print("🎯 Assignment 6: K-Means Clustering on California Housing Data")
    print("=" * 70)
    
    # Initialize the analysis
    analyzer = CaliforniaHousingClusterAnalysis()
    
    # Step 1: Load and prepare data
    analyzer.load_and_prepare_data()
    
    # Step 2: Find optimal number of clusters
    optimal_k, silhouette_scores, k_values = analyzer.find_optimal_clusters_silhouette()
    
    # Step 3: Perform K-Means clustering with different k values
    analyzer.perform_kmeans_clustering([3, 4, 5, 6, 7])
    
    # Step 4: Perform DBSCAN clustering for comparison
    analyzer.perform_dbscan_clustering()
    
    # Step 5: Analyze cluster characteristics
    analyzer.analyze_cluster_characteristics('kmeans_k5')
    
    # Step 6: Create visualizations
    analyzer.create_visualizations()
    
    # Step 7: Generate summary report
    analyzer.generate_summary_report()
    
    print(f"\n✅ Assignment 6 analysis completed successfully!")
    print(f"📁 Check the current directory for generated visualization files")


if __name__ == "__main__":
    main()
