"""
K-Means Clustering Implementation on MNIST Dataset
CS 8050 - Assignment 5
Group 6
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_score, accuracy_score
from sklearn.decomposition import PCA
import os

class KMeansAnalyzer:
    """
    K-Means clustering analyzer for MNIST dataset with accuracy optimization
    """
    
    def __init__(self):
        self.results = {}
        self.cluster_to_label_mapping = {}
        
    def load_mnist_data(self, n_samples=10000):
        """Load and preprocess MNIST dataset"""
        print("Loading MNIST dataset...")
        
        # Load MNIST data
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X = mnist.data.astype('float32')
        y = mnist.target.astype(int)
        
        # Normalize data to [0, 1]
        X = X / 255.0
        
        # Use subset for faster computation
        if n_samples is not None and n_samples < len(X):
            indices = np.random.choice(len(X), n_samples, replace=False)
            X = X[indices]
            y = y[indices]
        
        print(f"Dataset size: {len(X)} samples, {X.shape[1]} features")
        print(f"Data range: [{X.min():.3f}, {X.max():.3f}]\n")
        
        return X, y
    
    def retrieve_info(self, cluster_labels, y_train):
        """
        Associates most probable label with each cluster in KMeans model
        Returns: dictionary of clusters assigned to each label
        """
        reference_labels = {}
        
        for i in range(len(np.unique(cluster_labels))):
            index = np.where(cluster_labels == i, 1, 0)
            num = np.bincount(y_train[index == 1]).argmax()
            reference_labels[i] = num
        
        return reference_labels
    
    def perform_kmeans(self, X, n_clusters=256, random_state=42):
        """
        Perform K-Means clustering with optimal cluster count
        
        Time Complexity: O(n * k * d * iterations)
        Space Complexity: O(n*d + k*d)
        """
        print(f"Performing K-Means with k={n_clusters}...")
        
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=random_state,
            verbose=0
        )
        
        start_time = time.time()
        cluster_labels = kmeans.fit_predict(X)
        training_time = time.time() - start_time
        
        inertia = kmeans.inertia_
        centroids = kmeans.cluster_centers_
        
        print(f"✓ Training completed in {training_time:.3f}s")
        print(f"✓ Converged in {kmeans.n_iter_} iterations")
        print(f"✓ Final inertia: {inertia:.2f}\n")
        
        return kmeans, cluster_labels, centroids, inertia, training_time
    
    def calculate_accuracy(self, kmeans, X, y_true):
        """Calculate clustering accuracy with optimal cluster-to-label mapping"""
        cluster_labels = kmeans.labels_
        
        # Get mapping from clusters to labels
        reference_labels = self.retrieve_info(cluster_labels, y_true)
        self.cluster_to_label_mapping = reference_labels
        
        # Map cluster labels to digit labels
        number_labels = np.array([reference_labels[label] for label in cluster_labels])
        
        # Calculate accuracy
        accuracy = accuracy_score(number_labels, y_true)
        
        return accuracy, number_labels
    
    def evaluate_clustering(self, kmeans, X, y_true):
        """Evaluate clustering performance"""
        print("Evaluating clustering quality...")
        
        cluster_labels = kmeans.labels_
        
        # Silhouette Score (sample for speed)
        sample_size = min(5000, len(X))
        if len(X) > 5000:
            sample_idx = np.random.choice(len(X), sample_size, replace=False)
            silhouette = silhouette_score(X[sample_idx], cluster_labels[sample_idx])
        else:
            silhouette = silhouette_score(X, cluster_labels)
        
        # Adjusted Rand Index
        ari = adjusted_rand_score(y_true, cluster_labels)
        
        # Homogeneity
        homogeneity = homogeneity_score(y_true, cluster_labels)
        
        # Clustering Accuracy
        accuracy, number_labels = self.calculate_accuracy(kmeans, X, y_true)
        
        print(f"✓ Silhouette Score: {silhouette:.4f}")
        print(f"✓ Adjusted Rand Index: {ari:.4f}")
        print(f"✓ Homogeneity Score: {homogeneity:.4f}")
        print(f"✓ Clustering Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        
        return silhouette, ari, homogeneity, accuracy
    
    def optimize_cluster_count(self, X, y, cluster_range=[10, 16, 36, 64, 144, 256]):
        """
        Test different cluster counts to find optimal value
        Key insight: More clusters = better accuracy for handwritten digits
        """
        print("=" * 70)
        print("Optimizing Cluster Count")
        print("=" * 70)
        
        results = []
        
        for n_clusters in cluster_range:
            print(f"\nTesting k={n_clusters}...")
            
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', 
                                    n_init=10, max_iter=300, random_state=42)
            
            start_time = time.time()
            kmeans.fit(X)
            train_time = time.time() - start_time
            
            # Calculate metrics
            inertia = kmeans.inertia_
            homogeneity = homogeneity_score(y, kmeans.labels_)
            accuracy, _ = self.calculate_accuracy(kmeans, X, y)
            
            results.append({
                'n_clusters': n_clusters,
                'inertia': inertia,
                'homogeneity': homogeneity,
                'accuracy': accuracy,
                'train_time': train_time
            })
            
            print(f"  Inertia: {inertia:.2f}")
            print(f"  Homogeneity: {homogeneity:.4f}")
            print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  Time: {train_time:.3f}s")
        
        self.results['optimization'] = results
        
        # Find best
        best = max(results, key=lambda x: x['accuracy'])
        print(f"\n✓ Best accuracy: {best['accuracy']:.4f} with k={best['n_clusters']}")
        
        return results
    
    def full_analysis(self, X, y, optimal_k=256):
        """Perform complete K-Means analysis with optimal k"""
        print("\n" + "=" * 70)
        print(f"Full K-Means Analysis (k={optimal_k})")
        print("=" * 70)
        
        # Perform clustering
        kmeans, cluster_labels, centroids, inertia, train_time = self.perform_kmeans(
            X, n_clusters=optimal_k
        )
        
        # Evaluate clustering
        silhouette, ari, homogeneity, accuracy = self.evaluate_clustering(kmeans, X, y)
        
        # Analyze cluster distribution
        unique, counts = np.unique(cluster_labels, return_counts=True)
        cluster_sizes = dict(zip(unique, counts))
        
        print("Cluster Size Distribution (first 20):")
        for i, cluster_id in enumerate(sorted(cluster_sizes.keys())[:20]):
            size = cluster_sizes[cluster_id]
            percentage = (size / len(X)) * 100
            digit = self.cluster_to_label_mapping.get(cluster_id, '?')
            print(f"  Cluster {cluster_id:3d} → Digit {digit}: {size:4d} samples ({percentage:5.2f}%)")
        
        # Count clusters per digit
        digit_cluster_count = {}
        for cluster_id, digit in self.cluster_to_label_mapping.items():
            digit_cluster_count[digit] = digit_cluster_count.get(digit, 0) + 1
        
        print("\nClusters per Digit:")
        for digit in sorted(digit_cluster_count.keys()):
            print(f"  Digit {digit}: {digit_cluster_count[digit]} clusters")
        
        # Store results
        self.results['full_analysis'] = {
            'kmeans': kmeans,
            'cluster_labels': cluster_labels,
            'centroids': centroids,
            'inertia': inertia,
            'train_time': train_time,
            'silhouette': silhouette,
            'ari': ari,
            'homogeneity': homogeneity,
            'accuracy': accuracy,
            'cluster_sizes': cluster_sizes,
            'y_true': y
        }
        
        print("\n✓ Analysis complete!\n")
        return kmeans, cluster_labels, centroids
    
    def visualize_results(self, X):
        """Create comprehensive visualizations"""
        print("=" * 70)
        print("Generating Visualizations")
        print("=" * 70)
        
        os.makedirs('images', exist_ok=True)
        
        fig = plt.figure(figsize=(18, 12))
        
        # Plot 1: Cluster Count Optimization
        if 'optimization' in self.results:
            ax1 = plt.subplot(2, 3, 1)
            data = self.results['optimization']
            k_vals = [d['n_clusters'] for d in data]
            accuracies = [d['accuracy'] * 100 for d in data]
            
            ax1.plot(k_vals, accuracies, 'o-', linewidth=2, markersize=8, color='#2E86AB')
            ax1.set_xlabel('Number of Clusters (k)', fontsize=11)
            ax1.set_ylabel('Accuracy (%)', fontsize=11)
            ax1.set_title('Accuracy vs Number of Clusters\n(More clusters = Higher accuracy)', 
                         fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.set_xticks(k_vals)
            
            # Highlight best
            best_idx = accuracies.index(max(accuracies))
            ax1.plot(k_vals[best_idx], accuracies[best_idx], 'r*', markersize=20)
        
        # Plot 2: Time vs Cluster Count
        if 'optimization' in self.results:
            ax2 = plt.subplot(2, 3, 2)
            data = self.results['optimization']
            k_vals = [d['n_clusters'] for d in data]
            times = [d['train_time'] for d in data]
            
            ax2.plot(k_vals, times, 's-', linewidth=2, markersize=8, color='#E63946')
            ax2.set_xlabel('Number of Clusters (k)', fontsize=11)
            ax2.set_ylabel('Training Time (s)', fontsize=11)
            ax2.set_title('Time Complexity: O(n × k × d × i)', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.set_xticks(k_vals)
        
        # Plot 3: Inertia vs Cluster Count
        if 'optimization' in self.results:
            ax3 = plt.subplot(2, 3, 3)
            data = self.results['optimization']
            k_vals = [d['n_clusters'] for d in data]
            inertias = [d['inertia'] for d in data]
            
            ax3.plot(k_vals, inertias, '^-', linewidth=2, markersize=8, color='#F18F01')
            ax3.set_xlabel('Number of Clusters (k)', fontsize=11)
            ax3.set_ylabel('Inertia', fontsize=11)
            ax3.set_title('Inertia vs Clusters\n(Lower is better)', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.set_xticks(k_vals)
        
        # Plot 4: Performance Metrics
        if 'full_analysis' in self.results:
            ax4 = plt.subplot(2, 3, 4)
            data = self.results['full_analysis']
            
            metrics = ['Silhouette', 'ARI', 'Homogeneity', 'Accuracy']
            values = [data['silhouette'], data['ari'], data['homogeneity'], data['accuracy']]
            colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
            
            bars = ax4.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')
            ax4.set_ylabel('Score', fontsize=11)
            ax4.set_title(f'Clustering Quality Metrics\n(k={len(self.cluster_to_label_mapping)})', 
                         fontsize=12, fontweight='bold')
            ax4.set_ylim([0, 1])
            ax4.grid(True, alpha=0.3, axis='y')
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Plot 5: Clusters per Digit Distribution
        if 'full_analysis' in self.results:
            ax5 = plt.subplot(2, 3, 5)
            
            # Count clusters per digit
            digit_cluster_count = {}
            for cluster_id, digit in self.cluster_to_label_mapping.items():
                digit_cluster_count[digit] = digit_cluster_count.get(digit, 0) + 1
            
            digits = sorted(digit_cluster_count.keys())
            counts = [digit_cluster_count[d] for d in digits]
            
            bars = ax5.bar(digits, counts, color='#9D4EDD', alpha=0.8, edgecolor='black')
            ax5.set_xlabel('Digit Class', fontsize=11)
            ax5.set_ylabel('Number of Clusters', fontsize=11)
            ax5.set_title('Clusters per Digit\n(Different writing styles)', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='y')
            ax5.set_xticks(digits)
            
            for bar in bars:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        # Plot 6: 2D PCA Visualization
        if 'full_analysis' in self.results:
            ax6 = plt.subplot(2, 3, 6)
            data = self.results['full_analysis']
            
            print("Computing PCA for visualization...")
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X)
            
            # Use true labels for coloring
            scatter = ax6.scatter(X_pca[:, 0], X_pca[:, 1], 
                                c=data['y_true'], 
                                cmap='tab10', alpha=0.5, s=10, edgecolors='none')
            ax6.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
            ax6.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
            ax6.set_title('True Labels in 2D PCA Space', fontsize=12, fontweight='bold')
            plt.colorbar(scatter, ax=ax6, label='Digit')
        
        plt.tight_layout()
        plt.savefig('images/kmeans_analysis.png', dpi=150, bbox_inches='tight')
        print("✓ Main analysis saved as 'images/kmeans_analysis.png'")
        
        # Create separate figure for centroids
        self._visualize_centroids()
        
        print("✓ All visualizations complete!\n")
        
        return fig
    
    def _visualize_centroids(self):
        """Visualize cluster centroids as digit images"""
        if 'full_analysis' not in self.results:
            return
        
        print("Generating centroid visualization...")
        
        centroids = self.results['full_analysis']['centroids']
        n_clusters = min(30, len(centroids))  # Show first 30 centroids
        
        n_cols = 6
        n_rows = (n_clusters + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 2.5 * n_rows))
        axes = axes.flatten()
        
        for i in range(len(axes)):
            if i < n_clusters:
                centroid_img = centroids[i].reshape(28, 28)
                digit = self.cluster_to_label_mapping.get(i, '?')
                axes[i].imshow(centroid_img, cmap='gray', interpolation='nearest')
                axes[i].set_title(f'C{i}→{digit}', fontsize=10, fontweight='bold')
                axes[i].axis('off')
            else:
                axes[i].axis('off')
        
        plt.suptitle(f'K-Means Cluster Centroids (First {n_clusters} of {len(centroids)})', 
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig('images/kmeans_centroids.png', dpi=150, bbox_inches='tight')
        print("✓ Centroids saved as 'images/kmeans_centroids.png'")
        plt.close()


def main():
    """Main execution function"""
    print("\n" + "=" * 70)
    print("K-MEANS CLUSTERING ON MNIST - OPTIMIZED FOR HIGH ACCURACY")
    print("=" * 70)
    print("\nKey Insight: Use many clusters (256) to capture different")
    print("writing styles and orientations of the same digit!\n")
    
    np.random.seed(42)
    
    analyzer = KMeansAnalyzer()
    
    # Load MNIST data
    X, y = analyzer.load_mnist_data(n_samples=10000)
    
    # Optimize cluster count
    analyzer.optimize_cluster_count(X, y, cluster_range=[10, 16, 36, 64, 144, 256])
    
    # Perform full analysis with optimal k=256
    kmeans, cluster_labels, centroids = analyzer.full_analysis(X, y, optimal_k=256)
    
    # Generate visualizations
    analyzer.visualize_results(X)
    
    print("=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nFinal Accuracy: {analyzer.results['full_analysis']['accuracy']*100:.2f}%")
    print("\nOutput files:")
    print("  • images/kmeans_analysis.png - Main analysis dashboard")
    print("  • images/kmeans_centroids.png - Cluster centroid visualizations")
    print()
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()
