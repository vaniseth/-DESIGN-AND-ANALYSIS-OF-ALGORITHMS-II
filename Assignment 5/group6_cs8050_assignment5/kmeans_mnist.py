"""
K-Means Clustering Implementation on MNIST Dataset
Simplified version with essential visualizations
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_score
from sklearn.decomposition import PCA
import os

class KMeansAnalyzer:
    """
    Simplified K-Means clustering analyzer for MNIST dataset
    """
    
    def __init__(self):
        self.results = {}
        
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
    
    def perform_kmeans(self, X, n_clusters=10, random_state=42):
        """
        Perform K-Means clustering
        
        Time Complexity: O(n * k * d * iterations)
        - n: number of samples
        - k: number of clusters
        - d: number of features
        - iterations: number of iterations until convergence
        
        Space Complexity: O(n*d + k*d)
        """
        print(f"Performing K-Means with k={n_clusters}...")
        
        kmeans = KMeans(
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
        
        return cluster_labels, centroids, inertia, training_time
    
    def evaluate_clustering(self, X, y_true, cluster_labels):
        """Evaluate clustering performance"""
        print("Evaluating clustering quality...")
        
        # Silhouette Score: measures cluster cohesion and separation
        silhouette = silhouette_score(X, cluster_labels, sample_size=5000)
        
        # Adjusted Rand Index: measures similarity with ground truth
        ari = adjusted_rand_score(y_true, cluster_labels)
        
        # Homogeneity: measures if clusters contain only single class
        homogeneity = homogeneity_score(y_true, cluster_labels)
        
        # Calculate Clustering Accuracy by mapping clusters to true labels
        accuracy = self._calculate_clustering_accuracy(y_true, cluster_labels)
        
        print(f"✓ Silhouette Score: {silhouette:.4f} (range: [-1, 1], higher is better)")
        print(f"✓ Adjusted Rand Index: {ari:.4f} (range: [0, 1], higher is better)")
        print(f"✓ Homogeneity Score: {homogeneity:.4f} (range: [0, 1], higher is better)")
        print(f"✓ Clustering Accuracy: {accuracy:.4f} (range: [0, 1], higher is better)\n")
        
        return silhouette, ari, homogeneity, accuracy
    
    def _calculate_clustering_accuracy(self, y_true, cluster_labels):
        """
        Calculate clustering accuracy by finding the best mapping between
        clusters and true labels using the Hungarian algorithm approach.
        """
        from scipy.optimize import linear_sum_assignment
        
        # Create confusion matrix
        n_clusters = len(np.unique(cluster_labels))
        n_classes = len(np.unique(y_true))
        
        # Count matrix: rows=clusters, cols=true_labels
        confusion_matrix = np.zeros((n_clusters, n_classes), dtype=np.int64)
        
        for cluster_id in range(n_clusters):
            cluster_mask = (cluster_labels == cluster_id)
            for class_id in range(n_classes):
                confusion_matrix[cluster_id, class_id] = np.sum(
                    (y_true[cluster_mask] == class_id)
                )
        
        # Use Hungarian algorithm to find optimal mapping
        # We want to maximize, so negate the matrix
        row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
        
        # Calculate accuracy based on optimal mapping
        correct = confusion_matrix[row_ind, col_ind].sum()
        accuracy = correct / len(y_true)
        
        # Store the mapping for later use
        self.cluster_to_label_mapping = dict(zip(row_ind, col_ind))
        
        return accuracy
    
    def elbow_analysis(self, X, k_range=range(2, 15)):
        """
        Perform elbow method analysis to find optimal k
        """
        print("=" * 70)
        print("Elbow Method Analysis (Finding Optimal k)")
        print("=" * 70)
        
        inertias = []
        silhouette_scores = []
        training_times = []
        k_values = list(k_range)
        
        for k in k_values: 
            print(f"\nTesting k={k}...")
            
            start_time = time.time()
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, 
                          max_iter=300, random_state=42)
            labels = kmeans.fit_predict(X)
            train_time = time.time() - start_time
            
            inertia = kmeans.inertia_
            inertias.append(inertia)
            training_times.append(train_time)
            
            # Calculate silhouette score (use sample for speed)
            if len(X) > 5000:
                sample_idx = np.random.choice(len(X), 5000, replace=False)
                silhouette = silhouette_score(X[sample_idx], labels[sample_idx])
            else:
                silhouette = silhouette_score(X, labels)
            
            silhouette_scores.append(silhouette)
            
            print(f"  Inertia: {inertia:.2f}, Silhouette: {silhouette:.4f}, Time: {train_time:.3f}s")
        
        self.results['elbow'] = {
            'k_values': k_values,
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'training_times': training_times
        }
        
        print("\n✓ Elbow analysis complete!\n")
        return k_values, inertias, silhouette_scores
    
    def full_analysis(self, X, y, optimal_k=10):
        """Perform complete K-Means analysis with optimal k"""
        print("=" * 70)
        print(f"Full K-Means Analysis (k={optimal_k})")
        print("=" * 70)
        
        # Perform clustering
        cluster_labels, centroids, inertia, train_time = self.perform_kmeans(X, n_clusters=optimal_k)
        
        # Evaluate clustering
        silhouette, ari, homogeneity, accuracy = self.evaluate_clustering(X, y, cluster_labels)
        
        # Analyze cluster distribution
        unique, counts = np.unique(cluster_labels, return_counts=True)
        cluster_sizes = dict(zip(unique, counts))
        
        print("Cluster Size Distribution:")
        for cluster_id in sorted(cluster_sizes.keys()):
            size = cluster_sizes[cluster_id]
            percentage = (size / len(X)) * 100
            print(f"  Cluster {cluster_id}: {size:5d} samples ({percentage:5.2f}%)")
        
        # Store results
        self.results['full_analysis'] = {
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
        return cluster_labels, centroids
    
    def visualize_results(self, X):
        """Create essential visualizations"""
        print("=" * 70)
        print("Generating Visualizations")
        print("=" * 70)
        
        # Create output directory
        os.makedirs('images', exist_ok=True)
        
        fig = plt.figure(figsize=(18, 10))
        
        # Plot 1: Elbow Plot (Inertia vs k)
        if 'elbow' in self.results:
            ax1 = plt.subplot(2, 3, 1)
            data = self.results['elbow']
            ax1.plot(data['k_values'], data['inertias'], 'o-', linewidth=2, 
                    markersize=8, color='#2E86AB')
            ax1.set_xlabel('Number of Clusters (k)', fontsize=11)
            ax1.set_ylabel('Inertia (Within-cluster Sum of Squares)', fontsize=11)
            ax1.set_title('Elbow Method\n(Look for the "elbow" point)', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.set_xticks(data['k_values'])
        
        # Plot 2: Time Complexity Analysis (Training Time vs k)
        if 'elbow' in self.results:
            ax2 = plt.subplot(2, 3, 2)
            data = self.results['elbow']
            ax2.plot(data['k_values'], data['training_times'], 's-', linewidth=2, 
                    markersize=8, color='#E63946')
            ax2.set_xlabel('Number of Clusters (k)', fontsize=11)
            ax2.set_ylabel('Training Time (seconds)', fontsize=11)
            ax2.set_title('Time Complexity Analysis\nO(n × k × d × i)', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.set_xticks(data['k_values'])
            
            # Add annotation
            ax2.text(0.05, 0.95, 'n = samples\nk = clusters\nd = dimensions\ni = iterations', 
                    transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 3: Cluster Size Distribution
        if 'full_analysis' in self.results:
            ax3 = plt.subplot(2, 3, 3)
            data = self.results['full_analysis']
            cluster_ids = sorted(data['cluster_sizes'].keys())
            sizes = [data['cluster_sizes'][i] for i in cluster_ids]
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_ids)))
            bars = ax3.bar(cluster_ids, sizes, color=colors, alpha=0.8, edgecolor='black')
            ax3.set_xlabel('Cluster ID', fontsize=11)
            ax3.set_ylabel('Number of Samples', fontsize=11)
            ax3.set_title('Cluster Size Distribution', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.set_xticks(cluster_ids)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=9)
        
        # Plot 4: Performance Metrics Bar Chart
        if 'full_analysis' in self.results:
            ax4 = plt.subplot(2, 3, 4)
            data = self.results['full_analysis']
            
            metrics = ['Silhouette\nScore', 'Adjusted\nRand Index', 'Homogeneity\nScore', 'Clustering\nAccuracy']
            values = [data['silhouette'], data['ari'], data['homogeneity'], data['accuracy']]
            colors_metrics = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
            
            bars = ax4.bar(metrics, values, color=colors_metrics, alpha=0.8, edgecolor='black')
            ax4.set_ylabel('Score', fontsize=11)
            ax4.set_title('Clustering Quality Metrics', fontsize=12, fontweight='bold')
            ax4.set_ylim([0, 1])
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Plot 5: Space Complexity Visualization
        if 'elbow' in self.results:
            ax5 = plt.subplot(2, 3, 5)
            data = self.results['elbow']
            n_samples = len(X)
            n_features = X.shape[1]
            
            # Calculate space complexity for different k values
            # Space = n*d (data) + k*d (centroids) + n (labels)
            space_data = n_samples * n_features * 4 / (1024**2)  # 4 bytes per float32, convert to MB
            space_complexity = []
            for k in data['k_values']:
                space_centroids = k * n_features * 4 / (1024**2)
                space_labels = n_samples * 4 / (1024**2)
                total_space = space_data + space_centroids + space_labels
                space_complexity.append(total_space)
            
            ax5.plot(data['k_values'], space_complexity, 'd-', linewidth=2, 
                    markersize=8, color='#9D4EDD')
            ax5.set_xlabel('Number of Clusters (k)', fontsize=11)
            ax5.set_ylabel('Memory Usage (MB)', fontsize=11)
            ax5.set_title('Space Complexity Analysis\nO(n × d + k × d)', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3)
            ax5.set_xticks(data['k_values'])
            
            # Add breakdown annotation
            ax5.text(0.05, 0.95, f'Data: {space_data:.1f} MB\n+ Centroids (k×d)\n+ Labels (n)', 
                    transform=ax5.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 6: 2D PCA Visualization of Clusters
        if 'full_analysis' in self.results:
            ax6 = plt.subplot(2, 3, 6)
            data = self.results['full_analysis']
            
            print("Computing PCA for visualization...")
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X)
            
            scatter = ax6.scatter(X_pca[:, 0], X_pca[:, 1], 
                                c=data['cluster_labels'], 
                                cmap='tab10', alpha=0.5, s=10, edgecolors='none')
            ax6.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=11)
            ax6.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=11)
            ax6.set_title('Clusters in 2D PCA Space', fontsize=12, fontweight='bold')
            plt.colorbar(scatter, ax=ax6, label='Cluster ID')
        
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
        n_clusters = len(centroids)
        
        # Determine grid size
        n_cols = 5
        n_rows = (n_clusters + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.5 * n_rows))
        axes = axes.flatten() if n_clusters > 1 else [axes]
        
        for i in range(len(axes)):
            if i < n_clusters:
                centroid_img = centroids[i].reshape(28, 28)
                axes[i].imshow(centroid_img, cmap='gray', interpolation='nearest')
                axes[i].set_title(f'Cluster {i}', fontsize=11, fontweight='bold')
                axes[i].axis('off')
            else:
                axes[i].axis('off')
        
        plt.suptitle('K-Means Cluster Centroids (Average Digit Representations)', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('images/kmeans_centroids.png', dpi=150, bbox_inches='tight')
        print("✓ Centroids saved as 'images/kmeans_centroids.png'")
        plt.close()


def main():
    """Main execution function"""
    print("\n" + "=" * 70)
    print("K-MEANS CLUSTERING ON MNIST DATASET")
    print("=" * 70)
    print()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Initialize analyzer
    analyzer = KMeansAnalyzer()
    
    # Load MNIST data
    X, y = analyzer.load_mnist_data(n_samples=10000)

    # Perform elbow analysis to find optimal k
    analyzer.elbow_analysis(X, k_range=range(2, 15))
    
    # Perform full analysis with optimal k (10 for MNIST digits)
    cluster_labels, centroids = analyzer.full_analysis(X, y, optimal_k=10)
    
    # Generate visualizations
    analyzer.visualize_results(X)
    
    print("=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nOutput files:")
    print("  • images/kmeans_analysis.png - Main analysis dashboard")
    print("  • images/kmeans_centroids.png - Cluster centroid visualizations")
    print()
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()