"""
Random Forest Implementation on MNIST Dataset
CS 8050 - Assignment 5
Group 5
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from collections import Counter
import psutil
import os

class RandomForestAnalyzer:
    """
    Analyzes Random Forest algorithm complexity on MNIST dataset
    """
    
    def __init__(self):
        self.results = {}
        
    def load_mnist_data(self, sample_size=10000):
        """Load and preprocess MNIST dataset"""
        print("Loading MNIST dataset...")
        (X_full, y_full), (X_test_full, y_test_full) = mnist.load_data()
        
        # Flatten images
        X_full = X_full.reshape(-1, 784).astype('float32')
        X_test_full = X_test_full.reshape(-1, 784).astype('float32')
        
        # Combine and subsample
        X_combined = np.vstack([X_full, X_test_full])
        y_combined = np.hstack([y_full, y_test_full])
        
        indices = np.random.choice(len(X_combined), sample_size, replace=False)
        X, y = X_combined[indices], y_combined[indices]
        
        # Normalize features
        X = X / 255.0
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training samples: {len(X_train)}, Features: {X_train.shape[1]}")
        print(f"Testing samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def analyze_time_complexity_vs_samples(self, X_train, y_train, X_test, y_test):
        """
        Analyze time complexity: O(n_trees * n * log(n) * d)
        where n = samples, d = features
        """
        print("\n=== Time Complexity Analysis: Varying Sample Size ===")
        
        sample_sizes = [500, 1000, 2000, 4000, 8000]
        training_times = []
        prediction_times = []
        
        for n_samples in sample_sizes:
            if n_samples > len(X_train):
                break
                
            X_sub = X_train[:n_samples]
            y_sub = y_train[:n_samples]
            
            rf = RandomForestClassifier(
                n_estimators=10, 
                max_depth=10,
                random_state=42,
                n_jobs=1  # Single thread for fair timing
            )
            
            # Training time
            start = time.time()
            rf.fit(X_sub, y_sub)
            train_time = time.time() - start
            training_times.append(train_time)
            
            # Prediction time
            start = time.time()
            rf.predict(X_test[:100])
            pred_time = time.time() - start
            prediction_times.append(pred_time)
            
            print(f"n={n_samples}: Train={train_time:.3f}s, Predict={pred_time:.4f}s")
        
        self.results['sample_analysis'] = {
            'sizes': sample_sizes[:len(training_times)],
            'train_times': training_times,
            'pred_times': prediction_times
        }
        
        return training_times, prediction_times
    
    def analyze_time_complexity_vs_trees(self, X_train, y_train, X_test, y_test):
        """
        Analyze time complexity: O(n_trees * n * log(n) * d)
        Linear in number of trees
        """
        print("\n=== Time Complexity Analysis: Varying Number of Trees ===")
        
        n_trees_list = [1, 5, 10, 20, 50, 100]
        training_times = []
        prediction_times = []
        accuracies = []
        
        for n_trees in n_trees_list:
            rf = RandomForestClassifier(
                n_estimators=n_trees,
                max_depth=10,
                random_state=42,
                n_jobs=1
            )
            
            start = time.time()
            rf.fit(X_train[:2000], y_train[:2000])
            train_time = time.time() - start
            training_times.append(train_time)
            
            start = time.time()
            predictions = rf.predict(X_test[:100])
            pred_time = time.time() - start
            prediction_times.append(pred_time)
            
            accuracy = accuracy_score(y_test[:100], predictions)
            accuracies.append(accuracy)
            
            print(f"Trees={n_trees}: Train={train_time:.3f}s, "
                  f"Predict={pred_time:.4f}s, Acc={accuracy:.3f}")
        
        self.results['tree_analysis'] = {
            'n_trees': n_trees_list,
            'train_times': training_times,
            'pred_times': prediction_times,
            'accuracies': accuracies
        }
        
        return training_times, prediction_times, accuracies
    
    def analyze_time_complexity_vs_depth(self, X_train, y_train, X_test, y_test):
        """
        Analyze time complexity with varying tree depth
        Depth affects log(n) factor in O(n * log(n) * d)
        """
        print("\n=== Time Complexity Analysis: Varying Tree Depth ===")
        
        depths = [5, 10, 15, 20, None]  # None means unlimited
        training_times = []
        prediction_times = []
        accuracies = []
        
        for depth in depths:
            rf = RandomForestClassifier(
                n_estimators=10,
                max_depth=depth,
                random_state=42,
                n_jobs=1
            )
            
            start = time.time()
            rf.fit(X_train[:2000], y_train[:2000])
            train_time = time.time() - start
            training_times.append(train_time)
            
            start = time.time()
            predictions = rf.predict(X_test[:100])
            pred_time = time.time() - start
            prediction_times.append(pred_time)
            
            accuracy = accuracy_score(y_test[:100], predictions)
            accuracies.append(accuracy)
            
            depth_str = str(depth) if depth else "Unlimited"
            print(f"Depth={depth_str}: Train={train_time:.3f}s, "
                  f"Predict={pred_time:.4f}s, Acc={accuracy:.3f}")
        
        self.results['depth_analysis'] = {
            'depths': [str(d) if d else "Unlimited" for d in depths],
            'train_times': training_times,
            'pred_times': prediction_times,
            'accuracies': accuracies
        }
        
        return training_times, prediction_times, accuracies
    
    def analyze_space_complexity(self, X_train, y_train):
        """
        Analyze space complexity: O(n_trees * n_nodes * d)
        Model storage grows with trees and nodes
        """
        print("\n=== Space Complexity Analysis ===")
        
        process = psutil.Process(os.getpid())
        
        n_trees_list = [1, 10, 50, 100]
        memory_usage = []
        n_nodes_list = []
        
        for n_trees in n_trees_list:
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            rf = RandomForestClassifier(
                n_estimators=n_trees,
                max_depth=15,
                random_state=42
            )
            rf.fit(X_train[:2000], y_train[:2000])
            
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_increase = mem_after - mem_before
            memory_usage.append(mem_increase)
            
            # Count total nodes
            total_nodes = sum(tree.tree_.node_count for tree in rf.estimators_)
            n_nodes_list.append(total_nodes)
            
            print(f"Trees={n_trees}: Memory={mem_increase:.2f}MB, "
                  f"Total Nodes={total_nodes}")
        
        self.results['space_analysis'] = {
            'n_trees': n_trees_list,
            'memory_mb': memory_usage,
            'total_nodes': n_nodes_list
        }
        
        return memory_usage, n_nodes_list
    
    def analyze_feature_importance(self, X_train, y_train):
        """Analyze feature importance for interpretation"""
        print("\n=== Feature Importance Analysis ===")
        
        rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=15,
            random_state=42
        )
        rf.fit(X_train, y_train)
        
        feature_importance = rf.feature_importances_
        
        # Get top 20 most important features (pixels)
        top_indices = np.argsort(feature_importance)[-20:]
        
        print(f"Top 20 most important pixel positions:")
        for idx in reversed(top_indices):
            row = idx // 28
            col = idx % 28
            print(f"  Pixel ({row}, {col}): importance = {feature_importance[idx]:.5f}")
        
        self.results['feature_importance'] = feature_importance
        
        return feature_importance
    
    def full_model_evaluation(self, X_train, y_train, X_test, y_test):
        """Train and evaluate final model"""
        print("\n=== Full Model Evaluation ===")
        
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1  # Use all cores for final training
        )
        
        print("Training final model...")
        start = time.time()
        rf.fit(X_train, y_train)
        train_time = time.time() - start
        
        print(f"Training completed in {train_time:.2f}s")
        
        # Predictions
        start = time.time()
        y_pred = rf.predict(X_test)
        pred_time = time.time() - start
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Prediction time: {pred_time:.3f}s")
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.results['final_model'] = {
            'train_time': train_time,
            'pred_time': pred_time,
            'accuracy': accuracy,
            'y_pred': y_pred,
            'y_test': y_test
        }
        
        return rf, y_pred
    
    def visualize_results(self):
        """Create visualization plots"""
        print("\n=== Generating Visualizations ===")
        
        # Create images directory if it doesn't exist
        import os
        os.makedirs('images', exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Training time vs sample size
        if 'sample_analysis' in self.results:
            data = self.results['sample_analysis']
            axes[0, 0].plot(data['sizes'], data['train_times'], 'o-', linewidth=2)
            axes[0, 0].set_xlabel('Number of Training Samples (n)')
            axes[0, 0].set_ylabel('Training Time (seconds)')
            axes[0, 0].set_title('Training Time vs Sample Size\nComplexity: O(n log n)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Training time vs number of trees
        if 'tree_analysis' in self.results:
            data = self.results['tree_analysis']
            axes[0, 1].plot(data['n_trees'], data['train_times'], 's-', linewidth=2, color='green')
            axes[0, 1].set_xlabel('Number of Trees (k)')
            axes[0, 1].set_ylabel('Training Time (seconds)')
            axes[0, 1].set_title('Training Time vs Number of Trees\nComplexity: O(k * n log n)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Accuracy vs number of trees
        if 'tree_analysis' in self.results:
            data = self.results['tree_analysis']
            axes[0, 2].plot(data['n_trees'], data['accuracies'], '^-', linewidth=2, color='red')
            axes[0, 2].set_xlabel('Number of Trees (k)')
            axes[0, 2].set_ylabel('Accuracy')
            axes[0, 2].set_title('Accuracy vs Number of Trees')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Training time vs tree depth
        if 'depth_analysis' in self.results:
            data = self.results['depth_analysis']
            axes[1, 0].bar(range(len(data['depths'])), data['train_times'], color='purple', alpha=0.7)
            axes[1, 0].set_xticks(range(len(data['depths'])))
            axes[1, 0].set_xticklabels(data['depths'])
            axes[1, 0].set_xlabel('Maximum Tree Depth')
            axes[1, 0].set_ylabel('Training Time (seconds)')
            axes[1, 0].set_title('Training Time vs Tree Depth')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 5: Space complexity
        if 'space_analysis' in self.results:
            data = self.results['space_analysis']
            ax1 = axes[1, 1]
            ax1.plot(data['n_trees'], data['memory_mb'], 'o-', linewidth=2, color='orange')
            ax1.set_xlabel('Number of Trees')
            ax1.set_ylabel('Memory Usage (MB)', color='orange')
            ax1.tick_params(axis='y', labelcolor='orange')
            ax1.set_title('Space Complexity Analysis')
            ax1.grid(True, alpha=0.3)
            
            ax2 = ax1.twinx()
            ax2.plot(data['n_trees'], data['total_nodes'], 's-', linewidth=2, color='brown')
            ax2.set_ylabel('Total Nodes', color='brown')
            ax2.tick_params(axis='y', labelcolor='brown')
        
        # Plot 6: Confusion Matrix
        if 'final_model' in self.results:
            data = self.results['final_model']
            cm = confusion_matrix(data['y_test'], data['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 2])
            axes[1, 2].set_xlabel('Predicted Label')
            axes[1, 2].set_ylabel('True Label')
            axes[1, 2].set_title('Confusion Matrix\n(Final Model)')
        
        plt.tight_layout()
        
        # Create images directory if it doesn't exist
        import os
        os.makedirs('images', exist_ok=True)
        
        plt.savefig('images/random_forest_analysis.png', dpi=150, bbox_inches='tight')
        print("Visualization saved as 'images/random_forest_analysis.png'")
        
        return fig

def main():
    """Main execution function"""
    print("=" * 70)
    print("Random Forest Algorithm - Complexity Analysis on MNIST")
    print("=" * 70)
    
    analyzer = RandomForestAnalyzer()
    
    # Load data
    X_train, X_test, y_train, y_test = analyzer.load_mnist_data(sample_size=10000)
    
    # Run complexity analyses
    analyzer.analyze_time_complexity_vs_samples(X_train, y_train, X_test, y_test)
    analyzer.analyze_time_complexity_vs_trees(X_train, y_train, X_test, y_test)
    analyzer.analyze_time_complexity_vs_depth(X_train, y_train, X_test, y_test)
    analyzer.analyze_space_complexity(X_train, y_train)
    analyzer.analyze_feature_importance(X_train, y_train)
    
    # Full model evaluation
    rf_model, y_pred = analyzer.full_model_evaluation(X_train, y_train, X_test, y_test)
    
    # Generate visualizations
    analyzer.visualize_results()
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()