"""
Convolutional Neural Network (CNN) Implementation on MNIST Dataset
CS 8050 - Assignment 5
Group 6
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import psutil
import os

class CNNAnalyzer:
    """
    Analyzes CNN algorithm complexity on MNIST dataset
    """
    
    def __init__(self):
        self.results = {}
        
    def load_mnist_data(self):
        """Load and preprocess MNIST dataset"""
        print("Loading MNIST dataset...")
        
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        # Reshape for CNN (add channel dimension)
        X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        
        # One-hot encode labels
        y_train_cat = to_categorical(y_train, 10)
        y_test_cat = to_categorical(y_test, 10)
        
        print(f"Training samples: {len(X_train)}, Image shape: {X_train.shape[1:]}")
        print(f"Testing samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test, y_train_cat, y_test_cat
    
    def create_cnn_model(self, input_shape=(28, 28, 1), n_filters=32, kernel_size=3, 
                         n_conv_layers=2, n_dense_units=128):
        """
        Create CNN model with configurable architecture
        
        Complexity Analysis:
        - Convolution: O(n_filters * kernel_size^2 * input_channels * output_width * output_height)
        - Pooling: O(pool_size^2 * width * height * channels)
        - Dense: O(input_features * output_features)
        """
        model = models.Sequential()
        
        # First Conv layer
        model.add(layers.Conv2D(n_filters, (kernel_size, kernel_size), 
                                activation='relu', input_shape=input_shape, 
                                padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        # Additional Conv layers
        for i in range(1, n_conv_layers):
            model.add(layers.Conv2D(n_filters * (2**i), (kernel_size, kernel_size),
                                    activation='relu', padding='same'))
            model.add(layers.MaxPooling2D((2, 2)))
        
        # Flatten and Dense layers
        model.add(layers.Flatten())
        model.add(layers.Dense(n_dense_units, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(10, activation='softmax'))
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        return model
    
    def count_parameters(self, model):
        """Count total parameters in model"""
        total_params = model.count_params()
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
        
        print(f"\nModel Parameters:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        return total_params, trainable_params
    
    def analyze_time_complexity_vs_samples(self, X_train, y_train_cat, X_test, y_test_cat):
        """
        Analyze time complexity with varying training samples
        Per epoch: O(n * operations_per_sample)
        """
        print("\n=== Time Complexity Analysis: Varying Sample Size ===")
        
        sample_sizes = [1000, 5000, 10000, 20000, 40000]
        training_times_per_epoch = []
        prediction_times = []
        
        for n_samples in sample_sizes:
            if n_samples > len(X_train):
                break
            
            X_sub = X_train[:n_samples]
            y_sub = y_train_cat[:n_samples]
            
            model = self.create_cnn_model()
            
            # Training time (1 epoch for fair comparison)
            start = time.time()
            model.fit(X_sub, y_sub, epochs=1, batch_size=128, verbose=0)
            train_time = time.time() - start
            training_times_per_epoch.append(train_time)
            
            # Prediction time
            start = time.time()
            model.predict(X_test[:1000], verbose=0)
            pred_time = time.time() - start
            prediction_times.append(pred_time)
            
            print(f"n={n_samples}: Train(1 epoch)={train_time:.3f}s, Predict={pred_time:.3f}s")
        
        self.results['sample_analysis'] = {
            'sizes': sample_sizes[:len(training_times_per_epoch)],
            'train_times': training_times_per_epoch,
            'pred_times': prediction_times
        }
        
        return training_times_per_epoch, prediction_times
    
    def analyze_time_complexity_vs_filters(self, X_train, y_train_cat):
        """
        Analyze time complexity with varying number of filters
        Complexity increases linearly with number of filters
        """
        print("\n=== Time Complexity Analysis: Varying Number of Filters ===")
        
        n_filters_list = [16, 32, 64, 128]
        training_times = []
        total_params = []
        
        for n_filters in n_filters_list:
            model = self.create_cnn_model(n_filters=n_filters)
            params, _ = self.count_parameters(model)
            total_params.append(params)
            
            start = time.time()
            model.fit(X_train[:5000], y_train_cat[:5000], 
                     epochs=2, batch_size=128, verbose=0)
            train_time = time.time() - start
            training_times.append(train_time)
            
            print(f"Filters={n_filters}: Train(2 epochs)={train_time:.3f}s, Params={params:,}")
        
        self.results['filter_analysis'] = {
            'n_filters': n_filters_list,
            'train_times': training_times,
            'total_params': total_params
        }
        
        return training_times, total_params
    
    def analyze_time_complexity_vs_layers(self, X_train, y_train_cat):
        """
        Analyze time complexity with varying number of convolutional layers
        """
        print("\n=== Time Complexity Analysis: Varying Number of Conv Layers ===")
        
        n_layers_list = [1, 2, 3]
        training_times = []
        total_params = []
        
        for n_layers in n_layers_list:
            model = self.create_cnn_model(n_conv_layers=n_layers)
            params, _ = self.count_parameters(model)
            total_params.append(params)
            
            start = time.time()
            model.fit(X_train[:5000], y_train_cat[:5000],
                     epochs=2, batch_size=128, verbose=0)
            train_time = time.time() - start
            training_times.append(train_time)
            
            print(f"Layers={n_layers}: Train(2 epochs)={train_time:.3f}s, Params={params:,}")
        
        self.results['layer_analysis'] = {
            'n_layers': n_layers_list,
            'train_times': training_times,
            'total_params': total_params
        }
        
        return training_times, total_params
    
    def analyze_time_complexity_vs_kernel(self, X_train, y_train_cat):
        """
        Analyze time complexity with varying kernel sizes
        Complexity: O(kernel_size^2)
        """
        print("\n=== Time Complexity Analysis: Varying Kernel Size ===")
        
        kernel_sizes = [3, 5, 7]
        training_times = []
        
        for kernel_size in kernel_sizes:
            model = self.create_cnn_model(kernel_size=kernel_size)
            
            start = time.time()
            model.fit(X_train[:5000], y_train_cat[:5000],
                     epochs=2, batch_size=128, verbose=0)
            train_time = time.time() - start
            training_times.append(train_time)
            
            print(f"Kernel={kernel_size}x{kernel_size}: Train(2 epochs)={train_time:.3f}s")
        
        self.results['kernel_analysis'] = {
            'kernel_sizes': kernel_sizes,
            'train_times': training_times
        }
        
        return training_times
    
    def analyze_space_complexity(self, X_train, y_train_cat):
        """
        Analyze space complexity: model parameters and activations
        Space = O(sum of all layer parameters + batch_size * largest activation)
        """
        print("\n=== Space Complexity Analysis ===")
        
        process = psutil.Process(os.getpid())
        
        architectures = [
            {'n_filters': 16, 'n_conv_layers': 1, 'name': 'Small'},
            {'n_filters': 32, 'n_conv_layers': 2, 'name': 'Medium'},
            {'n_filters': 64, 'n_conv_layers': 2, 'name': 'Large'},
        ]
        
        memory_usage = []
        param_counts = []
        
        for arch in architectures:
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            model = self.create_cnn_model(
                n_filters=arch['n_filters'],
                n_conv_layers=arch['n_conv_layers']
            )
            
            params, _ = self.count_parameters(model)
            param_counts.append(params)
            
            # Train briefly to allocate memory
            model.fit(X_train[:1000], y_train_cat[:1000], 
                     epochs=1, batch_size=128, verbose=0)
            
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_increase = mem_after - mem_before
            memory_usage.append(mem_increase)
            
            print(f"{arch['name']} Model: Params={params:,}, Memory≈{mem_increase:.1f}MB")
        
        self.results['space_analysis'] = {
            'names': [a['name'] for a in architectures],
            'params': param_counts,
            'memory_mb': memory_usage
        }
        
        return memory_usage, param_counts
    
    def analyze_batch_size_effect(self, X_train, y_train_cat):
        """
        Analyze effect of batch size on training time and memory
        Larger batches: better GPU utilization but more memory
        """
        print("\n=== Batch Size Analysis ===")
        
        batch_sizes = [32, 64, 128, 256, 512]
        training_times = []
        
        for batch_size in batch_sizes:
            model = self.create_cnn_model()
            
            start = time.time()
            try:
                model.fit(X_train[:5000], y_train_cat[:5000],
                         epochs=2, batch_size=batch_size, verbose=0)
                train_time = time.time() - start
                training_times.append(train_time)
                print(f"Batch Size={batch_size}: Train(2 epochs)={train_time:.3f}s")
            except Exception as e:
                print(f"Batch Size={batch_size}: Failed - {e}")
                training_times.append(None)
        
        self.results['batch_analysis'] = {
            'batch_sizes': batch_sizes,
            'train_times': [t for t in training_times if t is not None]
        }
        
        return training_times
    
    def full_model_training(self, X_train, y_train_cat, X_test, y_test_cat, y_test):
        """Train and evaluate final CNN model"""
        print("\n=== Full Model Training and Evaluation ===")
        
        model = self.create_cnn_model(n_filters=32, n_conv_layers=2, n_dense_units=128)
        self.count_parameters(model)
        
        print("\nTraining model for 10 epochs...")
        start = time.time()
        history = model.fit(
            X_train, y_train_cat,
            epochs=10,
            batch_size=128,
            validation_split=0.1,
            verbose=1
        )
        train_time = time.time() - start
        
        print(f"\nTotal training time: {train_time:.2f}s ({train_time/10:.2f}s per epoch)")
        
        # Evaluation
        print("\nEvaluating on test set...")
        start = time.time()
        test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
        eval_time = time.time() - start
        
        # Predictions
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Evaluation time: {eval_time:.3f}s")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.results['final_model'] = {
            'model': model,
            'history': history,
            'train_time': train_time,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'y_pred': y_pred,
            'y_test': y_test
        }
        
        return model, history, y_pred
    
    def visualize_results(self):
        """Create comprehensive visualization plots"""
        print("\n=== Generating Visualizations ===")
        
        # Create images directory if it doesn't exist
        import os
        os.makedirs('images', exist_ok=True)
        
        fig = plt.figure(figsize=(20, 12))
        
        # Plot 1: Training time vs sample size
        if 'sample_analysis' in self.results:
            ax1 = plt.subplot(3, 3, 1)
            data = self.results['sample_analysis']
            ax1.plot(data['sizes'], data['train_times'], 'o-', linewidth=2, markersize=8)
            ax1.set_xlabel('Number of Training Samples (n)')
            ax1.set_ylabel('Training Time per Epoch (s)')
            ax1.set_title('Training Time vs Sample Size\nComplexity: O(n)')
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Training time vs number of filters
        if 'filter_analysis' in self.results:
            ax2 = plt.subplot(3, 3, 2)
            data = self.results['filter_analysis']
            ax2.plot(data['n_filters'], data['train_times'], 's-', 
                    linewidth=2, markersize=8, color='green')
            ax2.set_xlabel('Number of Filters')
            ax2.set_ylabel('Training Time (s)')
            ax2.set_title('Training Time vs Number of Filters\nComplexity: Linear in filters')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Parameters vs number of filters
        if 'filter_analysis' in self.results:
            ax3 = plt.subplot(3, 3, 3)
            data = self.results['filter_analysis']
            ax3.plot(data['n_filters'], np.array(data['total_params'])/1000, 
                    '^-', linewidth=2, markersize=8, color='red')
            ax3.set_xlabel('Number of Filters')
            ax3.set_ylabel('Total Parameters (thousands)')
            ax3.set_title('Model Size vs Number of Filters')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Training time vs number of layers
        if 'layer_analysis' in self.results:
            ax4 = plt.subplot(3, 3, 4)
            data = self.results['layer_analysis']
            ax4.bar(data['n_layers'], data['train_times'], color='purple', alpha=0.7)
            ax4.set_xlabel('Number of Convolutional Layers')
            ax4.set_ylabel('Training Time (s)')
            ax4.set_title('Training Time vs Network Depth')
            ax4.grid(True, alpha=0.3, axis='y')
        
        # Plot 5: Training time vs kernel size
        if 'kernel_analysis' in self.results:
            ax5 = plt.subplot(3, 3, 5)
            data = self.results['kernel_analysis']
            ax5.plot(data['kernel_sizes'], data['train_times'], 
                    'D-', linewidth=2, markersize=8, color='orange')
            ax5.set_xlabel('Kernel Size')
            ax5.set_ylabel('Training Time (s)')
            ax5.set_title('Training Time vs Kernel Size\nComplexity: O(kernel_size²)')
            ax5.grid(True, alpha=0.3)
        
        # Plot 6: Space complexity
        if 'space_analysis' in self.results:
            ax6 = plt.subplot(3, 3, 6)
            data = self.results['space_analysis']
            x_pos = np.arange(len(data['names']))
            ax6_twin = ax6.twinx()
            
            bars1 = ax6.bar(x_pos - 0.2, np.array(data['params'])/1000, 
                           0.4, label='Parameters (K)', color='skyblue', alpha=0.8)
            bars2 = ax6_twin.bar(x_pos + 0.2, data['memory_mb'], 
                                0.4, label='Memory (MB)', color='coral', alpha=0.8)
            
            ax6.set_xlabel('Model Architecture')
            ax6.set_ylabel('Parameters (thousands)', color='skyblue')
            ax6_twin.set_ylabel('Memory (MB)', color='coral')
            ax6.set_xticks(x_pos)
            ax6.set_xticklabels(data['names'])
            ax6.set_title('Space Complexity Analysis')
            ax6.tick_params(axis='y', labelcolor='skyblue')
            ax6_twin.tick_params(axis='y', labelcolor='coral')
            ax6.grid(True, alpha=0.3, axis='y')
        
        # Plot 7: Training history
        if 'final_model' in self.results:
            ax7 = plt.subplot(3, 3, 7)
            history = self.results['final_model']['history']
            ax7.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
            ax7.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
            ax7.set_xlabel('Epoch')
            ax7.set_ylabel('Accuracy')
            ax7.set_title('Training & Validation Accuracy')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # Plot 8: Loss history
        if 'final_model' in self.results:
            ax8 = plt.subplot(3, 3, 8)
            history = self.results['final_model']['history']
            ax8.plot(history.history['loss'], label='Training Loss', linewidth=2)
            ax8.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
            ax8.set_xlabel('Epoch')
            ax8.set_ylabel('Loss')
            ax8.set_title('Training & Validation Loss')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        
        # Plot 9: Confusion Matrix
        if 'final_model' in self.results:
            ax9 = plt.subplot(3, 3, 9)
            data = self.results['final_model']
            cm = confusion_matrix(data['y_test'], data['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax9, cbar_kws={'shrink': 0.8})
            ax9.set_xlabel('Predicted Label')
            ax9.set_ylabel('True Label')
            ax9.set_title('Confusion Matrix (Test Set)')
        
        plt.tight_layout()
        
        # Create images directory if it doesn't exist
        import os
        os.makedirs('images', exist_ok=True)
        
        plt.savefig('images/cnn_analysis.png', dpi=150, bbox_inches='tight')
        print("Visualization saved as 'images/cnn_analysis.png'")
        
        return fig

def main():
    """Main execution function"""
    print("=" * 70)
    print("Convolutional Neural Network - Complexity Analysis on MNIST")
    print("=" * 70)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    analyzer = CNNAnalyzer()
    
    # Load data
    X_train, X_test, y_train, y_test, y_train_cat, y_test_cat = analyzer.load_mnist_data()
    
    # Run complexity analyses
    analyzer.analyze_time_complexity_vs_samples(X_train, y_train_cat, X_test, y_test_cat)
    analyzer.analyze_time_complexity_vs_filters(X_train, y_train_cat)
    analyzer.analyze_time_complexity_vs_layers(X_train, y_train_cat)
    analyzer.analyze_time_complexity_vs_kernel(X_train, y_train_cat)
    analyzer.analyze_space_complexity(X_train, y_train_cat)
    analyzer.analyze_batch_size_effect(X_train, y_train_cat)
    
    # Full model training
    model, history, y_pred = analyzer.full_model_training(
        X_train, y_train_cat, X_test, y_test_cat, y_test
    )
    
    # Generate visualizations
    analyzer.visualize_results()
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()