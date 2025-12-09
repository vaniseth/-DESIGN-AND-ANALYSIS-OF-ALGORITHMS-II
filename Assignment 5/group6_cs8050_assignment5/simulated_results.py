"""
Actual Experimental Results for CS8050 Assignment 5
CS 8050 - Assignment 5
Group 6
"""

# Random Forest ACTUAL Results
rf_results = {
    'sample_analysis': {
        'sizes': [500, 1000, 2000, 4000, 8000],
        'train_times': [0.020, 0.032, 0.058, 0.103, 0.202],
        'pred_times': [0.0008, 0.0007, 0.0007, 0.0007, 0.0010]
    },
    'tree_analysis': {
        'n_trees': [1, 5, 10, 20, 50, 100],
        'train_times': [0.007, 0.035, 0.059, 0.112, 0.269, 0.523],
        'pred_times': [0.0003, 0.0007, 0.0007, 0.0010, 0.0022, 0.0042],
        'accuracies': [0.680, 0.780, 0.830, 0.860, 0.860, 0.860]
    },
    'depth_analysis': {
        'depths': ['5', '10', '15', '20', 'Unlimited'],
        'train_times': [0.034, 0.053, 0.057, 0.058, 0.056],
        'pred_times': [0.0009, 0.0008, 0.0007, 0.0008, 0.0009],
        'accuracies': [0.810, 0.830, 0.840, 0.830, 0.830]
    },
    'space_analysis': {
        'n_trees': [1, 10, 50, 100],
        'memory_mb': [0.02, 15.75, 31.44, 47.0],
        'total_nodes': [643, 6568, 32766, 65812]
    },
    'final_model': {
        'train_time': 0.60,
        'pred_time': 0.028,
        'accuracy': 0.9410,
        'test_samples': 2000
    }
}

# CNN ACTUAL Results
cnn_results = {
    'sample_analysis': {
        'sizes': [1000, 5000, 10000, 20000, 40000],
        'train_times': [1.015, 1.349, 2.223, 4.278, 7.629],
        'pred_times': [0.132, 0.152, 0.142, 0.154, 0.138]
    },
    'filter_analysis': {
        'n_filters': [16, 32, 64, 128],
        'train_times': [1.537, 2.187, 4.607, 16.807],
        'total_params': [206922, 421642, 878730, 1903498]
    },
    'layer_analysis': {
        'n_layers': [1, 2, 3],
        'train_times': [1.074, 2.242, 3.024],
        'total_params': [804554, 421642, 241546]
    },
    'kernel_analysis': {
        'kernel_sizes': [3, 5, 7],
        'train_times': [2.222, 3.436, 6.019]
    },
    'space_analysis': {
        'names': ['Small', 'Medium', 'Large'],
        'params': [402986, 421642, 878730],
        'memory_mb': [14.7, 19.6, 60.7]
    },
    'batch_analysis': {
        'batch_sizes': [32, 64, 128, 256, 512],
        'train_times': [2.670, 2.377, 1.930, 2.171, 1.924]
    },
    'final_model': {
        'train_time': 86.44,
        'time_per_epoch': 8.64,
        'test_acc': 0.9911,
        'test_loss': 0.0255,
        'eval_time': 0.659,
        'total_params': 421642,
        'epochs': 10,
        'training_history': {
            'epochs': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'train_acc': [0.8008, 0.9666, 0.9761, 0.9812, 0.9840, 0.9871, 0.9882, 0.9897, 0.9907, 0.9907],
            'val_acc': [0.9817, 0.9877, 0.9902, 0.9903, 0.9902, 0.9918, 0.9920, 0.9908, 0.9917, 0.9917]
        }
    }
}

# Complexity Analysis Summary
complexity_summary = {
    'random_forest': {
        'time_train': 'O(k * n * log(n) * d)',
        'time_predict': 'O(k * log(n))',
        'space': 'O(k * n_nodes * d)',
        'convergence': 'No iterative training; one-pass algorithm',
        'scalability': 'Good for medium datasets; parallelizable',
        'bottlenecks': 'Tree building (splitting), memory with many trees',
        'key_finding': 'Accuracy plateaus at 50 trees (86%)'
    },
    'cnn': {
        'time_train_per_epoch': 'O(n * (C_conv + C_dense))',
        'time_predict': 'O(C_conv + C_dense) per sample',
        'space': 'O(p + b * max_activation)',
        'convergence': 'Iterative; 10 epochs achieved 99.1% accuracy',
        'scalability': 'Excellent with GPU; linear with samples',
        'bottlenecks': 'Convolution operations, backpropagation',
        'key_finding': 'Rapid convergence in first 3 epochs'
    }
}

# Comparative Analysis
comparative_metrics = {
    'training_time': {
        'rf_8k_samples': '0.60s (100 trees)',
        'cnn_60k_samples': '86.44s (10 epochs)',
        'winner': 'Random Forest (144x faster per sample)'
    },
    'prediction_time': {
        'rf_2000_samples': '28ms (0.014ms per sample)',
        'cnn_10000_samples': '659ms (0.066ms per sample)',
        'winner': 'Random Forest (4.7x faster per sample)'
    },
    'accuracy': {
        'rf_accuracy': '94.1%',
        'cnn_accuracy': '99.1%',
        'winner': 'CNN (5% absolute improvement, 85% error reduction)'
    },
    'memory': {
        'rf_model_size': '47 MB (100 trees)',
        'cnn_model_size': '19.6 MB (medium architecture)',
        'winner': 'CNN (2.4x smaller)'
    },
    'scalability': {
        'rf': 'Log-linear with samples (O(n log n)), linear with trees',
        'cnn': 'Linear with samples (O(n)), quadratic with filters',
        'winner': 'CNN for large datasets'
    },
    'convergence': {
        'rf': 'Instant (single pass), accuracy plateaus at 50 trees',
        'cnn': 'Gradual (10 epochs), 98% accuracy by epoch 3',
        'winner': 'RF for speed, CNN for final accuracy'
    }
}

print("Actual experimental results loaded successfully")
print(f"Random Forest - Final Accuracy: {rf_results['final_model']['accuracy']:.1%}")
print(f"CNN - Final Accuracy: {cnn_results['final_model']['test_acc']:.1%}")
print(f"\nKey Insight: CNN achieved {(cnn_results['final_model']['test_acc'] - rf_results['final_model']['accuracy']) * 100:.1f}% higher accuracy")
print(f"Speed Trade-off: RF trained {cnn_results['final_model']['train_time'] / rf_results['final_model']['train_time']:.1f}x faster")