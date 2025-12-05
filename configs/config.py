"""
Configurações gerais do projeto de reconhecimento de Libras
"""

import os
from pathlib import Path

                       
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

                                   
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

                          
DATASET_CONFIG = {
    "name": "libras_mnist",
    "kaggle_url": "datamoon/libras-mnist",
    "csv_file": "libras_mnist.csv",
    "n_classes": 24,
    "original_size": (28, 28),
    "target_size": (224, 224),                    
    "channels": 3,       
    "batch_size": 32,
    "test_split": 0.2,
    "validation_split": 0.1,
    "random_seed": 42
}

                         
MODEL_CONFIG = {
    "base_model": "MobileNetV2",
    "input_shape": (224, 224, 3),
    "include_top": False,
    "weights": "imagenet",
    "pooling": "avg",
    "dropout_rate": 0.5,
    "dense_units": 128,
    "activation": "relu",
    "final_activation": "softmax"
}

                                         
IMPROVED_TRAINING_CONFIG = {
    "epochs": 100,                            
    "batch_size": 16,                                      
    "learning_rate": 0.0001,                                          
    "optimizer": "adam",
    "loss": "categorical_crossentropy",
    "metrics": ["accuracy"],
    "weight_decay": 1e-4,                          
    "early_stopping": {
        "monitor": "val_accuracy",
        "patience": 15,                           
        "restore_best_weights": True
    },
    "reduce_lr": {
        "monitor": "val_loss",
        "factor": 0.3,                          
        "patience": 8,                         
        "min_lr": 1e-7
    },
    "cosine_scheduler": True                         
}

                                            
MODEL_ARCHITECTURES = {
    "mobilenet_v2": {
        "base_model": "MobileNetV2",
        "input_shape": (224, 224, 3),
        "include_top": False,
        "weights": "imagenet",
        "pooling": "avg",
        "dense_units": [256, 128],
        "dropout_rate": 0.5
    },
    "efficientnet": {
        "base_model": "EfficientNetB0",
        "input_shape": (224, 224, 3),
        "include_top": False,
        "weights": "imagenet",
        "pooling": "avg",
        "dense_units": [256, 128],
        "dropout_rate": 0.5
    },
    "resnet": {
        "base_model": "ResNet50V2",
        "input_shape": (224, 224, 3),
        "include_top": False,
        "weights": "imagenet",
        "pooling": "avg",
        "dense_units": [256, 128],
        "dropout_rate": 0.5
    }
}

                                                
HYPERPARAMETER_OPTIMIZATION = {
    "n_trials": 50,
    "direction": "maximize",
    "pruner": "MedianPruner",
    "study_name": "libras_recognition_optimization",
    "timeout": 3600,                           
    "parameters": {
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
        "dropout_rate": {"type": "float", "low": 0.2, "high": 0.7},
        "dense_units_1": {"type": "categorical", "choices": [64, 128, 256, 512]},
        "dense_units_2": {"type": "categorical", "choices": [32, 64, 128, 256]},
        "optimizer": {"type": "categorical", "choices": ["adam", "sgd", "rmsprop"]},
        "model_type": {"type": "categorical", "choices": ["mobilenet_v2", "efficientnet"]}
    }
}

                                               
AUGMENTATION_CONFIG = {
    "rotation_range": 20,                           
    "width_shift_range": 0.15,                              
    "height_shift_range": 0.15,                              
    "shear_range": 0.15,                              
    "zoom_range": 0.2,                             
    "brightness_range": [0.8, 1.2],                            
    "channel_shift_range": 0.1,                           
    "horizontal_flip": False,                                     
    "fill_mode": "nearest"
}

                                              
ADVANCED_AUGMENTATION_CONFIG = {
    "rotation_range": 25,
    "width_shift_range": 0.2,
    "height_shift_range": 0.2,
    "shear_range": 0.2,
    "zoom_range": 0.25,
    "brightness_range": [0.7, 1.3],
    "channel_shift_range": 0.15,
    "horizontal_flip": False,
    "fill_mode": "nearest",
    "rescale": 1./255
}

                               
VISUALIZATION_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 100,
    "style": "seaborn-v0_8",
    "palette": "husl",
    "font_size": 12
}

                                          
REALTIME_CONFIG = {
    "camera_index": 0,
    "frame_width": 640,
    "frame_height": 480,
    "fps": 30,
    "confidence_threshold": 0.7,
    "prediction_interval": 5                          
}

                                           
LIBRAS_CLASSES = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F",
    6: "G", 7: "H", 8: "I", 9: "J", 10: "K", 11: "L",
    12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R",
    18: "S", 19: "T", 20: "U", 21: "V", 22: "W", 23: "X"
}

                          
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "libras_recognition.log"
}

                               
COLAB_CONFIG = {
    "gpu_enabled": True,
    "kaggle_json_path": "/content/kaggle.json",
    "kaggle_dir": "/root/.kaggle",
    "data_dir": "/content/data"
}
