"""
Configurações gerais do projeto de reconhecimento de Libras
"""

import os
from pathlib import Path

# Diretórios do projeto
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Criar diretórios se não existirem
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Configurações do Dataset
DATASET_CONFIG = {
    "name": "libras_mnist",
    "kaggle_url": "datamoon/libras-mnist",
    "csv_file": "libras_mnist.csv",
    "n_classes": 24,
    "original_size": (28, 28),
    "target_size": (224, 224),  # Para MobileNetV2
    "channels": 3,  # RGB
    "batch_size": 32,
    "test_split": 0.2,
    "validation_split": 0.1,
    "random_seed": 42
}

# Configurações do Modelo
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

# Configurações de Treinamento
TRAINING_CONFIG = {
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "loss": "categorical_crossentropy",
    "metrics": ["accuracy"],
    "early_stopping": {
        "monitor": "val_accuracy",
        "patience": 10,
        "restore_best_weights": True
    },
    "reduce_lr": {
        "monitor": "val_loss",
        "factor": 0.5,
        "patience": 5,
        "min_lr": 1e-7
    }
}

# Configurações de Data Augmentation
AUGMENTATION_CONFIG = {
    "rotation_range": 15,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "shear_range": 0.1,
    "zoom_range": 0.1,
    "horizontal_flip": False,  # Não fazer flip para sinais de mão
    "fill_mode": "nearest"
}

# Configurações de Visualização
VISUALIZATION_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 100,
    "style": "seaborn-v0_8",
    "palette": "husl",
    "font_size": 12
}

# Configurações da Aplicação em Tempo Real
REALTIME_CONFIG = {
    "camera_index": 0,
    "frame_width": 640,
    "frame_height": 480,
    "fps": 30,
    "confidence_threshold": 0.7,
    "prediction_interval": 5  # Frames entre predições
}

# Mapeamento das classes (letras de Libras)
LIBRAS_CLASSES = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F",
    6: "G", 7: "H", 8: "I", 9: "J", 10: "K", 11: "L",
    12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R",
    18: "S", 19: "T", 20: "U", 21: "V", 22: "W", 23: "X"
}

# Configurações de Logging
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "libras_recognition.log"
}

# Configurações do Google Colab
COLAB_CONFIG = {
    "gpu_enabled": True,
    "kaggle_json_path": "/content/kaggle.json",
    "kaggle_dir": "/root/.kaggle",
    "data_dir": "/content/data"
}
