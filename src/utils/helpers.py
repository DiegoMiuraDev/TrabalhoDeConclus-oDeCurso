"""
Funções auxiliares para o projeto
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import os
from pathlib import Path


def check_gpu_availability():
    """
    Verifica se GPU está disponível para TensorFlow
    
    Returns:
        bool: True se GPU estiver disponível
    """
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU detectada: {len(gpus)} dispositivo(s)")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
            return True
        else:
            print("⚠️  Nenhuma GPU detectada")
            return False
    except ImportError:
        print("❌ TensorFlow não instalado")
        return False


def print_system_info():
    """
    Imprime informações do sistema e versões das bibliotecas
    """
    print("🖥️  Informações do Sistema:")
    print(f"   Python: {os.sys.version}")
    
    try:
        import tensorflow as tf
        print(f"   TensorFlow: {tf.__version__}")
    except ImportError:
        print("   TensorFlow: Não instalado")
    
    try:
        import cv2
        print(f"   OpenCV: {cv2.__version__}")
    except ImportError:
        print("   OpenCV: Não instalado")
    
    try:
        import numpy as np
        print(f"   NumPy: {np.__version__}")
    except ImportError:
        print("   NumPy: Não instalado")
    
    try:
        import pandas as pd
        print(f"   Pandas: {pd.__version__}")
    except ImportError:
        print("   Pandas: Não instalado")


def create_directory_structure(base_path: str):
    """
    Cria estrutura de diretórios do projeto
    
    Args:
        base_path (str): Caminho base do projeto
    """
    directories = [
        "data/raw",
        "data/processed",
        "models",
        "results/plots",
        "results/metrics",
        "logs",
        "notebooks",
        "scripts"
    ]
    
    for directory in directories:
        dir_path = Path(base_path) / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Criado: {dir_path}")


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Redimensiona uma imagem
    
    Args:
        image (np.ndarray): Imagem de entrada
        target_size (Tuple[int, int]): Tamanho alvo (width, height)
    
    Returns:
        np.ndarray: Imagem redimensionada
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def grayscale_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Converte imagem em escala de cinza para RGB
    
    Args:
        image (np.ndarray): Imagem em escala de cinza
    
    Returns:
        np.ndarray: Imagem RGB
    """
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image


def normalize_image(image: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
    """
    Normaliza imagem para o range especificado
    
    Args:
        image (np.ndarray): Imagem de entrada
        min_val (float): Valor mínimo
        max_val (float): Valor máximo
    
    Returns:
        np.ndarray: Imagem normalizada
    """
    img_min = image.min()
    img_max = image.max()
    
    if img_max == img_min:
        return np.zeros_like(image)
    
    normalized = (image - img_min) / (img_max - img_min)
    return normalized * (max_val - min_val) + min_val


def save_image(image: np.ndarray, filepath: str, cmap: Optional[str] = None):
    """
    Salva imagem em arquivo
    
    Args:
        image (np.ndarray): Imagem para salvar
        filepath (str): Caminho do arquivo
        cmap (str): Mapa de cores (opcional)
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


def calculate_image_statistics(images: np.ndarray) -> dict:
    """
    Calcula estatísticas de um conjunto de imagens
    
    Args:
        images (np.ndarray): Array de imagens
    
    Returns:
        dict: Estatísticas das imagens
    """
    stats = {
        'shape': images.shape,
        'dtype': images.dtype,
        'min': images.min(),
        'max': images.max(),
        'mean': images.mean(),
        'std': images.std(),
        'median': np.median(images)
    }
    
    return stats


def print_image_statistics(images: np.ndarray, title: str = "Estatísticas das Imagens"):
    """
    Imprime estatísticas de um conjunto de imagens
    
    Args:
        images (np.ndarray): Array de imagens
        title (str): Título para exibição
    """
    stats = calculate_image_statistics(images)
    
    print(f"\n📊 {title}:")
    print(f"   Forma: {stats['shape']}")
    print(f"   Tipo: {stats['dtype']}")
    print(f"   Mínimo: {stats['min']:.3f}")
    print(f"   Máximo: {stats['max']:.3f}")
    print(f"   Média: {stats['mean']:.3f}")
    print(f"   Desvio Padrão: {stats['std']:.3f}")
    print(f"   Mediana: {stats['median']:.3f}")


def format_time(seconds: float) -> str:
    """
    Formata tempo em segundos para string legível
    
    Args:
        seconds (float): Tempo em segundos
    
    Returns:
        str: Tempo formatado
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {secs:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {secs:.1f}s"


def get_class_names(class_mapping: dict) -> List[str]:
    """
    Obtém nomes das classes a partir do mapeamento
    
    Args:
        class_mapping (dict): Mapeamento de índices para nomes
    
    Returns:
        List[str]: Lista de nomes das classes
    """
    return [class_mapping[i] for i in sorted(class_mapping.keys())]


def create_confusion_matrix_plot(cm: np.ndarray, class_names: List[str], 
                                title: str = "Matriz de Confusão", 
                                save_path: Optional[str] = None):
    """
    Cria plot da matriz de confusão
    
    Args:
        cm (np.ndarray): Matriz de confusão
        class_names (List[str]): Nomes das classes
        title (str): Título do gráfico
        save_path (str): Caminho para salvar (opcional)
    """
    import seaborn as sns
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=16)
    plt.xlabel('Predição', fontsize=12)
    plt.ylabel('Verdadeiro', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
                       
    print("🧪 Testando funções auxiliares...")
    
                   
    check_gpu_availability()
    
                            
    print_system_info()
    
                                   
    create_directory_structure("/tmp/test_project")
    
    print("✅ Testes concluídos!")
