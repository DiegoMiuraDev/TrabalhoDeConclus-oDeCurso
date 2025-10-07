"""
Módulo para carregamento e manipulação do dataset Libras MNIST
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import os
from configs.config import DATASET_CONFIG, LIBRAS_CLASSES


class LibrasDatasetLoader:
    """
    Classe para carregar e manipular o dataset Libras MNIST
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Inicializa o carregador de dataset
        
        Args:
            data_dir (str): Diretório onde estão os dados
        """
        self.data_dir = Path(data_dir)
        self.df = None
        self.X = None
        self.y = None
        self.class_names = list(LIBRAS_CLASSES.values())
        self.n_classes = len(self.class_names)
        
    def load_dataset(self, csv_file: Optional[str] = None) -> pd.DataFrame:
        """
        Carrega o dataset a partir do arquivo CSV
        
        Args:
            csv_file (str): Nome do arquivo CSV (opcional)
        
        Returns:
            pd.DataFrame: Dataset carregado
        """
        if csv_file is None:
            csv_file = DATASET_CONFIG["csv_file"]
        
        csv_path = self.data_dir / csv_file
        
        if not csv_path.exists():
            # Tentar encontrar arquivos CSV no diretório
            csv_files = list(self.data_dir.glob("*.csv"))
            if csv_files:
                csv_path = csv_files[0]
                print(f"📁 Usando arquivo encontrado: {csv_path.name}")
            else:
                raise FileNotFoundError(f"Arquivo CSV não encontrado em: {self.data_dir}")
        
        print(f"📊 Carregando dataset: {csv_path}")
        self.df = pd.read_csv(csv_path)
        
        print(f"✅ Dataset carregado com sucesso!")
        print(f"   Dimensões: {self.df.shape}")
        print(f"   Colunas: {list(self.df.columns)}")
        
        return self.df
    
    def explore_dataset(self) -> Dict[str, Any]:
        """
        Explora o dataset e retorna informações básicas
        
        Returns:
            Dict[str, Any]: Informações do dataset
        """
        if self.df is None:
            raise ValueError("Dataset não carregado. Execute load_dataset() primeiro.")
        
        # Informações básicas
        info = {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.to_dict(),
            "null_counts": self.df.isnull().sum().to_dict(),
            "memory_usage": self.df.memory_usage(deep=True).sum()
        }
        
        # Identificar coluna de labels
        label_column = self.df.columns[0]  # Assumindo que a primeira coluna é o label
        info["label_column"] = label_column
        
        # Análise das classes
        unique_labels = self.df[label_column].unique()
        class_counts = self.df[label_column].value_counts().sort_index()
        
        info.update({
            "n_classes": len(unique_labels),
            "unique_labels": sorted(unique_labels),
            "class_counts": class_counts.to_dict(),
            "class_distribution": {
                "mean": class_counts.mean(),
                "std": class_counts.std(),
                "min": class_counts.min(),
                "max": class_counts.max()
            }
        })
        
        return info
    
    def prepare_data(self, label_column: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara os dados para treinamento
        
        Args:
            label_column (str): Nome da coluna de labels
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features (X) e labels (y)
        """
        if self.df is None:
            raise ValueError("Dataset não carregado. Execute load_dataset() primeiro.")
        
        if label_column is None:
            label_column = self.df.columns[0]
        
        # Separar features e labels
        pixel_columns = [col for col in self.df.columns if col != label_column]
        
        print(f"🏷️  Coluna de labels: {label_column}")
        print(f"🖼️  Colunas de pixels: {len(pixel_columns)}")
        
        # Extrair dados
        X = self.df[pixel_columns].values
        y = self.df[label_column].values
        
        # Determinar dimensões da imagem
        n_pixels = len(pixel_columns)
        img_size = int(np.sqrt(n_pixels))
        
        print(f"📐 Dimensões da imagem: {img_size}x{img_size}")
        print(f"📊 Dados extraídos:")
        print(f"   Features: {X.shape}")
        print(f"   Labels: {y.shape}")
        
        # Armazenar para uso posterior
        self.X = X
        self.y = y
        self.img_size = img_size
        self.pixel_columns = pixel_columns
        self.label_column = label_column
        
        return X, y
    
    def reshape_images(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Redimensiona as imagens para formato 2D
        
        Args:
            X (np.ndarray): Features (opcional, usa self.X se não fornecido)
        
        Returns:
            np.ndarray: Imagens redimensionadas
        """
        if X is None:
            if self.X is None:
                raise ValueError("Dados não preparados. Execute prepare_data() primeiro.")
            X = self.X
        
        if not hasattr(self, 'img_size'):
            n_pixels = X.shape[1]
            img_size = int(np.sqrt(n_pixels))
        else:
            img_size = self.img_size
        
        X_images = X.reshape(-1, img_size, img_size)
        print(f"🖼️  Imagens redimensionadas: {X_images.shape}")
        
        return X_images
    
    def normalize_images(self, X: Optional[np.ndarray] = None, 
                        min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
        """
        Normaliza as imagens para o range especificado
        
        Args:
            X (np.ndarray): Imagens (opcional)
            min_val (float): Valor mínimo
            max_val (float): Valor máximo
        
        Returns:
            np.ndarray: Imagens normalizadas
        """
        if X is None:
            if self.X is None:
                raise ValueError("Dados não preparados. Execute prepare_data() primeiro.")
            X = self.X
        
        # Normalizar para [0, 1] se os valores estiverem em [0, 255]
        if X.max() > 1.0:
            X_normalized = X.astype('float32') / 255.0
        else:
            X_normalized = X.astype('float32')
        
        # Ajustar para o range desejado
        if min_val != 0.0 or max_val != 1.0:
            X_normalized = X_normalized * (max_val - min_val) + min_val
        
        print(f"✅ Imagens normalizadas para [{min_val}, {max_val}]")
        print(f"   Range atual: [{X_normalized.min():.3f}, {X_normalized.max():.3f}]")
        
        return X_normalized
    
    def get_class_samples(self, n_samples: int = 1) -> Dict[int, np.ndarray]:
        """
        Obtém amostras de cada classe
        
        Args:
            n_samples (int): Número de amostras por classe
        
        Returns:
            Dict[int, np.ndarray]: Amostras por classe
        """
        if self.df is None:
            raise ValueError("Dataset não carregado. Execute load_dataset() primeiro.")
        
        samples = {}
        unique_labels = sorted(self.df[self.label_column].unique())
        
        for label in unique_labels:
            class_data = self.df[self.df[self.label_column] == label]
            if len(class_data) >= n_samples:
                samples[label] = class_data.head(n_samples)
            else:
                samples[label] = class_data
        
        return samples
    
    def get_dataset_info(self) -> str:
        """
        Retorna informações resumidas do dataset
        
        Returns:
            str: Informações formatadas
        """
        if self.df is None:
            return "Dataset não carregado"
        
        info = self.explore_dataset()
        
        summary = f"""
📊 Informações do Dataset Libras MNIST:
   Dimensões: {info['shape']}
   Classes: {info['n_classes']}
   Amostras por classe: {info['class_distribution']['mean']:.1f} ± {info['class_distribution']['std']:.1f}
   Range: {info['class_distribution']['min']} - {info['class_distribution']['max']}
   Memória: {info['memory_usage'] / 1024 / 1024:.1f} MB
        """
        
        return summary.strip()


def load_libras_dataset(data_dir: str = "data", csv_file: Optional[str] = None) -> LibrasDatasetLoader:
    """
    Função de conveniência para carregar o dataset
    
    Args:
        data_dir (str): Diretório dos dados
        csv_file (str): Arquivo CSV (opcional)
    
    Returns:
        LibrasDatasetLoader: Carregador configurado
    """
    loader = LibrasDatasetLoader(data_dir)
    loader.load_dataset(csv_file)
    return loader


if __name__ == "__main__":
    # Exemplo de uso
    print("🧪 Testando carregador de dataset...")
    
    # Criar diretório de teste
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Simular dados de teste
    n_samples = 100
    n_pixels = 784  # 28x28
    
    # Criar dados sintéticos
    data = {
        'label': np.random.randint(0, 24, n_samples),
        **{f'pixel_{i}': np.random.randint(0, 256, n_samples) for i in range(n_pixels)}
    }
    
    df = pd.DataFrame(data)
    df.to_csv(test_dir / "test_libras.csv", index=False)
    
    # Testar carregador
    loader = LibrasDatasetLoader(str(test_dir))
    loader.load_dataset("test_libras.csv")
    
    # Explorar dataset
    info = loader.explore_dataset()
    print(loader.get_dataset_info())
    
    # Preparar dados
    X, y = loader.prepare_data()
    X_images = loader.reshape_images()
    X_normalized = loader.normalize_images()
    
    print("✅ Teste concluído com sucesso!")
    
    # Limpar arquivos de teste
    import shutil
    shutil.rmtree(test_dir)
