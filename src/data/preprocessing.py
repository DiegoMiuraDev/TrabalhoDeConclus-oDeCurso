"""
Módulo de pré-processamento de imagens para o projeto de reconhecimento de Libras
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from configs.config import DATASET_CONFIG, MODEL_CONFIG


class ImagePreprocessor:
    """
    Classe para pré-processamento de imagens do dataset Libras
    """
    
    def __init__(self, target_size: Tuple[int, int] = None, channels: int = 3):
        """
        Inicializa o pré-processador
        
        Args:
            target_size (Tuple[int, int]): Tamanho alvo das imagens
            channels (int): Número de canais (1 para grayscale, 3 para RGB)
        """
        self.target_size = target_size or DATASET_CONFIG["target_size"]
        self.channels = channels
        self.original_size = DATASET_CONFIG["original_size"]
        
    def grayscale_to_rgb(self, images: np.ndarray) -> np.ndarray:
        """
        Converte imagens em escala de cinza para RGB
        
        Args:
            images (np.ndarray): Imagens em escala de cinza (N, H, W)
        
        Returns:
            np.ndarray: Imagens RGB (N, H, W, 3)
        """
        if len(images.shape) == 3:
            # Adicionar dimensão de canal
            images = np.expand_dims(images, axis=-1)
        
        # Repetir canal para criar RGB
        rgb_images = np.repeat(images, 3, axis=-1)
        
        print(f"🔄 Convertido para RGB: {images.shape} -> {rgb_images.shape}")
        return rgb_images
    
    def resize_images(self, images: np.ndarray, 
                     target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Redimensiona imagens para o tamanho alvo
        
        Args:
            images (np.ndarray): Imagens de entrada
            target_size (Tuple[int, int]): Tamanho alvo (opcional)
        
        Returns:
            np.ndarray: Imagens redimensionadas
        """
        if target_size is None:
            target_size = self.target_size
        
        if images.shape[1:3] == target_size:
            print(f"✅ Imagens já estão no tamanho correto: {target_size}")
            return images
        
        resized_images = []
        for img in images:
            if len(img.shape) == 2:  # Grayscale
                resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            else:  # RGB
                resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            resized_images.append(resized)
        
        resized_images = np.array(resized_images)
        print(f"📐 Redimensionado: {images.shape} -> {resized_images.shape}")
        
        return resized_images
    
    def normalize_images(self, images: np.ndarray, 
                        min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
        """
        Normaliza imagens para o range especificado
        
        Args:
            images (np.ndarray): Imagens de entrada
            min_val (float): Valor mínimo
            max_val (float): Valor máximo
        
        Returns:
            np.ndarray: Imagens normalizadas
        """
        # Converter para float32
        images = images.astype('float32')
        
        # Normalizar para [0, 1] se necessário
        if images.max() > 1.0:
            images = images / 255.0
        
        # Ajustar para o range desejado
        if min_val != 0.0 or max_val != 1.0:
            images = images * (max_val - min_val) + min_val
        
        print(f"✅ Normalizado para [{min_val}, {max_val}]")
        print(f"   Range atual: [{images.min():.3f}, {images.max():.3f}]")
        
        return images
    
    def preprocess_for_mobilenet(self, images: np.ndarray) -> np.ndarray:
        """
        Pré-processa imagens para o modelo MobileNetV2
        
        Args:
            images (np.ndarray): Imagens de entrada (N, H, W) ou (N, H, W, C)
        
        Returns:
            np.ndarray: Imagens pré-processadas (N, 224, 224, 3)
        """
        print("🔄 Pré-processando para MobileNetV2...")
        
        # 1. Converter para RGB se necessário
        if len(images.shape) == 3 or images.shape[-1] == 1:
            images = self.grayscale_to_rgb(images)
        
        # 2. Redimensionar para 224x224
        images = self.resize_images(images, (224, 224))
        
        # 3. Normalizar para [0, 1]
        images = self.normalize_images(images)
        
        print(f"✅ Pré-processamento concluído: {images.shape}")
        return images
    
    def augment_images(self, images: np.ndarray, labels: np.ndarray,
                      augmentation_config: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplica data augmentation nas imagens
        
        Args:
            images (np.ndarray): Imagens de entrada
            labels (np.ndarray): Labels correspondentes
            augmentation_config (dict): Configuração de augmentation
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Imagens e labels aumentados
        """
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        if augmentation_config is None:
            augmentation_config = {
                "rotation_range": 15,
                "width_shift_range": 0.1,
                "height_shift_range": 0.1,
                "shear_range": 0.1,
                "zoom_range": 0.1,
                "horizontal_flip": False,
                "fill_mode": "nearest"
            }
        
        # Criar gerador de augmentation
        datagen = ImageDataGenerator(**augmentation_config)
        
        # Aplicar augmentation
        augmented_images = []
        augmented_labels = []
        
        for i in range(len(images)):
            img = images[i]
            label = labels[i]
            
            # Expandir dimensões para o gerador
            img_expanded = np.expand_dims(img, axis=0)
            label_expanded = np.expand_dims(label, axis=0)
            
            # Gerar amostra aumentada
            aug_img, aug_label = next(datagen.flow(img_expanded, label_expanded, batch_size=1))
            
            augmented_images.append(aug_img[0])
            augmented_labels.append(aug_label[0])
        
        augmented_images = np.array(augmented_images)
        augmented_labels = np.array(augmented_labels)
        
        print(f"🔄 Data augmentation aplicado: {images.shape} -> {augmented_images.shape}")
        
        return augmented_images, augmented_labels
    
    def prepare_training_data(self, X: np.ndarray, y: np.ndarray,
                            test_size: float = 0.2, validation_size: float = 0.1,
                            random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepara dados para treinamento com divisão train/validation/test
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Labels
            test_size (float): Proporção do conjunto de teste
            validation_size (float): Proporção do conjunto de validação
            random_state (int): Seed para reprodutibilidade
        
        Returns:
            Tuple: X_train, X_val, X_test, y_train, y_val, y_test
        """
        print("📚 Preparando dados para treinamento...")
        
        # 1. Pré-processar imagens
        X_processed = self.preprocess_for_mobilenet(X)
        
        # 2. Converter labels para categorical
        y_categorical = to_categorical(y, num_classes=len(np.unique(y)))
        
        # 3. Dividir em train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_categorical,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        # 4. Dividir train em train/validation
        if validation_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=validation_size,
                random_state=random_state,
                stratify=np.argmax(y_train, axis=1)
            )
        else:
            X_val, y_val = None, None
        
        print(f"✅ Dados preparados:")
        print(f"   Treino: {X_train.shape[0]} amostras")
        if X_val is not None:
            print(f"   Validação: {X_val.shape[0]} amostras")
        print(f"   Teste: {X_test.shape[0]} amostras")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_data_generator(self, X: np.ndarray, y: np.ndarray,
                            batch_size: int = 32, shuffle: bool = True,
                            augmentation: bool = False) -> 'ImageDataGenerator':
        """
        Cria gerador de dados para treinamento
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Labels
            batch_size (int): Tamanho do batch
            shuffle (bool): Se deve embaralhar os dados
            augmentation (bool): Se deve aplicar augmentation
        
        Returns:
            ImageDataGenerator: Gerador de dados
        """
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        if augmentation:
            datagen = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=False,
                fill_mode="nearest"
            )
        else:
            datagen = ImageDataGenerator()
        
        return datagen.flow(X, y, batch_size=batch_size, shuffle=shuffle)


def preprocess_libras_images(images: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Função de conveniência para pré-processar imagens de Libras
    
    Args:
        images (np.ndarray): Imagens de entrada
        target_size (Tuple[int, int]): Tamanho alvo
    
    Returns:
        np.ndarray: Imagens pré-processadas
    """
    preprocessor = ImagePreprocessor(target_size)
    return preprocessor.preprocess_for_mobilenet(images)


if __name__ == "__main__":
    # Exemplo de uso
    print("🧪 Testando pré-processador...")
    
    # Criar dados sintéticos
    n_samples = 10
    original_size = (28, 28)
    target_size = (224, 224)
    
    # Imagens em escala de cinza
    images = np.random.randint(0, 256, (n_samples, *original_size), dtype=np.uint8)
    labels = np.random.randint(0, 24, n_samples)
    
    print(f"📊 Dados de entrada: {images.shape}")
    
    # Testar pré-processador
    preprocessor = ImagePreprocessor(target_size)
    
    # Pré-processar para MobileNet
    processed_images = preprocessor.preprocess_for_mobilenet(images)
    print(f"✅ Imagens processadas: {processed_images.shape}")
    
    # Preparar dados para treinamento
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_training_data(
        images, labels, test_size=0.2, validation_size=0.1
    )
    
    print(f"📚 Divisão dos dados:")
    print(f"   Treino: {X_train.shape}")
    print(f"   Validação: {X_val.shape}")
    print(f"   Teste: {X_test.shape}")
    
    print("✅ Teste concluído com sucesso!")
