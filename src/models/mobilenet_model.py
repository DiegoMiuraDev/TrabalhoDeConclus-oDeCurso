"""
Módulo para criação e gerenciamento do modelo MobileNetV2 para reconhecimento de Libras
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from typing import Tuple, Optional
import numpy as np


class MobileNetLibrasModel:
    """
    Classe para gerenciar o modelo MobileNetV2 para reconhecimento de Libras
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 3),
                 n_classes: int = 24, dropout_rate: float = 0.5,
                 dense_units: int = 128):
        """
        Inicializa o modelo
        
        Args:
            input_shape (Tuple[int, int, int]): Formato de entrada (altura, largura, canais)
            n_classes (int): Número de classes
            dropout_rate (float): Taxa de dropout
            dense_units (int): Unidades na camada densa
        """
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.model = None
        self.base_model = None
        
    def build_model(self, trainable_base: bool = False) -> keras.Model:
        """
        Constrói o modelo MobileNetV2 com Transfer Learning
        
        Args:
            trainable_base (bool): Se o modelo base deve ser treinável
        
        Returns:
            keras.Model: Modelo compilado
        """
        print("🏗️  Construindo modelo MobileNetV2...")
        
        # 1. Carregar modelo base pré-treinado
        self.base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        
        # Congelar camadas do modelo base
        self.base_model.trainable = trainable_base
        
        print(f"   Base model: MobileNetV2 (imagenet)")
        print(f"   Camadas base treináveis: {trainable_base}")
        print(f"   Total de camadas base: {len(self.base_model.layers)}")
        
        # 2. Construir o modelo completo
        inputs = keras.Input(shape=self.input_shape)
        
        # Pré-processamento do MobileNetV2
        x = preprocess_input(inputs)
        
        # Base model
        x = self.base_model(x, training=False)
        
        # Camadas personalizadas
        x = layers.Dense(self.dense_units, activation='relu', name='dense_1')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_1')(x)
        x = layers.Dense(self.dense_units // 2, activation='relu', name='dense_2')(x)
        x = layers.Dropout(self.dropout_rate / 2, name='dropout_2')(x)
        
        # Camada de saída
        outputs = layers.Dense(self.n_classes, activation='softmax', name='predictions')(x)
        
        # Criar modelo
        self.model = keras.Model(inputs, outputs, name='MobileNetV2_Libras')
        
        print(f"✅ Modelo construído com sucesso!")
        print(f"   Parâmetros totais: {self.model.count_params():,}")
        print(f"   Parâmetros treináveis: {sum([tf.size(w).numpy() for w in self.model.trainable_weights]):,}")
        
        return self.model
    
    def unfreeze_base_layers(self, n_layers: int = 20) -> None:
        """
        Descongela as últimas n camadas do modelo base para fine-tuning
        
        Args:
            n_layers (int): Número de camadas a descongelar
        """
        if self.base_model is None:
            raise ValueError("Modelo não foi construído ainda. Execute build_model() primeiro.")
        
        # Congelar todas as camadas primeiro
        self.base_model.trainable = True
        
        # Descongelar apenas as últimas n camadas
        for layer in self.base_model.layers[:-n_layers]:
            layer.trainable = False
        
        print(f"🔓 Descongeladas as últimas {n_layers} camadas do modelo base")
        print(f"   Parâmetros treináveis: {sum([tf.size(w).numpy() for w in self.model.trainable_weights]):,}")
    
    def compile_model(self, learning_rate: float = 0.001,
                     optimizer: str = 'adam',
                     loss: str = 'categorical_crossentropy',
                     metrics: list = ['accuracy']) -> None:
        """
        Compila o modelo
        
        Args:
            learning_rate (float): Taxa de aprendizado
            optimizer (str): Otimizador
            loss (str): Função de perda
            metrics (list): Métricas a serem avaliadas
        """
        if self.model is None:
            raise ValueError("Modelo não foi construído ainda. Execute build_model() primeiro.")
        
        # Configurar otimizador
        if optimizer.lower() == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer.lower() == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        # Compilar modelo
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )
        
        print(f"✅ Modelo compilado!")
        print(f"   Otimizador: {optimizer}")
        print(f"   Taxa de aprendizado: {learning_rate}")
        print(f"   Função de perda: {loss}")
        print(f"   Métricas: {metrics}")
    
    def get_model_summary(self) -> str:
        """
        Retorna o resumo do modelo
        
        Returns:
            str: Resumo do modelo
        """
        if self.model is None:
            return "Modelo não construído ainda"
        
        from io import StringIO
        import sys
        
        # Capturar o output do summary
        old_stdout = sys.stdout
        sys.stdout = summary_buffer = StringIO()
        self.model.summary()
        sys.stdout = old_stdout
        
        return summary_buffer.getvalue()
    
    def save_model(self, filepath: str, save_format: str = 'h5') -> None:
        """
        Salva o modelo
        
        Args:
            filepath (str): Caminho para salvar o modelo
            save_format (str): Formato de salvamento ('h5' ou 'tf')
        """
        if self.model is None:
            raise ValueError("Modelo não foi construído ainda.")
        
        self.model.save(filepath, save_format=save_format)
        print(f"💾 Modelo salvo em: {filepath}")
    
    def load_model(self, filepath: str) -> keras.Model:
        """
        Carrega um modelo salvo
        
        Args:
            filepath (str): Caminho do modelo
        
        Returns:
            keras.Model: Modelo carregado
        """
        self.model = keras.models.load_model(filepath)
        print(f"📂 Modelo carregado de: {filepath}")
        return self.model
    
    def predict(self, images: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Faz predições nas imagens
        
        Args:
            images (np.ndarray): Imagens para predição
            batch_size (int): Tamanho do batch
        
        Returns:
            np.ndarray: Predições (probabilidades)
        """
        if self.model is None:
            raise ValueError("Modelo não foi construído ou carregado ainda.")
        
        return self.model.predict(images, batch_size=batch_size, verbose=0)
    
    def predict_classes(self, images: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Prediz as classes das imagens
        
        Args:
            images (np.ndarray): Imagens para predição
            batch_size (int): Tamanho do batch
        
        Returns:
            np.ndarray: Classes preditas
        """
        predictions = self.predict(images, batch_size)
        return np.argmax(predictions, axis=1)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                batch_size: int = 32, verbose: int = 1) -> Tuple[float, float]:
        """
        Avalia o modelo
        
        Args:
            X_test (np.ndarray): Dados de teste
            y_test (np.ndarray): Labels de teste
            batch_size (int): Tamanho do batch
            verbose (int): Nível de verbosidade
        
        Returns:
            Tuple[float, float]: (loss, accuracy)
        """
        if self.model is None:
            raise ValueError("Modelo não foi construído ou carregado ainda.")
        
        return self.model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)


def create_mobilenet_model(input_shape: Tuple[int, int, int] = (224, 224, 3),
                          n_classes: int = 24,
                          dropout_rate: float = 0.5,
                          dense_units: int = 128,
                          trainable_base: bool = False) -> keras.Model:
    """
    Função de conveniência para criar um modelo MobileNetV2 para Libras
    
    Args:
        input_shape (Tuple[int, int, int]): Formato de entrada
        n_classes (int): Número de classes
        dropout_rate (float): Taxa de dropout
        dense_units (int): Unidades na camada densa
        trainable_base (bool): Se o modelo base deve ser treinável
    
    Returns:
        keras.Model: Modelo MobileNetV2
    """
    model_builder = MobileNetLibrasModel(
        input_shape=input_shape,
        n_classes=n_classes,
        dropout_rate=dropout_rate,
        dense_units=dense_units
    )
    
    model = model_builder.build_model(trainable_base=trainable_base)
    
    return model


def create_mobilenet_with_config(config: dict) -> keras.Model:
    """
    Cria modelo MobileNetV2 a partir de um dicionário de configuração
    
    Args:
        config (dict): Configurações do modelo
    
    Returns:
        keras.Model: Modelo criado
    """
    return create_mobilenet_model(
        input_shape=config.get('input_shape', (224, 224, 3)),
        n_classes=config.get('n_classes', 24),
        dropout_rate=config.get('dropout_rate', 0.5),
        dense_units=config.get('dense_units', 128),
        trainable_base=config.get('trainable_base', False)
    )


if __name__ == "__main__":
    # Exemplo de uso
    print("🧪 Testando modelo MobileNetV2...")
    
    # Criar modelo
    model_builder = MobileNetLibrasModel(
        input_shape=(224, 224, 3),
        n_classes=24,
        dropout_rate=0.5,
        dense_units=128
    )
    
    # Construir modelo
    model = model_builder.build_model(trainable_base=False)
    
    # Compilar modelo
    model_builder.compile_model(learning_rate=0.001)
    
    # Mostrar resumo
    print("\n" + "="*60)
    print("RESUMO DO MODELO")
    print("="*60)
    print(model_builder.get_model_summary())
    
    # Testar predição com dados sintéticos
    print("\n🧪 Testando predição...")
    test_images = np.random.rand(5, 224, 224, 3).astype(np.float32)
    predictions = model_builder.predict(test_images)
    predicted_classes = model_builder.predict_classes(test_images)
    
    print(f"✅ Predições: {predicted_classes}")
    print(f"   Forma das probabilidades: {predictions.shape}")
    
    print("\n✅ Teste concluído com sucesso!")

