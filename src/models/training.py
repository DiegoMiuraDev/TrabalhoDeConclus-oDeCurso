"""
Módulo para treinamento de modelos de reconhecimento de Libras
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, 
    TensorBoard, Callback
)
from typing import Tuple, Dict, List, Optional, Any
import numpy as np
import time
from pathlib import Path
from datetime import datetime


class TrainingMetricsLogger(Callback):
    """
    Callback personalizado para logar métricas durante o treinamento
    """
    
    def __init__(self):
        super().__init__()
        self.epoch_times = []
        self.start_time = None
        
    def on_train_begin(self, logs=None):
        """Início do treinamento"""
        self.start_time = time.time()
        print("🚀 Iniciando treinamento...")
        
    def on_epoch_begin(self, epoch, logs=None):
        """Início de cada época"""
        self.epoch_start_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        """Fim de cada época"""
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        
                                          
        avg_time = np.mean(self.epoch_times)
        remaining_epochs = self.params['epochs'] - (epoch + 1)
        estimated_time = avg_time * remaining_epochs
        
        print(f"\n⏱️  Tempo da época: {epoch_time:.1f}s | Tempo estimado restante: {estimated_time/60:.1f}min")
        
    def on_train_end(self, logs=None):
        """Fim do treinamento"""
        total_time = time.time() - self.start_time
        print(f"\n✅ Treinamento concluído em {total_time/60:.1f} minutos")


class LibrasModelTrainer:
    """
    Classe para gerenciar o treinamento de modelos de reconhecimento de Libras
    """
    
    def __init__(self, model: keras.Model, model_name: str = "libras_model"):
        """
        Inicializa o treinador
        
        Args:
            model (keras.Model): Modelo a ser treinado
            model_name (str): Nome do modelo
        """
        self.model = model
        self.model_name = model_name
        self.history = None
        self.callbacks = []
        
    def setup_callbacks(self, 
                       checkpoint_path: Optional[str] = None,
                       early_stopping_patience: int = 10,
                       reduce_lr_patience: int = 5,
                       min_lr: float = 1e-7,
                       tensorboard_log_dir: Optional[str] = None,
                       custom_callbacks: Optional[List[Callback]] = None) -> List[Callback]:
        """
        Configura callbacks para o treinamento
        
        Args:
            checkpoint_path (str): Caminho para salvar checkpoints
            early_stopping_patience (int): Paciência para early stopping
            reduce_lr_patience (int): Paciência para redução de LR
            min_lr (float): Taxa de aprendizado mínima
            tensorboard_log_dir (str): Diretório para logs do TensorBoard
            custom_callbacks (List[Callback]): Callbacks personalizados
        
        Returns:
            List[Callback]: Lista de callbacks configurados
        """
        callbacks = []
        
                           
        early_stop = EarlyStopping(
            monitor='val_accuracy',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        )
        callbacks.append(early_stop)
        print(f"✅ Early Stopping configurado (patience={early_stopping_patience})")
        
                                            
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            verbose=1,
            mode='min'
        )
        callbacks.append(reduce_lr)
        print(f"✅ Reduce LR on Plateau configurado (patience={reduce_lr_patience})")
        
                             
        if checkpoint_path:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            checkpoint = ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            )
            callbacks.append(checkpoint)
            print(f"✅ Model Checkpoint configurado: {checkpoint_path}")
        
                        
        if tensorboard_log_dir:
            Path(tensorboard_log_dir).mkdir(parents=True, exist_ok=True)
            tensorboard = TensorBoard(
                log_dir=tensorboard_log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                update_freq='epoch'
            )
            callbacks.append(tensorboard)
            print(f"✅ TensorBoard configurado: {tensorboard_log_dir}")
        
                                  
        metrics_logger = TrainingMetricsLogger()
        callbacks.append(metrics_logger)
        
                                     
        if custom_callbacks:
            callbacks.extend(custom_callbacks)
            print(f"✅ {len(custom_callbacks)} callback(s) personalizado(s) adicionado(s)")
        
        self.callbacks = callbacks
        return callbacks
    
    def train(self,
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None,
             epochs: int = 50,
             batch_size: int = 32,
             validation_split: float = 0.1,
             verbose: int = 1,
             use_data_augmentation: bool = False) -> keras.callbacks.History:
        """
        Treina o modelo
        
        Args:
            X_train (np.ndarray): Dados de treino
            y_train (np.ndarray): Labels de treino
            X_val (np.ndarray): Dados de validação (opcional)
            y_val (np.ndarray): Labels de validação (opcional)
            epochs (int): Número de épocas
            batch_size (int): Tamanho do batch
            validation_split (float): Proporção para validação se X_val não fornecido
            verbose (int): Nível de verbosidade
            use_data_augmentation (bool): Se deve usar data augmentation
        
        Returns:
            keras.callbacks.History: Histórico de treinamento
        """
        print("\n" + "="*60)
        print("🎯 INICIANDO TREINAMENTO")
        print("="*60)
        print(f"📊 Dados de treino: {X_train.shape}")
        if X_val is not None:
            print(f"📊 Dados de validação: {X_val.shape}")
        print(f"⚙️  Épocas: {epochs}")
        print(f"⚙️  Batch size: {batch_size}")
        print(f"⚙️  Data augmentation: {use_data_augmentation}")
        print("="*60 + "\n")
        
                                    
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            validation_split = 0.0
        else:
            validation_data = None
        
                           
        if use_data_augmentation:
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            
            datagen = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=False,
                fill_mode='nearest'
            )
            
            datagen.fit(X_train)
            
                                           
            self.history = self.model.fit(
                datagen.flow(X_train, y_train, batch_size=batch_size),
                validation_data=validation_data,
                epochs=epochs,
                steps_per_epoch=len(X_train) // batch_size,
                callbacks=self.callbacks,
                verbose=verbose
            )
        else:
                                           
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=validation_data,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=self.callbacks,
                verbose=verbose
            )
        
        return self.history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                batch_size: int = 32, verbose: int = 1) -> Dict[str, float]:
        """
        Avalia o modelo no conjunto de teste
        
        Args:
            X_test (np.ndarray): Dados de teste
            y_test (np.ndarray): Labels de teste
            batch_size (int): Tamanho do batch
            verbose (int): Nível de verbosidade
        
        Returns:
            Dict[str, float]: Métricas de avaliação
        """
        print("\n📈 Avaliando modelo no conjunto de teste...")
        
        results = self.model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)
        
        metrics = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics[metric_name] = results[i]
        
        print(f"\n✅ Resultados da avaliação:")
        for metric_name, value in metrics.items():
            print(f"   {metric_name}: {value:.4f}")
        
        return metrics
    
    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Faz predições
        
        Args:
            X (np.ndarray): Dados para predição
            batch_size (int): Tamanho do batch
        
        Returns:
            np.ndarray: Predições (probabilidades)
        """
        return self.model.predict(X, batch_size=batch_size, verbose=0)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Retorna resumo do treinamento
        
        Returns:
            Dict[str, Any]: Resumo do treinamento
        """
        if self.history is None:
            return {"status": "Modelo não foi treinado ainda"}
        
        history_dict = self.history.history
        
        summary = {
            "epochs_trained": len(history_dict['loss']),
            "final_train_loss": history_dict['loss'][-1],
            "final_train_accuracy": history_dict['accuracy'][-1],
            "best_train_accuracy": max(history_dict['accuracy']),
            "best_train_accuracy_epoch": np.argmax(history_dict['accuracy']) + 1
        }
        
        if 'val_loss' in history_dict:
            summary.update({
                "final_val_loss": history_dict['val_loss'][-1],
                "final_val_accuracy": history_dict['val_accuracy'][-1],
                "best_val_accuracy": max(history_dict['val_accuracy']),
                "best_val_accuracy_epoch": np.argmax(history_dict['val_accuracy']) + 1
            })
        
        return summary
    
    def save_model(self, filepath: str) -> None:
        """
        Salva o modelo
        
        Args:
            filepath (str): Caminho para salvar
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)
        print(f"💾 Modelo salvo em: {filepath}")
    
    def save_history(self, filepath: str) -> None:
        """
        Salva o histórico de treinamento
        
        Args:
            filepath (str): Caminho para salvar
        """
        if self.history is None:
            print("⚠️  Nenhum histórico para salvar")
            return
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        np.save(filepath, self.history.history)
        print(f"💾 Histórico salvo em: {filepath}")


def train_libras_model(model: keras.Model,
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_val: np.ndarray,
                      y_val: np.ndarray,
                      epochs: int = 50,
                      batch_size: int = 32,
                      checkpoint_path: str = "models/libras_model.h5",
                      early_stopping_patience: int = 10,
                      use_data_augmentation: bool = False) -> Tuple[keras.Model, keras.callbacks.History]:
    """
    Função de conveniência para treinar um modelo de Libras
    
    Args:
        model (keras.Model): Modelo a ser treinado
        X_train (np.ndarray): Dados de treino
        y_train (np.ndarray): Labels de treino
        X_val (np.ndarray): Dados de validação
        y_val (np.ndarray): Labels de validação
        epochs (int): Número de épocas
        batch_size (int): Tamanho do batch
        checkpoint_path (str): Caminho para salvar checkpoints
        early_stopping_patience (int): Paciência para early stopping
        use_data_augmentation (bool): Se deve usar data augmentation
    
    Returns:
        Tuple[keras.Model, keras.callbacks.History]: Modelo treinado e histórico
    """
                   
    trainer = LibrasModelTrainer(model)
    
                          
    trainer.setup_callbacks(
        checkpoint_path=checkpoint_path,
        early_stopping_patience=early_stopping_patience
    )
    
             
    history = trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        use_data_augmentation=use_data_augmentation
    )
    
    return model, history


if __name__ == "__main__":
                    
    print("🧪 Testando módulo de treinamento...")
    
    from models.mobilenet_model import create_mobilenet_model
    
                  
    model = create_mobilenet_model(
        input_shape=(224, 224, 3),
        n_classes=24,
        dropout_rate=0.5,
        dense_units=128
    )
    
                     
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
                            
    print("\n📊 Criando dados sintéticos...")
    X_train = np.random.rand(100, 224, 224, 3).astype(np.float32)
    y_train = tf.keras.utils.to_categorical(np.random.randint(0, 24, 100), 24)
    X_val = np.random.rand(20, 224, 224, 3).astype(np.float32)
    y_val = tf.keras.utils.to_categorical(np.random.randint(0, 24, 20), 24)
    
                   
    trainer = LibrasModelTrainer(model, "test_model")
    
                          
    trainer.setup_callbacks(
        checkpoint_path="/tmp/test_model.h5",
        early_stopping_patience=3,
        reduce_lr_patience=2
    )
    
                                
    print("\n🎯 Treinando modelo de teste...")
    history = trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=5,
        batch_size=16,
        verbose=1
    )
    
                           
    print("\n📊 Resumo do Treinamento:")
    summary = trainer.get_training_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Teste concluído com sucesso!")

