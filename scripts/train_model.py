#!/usr/bin/env python3
"""
Script para treinar o modelo de reconhecimento de Libras
"""

import sys
import os
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from data.dataset_loader import LibrasDatasetLoader
from data.preprocessing import ImagePreprocessor
from models.mobilenet_model import create_mobilenet_model
from utils.helpers import check_gpu_availability, print_system_info
from visualization.plots import LibrasVisualizer
from configs.config import TRAINING_CONFIG, MODEL_CONFIG, DATASET_CONFIG


def main():
    """
    Função principal para treinar o modelo
    """
    print("🚀 Iniciando treinamento do modelo de reconhecimento de Libras")
    print("=" * 60)
    
    # Verificar sistema
    print_system_info()
    check_gpu_availability()
    
    # Configurações
    data_dir = "data"
    model_save_path = "models/libras_model.h5"
    results_dir = "results"
    
    # Criar diretórios
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(exist_ok=True)
    
    try:
        # 1. Carregar dataset
        print("\n📊 Carregando dataset...")
        loader = LibrasDatasetLoader(data_dir)
        df = loader.load_dataset()
        
        # Explorar dataset
        info = loader.explore_dataset()
        print(loader.get_dataset_info())
        
        # 2. Preparar dados
        print("\n🔄 Preparando dados...")
        X, y = loader.prepare_data()
        
        # 3. Pré-processar imagens
        print("\n🖼️  Pré-processando imagens...")
        preprocessor = ImagePreprocessor()
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_training_data(
            X, y, 
            test_size=DATASET_CONFIG["test_split"],
            validation_size=DATASET_CONFIG["validation_split"]
        )
        
        # 4. Criar modelo
        print("\n🏗️  Criando modelo...")
        model = create_mobilenet_model(
            input_shape=MODEL_CONFIG["input_shape"],
            n_classes=DATASET_CONFIG["n_classes"],
            dropout_rate=MODEL_CONFIG["dropout_rate"],
            dense_units=MODEL_CONFIG["dense_units"]
        )
        
        # Compilar modelo
        model.compile(
            optimizer=Adam(learning_rate=TRAINING_CONFIG["learning_rate"]),
            loss=TRAINING_CONFIG["loss"],
            metrics=TRAINING_CONFIG["metrics"]
        )
        
        print(f"✅ Modelo criado com {model.count_params():,} parâmetros")
        
        # 5. Callbacks
        callbacks = [
            EarlyStopping(
                monitor=TRAINING_CONFIG["early_stopping"]["monitor"],
                patience=TRAINING_CONFIG["early_stopping"]["patience"],
                restore_best_weights=TRAINING_CONFIG["early_stopping"]["restore_best_weights"],
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor=TRAINING_CONFIG["reduce_lr"]["monitor"],
                factor=TRAINING_CONFIG["reduce_lr"]["factor"],
                patience=TRAINING_CONFIG["reduce_lr"]["patience"],
                min_lr=TRAINING_CONFIG["reduce_lr"]["min_lr"],
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # 6. Treinar modelo
        print("\n🎯 Iniciando treinamento...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=TRAINING_CONFIG["epochs"],
            batch_size=TRAINING_CONFIG["batch_size"],
            callbacks=callbacks,
            verbose=1
        )
        
        # 7. Avaliar modelo
        print("\n📈 Avaliando modelo...")
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"📊 Resultados:")
        print(f"   Treino - Acurácia: {train_acc:.4f}, Perda: {train_loss:.4f}")
        print(f"   Validação - Acurácia: {val_acc:.4f}, Perda: {val_loss:.4f}")
        print(f"   Teste - Acurácia: {test_acc:.4f}, Perda: {test_loss:.4f}")
        
        # 8. Salvar resultados
        print("\n💾 Salvando resultados...")
        
        # Salvar histórico
        np.save(f"{results_dir}/training_history.npy", history.history)
        
        # Salvar métricas
        metrics = {
            'train_acc': train_acc,
            'train_loss': train_loss,
            'val_acc': val_acc,
            'val_loss': val_loss,
            'test_acc': test_acc,
            'test_loss': test_loss
        }
        np.save(f"{results_dir}/metrics.npy", metrics)
        
        # 9. Visualizações
        print("\n📊 Gerando visualizações...")
        visualizer = LibrasVisualizer()
        
        # Histórico de treinamento
        visualizer.plot_training_history(
            history.history,
            save_path=f"{results_dir}/training_history.png"
        )
        
        # Matriz de confusão
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        visualizer.plot_confusion_matrix(
            cm,
            save_path=f"{results_dir}/confusion_matrix.png"
        )
        
        print(f"✅ Treinamento concluído com sucesso!")
        print(f"📁 Modelo salvo em: {model_save_path}")
        print(f"📊 Resultados salvos em: {results_dir}")
        
    except Exception as e:
        print(f"❌ Erro durante o treinamento: {e}")
        raise


if __name__ == "__main__":
    main()
