#!/usr/bin/env python3
"""
Script para demonstração em tempo real do reconhecimento de Libras
"""

import sys
import os
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from configs.config import REALTIME_CONFIG, LIBRAS_CLASSES
from utils.helpers import resize_image, grayscale_to_rgb, normalize_image


class LibrasRealTimeRecognizer:
    """
    Classe para reconhecimento em tempo real de sinais de Libras
    """
    
    def __init__(self, model_path: str = "models/libras_model.h5"):
        """
        Inicializa o reconhecedor
        
        Args:
            model_path (str): Caminho para o modelo treinado
        """
        self.model_path = model_path
        self.model = None
        self.class_names = list(LIBRAS_CLASSES.values())
        self.confidence_threshold = REALTIME_CONFIG["confidence_threshold"]
        self.prediction_interval = REALTIME_CONFIG["prediction_interval"]
        self.frame_count = 0
        
        # Carregar modelo
        self.load_model()
    
    def load_model(self):
        """
        Carrega o modelo treinado
        """
        try:
            print(f"📥 Carregando modelo: {self.model_path}")
            self.model = load_model(self.model_path)
            print("✅ Modelo carregado com sucesso!")
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            raise
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Pré-processa um frame da webcam
        
        Args:
            frame (np.ndarray): Frame da webcam
        
        Returns:
            np.ndarray: Frame pré-processado
        """
        # Converter para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Redimensionar para 224x224
        resized = resize_image(gray, (224, 224))
        
        # Converter para RGB
        rgb = grayscale_to_rgb(resized)
        
        # Normalizar
        normalized = normalize_image(rgb, 0.0, 1.0)
        
        # Adicionar dimensão de batch
        processed = np.expand_dims(normalized, axis=0)
        
        return processed
    
    def predict(self, frame: np.ndarray) -> tuple:
        """
        Faz predição em um frame
        
        Args:
            frame (np.ndarray): Frame da webcam
        
        Returns:
            tuple: (classe_predita, confiança)
        """
        # Pré-processar frame
        processed_frame = self.preprocess_frame(frame)
        
        # Fazer predição
        predictions = self.model.predict(processed_frame, verbose=0)
        
        # Obter classe e confiança
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        return class_idx, confidence
    
    def draw_prediction(self, frame: np.ndarray, class_idx: int, 
                       confidence: float) -> np.ndarray:
        """
        Desenha a predição no frame
        
        Args:
            frame (np.ndarray): Frame original
            class_idx (int): Índice da classe predita
            confidence (float): Confiança da predição
        
        Returns:
            np.ndarray: Frame com predição desenhada
        """
        # Obter nome da classe
        class_name = self.class_names[class_idx]
        
        # Determinar cor baseada na confiança
        if confidence >= self.confidence_threshold:
            color = (0, 255, 0)  # Verde
            status = "ALTO"
        else:
            color = (0, 0, 255)  # Vermelho
            status = "BAIXO"
        
        # Desenhar retângulo de fundo
        cv2.rectangle(frame, (10, 10), (400, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 100), color, 2)
        
        # Desenhar texto
        cv2.putText(frame, f"Letra: {class_name}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Confianca: {confidence:.3f}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Status: {status}", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame
    
    def run(self):
        """
        Executa o reconhecimento em tempo real
        """
        print("🎥 Iniciando reconhecimento em tempo real...")
        print("📋 Instruções:")
        print("   - Posicione sua mão na frente da câmera")
        print("   - Faça o sinal da letra desejada")
        print("   - Pressione 'q' para sair")
        print("   - Pressione 's' para salvar frame")
        print("=" * 50)
        
        # Inicializar câmera
        cap = cv2.VideoCapture(REALTIME_CONFIG["camera_index"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, REALTIME_CONFIG["frame_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REALTIME_CONFIG["frame_height"])
        
        if not cap.isOpened():
            print("❌ Erro ao abrir câmera")
            return
        
        # Variáveis para controle
        last_prediction = None
        last_confidence = 0.0
        
        try:
            while True:
                # Capturar frame
                ret, frame = cap.read()
                if not ret:
                    print("❌ Erro ao capturar frame")
                    break
                
                # Fazer predição a cada N frames
                if self.frame_count % self.prediction_interval == 0:
                    try:
                        class_idx, confidence = self.predict(frame)
                        last_prediction = class_idx
                        last_confidence = confidence
                    except Exception as e:
                        print(f"⚠️  Erro na predição: {e}")
                
                # Desenhar predição
                if last_prediction is not None:
                    frame = self.draw_prediction(frame, last_prediction, last_confidence)
                
                # Desenhar instruções
                cv2.putText(frame, "Pressione 'q' para sair, 's' para salvar", 
                           (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (255, 255, 255), 1)
                
                # Mostrar frame
                cv2.imshow('Reconhecimento de Libras', frame)
                
                # Controle de teclado
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Salvar frame
                    filename = f"captured_frame_{self.frame_count}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Frame salvo: {filename}")
                
                self.frame_count += 1
        
        except KeyboardInterrupt:
            print("\n⏹️  Interrompido pelo usuário")
        
        finally:
            # Limpar recursos
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Reconhecimento finalizado")


def main():
    """
    Função principal
    """
    print("🚀 Sistema de Reconhecimento de Libras em Tempo Real")
    print("=" * 60)
    
    # Verificar se o modelo existe
    model_path = "models/libras_model.h5"
    if not Path(model_path).exists():
        print(f"❌ Modelo não encontrado: {model_path}")
        print("📋 Execute primeiro o script de treinamento:")
        print("   python scripts/train_model.py")
        return
    
    try:
        # Criar reconhecedor
        recognizer = LibrasRealTimeRecognizer(model_path)
        
        # Executar reconhecimento
        recognizer.run()
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        raise


if __name__ == "__main__":
    main()
