#!/usr/bin/env python3
"""
Aplicativo Console para Reconhecimento de Libras
Funciona sem interface gráfica (para WSL/terminais)
"""

import cv2
import numpy as np
import time
import tensorflow as tf
from collections import deque

# Configurações
MODEL_PATH = "dataset/model_unquant.tflite"
LABELS_PATH = "dataset/labels.txt"
IMAGE_SIZE = 224

class LibrasConsoleApp:
    def __init__(self):
        self.cap = None
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.labels = []
        # Buffers para suavização
        self.pred_buffer = deque(maxlen=12)
        self.conf_buffer = deque(maxlen=12)
        self.current_label = None
        self.stable_count = 0
        self.required_stable_frames = 6
        self.ema_confidence = 0.0
        
        self.load_labels()
        self.load_model()
    
    def load_labels(self):
        """Carrega as labels"""
        try:
            with open(LABELS_PATH, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
            print(f"✅ Labels: {self.labels}")
        except Exception as e:
            print(f"❌ Erro ao carregar labels: {e}")
            self.labels = ["A", "E", "I", "O", "U"]
    
    def load_model(self):
        """Carrega modelo TensorFlow Lite"""
        try:
            print(f"\n📥 Carregando modelo TensorFlow Lite...")
            self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print(f"✅ Modelo carregado!")
            print(f"   Input shape: {self.input_details[0]['shape']}")
            print(f"   Output shape: {self.output_details[0]['shape']}")
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            raise
    
    def preprocess_frame(self, frame):
        """Pré-processa frame"""
        img = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(float) / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    
    def predict(self, frame):
        """Faz predição"""
        if self.interpreter is None:
            return None, 0.0, []
        
        try:
            processed = self.preprocess_frame(frame)
            input_data = processed.astype(np.float32)
            
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            
            max_idx = np.argmax(output)
            confidence = float(output[max_idx])
            predicted_class = self.labels[max_idx] if max_idx < len(self.labels) else "Desconhecido"
            
            all_predictions = []
            for i, conf in enumerate(output):
                class_name = self.labels[i] if i < len(self.labels) else f"Classe {i}"
                all_predictions.append((class_name, float(conf)))
            all_predictions.sort(key=lambda x: x[1], reverse=True)
            
            return predicted_class, confidence, all_predictions
        except Exception as e:
            print(f"Erro na predição: {e}")
            return None, 0.0, []

    def _smooth_and_confirm(self, prediction, confidence, min_confidence=0.55):
        """Aplica suavização temporal (maioria ponderada) e histerese de confirmação."""
        if prediction and confidence >= min_confidence:
            self.pred_buffer.append(prediction)
            self.conf_buffer.append(confidence)

        if not self.pred_buffer:
            return None, 0.0

        # Maioria ponderada por confiança
        weights = {}
        for p, c in zip(self.pred_buffer, self.conf_buffer):
            weights[p] = weights.get(p, 0.0) + c
        cand_label, cand_weight = max(weights.items(), key=lambda x: x[1])
        avg_conf = cand_weight / len(self.pred_buffer)

        # Histerese: só troca após estabilidade por N frames
        if self.current_label == cand_label:
            self.stable_count = min(self.required_stable_frames, self.stable_count + 1)
        else:
            # Permite troca mais rápida se confiança muito alta
            if avg_conf >= 0.8:
                self.stable_count = self.required_stable_frames
            else:
                self.stable_count = max(0, self.stable_count - 1)

            if self.stable_count <= 0:
                # iniciar nova candidatura
                self.current_label = cand_label
                self.stable_count = 1

        confirmed = self.stable_count >= self.required_stable_frames

        # EMA para confiança (suaviza variação visual)
        alpha = 0.2
        self.ema_confidence = (1 - alpha) * self.ema_confidence + alpha * avg_conf

        if confirmed:
            return self.current_label, self.ema_confidence
        else:
            return None, self.ema_confidence
    
    def run(self):
        """Loop principal"""
        print("\n" + "=" * 60)
        print("🤟 RECONHECIMENTO DE LIBRAS - CONSOLE")
        print("=" * 60)
        print("\nPressione 'q' para sair\n")
        
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("❌ Não foi possível acessar a câmera")
                return
            
            print("🎥 Câmera iniciada!")
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                
                # Predição
                prediction, confidence, all_predictions = self.predict(frame)
                confirmed_label, smooth_conf = self._smooth_and_confirm(prediction, confidence)
                
                # Desenha no frame
                if confirmed_label:
                    text = f"{confirmed_label} - {smooth_conf*100:.1f}%"
                    cv2.putText(frame, text, (20, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                
                # Mostra todas as predições
                y_offset = 100
                for i, (class_name, conf) in enumerate(all_predictions[:3]):
                    color = (0, 255, 0) if i == 0 else (255, 255, 255)
                    cv2.putText(frame, f"{class_name}: {conf*100:.1f}%",
                               (20, y_offset + i * 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                cv2.imshow("Reconhecimento de Libras", frame)
                
                # Print no console
                if confirmed_label:
                    print(f"\r{confirmed_label} ({smooth_conf*100:.1f}%) " + " " * 20, end='', flush=True)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                time.sleep(0.03)
        
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrompido pelo usuário")
        except Exception as e:
            print(f"\n❌ Erro: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("\n✅ Encerrado")

def main():
    print("=" * 60)
    print("🚀 Iniciando aplicativo console...")
    print("=" * 60)
    
    app = LibrasConsoleApp()
    app.run()

if __name__ == '__main__':
    main()

