#!/usr/bin/env python3
"""
Aplicativo Desktop para Reconhecimento de Libras
Interface gráfica moderna usando Tkinter
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import time
import os

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Configurações
MODEL_PATH = "dataset/model_unquant.tflite"
LABELS_PATH = "dataset/labels.txt"
IMAGE_SIZE = 224

class LibrasDesktopApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🤟 Reconhecimento de Libras - Desktop")
        self.root.geometry("900x750")
        self.root.configure(bg='#f0f0f0')
        
        # Variáveis
        self.cap = None
        self.is_running = False
        self.interpreter = None  # TensorFlow Lite interpreter
        self.input_details = None
        self.output_details = None
        self.labels = []
        
        # Carrega labels
        self.load_labels()
        
        # Cria interface
        self.create_ui()
        
        # Carrega modelo em thread separada
        threading.Thread(target=self.load_model, daemon=True).start()
    
    def load_labels(self):
        """Carrega as labels do modelo"""
        try:
            if os.path.exists(LABELS_PATH):
                with open(LABELS_PATH, 'r') as f:
                    self.labels = [line.strip() for line in f.readlines()]
                print(f"✅ Labels: {self.labels}")
        except Exception as e:
            print(f"❌ Erro ao carregar labels: {e}")
            self.labels = ["Classe 0", "Classe 1", "Classe 2", "Classe 3", "Classe 4"]
    
    def load_model(self):
        """Carrega o modelo TensorFlow Lite"""
        if not TENSORFLOW_AVAILABLE:
            self.status_label.config(text="❌ TensorFlow não disponível")
            return
        
        if not os.path.exists(MODEL_PATH):
            self.status_label.config(text="⚠️ Modelo não encontrado.")
            return
        
        try:
            self.status_label.config(text="📥 Carregando modelo TensorFlow Lite...")
            
            # Carrega TensorFlow Lite
            self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
            self.interpreter.allocate_tensors()
            
            # Pega detalhes de entrada e saída
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Print para debug
            print(f"✅ Modelo TensorFlow Lite carregado!")
            print(f"Input: {self.input_details[0]['shape']}")
            print(f"Output: {self.output_details[0]['shape']}")
            
            self.status_label.config(text="✅ Modelo carregado com sucesso!")
            self.start_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            self.status_label.config(text=f"❌ Erro ao carregar modelo")
            print(f"❌ Erro: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror(
                "Erro ao Carregar Modelo",
                f"Erro: {str(e)}\n\nVerifique o arquivo do modelo."
            )
    
    def create_ui(self):
        """Cria a interface gráfica"""
        
        # Container principal
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ==========================================
        # HEADER
        # ==========================================
        header_frame = tk.Frame(main_container, bg='#667eea', relief=tk.FLAT)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = tk.Label(
            header_frame,
            text="🤟 Reconhecimento de Libras",
            font=("Arial", 24, "bold"),
            bg='#667eea',
            fg='white',
            padx=20,
            pady=15
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            header_frame,
            text="Reconhecimento em Tempo Real com IA",
            font=("Arial", 11),
            bg='#667eea',
            fg='#e0e0ff'
        )
        subtitle_label.pack(pady=(0, 15))
        
        # ==========================================
        # VIDEO SECTION
        # ==========================================
        video_frame = tk.LabelFrame(
            main_container,
            text="📷 Câmera",
            font=("Arial", 12, "bold"),
            bg='white',
            fg='#333',
            padx=10,
            pady=10
        )
        video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.video_label = tk.Label(
            video_frame,
            text="Clique em 'Iniciar Câmera' para começar",
            bg='black',
            fg='white',
            font=("Arial", 14),
            width=80,
            height=20
        )
        self.video_label.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        # ==========================================
        # CONTROLS
        # ==========================================
        controls_frame = tk.Frame(main_container, bg='white')
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.start_btn = tk.Button(
            controls_frame,
            text="🎥 Iniciar Câmera",
            command=self.start_camera,
            font=("Arial", 12, "bold"),
            bg='#4CAF50',
            fg='white',
            padx=25,
            pady=12,
            relief=tk.RAISED,
            borderwidth=2,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(
            controls_frame,
            text="⏹️ Parar Câmera",
            command=self.stop_camera,
            font=("Arial", 12, "bold"),
            bg='#f44336',
            fg='white',
            padx=25,
            pady=12,
            relief=tk.RAISED,
            borderwidth=2,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # ==========================================
        # RESULT SECTION
        # ==========================================
        result_frame = tk.LabelFrame(
            main_container,
            text="🎯 Resultado",
            font=("Arial", 12, "bold"),
            bg='white',
            fg='#333',
            padx=10,
            pady=10
        )
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        # Resultado principal
        self.result_label = tk.Label(
            result_frame,
            text="Aguardando...",
            font=("Arial", 64, "bold"),
            fg='#667eea',
            bg='white'
        )
        self.result_label.pack(pady=(10, 5))
        
        # Confiança
        self.confidence_label = tk.Label(
            result_frame,
            text="",
            font=("Arial", 18),
            fg='#666',
            bg='white'
        )
        self.confidence_label.pack(pady=(0, 10))
        
        # Todas as predições
        self.predictions_frame = tk.Frame(result_frame, bg='white')
        self.predictions_frame.pack(fill=tk.X, padx=10)
        
        # Status
        self.status_label = tk.Label(
            main_container,
            text="⏳ Carregando modelo...",
            font=("Arial", 10),
            fg='#666',
            bg='#f0f0f0'
        )
        self.status_label.pack(pady=(5, 0))
    
    def start_camera(self):
        """Inicia a captura de vídeo"""
        if self.interpreter is None:
            messagebox.showwarning(
                "Modelo Não Disponível",
                "O modelo não foi carregado.\n\n"
                "Por favor, verifique o arquivo do modelo."
            )
            return
        
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Erro", "Não foi possível acessar a câmera.")
                return
            
            # Configurar resolução
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.is_running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_label.config(text="🟢 Câmera ativa")
            
            self.update_video()
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao iniciar câmera:\n{e}")
    
    def stop_camera(self):
        """Para a captura de vídeo"""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.video_label.config(image='', text="Câmera desligada")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="⚫ Câmera desligada")
        
        self.result_label.config(text="Aguardando...")
        self.confidence_label.config(text="")
        
        # Limpa predições
        for widget in self.predictions_frame.winfo_children():
            widget.destroy()
    
    def preprocess_frame(self, frame):
        """Pré-processa o frame para o modelo"""
        img = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(float) / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    
    def predict(self, frame):
        """Faz predição no frame usando TensorFlow Lite"""
        if self.interpreter is None:
            return None, 0.0, []
        
        try:
            # Pré-processa
            processed = self.preprocess_frame(frame)
            
            # TensorFlow Lite usa float32
            input_data = processed.astype(np.float32)
            
            # Define tensor de entrada
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # Faz predição
            self.interpreter.invoke()
            
            # Pega resultado
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            
            max_idx = np.argmax(output_data)
            confidence = float(output_data[max_idx])
            predicted_class = self.labels[max_idx] if max_idx < len(self.labels) else "Desconhecido"
            
            # Todas as predições ordenadas
            all_predictions = []
            for i, conf in enumerate(output_data):
                class_name = self.labels[i] if i < len(self.labels) else f"Classe {i}"
                all_predictions.append((class_name, float(conf)))
            all_predictions.sort(key=lambda x: x[1], reverse=True)
            
            return predicted_class, confidence, all_predictions
        except Exception as e:
            print(f"Erro na predição: {e}")
            import traceback
            traceback.print_exc()
            return None, 0.0, []
    
    def update_video(self):
        """Atualiza o frame do vídeo"""
        if not self.is_running or not self.cap:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.stop_camera()
            return
        
        # Espelha o frame
        frame = cv2.flip(frame, 1)
        
        # Faz predição
        prediction, confidence, all_predictions = self.predict(frame)
        
        # Desenha informações no frame
        if prediction:
            text = f"{prediction}"
            cv2.putText(frame, text, (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            conf_text = f"{confidence*100:.1f}%"
            cv2.putText(frame, conf_text, (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Converte para exibição
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_pil = frame_pil.resize((640, 480), Image.Resampling.LANCZOS)
        frame_tk = ImageTk.PhotoImage(image=frame_pil)
        
        self.video_label.config(image=frame_tk, text="")
        self.video_label.image = frame_tk
        
        # Atualiza resultados
        if prediction:
            self.result_label.config(text=prediction)
            self.confidence_label.config(text=f"Confiança: {confidence*100:.1f}%")
            self.update_predictions(all_predictions)
        
        # Próximo frame
        self.root.after(30, self.update_video)
    
    def update_predictions(self, predictions):
        """Atualiza lista de predições"""
        # Limpa anteriores
        for widget in self.predictions_frame.winfo_children():
            widget.destroy()
        
        # Mostra top 3
        for i, (class_name, conf) in enumerate(predictions[:3]):
            if i == 0:
                bg_color = '#e7f3ff'
                text_color = '#0066cc'
                font_weight = 'bold'
            else:
                bg_color = '#f5f5f5'
                text_color = '#666'
                font_weight = 'normal'
            
            item_frame = tk.Frame(
                self.predictions_frame,
                bg=bg_color,
                relief=tk.RAISED,
                borderwidth=1
            )
            item_frame.pack(fill=tk.X, padx=5, pady=2)
            
            tk.Label(
                item_frame,
                text=class_name,
                font=("Arial", 11, font_weight),
                bg=bg_color,
                fg=text_color,
                anchor='w'
            ).pack(side=tk.LEFT, padx=10, pady=5)
            
            tk.Label(
                item_frame,
                text=f"{conf*100:.1f}%",
                font=("Arial", 11, font_weight),
                bg=bg_color,
                fg=text_color,
                anchor='e'
            ).pack(side=tk.RIGHT, padx=10)

def main():
    """Função principal"""
    root = tk.Tk()
    app = LibrasDesktopApp(root)
    
    def on_closing():
        if app.is_running:
            app.stop_camera()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nEncerrando aplicação...")
        if app.is_running:
            app.stop_camera()
        root.destroy()

if __name__ == '__main__':
    print("=" * 60)
    print("🤟 Aplicativo Desktop - Reconhecimento de Libras")
    print("=" * 60)
    main()

