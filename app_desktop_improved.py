                      
"""
Aplicativo Desktop Melhorado para Reconhecimento de Libras
Com detecção de mãos via MediaPipe, pré-processamento avançado e UI moderna
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
import time
import os
from collections import deque
import math

                        
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe não disponível. Instale com: pip install mediapipe")

               
KERAS_MODEL_PATH = os.environ.get("KERAS_MODEL_PATH", os.path.join(os.path.dirname(__file__), "dataset", "keras_model.h5"))
LABELS_PATH = "dataset/labels.txt"
IMAGE_SIZE = 224
MIN_CONFIDENCE = 0.55                                                 
PREDICT_EVERY_N_FRAMES = 3                                                 
NORMALIZATION = os.environ.get("INPUT_NORMALIZATION", "0_1")                 
FALLBACK_VOWEL_LABELS = ["A", "E", "I", "O", "U"]
TFLITE_MODEL_PATH = os.environ.get("TFLITE_MODEL_PATH", os.path.join(os.path.dirname(__file__), "dataset", "model_unquant.tflite"))

class HandDetector:
    """Classe para detecção e isolamento de mãos usando MediaPipe"""
    
    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            self.mp_hands = None
            self.hands = None
            return
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,                  
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def detect_and_crop_hand(self, frame):
        """
        Detecta mão no frame e retorna ROI (Region of Interest) da mão
        
        Returns:
            tuple: (roi_cropped, hand_landmarks, bbox)
            - roi_cropped: Imagem da mão isolada
            - hand_landmarks: Landmarks da mão
            - bbox: Bounding box (x, y, w, h)
        """
        if not self.hands:
            return None, None, None
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if not results.multi_hand_landmarks:
            return None, None, None
        
        hand_landmarks = results.multi_hand_landmarks[0]
        
        h, w = frame.shape[:2]
        x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        margin_x = int((x_max - x_min) * 0.2)
        margin_y = int((y_max - y_min) * 0.2)
        
        x_min = max(0, x_min - margin_x)
        y_min = max(0, y_min - margin_y)
        x_max = min(w, x_max + margin_x)
        y_max = min(h, y_max + margin_y)
        
        roi = frame[y_min:y_max, x_min:x_max]
        
        if roi.size == 0:
            return None, hand_landmarks, (x_min, y_min, x_max - x_min, y_max - y_min)
        
        return roi, hand_landmarks, (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def draw_landmarks(self, frame, hand_landmarks):
        """Desenha landmarks da mão no frame"""
        if not self.hands or not hand_landmarks:
            return frame
        
        annotated_frame = frame.copy()
        self.mp_drawing.draw_landmarks(
            annotated_frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
        )
        return annotated_frame

class PredictionSmoother:
    """Suaviza predições para evitar flickering"""
    
    def __init__(self, buffer_size=5, threshold=0.6):
        self.buffer_size = buffer_size
        self.threshold = threshold
        self.prediction_buffer = deque(maxlen=buffer_size)
        self.confidence_buffer = deque(maxlen=buffer_size)
    
    def add_prediction(self, prediction, confidence):
        """Adiciona nova predição ao buffer"""
        self.prediction_buffer.append(prediction)
        self.confidence_buffer.append(confidence)
    
    def get_smoothed_prediction(self):
        """Retorna predição suavizada baseada no buffer"""
        if not self.prediction_buffer:
            return None, 0.0
        
                                            
        prediction_counts = {}
        total_confidence = 0.0
        
        for pred, conf in zip(self.prediction_buffer, self.confidence_buffer):
            if pred in prediction_counts:
                prediction_counts[pred] += conf
            else:
                prediction_counts[pred] = conf
            total_confidence += conf
        
                                                          
        if prediction_counts:
            smoothed_pred = max(prediction_counts.items(), key=lambda x: x[1])
            avg_confidence = smoothed_pred[1] / len(self.prediction_buffer)
            
                                                                   
            if avg_confidence >= self.threshold:
                return smoothed_pred[0], avg_confidence
        
        return None, 0.0

class LibrasDesktopAppImproved:
    def __init__(self, root):
        self.root = root
        self.root.title("INTELIGÊNCIA ARTIFICIAL E VISÃO COMPUTACIONAL PARA RECONHECIMENTO AUTOMÁTICO DE LIBRAS")
        self.root.geometry("1100x850")
        self.root.configure(bg='#1a1a2e')
        
                   
        self.cap = None
        self.is_running = False
        self.interpreter = None                                        
        self.keras_model = None
        self.use_keras = False
        self.input_details = None
        self.output_details = None
        self.labels = []
        self.camera_index = 0                                
        self.available_cameras = []                                
        
                          
        self.hand_detector = HandDetector()
        
                                                                     
        self.prediction_smoother = PredictionSmoother(buffer_size=15, threshold=0.65)
        
                                
        self.last_prediction = None
        self.last_confidence = 0.0
        self.pending_prediction = None
        self.stable_counter = 0
        self.required_stable_frames = 5
        self.ema_confidence = 0.0
        self.animation_frame = 0
        
                        
        self.load_labels()
        
                                     
        self.detect_available_cameras()
        
                        
        self.create_ui()
        
                                           
        threading.Thread(target=self.load_model, daemon=True).start()
    
    def load_labels(self):
        """Carrega as labels do modelo"""
        try:
                                                                   
            npy_path = os.path.join(os.path.dirname(__file__), "models", "libras_classes.npy")
            if os.path.exists(npy_path):
                try:
                    labels_np = np.load(npy_path, allow_pickle=True)
                    self.labels = [str(x) for x in labels_np.tolist()]
                    print(f"✅ Labels (npy) carregadas: {self.labels}")
                    return
                except Exception:
                    pass

            if os.path.exists(LABELS_PATH):
                with open(LABELS_PATH, 'r') as f:
                    lines = f.readlines()
                                                          
                    self.labels = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            self.labels.append(parts[1])
                        else:
                            self.labels.append(parts[0])
                print(f"✅ Labels carregadas: {self.labels}")
                                                      
            if not self.labels:
                self.labels = FALLBACK_VOWEL_LABELS.copy()
                print(f"ℹ️ Usando labels padrão: {self.labels}")
        except Exception as e:
            print(f"❌ Erro ao carregar labels: {e}")
            self.labels = ["A", "E", "I", "O", "U"]
    
    def detect_available_cameras(self):
        """Detecta câmeras disponíveis no sistema"""
        self.available_cameras = []
        max_cameras_to_test = 10
        
        for i in range(max_cameras_to_test):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                                                                 
                ret, _ = cap.read()
                if ret:
                    self.available_cameras.append(i)
                cap.release()
        
        if not self.available_cameras:
                                                                           
            self.available_cameras = [0]
        
        print(f"📹 Câmeras disponíveis: {self.available_cameras}")
    
    def load_model(self):
        """Carrega exclusivamente o modelo Keras (.h5) do diretório dataset/"""
        global IMAGE_SIZE
        if not TENSORFLOW_AVAILABLE:
            self.status_label.config(text="TensorFlow não disponível")
            return
        
        try:
            self.status_label.config(text="📥 Carregando modelo...")
                                                   
            if os.environ.get("USE_TFLITE", "0") == "1":
                raise RuntimeError("Forçando carregamento TFLite via USE_TFLITE=1")
            keras_path = KERAS_MODEL_PATH
            if keras_path and os.path.exists(keras_path):
                                                                                                       
                def depthwise_conv2d_compat(*args, **kwargs):
                    try:
                        kwargs.pop('groups', None)
                    except Exception:
                        pass
                    return tf.keras.layers.DepthwiseConv2D(*args, **kwargs)

                                                                                              
                self.keras_model = tf.keras.models.load_model(
                    keras_path,
                    compile=False,
                    custom_objects={
                        'DepthwiseConv2D': depthwise_conv2d_compat
                    }
                )
                self.use_keras = True
                print(f"✅ Modelo Keras carregado: {os.path.basename(keras_path)}")
                                                                        
                try:
                    model_input_shape = self.keras_model.input_shape
                    if isinstance(model_input_shape, (list, tuple)) and len(model_input_shape) == 4:
                        h, w = model_input_shape[1], model_input_shape[2]
                        if isinstance(h, int) and isinstance(w, int) and h == w and h > 0:
                            global IMAGE_SIZE
                            IMAGE_SIZE = h
                            print(f"ℹ️ Ajustando IMAGE_SIZE para {IMAGE_SIZE} a partir do modelo Keras")
                except Exception:
                    pass
                self.status_label.config(text="✅ Modelo Keras carregado! Clique em 'Iniciar'.")
                self.start_btn.config(state=tk.NORMAL)
                return
            
                                                  
            raise FileNotFoundError("keras_model.h5 não encontrado, tentando TFLite")
            
        except Exception as e:
                                              
            try:
                if not os.path.exists(TFLITE_MODEL_PATH):
                    raise FileNotFoundError(f"TFLite não encontrado em {TFLITE_MODEL_PATH}")
                self.interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.use_keras = False
                self.status_label.config(text="✅ Modelo TFLite carregado! Clique em 'Iniciar'.")
                print(f"✅ Modelo TFLite carregado: {os.path.basename(TFLITE_MODEL_PATH)}")
                                                                               
                try:
                    shape = self.input_details[0]['shape']
                    if len(shape) == 4 and shape[1] == shape[2] and shape[1] > 0:
                        IMAGE_SIZE = int(shape[1])
                        print(f"ℹ️ Ajustando IMAGE_SIZE para {IMAGE_SIZE} a partir do modelo TFLite")
                except Exception:
                    pass
                self.start_btn.config(state=tk.NORMAL)
                return
            except Exception as e2:
                err_msg = f"❌ Erro ao carregar modelo (Keras e TFLite falharam): {e2}"
                self.status_label.config(text=err_msg)
                print(err_msg)
                import traceback
                traceback.print_exc()
    
    def create_ui(self):
        """Cria a interface gráfica melhorada"""
        
                             
        main_container = tk.Frame(self.root, bg='#020617')                                    
        main_container.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)
        
                                                    
                
                                                    
        header_frame = tk.Frame(main_container, bg='#020617', relief=tk.FLAT)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = tk.Label(
            header_frame,
            text="INTELIGÊNCIA ARTIFICIAL E VISÃO COMPUTACIONAL PARA RECONHECIMENTO AUTOMÁTICO DE LIBRAS",
            font=("Segoe UI", 20, "bold"),
            bg='#020617',
            fg='#e5e7eb',
            padx=25,
            pady=14
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            header_frame,
            text="Aplicação de inteligência artificial e visão computacional para reconhecimento de sinais estáticos de Libras",
            font=("Segoe UI", 10),
            bg='#020617',
            fg='#9ca3af'
        )
        subtitle_label.pack(pady=(0, 18))
        
                                                    
                                     
                                                    
        content_frame = tk.Frame(main_container, bg='#020617')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
                                
        left_column = tk.Frame(content_frame, bg='#020617')
        left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        video_frame = tk.LabelFrame(
            left_column,
            text="Câmera ao vivo",
            font=("Segoe UI", 12, "bold"),
            bg='#020617',
            fg='#e5e7eb',
            padx=10,
            pady=10,
            relief=tk.FLAT
        )
        video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(
            video_frame,
            text="Aguardando inicialização da câmera...",
            bg='#020617',
            fg='#e5e7eb',
            font=("Segoe UI", 12),
            width=50,
            height=25
        )
        self.video_label.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        
                                      
        self.hand_status_label = tk.Label(
            video_frame,
            text="Nenhuma mão detectada",
            bg='#020617',
            fg='#f97316',
            font=("Segoe UI", 10),
            anchor='w'
        )
        self.hand_status_label.pack(fill=tk.X, padx=5, pady=(5, 0))
        
                                    
        right_column = tk.Frame(content_frame, bg='#020617', width=340)
        right_column.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_column.pack_propagate(False)
        
                                                
        result_main_frame = tk.LabelFrame(
            right_column,
            text="Letra reconhecida",
            font=("Segoe UI", 12, "bold"),
            bg='#020617',
            fg='#e5e7eb',
            padx=12,
            pady=12,
            relief=tk.FLAT
        )
        result_main_frame.pack(fill=tk.X, pady=(0, 10))
        
                                       
        self.letter_canvas = tk.Canvas(
            result_main_frame,
            bg='#020617',
            width=300,
            height=170,
            highlightthickness=0
        )
        self.letter_canvas.pack(pady=10)
        
                                             
        self.confidence_frame = tk.Frame(result_main_frame, bg='#020617')
        self.confidence_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.confidence_label = tk.Label(
            self.confidence_frame,
            text="Confiança: 0%",
            font=("Segoe UI", 14),
            fg='#e5e7eb',
            bg='#020617'
        )
        self.confidence_label.pack()
        
                                         
        self.confidence_bar = tk.Canvas(
            self.confidence_frame,
            bg='#020617',
            height=25,
            highlightthickness=0
        )
        self.confidence_bar.pack(fill=tk.X, pady=(5, 0))
        
                                                                           
        self.predictions_frame = None
        
                                                    
                  
                                                    
        controls_frame = tk.Frame(main_container, bg='#020617')
        controls_frame.pack(fill=tk.X, pady=(15, 0))
        
                                        
        camera_controls_frame = tk.Frame(controls_frame, bg='#020617')
        camera_controls_frame.pack(side=tk.LEFT, padx=5)
        
                                                 
        camera_label = tk.Label(
            camera_controls_frame,
            text="Câmera:",
            font=("Segoe UI", 11),
            fg='#e5e7eb',
            bg='#020617'
        )
        camera_label.pack(side=tk.LEFT, padx=(0, 5))
        
                                         
        camera_options = [f"Câmera {i}" for i in self.available_cameras]
        if not camera_options:
            camera_options = ["Câmera 0"]
        
        self.camera_var = tk.StringVar()
        self.camera_var.set(camera_options[0] if camera_options else "Câmera 0")
        
        self.camera_combo = ttk.Combobox(
            camera_controls_frame,
            textvariable=self.camera_var,
            values=camera_options,
            state="readonly",
            width=12,
            font=("Segoe UI", 11)
        )
        self.camera_combo.pack(side=tk.LEFT, padx=5)
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_selected)
        
        self.start_btn = tk.Button(
            controls_frame,
            text="Iniciar câmera",
            command=self.start_camera,
            font=("Segoe UI", 13, "bold"),
            bg='#2563eb',
            fg='white',
            padx=30,
            pady=14,
            relief=tk.FLAT,
            cursor='hand2',
            state=tk.DISABLED,
            activebackground='#1d4ed8'
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(
            controls_frame,
            text="Parar",
            command=self.stop_camera,
            font=("Segoe UI", 13, "bold"),
            bg='#ef4444',
            fg='white',
            padx=30,
            pady=14,
            relief=tk.FLAT,
            cursor='hand2',
            state=tk.DISABLED,
            activebackground='#dc2626'
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
                
        self.status_label = tk.Label(
            controls_frame,
            text="Carregando modelo...",
            font=("Segoe UI", 11),
            fg='#9ca3af',
            bg='#020617',
            anchor='w'
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=15)
    
    def draw_animated_letter(self, letter, confidence):
        """Desenha letra animada no canvas"""
        self.letter_canvas.delete("all")
        
        width = self.letter_canvas.winfo_width()
        height = self.letter_canvas.winfo_height()
        
        if width <= 1 or height <= 1:
            return
        
                                                                    
        if confidence >= 0.8:
            color = '#22c55e'         
        elif confidence >= 0.6:
            color = '#eab308'           
        else:
            color = '#ef4444'            
        
                           
        pulse_size = 5 + math.sin(self.animation_frame * 0.3) * 3
        
                              
        self.letter_canvas.create_text(
            width // 2,
            height // 2,
            text=letter,
            font=("Arial", 120, "bold"),
            fill=color,
            tags="letter"
        )
        
                                          
        for i in range(3):
            radius = 80 + (pulse_size * i)
            alpha = 0.3 - (i * 0.1)
            self.letter_canvas.create_oval(
                width // 2 - radius,
                height // 2 - radius,
                width // 2 + radius,
                height // 2 + radius,
                outline=color,
                width=2,
                tags="pulse"
            )
        
        self.animation_frame += 1
    
    def update_confidence_bar(self, confidence):
        """Atualiza barra de confiança visual"""
        self.confidence_bar.delete("all")
        
        width = self.confidence_bar.winfo_width()
        height = self.confidence_bar.winfo_height()
        
        if width <= 1:
            return
        
        bar_width = int(width * confidence)
        
                                                          
        if confidence >= 0.8:
            color = '#22c55e'
        elif confidence >= 0.6:
            color = '#eab308'
        else:
            color = '#ef4444'
        
                                    
        self.confidence_bar.create_rectangle(
            0, 0, bar_width, height,
            fill=color,
            outline='',
            tags="progress"
        )
        
                         
        self.confidence_bar.create_text(
            width // 2,
            height // 2,
            text=f"{confidence * 100:.1f}%",
            font=("Segoe UI", 11, "bold"),
            fill='#ffffff',
            tags="text"
        )
    
    def on_camera_selected(self, event=None):
        """Callback quando uma câmera é selecionada"""
        selected = self.camera_var.get()
                                                       
        try:
            camera_num = int(selected.split()[-1])
            self.camera_index = camera_num
            print(f"📹 Câmera selecionada: índice {self.camera_index}")
        except (ValueError, IndexError):
            self.camera_index = 0
            print(f"⚠️ Erro ao parsear seleção de câmera, usando índice 0")
    
    def start_camera(self):
        """Inicia a captura de vídeo"""
        has_keras = (self.use_keras and self.keras_model is not None)
        has_tflite = (not self.use_keras and self.interpreter is not None)
        if not (has_keras or has_tflite):
            messagebox.showwarning(
                "Modelo Não Disponível",
                "Nenhum modelo carregado.\n\nColoque 'keras_model.h5' em dataset/ ou 'model_unquant.tflite' e tente novamente."
            )
            return
        
        self.on_camera_selected()
        
        try:
            try:
                cv2.setUseOptimized(True)
                cv2.setNumThreads(0)
            except Exception:
                pass

            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                messagebox.showerror(
                    "Erro", 
                    f"Não foi possível acessar a câmera índice {self.camera_index}.\n\n"
                    f"Verifique se a câmera está conectada e não está sendo usada por outro programa."
                )
                return
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.camera_combo.config(state=tk.DISABLED)  
            self.status_label.config(text=f"Câmera {self.camera_index} ativa - posicione sua mão na frente da câmera")
            
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
        self.camera_combo.config(state="readonly")                               
        self.status_label.config(text="Câmera desligada")
        self.hand_status_label.config(text="Nenhuma mão detectada", fg='#ff6b6b')
        
                        
        self.letter_canvas.delete("all")
        self.letter_canvas.create_text(
            self.letter_canvas.winfo_width() // 2,
            self.letter_canvas.winfo_height() // 2,
            text="?",
            font=("Arial", 120, "bold"),
            fill='#666666'
        )
        self.confidence_label.config(text="Confiança: 0%")
        self.confidence_bar.delete("all")
        
                         
        for widget in self.predictions_frame.winfo_children():
            widget.destroy()
    
    def preprocess_hand_image(self, hand_roi):
        """Pré-processa imagem da mão"""
        if hand_roi is None or hand_roi.size == 0:
            return None
        
                                            
        h, w = hand_roi.shape[:2]
        aspect = w / h
        
        if aspect > 1:
            new_w = IMAGE_SIZE
            new_h = int(IMAGE_SIZE / aspect)
        else:
            new_h = IMAGE_SIZE
            new_w = int(IMAGE_SIZE * aspect)
        
        resized = cv2.resize(hand_roi, (new_w, new_h))
        
                                    
        canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        
                           
        y_offset = (IMAGE_SIZE - new_h) // 2
        x_offset = (IMAGE_SIZE - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
                   
        img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        if NORMALIZATION.lower() == "neg1_1":
            img_normalized = (img_normalized * 2.0) - 1.0
        img_expanded = np.expand_dims(img_normalized, axis=0)
        
        return img_expanded
    
    def preprocess_frame(self, frame):
        """Pré-processa frame completo (fallback quando não há mão detectada)"""
        img = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        if NORMALIZATION.lower() == "neg1_1":
            img = (img * 2.0) - 1.0
        img = np.expand_dims(img, axis=0)
        return img
    
    def predict(self, processed_img):
        """Faz predição usando Keras (.h5) ou TensorFlow Lite"""
        if processed_img is None:
            return None, 0.0, []
        
        try:
            input_data = processed_img.astype(np.float32)

            if self.use_keras and self.keras_model is not None:
                output_data = self.keras_model.predict(input_data, verbose=0)[0]
            else:
                if self.interpreter is None:
                    return None, 0.0, []
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
                                                             
            try:
                s = float(np.sum(output_data))
                if not (0.99 <= s <= 1.01):
                    output_exp = np.exp(output_data - np.max(output_data))
                    output_data = output_exp / np.sum(output_exp)
            except Exception:
                pass
            
            max_idx = np.argmax(output_data)
            confidence = float(output_data[max_idx])
            
                                                                 
            if len(self.labels) != len(output_data):
                if len(output_data) == 5:
                    self.labels = FALLBACK_VOWEL_LABELS.copy()
                else:
                    self.labels = [f"Classe {i}" for i in range(len(output_data))]
            predicted_class = self.labels[max_idx] if max_idx < len(self.labels) else str(max_idx)
            
                                
            all_predictions = []
            for i, conf in enumerate(output_data):
                if i < len(self.labels):
                    class_name = self.labels[i]
                else:
                    class_name = f"Classe {i}"
                all_predictions.append((class_name, float(conf)))
            all_predictions.sort(key=lambda x: x[1], reverse=True)
            
            return predicted_class, confidence, all_predictions
            
        except Exception as e:
            print(f"Erro na predição: {e}")
            return None, 0.0, []
    
    def update_video(self):
        """Atualiza frame do vídeo"""
        if not self.is_running or not self.cap:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.stop_camera()
            return
        
                       
        frame = cv2.flip(frame, 1)
        
                     
        hand_roi, hand_landmarks, bbox = self.hand_detector.detect_and_crop_hand(frame)
        hand_detected = hand_roi is not None
        
                                     
        if hand_detected:
            self.hand_status_label.config(text="Mão detectada", fg='#10b981')
            
                                      
            prediction = None
            confidence = 0.0
            all_predictions = []
            if (self.animation_frame % PREDICT_EVERY_N_FRAMES) == 0:
                processed = self.preprocess_hand_image(hand_roi)
                prediction, confidence, all_predictions = self.predict(processed)
            
                                          
            if prediction and confidence >= MIN_CONFIDENCE:
                self.prediction_smoother.add_prediction(prediction, confidence)
                smoothed_pred, smoothed_conf = self.prediction_smoother.get_smoothed_prediction()
                if smoothed_pred:
                                                                       
                    if self.last_prediction == smoothed_pred:
                        self.stable_counter = min(self.required_stable_frames, self.stable_counter + 1)
                    else:
                                                        
                        if smoothed_conf >= 0.8:
                            self.stable_counter = self.required_stable_frames
                        else:
                            self.stable_counter = max(0, self.stable_counter - 1)
                        if self.stable_counter <= 0:
                            self.pending_prediction = smoothed_pred
                            self.stable_counter = 1
                                             
                    if self.stable_counter >= self.required_stable_frames and self.pending_prediction is not None:
                        self.last_prediction = self.pending_prediction
                        self.pending_prediction = None
                                      
                    alpha = 0.2
                    self.ema_confidence = (1 - alpha) * self.ema_confidence + alpha * smoothed_conf
                    self.last_confidence = self.ema_confidence

                                                                       
            if all_predictions:
                top_label, top_conf = all_predictions[0]
                self.last_prediction = top_label
                self.last_confidence = top_conf
            
                                        
            if hand_landmarks:
                frame = self.hand_detector.draw_landmarks(frame, hand_landmarks)
            
                                  
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "MAO DETECTADA", (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
                                                       
            if self.last_prediction and self.last_confidence >= MIN_CONFIDENCE:
                                  
                text = f"{self.last_prediction} ({self.last_confidence*100:.1f}%)"
                cv2.putText(frame, text, (20, 50),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
        else:
            self.hand_status_label.config(text="Nenhuma mão detectada", fg='#ff6b6b')
                                                                            
            prediction = None
            confidence = 0.0
            all_predictions = []
            if (self.animation_frame % PREDICT_EVERY_N_FRAMES) == 0:
                processed = self.preprocess_frame(frame)
                prediction, confidence, all_predictions = self.predict(processed)
            
            if prediction and confidence >= MIN_CONFIDENCE:
                self.prediction_smoother.add_prediction(prediction, confidence)
                smoothed_pred, smoothed_conf = self.prediction_smoother.get_smoothed_prediction()
                if smoothed_pred:
                    if self.last_prediction == smoothed_pred:
                        self.stable_counter = min(self.required_stable_frames, self.stable_counter + 1)
                    else:
                        if smoothed_conf >= 0.8:
                            self.stable_counter = self.required_stable_frames
                        else:
                            self.stable_counter = max(0, self.stable_counter - 1)
                        if self.stable_counter <= 0:
                            self.pending_prediction = smoothed_pred
                            self.stable_counter = 1
                    if self.stable_counter >= self.required_stable_frames and self.pending_prediction is not None:
                        self.last_prediction = self.pending_prediction
                        self.pending_prediction = None
                    alpha = 0.2
                    self.ema_confidence = (1 - alpha) * self.ema_confidence + alpha * smoothed_conf
                    self.last_confidence = self.ema_confidence

                                                                                   
            if all_predictions:
                top_label, top_conf = all_predictions[0]
                self.last_prediction = top_label
                self.last_confidence = top_conf
        
                                    
        if self.last_prediction and self.last_confidence >= MIN_CONFIDENCE:
            self.draw_animated_letter(self.last_prediction, self.last_confidence)
            self.confidence_label.config(
                text=f"Confiança: {self.last_confidence*100:.1f}%"
            )
            self.update_confidence_bar(self.last_confidence)
            self.update_predictions(all_predictions if hand_detected else [])
        
                                      
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_pil = frame_pil.resize((640, 480), Image.Resampling.LANCZOS)
        frame_tk = ImageTk.PhotoImage(image=frame_pil)
        
        self.video_label.config(image=frame_tk, text="")
        self.video_label.image = frame_tk
        
                       
        self.root.after(15, self.update_video)
    
    def update_predictions(self, predictions):
        """
        Método mantido por compatibilidade, mas a interface não exibe mais
        a lista de predições. A letra principal e a confiança já são mostradas
        no painel à direita.
        """
        return

def main():
    """Função principal"""
    root = tk.Tk()
    app = LibrasDesktopAppImproved(root)
    
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
    print("=" * 70)
    print("Aplicativo Desktop - Reconhecimento Automático de Libras")
    print("=" * 70)
    if not MEDIAPIPE_AVAILABLE:
        print("AVISO: MediaPipe não está instalado.")
        print("   Instale com: pip install mediapipe")
        print("   O app funcionará sem detecção de mãos (modo fallback)")
        print()
    main()

