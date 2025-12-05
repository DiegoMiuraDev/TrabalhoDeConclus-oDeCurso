#!/usr/bin/env python3
"""
Script para coletar dados de teste usando webcam
Organiza as imagens por classe (A, E, I, O, U) para avaliação posterior
"""

import cv2
import numpy as np
import os
from pathlib import Path
import time

# Configurações
TEST_DATA_DIR = "dataset/test_images"
IMAGE_SIZE = 224
CLASSES = ["A", "E", "I", "O", "U"]
CAMERA_INDEX = 1  # Índice da câmera (0 = primeira, 1 = segunda, etc.)

class TestDataCollector:
    def __init__(self, camera_index=None):
        self.cap = None
        self.current_class = 0
        self.image_count = {cls: 0 for cls in CLASSES}
        self.collecting = False
        self.save_interval = 0.5  # Salvar a cada 0.5 segundos
        self.last_save_time = {cls: 0 for cls in CLASSES}
        self.camera_index = camera_index if camera_index is not None else CAMERA_INDEX
        
        # Criar diretórios
        self.test_dir = Path(TEST_DATA_DIR)
        for cls in CLASSES:
            (self.test_dir / cls).mkdir(parents=True, exist_ok=True)
        
        print("📸 Coletor de Dados de Teste", flush=True)
        print("="*60, flush=True)
        print("\n📋 Instruções:", flush=True)
        print("  - Pressione 1-5 para selecionar a classe (A, E, I, O, U)", flush=True)
        print("  - Pressione ESPAÇO para começar/parar coleta", flush=True)
        print("  - Pressione 'q' para sair", flush=True)
        print(f"\n📁 Imagens serão salvas em: {TEST_DATA_DIR}/", flush=True)
        print("="*60, flush=True)
    
    def start_camera(self):
        """Inicia a câmera"""
        print(f"Tentando abrir câmera (índice {self.camera_index})...", flush=True)
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"❌ Câmera índice {self.camera_index} não disponível.", flush=True)
            print("💡 Dica: Tente mudar CAMERA_INDEX no início do script ou use:", flush=True)
            print("   python scripts/collect_test_data.py --camera 1", flush=True)
            raise RuntimeError(f"Não foi possível abrir a câmera índice {self.camera_index}. Verifique se a câmera está conectada e não está sendo usada por outro programa.")
        
        # Configurar resolução
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"✅ Câmera índice {self.camera_index} iniciada com sucesso!", flush=True)
    
    def preprocess_image(self, frame):
        """Pré-processa frame para salvar"""
        # Redimensionar
        img = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        return img
    
    def save_image(self, frame, class_name):
        """Salva imagem na pasta da classe"""
        current_time = time.time()
        
        # Verificar intervalo mínimo
        if current_time - self.last_save_time[class_name] < self.save_interval:
            return False
        
        self.last_save_time[class_name] = current_time
        
        # Pré-processar
        img = self.preprocess_image(frame)
        
        # Gerar nome do arquivo
        count = self.image_count[class_name]
        filename = f"{class_name}_{count:04d}.jpg"
        filepath = self.test_dir / class_name / filename
        
        # Salvar
        cv2.imwrite(str(filepath), img)
        self.image_count[class_name] += 1
        
        return True
    
    def run(self):
        """Loop principal"""
        self.start_camera()
        
        print("\n🖥️  Abrindo janela da câmera...", flush=True)
        print("   (Se a janela não aparecer, verifique se há permissões de câmera)", flush=True)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠️  Não foi possível ler frame da câmera", flush=True)
                    break
                
                # Espelhar frame (mais natural)
                frame = cv2.flip(frame, 1)
                
                # Obter classe atual
                class_name = CLASSES[self.current_class]
                
                # Status na tela
                status_text = f"Classe: {class_name} | Coletando: {'SIM' if self.collecting else 'NÃO'}"
                count_text = f"Imagens salvas: {self.image_count[class_name]}"
                
                # Desenhar informações
                cv2.putText(frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, count_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Instruções
                cv2.putText(frame, "1-5: Classe | ESPACO: Coletar | Q: Sair", 
                           (10, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Retângulo indicando classe
                color = (0, 255, 0) if self.collecting else (0, 0, 255)
                cv2.rectangle(frame, (10, 80), (200, 120), color, 2)
                cv2.putText(frame, class_name, (20, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Salvar se coletando
                if self.collecting:
                    if self.save_image(frame, class_name):
                        # Feedback visual
                        cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 255, 0), -1)
                
                # Mostrar frame
                cv2.imshow('Coletor de Dados de Teste', frame)
                
                # Processar teclas
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    self.collecting = not self.collecting
                    print(f"{'▶️  Coletando' if self.collecting else '⏸️  Pausado'} - Classe: {class_name}")
                elif key == ord('1'):
                    self.current_class = 0
                    self.collecting = False
                    print(f"📁 Classe selecionada: {CLASSES[0]}")
                elif key == ord('2'):
                    self.current_class = 1
                    self.collecting = False
                    print(f"📁 Classe selecionada: {CLASSES[1]}")
                elif key == ord('3'):
                    self.current_class = 2
                    self.collecting = False
                    print(f"📁 Classe selecionada: {CLASSES[2]}")
                elif key == ord('4'):
                    self.current_class = 3
                    self.collecting = False
                    print(f"📁 Classe selecionada: {CLASSES[3]}")
                elif key == ord('5'):
                    self.current_class = 4
                    self.collecting = False
                    print(f"📁 Classe selecionada: {CLASSES[4]}")
        
        finally:
            self.cleanup()
            self.print_summary()
    
    def cleanup(self):
        """Limpa recursos"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def print_summary(self):
        """Imprime resumo da coleta"""
        print("\n" + "="*60)
        print("📊 RESUMO DA COLETA")
        print("="*60)
        total = 0
        for cls in CLASSES:
            count = self.image_count[cls]
            total += count
            print(f"   {cls}: {count} imagens")
        print(f"\n   Total: {total} imagens")
        print(f"   Salvas em: {TEST_DATA_DIR}/")
        print("="*60)
        print("\n✅ Agora você pode executar:")
        print("   python scripts/generate_metrics_table.py")
        print("   para calcular as métricas reais!")


if __name__ == "__main__":
    import sys
    import argparse
    
    # Configurar argumentos de linha de comando
    parser = argparse.ArgumentParser(description='Coletor de dados de teste para Libras')
    parser.add_argument('--camera', type=int, default=None,
                        help=f'Índice da câmera a usar (padrão: {CAMERA_INDEX})')
    args = parser.parse_args()
    
    try:
        print("Iniciando coletor de dados de teste...", flush=True)
        camera_index = args.camera if args.camera is not None else CAMERA_INDEX
        if args.camera is not None:
            print(f"📹 Usando câmera índice {camera_index} (especificada via argumento)", flush=True)
        collector = TestDataCollector(camera_index=camera_index)
        print("Coletor criado. Iniciando execução...", flush=True)
        collector.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrompido pelo usuário", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ ERRO: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

