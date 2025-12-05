#!/usr/bin/env python3
"""
Script para coletar dados de teste usando webcam (versão com debug)
Organiza as imagens por classe (A, E, I, O, U) para avaliação posterior
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path
import time
import traceback

# Configurações
TEST_DATA_DIR = "dataset/test_images"
IMAGE_SIZE = 224
CLASSES = ["A", "E", "I", "O", "U"]

# Função para log
def log(message):
    """Escreve mensagem no console e no arquivo de log"""
    print(message, flush=True)
    with open("collect_test_data.log", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

class TestDataCollector:
    def __init__(self):
        self.cap = None
        self.current_class = 0
        self.image_count = {cls: 0 for cls in CLASSES}
        self.collecting = False
        self.save_interval = 0.5  # Salvar a cada 0.5 segundos
        self.last_save_time = {cls: 0 for cls in CLASSES}
        
        # Criar diretórios
        self.test_dir = Path(TEST_DATA_DIR)
        for cls in CLASSES:
            (self.test_dir / cls).mkdir(parents=True, exist_ok=True)
        
        log("📸 Coletor de Dados de Teste")
        log("="*60)
        log("\n📋 Instruções:")
        log("  - Pressione 1-5 para selecionar a classe (A, E, I, O, U)")
        log("  - Pressione ESPAÇO para começar/parar coleta")
        log("  - Pressione 'q' para sair")
        log(f"\n📁 Imagens serão salvas em: {TEST_DATA_DIR}/")
        log("="*60)
    
    def start_camera(self):
        """Inicia a câmera"""
        log("Tentando abrir câmera...")
        
        # Tentar diferentes índices de câmera
        for camera_index in [0, 1, 2]:
            log(f"Tentando câmera índice {camera_index}...")
            self.cap = cv2.VideoCapture(camera_index)
            
            if self.cap.isOpened():
                # Testar se consegue ler um frame
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    log(f"✅ Câmera {camera_index} funcionando! Dimensões: {frame.shape}")
                    # Configurar resolução
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    return
                else:
                    log(f"⚠️  Câmera {camera_index} aberta mas não consegue ler frames")
                    self.cap.release()
            else:
                log(f"❌ Câmera {camera_index} não disponível")
        
        raise RuntimeError(
            "Não foi possível abrir nenhuma câmera.\n"
            "Verifique:\n"
            "  - Se a câmera está conectada\n"
            "  - Se nenhum outro programa está usando a câmera\n"
            "  - Se há permissões de acesso à câmera no Windows"
        )
    
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
        success = cv2.imwrite(str(filepath), img)
        if success:
            self.image_count[class_name] += 1
            log(f"💾 Imagem salva: {filename}")
        else:
            log(f"❌ Erro ao salvar: {filename}")
        
        return success
    
    def run(self):
        """Loop principal"""
        try:
            self.start_camera()
            
            log("\n🖥️  Abrindo janela da câmera...")
            log("   (Aguarde alguns segundos para a janela aparecer)")
            log("   (Se não aparecer, verifique permissões de câmera no Windows)")
            
            frame_count = 0
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    log("⚠️  Não foi possível ler frame da câmera")
                    break
                
                frame_count += 1
                if frame_count == 1:
                    log(f"✅ Primeiro frame lido com sucesso! Dimensões: {frame.shape}")
                
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
                    log("👋 Saindo...")
                    break
                elif key == ord(' '):
                    self.collecting = not self.collecting
                    status = '▶️  Coletando' if self.collecting else '⏸️  Pausado'
                    log(f"{status} - Classe: {class_name}")
                elif key == ord('1'):
                    self.current_class = 0
                    self.collecting = False
                    log(f"📁 Classe selecionada: {CLASSES[0]}")
                elif key == ord('2'):
                    self.current_class = 1
                    self.collecting = False
                    log(f"📁 Classe selecionada: {CLASSES[1]}")
                elif key == ord('3'):
                    self.current_class = 2
                    self.collecting = False
                    log(f"📁 Classe selecionada: {CLASSES[2]}")
                elif key == ord('4'):
                    self.current_class = 3
                    self.collecting = False
                    log(f"📁 Classe selecionada: {CLASSES[3]}")
                elif key == ord('5'):
                    self.current_class = 4
                    self.collecting = False
                    log(f"📁 Classe selecionada: {CLASSES[4]}")
        
        except Exception as e:
            log(f"❌ ERRO durante execução: {e}")
            traceback.print_exc()
        finally:
            self.cleanup()
            self.print_summary()
    
    def cleanup(self):
        """Limpa recursos"""
        log("Limpando recursos...")
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        log("✅ Recursos liberados")
    
    def print_summary(self):
        """Imprime resumo da coleta"""
        log("\n" + "="*60)
        log("📊 RESUMO DA COLETA")
        log("="*60)
        total = 0
        for cls in CLASSES:
            count = self.image_count[cls]
            total += count
            log(f"   {cls}: {count} imagens")
        log(f"\n   Total: {total} imagens")
        log(f"   Salvas em: {TEST_DATA_DIR}/")
        log("="*60)
        log("\n✅ Agora você pode executar:")
        log("   python scripts/generate_metrics_table.py")
        log("   para calcular as métricas reais!")


if __name__ == "__main__":
    # Limpar log anterior
    if os.path.exists("collect_test_data.log"):
        os.remove("collect_test_data.log")
    
    try:
        log("="*60)
        log("INICIANDO COLETOR DE DADOS DE TESTE")
        log("="*60)
        log(f"Python: {sys.version}")
        log(f"OpenCV: {cv2.__version__}")
        log(f"Diretório de trabalho: {os.getcwd()}")
        
        collector = TestDataCollector()
        collector.run()
        
    except KeyboardInterrupt:
        log("\n\n⚠️  Interrompido pelo usuário")
        sys.exit(0)
    except Exception as e:
        log(f"\n\n❌ ERRO FATAL: {e}")
        traceback.print_exc()
        sys.exit(1)




