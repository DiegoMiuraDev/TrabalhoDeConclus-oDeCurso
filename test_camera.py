#!/usr/bin/env python3
"""Teste rápido da câmera"""
import sys
import cv2

print("="*60)
print("TESTE DE CÂMERA")
print("="*60)

try:
    print("\n1. Importando OpenCV...")
    print(f"   OpenCV version: {cv2.__version__}")
    
    print("\n2. Tentando abrir câmera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("   ❌ ERRO: Não foi possível abrir a câmera!")
        print("   Verifique se:")
        print("   - A câmera está conectada")
        print("   - Nenhum outro programa está usando a câmera")
        sys.exit(1)
    
    print("   ✅ Câmera aberta com sucesso!")
    
    print("\n3. Lendo um frame...")
    ret, frame = cap.read()
    
    if not ret:
        print("   ❌ ERRO: Não foi possível ler frame da câmera!")
        cap.release()
        sys.exit(1)
    
    print(f"   ✅ Frame lido! Dimensões: {frame.shape}")
    
    print("\n4. Tentando mostrar janela...")
    cv2.imshow('Teste', frame)
    print("   ✅ Janela criada! Pressione qualquer tecla para fechar...")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()
    
    print("\n✅ Teste concluído com sucesso!")
    
except Exception as e:
    print(f"\n❌ ERRO: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


