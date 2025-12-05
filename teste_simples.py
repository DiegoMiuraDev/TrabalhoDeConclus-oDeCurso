import sys
print("Teste 1: Saída básica", flush=True)
sys.stdout.flush()

import cv2
print(f"Teste 2: OpenCV importado - versão {cv2.__version__}", flush=True)
sys.stdout.flush()

print("Teste 3: Tentando abrir câmera...", flush=True)
sys.stdout.flush()

cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("✅ Câmera aberta!", flush=True)
    ret, frame = cap.read()
    if ret:
        print(f"✅ Frame lido! Shape: {frame.shape}", flush=True)
        print("Abrindo janela... Pressione qualquer tecla para fechar.", flush=True)
        cv2.imshow('Teste', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("❌ Não conseguiu ler frame", flush=True)
    cap.release()
else:
    print("❌ Não conseguiu abrir câmera", flush=True)
    print("Tentando câmera índice 1...", flush=True)
    cap = cv2.VideoCapture(1)
    if cap.isOpened():
        print("✅ Câmera 1 aberta!", flush=True)
        cap.release()
    else:
        print("❌ Câmera 1 também não funcionou", flush=True)

print("Fim do teste", flush=True)




