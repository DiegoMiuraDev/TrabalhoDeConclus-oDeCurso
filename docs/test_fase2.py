#!/usr/bin/env python3
"""
Script de teste rápido da Fase 2
Verifica se todos os módulos estão funcionando corretamente
"""

import sys
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / "src"))

print("="*60)
print("🧪 TESTE DA FASE 2 - MÓDULOS")
print("="*60)

# Teste 1: Importações
print("\n1️⃣  Testando importações...")
try:
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    print("   ✅ TensorFlow:", tf.__version__)
    print("   ✅ Keras:", keras.__version__)
    print("   ✅ NumPy:", np.__version__)
except Exception as e:
    print(f"   ❌ Erro nas importações básicas: {e}")
    sys.exit(1)

# Teste 2: Módulos do projeto
print("\n2️⃣  Testando módulos do projeto...")
try:
    from data.dataset_loader import LibrasDatasetLoader
    from data.preprocessing import ImagePreprocessor
    from models.mobilenet_model import MobileNetLibrasModel, create_mobilenet_model
    from models.training import LibrasModelTrainer
    from utils.helpers import check_gpu_availability
    from visualization.plots import LibrasVisualizer
    print("   ✅ Todos os módulos importados com sucesso!")
except Exception as e:
    print(f"   ❌ Erro ao importar módulos: {e}")
    sys.exit(1)

# Teste 3: Verificar GPU
print("\n3️⃣  Verificando GPU...")
has_gpu = check_gpu_availability()
if has_gpu:
    print("   ✅ GPU disponível para treinamento!")
else:
    print("   ⚠️  Nenhuma GPU detectada (usará CPU)")

# Teste 4: Criar modelo de teste
print("\n4️⃣  Testando criação do modelo...")
try:
    print("   Criando modelo MobileNetV2...")
    model_builder = MobileNetLibrasModel(
        input_shape=(224, 224, 3),
        n_classes=24,
        dropout_rate=0.5,
        dense_units=128
    )
    
    model = model_builder.build_model(trainable_base=False)
    print(f"   ✅ Modelo criado com {model.count_params():,} parâmetros")
    
    # Compilar
    model_builder.compile_model(learning_rate=0.001)
    print("   ✅ Modelo compilado com sucesso!")
    
except Exception as e:
    print(f"   ❌ Erro ao criar modelo: {e}")
    sys.exit(1)

# Teste 5: Predição de teste
print("\n5️⃣  Testando predição com dados sintéticos...")
try:
    # Criar dados de teste
    test_images = np.random.rand(5, 224, 224, 3).astype(np.float32)
    
    # Fazer predição
    predictions = model_builder.predict(test_images)
    predicted_classes = model_builder.predict_classes(test_images)
    
    print(f"   ✅ Predições realizadas!")
    print(f"   Shape das predições: {predictions.shape}")
    print(f"   Classes preditas: {predicted_classes}")
    
except Exception as e:
    print(f"   ❌ Erro na predição: {e}")
    sys.exit(1)

# Teste 6: Pré-processador
print("\n6️⃣  Testando pré-processador de imagens...")
try:
    preprocessor = ImagePreprocessor()
    
    # Criar imagens de teste
    test_imgs = np.random.randint(0, 256, (10, 28, 28), dtype=np.uint8)
    
    # Pré-processar
    processed = preprocessor.preprocess_for_mobilenet(test_imgs)
    
    print(f"   ✅ Pré-processamento OK!")
    print(f"   Shape: {test_imgs.shape} → {processed.shape}")
    
except Exception as e:
    print(f"   ❌ Erro no pré-processamento: {e}")
    sys.exit(1)

# Teste 7: Visualizador
print("\n7️⃣  Testando visualizador...")
try:
    visualizer = LibrasVisualizer()
    print("   ✅ Visualizador criado com sucesso!")
except Exception as e:
    print(f"   ❌ Erro ao criar visualizador: {e}")
    sys.exit(1)

# Resumo final
print("\n" + "="*60)
print("✅ TODOS OS TESTES PASSARAM!")
print("="*60)
print("\n📋 RESUMO:")
print("   ✅ Bibliotecas instaladas")
print("   ✅ Módulos funcionando")
print("   ✅ Modelo pode ser criado")
print("   ✅ Predições funcionam")
print("   ✅ Pré-processamento OK")
print("   ✅ Visualizações OK")

if has_gpu:
    print("\n🚀 Sistema pronto para treinamento COM GPU!")
    print("   Tempo estimado: 30-45 minutos")
else:
    print("\n⚠️  Sistema pronto para treinamento SEM GPU")
    print("   Tempo estimado: 2-3 horas")

print("\n📚 PRÓXIMOS PASSOS:")
print("   1. Executar Fase 1 (explorar dados)")
print("   2. Executar Fase 2 (treinar modelo)")
print("   3. Executar Fase 3 (usar na webcam)")

print("\n💡 COMANDOS ÚTEIS:")
print("   # Ver dados:")
print("   jupyter notebook notebooks/01_data_exploration_simples.ipynb")
print("\n   # Treinar modelo:")
print("   python scripts/train_model.py")
print("\n   # Ou usar notebook:")
print("   jupyter notebook notebooks/02_model_training.ipynb")

print("\n" + "="*60)
print("✨ Fase 2 está pronta para uso!")
print("="*60)

