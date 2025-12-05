#!/usr/bin/env python3
"""
Script para testar o mapeamento real do modelo
Testa algumas imagens conhecidas para descobrir a ordem correta das classes
"""

import sys
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import cv2

# Adicionar diretório raiz ao path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def preprocess_image(image_path: str, target_size: tuple = (224, 224)):
    """Pré-processa uma imagem"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Não foi possível carregar: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def test_model_mapping():
    """Testa o mapeamento do modelo com imagens conhecidas"""
    
    print("="*80)
    print("🔍 TESTE DE MAPEAMENTO DO MODELO")
    print("="*80)
    
    # Carregar modelo
    model_path = "dataset/keras_model.h5"
    if not os.path.exists(model_path):
        print(f"❌ Modelo não encontrado: {model_path}")
        return
    
    print(f"\n📦 Carregando modelo: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"✅ Modelo carregado!")
        print(f"   Output shape: {model.output_shape}")
        n_classes = model.output_shape[1] if len(model.output_shape) > 1 else 5
        print(f"   Número de classes: {n_classes}")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return
    
    # Carregar labels
    labels_path = "dataset/labels.txt"
    labels = []
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) > 1:
                        labels.append(parts[1])
                    else:
                        labels.append(parts[0])
    else:
        labels = ["A", "E", "I", "O", "U"]
    
    print(f"\n📋 Labels do labels.txt: {labels}")
    print(f"   Ordem assumida: índice 0 = {labels[0]}, índice 1 = {labels[1]}, etc.")
    
    # Testar com imagens conhecidas
    test_dir = Path("dataset/test_images")
    if not test_dir.exists():
        print(f"\n⚠️  Diretório de teste não encontrado: {test_dir}")
        print("   Crie pastas A/, E/, I/, O/, U/ com imagens de teste")
        return
    
    print(f"\n🧪 Testando com imagens conhecidas...")
    print("="*80)
    
    # Para cada classe, testar uma imagem
    results = {}
    for label in labels:
        label_dir = test_dir / label
        if not label_dir.exists():
            print(f"   ⚠️  Pasta {label}/ não encontrada - pulando")
            continue
        
        # Pegar primeira imagem da pasta
        image_files = list(label_dir.glob("*.jpg")) + list(label_dir.glob("*.JPG"))
        if not image_files:
            print(f"   ⚠️  Nenhuma imagem encontrada em {label}/ - pulando")
            continue
        
        test_image = image_files[0]
        print(f"\n   📸 Testando imagem de {label}: {test_image.name}")
        
        try:
            img = preprocess_image(test_image)
            pred_proba = model.predict(img, verbose=0)[0]
            pred_idx = np.argmax(pred_proba)
            confidence = pred_proba[pred_idx]
            
            print(f"      → Modelo prediz: índice {pred_idx} com confiança {confidence:.2%}")
            if pred_idx < len(labels):
                print(f"      → Interpretado como: {labels[pred_idx]}")
            else:
                print(f"      → ⚠️  Índice {pred_idx} está fora do range de labels!")
            
            # Mostrar todas as probabilidades
            print(f"      Probabilidades para todas as classes:")
            for i, prob in enumerate(pred_proba):
                label_name = labels[i] if i < len(labels) else f"Classe {i}"
                marker = " ⭐" if i == pred_idx else ""
                print(f"         {label_name} (índice {i}): {prob:.2%}{marker}")
            
            results[label] = {
                'predicted_idx': int(pred_idx),
                'confidence': float(confidence),
                'expected_idx': labels.index(label)
            }
            
        except Exception as e:
            print(f"      ❌ Erro ao processar imagem: {e}")
            continue
    
    # Análise final
    print("\n" + "="*80)
    print("📊 ANÁLISE DO MAPEAMENTO")
    print("="*80)
    
    correct = 0
    total = 0
    for label, result in results.items():
        expected = result['expected_idx']
        predicted = result['predicted_idx']
        match = expected == predicted
        if match:
            correct += 1
        total += 1
        status = "✅" if match else "❌"
        print(f"   {status} {label}: esperado índice {expected}, modelo retornou índice {predicted}")
    
    accuracy = correct / total if total > 0 else 0
    print(f"\n   Precisão do mapeamento: {accuracy:.1%} ({correct}/{total})")
    
    if accuracy < 0.8:
        print(f"\n   ⚠️  ATENÇÃO: Mapeamento pode estar incorreto!")
        print(f"   O modelo pode ter sido treinado com ordem diferente de classes.")
        print(f"   Verifique a ordem real das classes no Teachable Machine.")
    else:
        print(f"\n   ✅ Mapeamento parece correto!")

if __name__ == "__main__":
    test_model_mapping()

