#!/usr/bin/env python3
"""
Script para verificar a ordem real das classes do modelo
Testa imagens conhecidas para descobrir qual índice corresponde a qual classe
"""

import sys
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import cv2

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

def main():
    print("="*80)
    print("🔍 VERIFICAÇÃO DA ORDEM REAL DAS CLASSES DO MODELO")
    print("="*80)
    
    # Carregar modelo
    model_path = "dataset/keras_model.h5"
    if not os.path.exists(model_path):
        print(f"❌ Modelo não encontrado: {model_path}")
        return
    
    print(f"\n📦 Carregando modelo: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        n_classes = model.output_shape[1] if len(model.output_shape) > 1 else 5
        print(f"✅ Modelo carregado! Número de classes: {n_classes}")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Carregar labels do labels.txt
    labels_path = "dataset/labels.txt"
    labels_from_file = []
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) > 1:
                        labels_from_file.append(parts[1])
                    else:
                        labels_from_file.append(parts[0])
    else:
        labels_from_file = ["A", "E", "I", "O", "U"]
    
    print(f"\n📋 Labels do labels.txt: {labels_from_file}")
    print(f"   Ordem assumida: índice 0 = {labels_from_file[0]}, índice 1 = {labels_from_file[1]}, etc.")
    
    # Testar com imagens conhecidas
    test_dir = Path("dataset/test_images")
    if not test_dir.exists():
        print(f"\n⚠️  Diretório de teste não encontrado: {test_dir}")
        return
    
    print(f"\n🧪 Testando modelo com imagens conhecidas...")
    print("="*80)
    
    # Mapeamento descoberto: {índice_modelo: nome_classe_real}
    discovered_mapping = {}
    
    # Para cada classe, testar algumas imagens
    for label in labels_from_file:
        label_dir = test_dir / label
        if not label_dir.exists():
            print(f"   ⚠️  Pasta {label}/ não encontrada - pulando")
            continue
        
        # Pegar algumas imagens da pasta
        image_files = list(label_dir.glob("*.jpg")) + list(label_dir.glob("*.JPG"))
        if not image_files:
            print(f"   ⚠️  Nenhuma imagem encontrada em {label}/ - pulando")
            continue
        
        # Testar até 5 imagens
        test_images = image_files[:5]
        predictions = []
        
        for test_image in test_images:
            try:
                img = preprocess_image(test_image)
                pred_proba = model.predict(img, verbose=0)[0]
                pred_idx = np.argmax(pred_proba)
                confidence = pred_proba[pred_idx]
                predictions.append((pred_idx, confidence))
            except Exception as e:
                print(f"      ⚠️  Erro ao processar {test_image.name}: {e}")
                continue
        
        if predictions:
            # Pegar o índice mais frequente
            pred_indices = [p[0] for p in predictions]
            most_common_idx = max(set(pred_indices), key=pred_indices.count)
            avg_confidence = np.mean([p[1] for p in predictions if p[0] == most_common_idx])
            
            discovered_mapping[most_common_idx] = label
            print(f"   ✅ {label}: modelo retorna índice {most_common_idx} (confiança média: {avg_confidence:.1%})")
    
    # Análise final
    print("\n" + "="*80)
    print("📊 MAPEAMENTO DESCOBERTO")
    print("="*80)
    
    print(f"\n   Ordem REAL do modelo (descoberta):")
    for idx in sorted(discovered_mapping.keys()):
        print(f"      Índice {idx} = {discovered_mapping[idx]}")
    
    print(f"\n   Ordem do labels.txt (assumida):")
    for idx, label in enumerate(labels_from_file):
        print(f"      Índice {idx} = {label}")
    
    # Verificar se há diferença
    mismatch = False
    for idx in range(n_classes):
        expected = labels_from_file[idx] if idx < len(labels_from_file) else None
        actual = discovered_mapping.get(idx, None)
        if expected != actual:
            mismatch = True
            print(f"\n   ❌ MISMATCH no índice {idx}:")
            print(f"      labels.txt diz: {expected}")
            print(f"      Modelo retorna: {actual}")
    
    if mismatch:
        print(f"\n   ⚠️  ORDEM DIFERENTE DETECTADA!")
        print(f"   O modelo foi treinado com ordem diferente do labels.txt")
        print(f"\n   💡 SOLUÇÃO: Ajuste o labels.txt para refletir a ordem real do modelo:")
        print(f"      (ordem descoberta)")
        for idx in sorted(discovered_mapping.keys()):
            print(f"      {idx} {discovered_mapping[idx]}")
    else:
        print(f"\n   ✅ Ordem está correta!")

if __name__ == "__main__":
    main()

