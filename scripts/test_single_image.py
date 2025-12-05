#!/usr/bin/env python3
"""
Script para testar UMA imagem específica com o MESMO pipeline do generate_metrics_table.py
Usa load_model, load_labels e preprocess_image do próprio script de métricas.
"""

import os
from pathlib import Path
import numpy as np

import tensorflow as tf  # garante que tf esteja disponível aqui também

# Importa funções do script de métricas
from scripts.generate_metrics_table import load_model, load_labels, preprocess_image


def test_single_image(image_path: str):
    """Testa uma única imagem e imprime as probabilidades para cada classe."""
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"❌ Imagem não encontrada: {image_path}")
        return

    print("=" * 80)
    print("🔍 TESTE DE IMAGEM ÚNICA")
    print("=" * 80)
    print(f"📸 Imagem: {image_path}")

    # Carregar modelo e labels usando o mesmo código do generate_metrics_table.py
    model_path = "dataset/keras_model.h5"
    if not os.path.exists(model_path):
        print(f"❌ Modelo não encontrado em: {model_path}")
        return

    model = load_model(model_path)
    labels = load_labels("dataset/labels.txt")

    print(f"\n📋 Labels carregadas: {labels}")

    # Pré-processar imagem (mesma função usada no script de métricas)
    img = preprocess_image(str(image_path))

    # Fazer predição
    print("\n🔮 Fazendo predição na imagem...")
    pred_proba = model.predict(img, verbose=0)[0]
    pred_idx = int(np.argmax(pred_proba))
    confidence = float(pred_proba[pred_idx])

    pred_label = labels[pred_idx] if pred_idx < len(labels) else f"índice {pred_idx}"

    print("\n📊 Probabilidades por classe:")
    for i, prob in enumerate(pred_proba):
        label_name = labels[i] if i < len(labels) else f"Classe {i}"
        marker = "  ⭐" if i == pred_idx else ""
        print(f"   {label_name} (índice {i}): {prob:.4f}{marker}")

    print("\n🎯 Resultado final:")
    print(f"   Classe predita: {pred_label} (índice {pred_idx}) com confiança {confidence:.2%}")
    print("=" * 80)


if __name__ == "__main__":
    # Caminho padrão que você passou: A_0006.jpg
    default_image = "dataset/test_images/A/A_0006.jpg"
    test_single_image(default_image)


