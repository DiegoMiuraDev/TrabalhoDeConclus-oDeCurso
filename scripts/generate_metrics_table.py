                      
"""
Script para gerar tabela de métricas de precisão do modelo de reconhecimento de Libras
Funciona com modelo do Teachable Machine usando imagens reais organizadas por classe
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    confusion_matrix
)

                                  
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.append(str(project_root / "src"))

from configs.config import LIBRAS_CLASSES, DATASET_CONFIG, MODEL_CONFIG

                                              
DATASET_CONFIG["n_classes"] = 5


def load_model(model_path: str):
    """
    Carrega o modelo treinado (compatível com Teachable Machine)
    
    Args:
        model_path (str): Caminho para o modelo
        
    Returns:
        Modelo carregado
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado em: {model_path}")
    
    print(f"📦 Carregando modelo de: {model_path}")
    
                                                             
    try:
                                                                  
        import h5py
        import json
        
                          
        with h5py.File(model_path, 'r') as f:
                                                        
            if 'model_config' in f.attrs:
                model_config = json.loads(f.attrs['model_config'])
                
                                                                
                def clean_config(config):
                    if isinstance(config, dict):
                        if 'config' in config and isinstance(config['config'], dict):
                                                         
                            config['config'].pop('groups', None)
                                                           
                        for key, value in config.items():
                            if isinstance(value, (dict, list)):
                                clean_config(value)
                    elif isinstance(config, list):
                        for item in config:
                            clean_config(item)
                
                clean_config(model_config)
        
                                              
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
        except (TypeError, ValueError) as e:
                                                                              
            print("   ⚠️  Tentando carregar com tratamento de compatibilidade...")
            
                                                                      
                                           
            try:
                                                                               
                import warnings
                warnings.filterwarnings('ignore')
                
                                                                             
                model = tf.keras.models.load_model(
                    model_path, 
                    compile=False,
                    custom_objects={}
                )
            except Exception:
                                                                      
                tflite_path = model_path.replace('.h5', '.tflite').replace('keras_model', 'model_unquant')
                if os.path.exists(tflite_path):
                    print(f"   💡 Tentando carregar modelo TFLite: {tflite_path}")
                    return load_tflite_model(tflite_path)
                else:
                    raise e
        
    except Exception as e:
                                               
        print(f"   ⚠️  Erro ao carregar: {e}")
        print("   💡 Tentando método alternativo...")
        model = tf.keras.models.load_model(model_path, compile=False)
    
    print(f"✅ Modelo carregado com sucesso!")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    return model


def load_tflite_model(tflite_path: str):
    """
    Carrega modelo TensorFlow Lite como wrapper Keras
    
    Args:
        tflite_path: Caminho para modelo .tflite
        
    Returns:
        Wrapper do modelo
    """
    class TFLiteModelWrapper:
        def __init__(self, tflite_path):
            self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_shape = self.input_details[0]['shape']
            self.output_shape = self.output_details[0]['shape']
        
        def predict(self, X, verbose=0, batch_size=32):
            predictions = []
            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                batch_preds = []
                for img in batch:
                                    
                    input_data = np.expand_dims(img, axis=0).astype(np.float32)
                    self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                    self.interpreter.invoke()
                                  
                    output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                    batch_preds.append(output_data[0])
                predictions.extend(batch_preds)
            return np.array(predictions)
    
    return TFLiteModelWrapper(tflite_path)


def load_labels(labels_path: str = "dataset/labels.txt"):
    """
    Carrega labels do modelo de vogais (A, E, I, O, U).
    Mantemos isso o mais simples e determinístico possível:
    1. Usa dataset/labels.txt, que é o arquivo exportado pelo Teachable Machine
    2. Se não existir ou estiver vazio, usa fallback ["A", "E", "I", "O", "U"]
    
    Args:
        labels_path (str): Caminho para o arquivo de labels
        
    Returns:
        Lista de labels
    """
                                                                          
    if os.path.exists(labels_path):
        labels = []
        with open(labels_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                                                  
                    parts = line.split()
                    if len(parts) > 1:
                        labels.append(parts[1])
                    else:
                        labels.append(parts[0])
        if labels:
            print(f"✅ Labels carregadas de {labels_path}: {labels}")
            return labels
    
                                         
    fallback = ["A", "E", "I", "O", "U"]
    print(f"⚠️  Nenhum arquivo de labels encontrado, usando fallback: {fallback}")
    return fallback


                                                                               
                                          
                                                                               
                                                                               


def preprocess_image(image_path: str, target_size: tuple = (224, 224)):
    """
    Carrega e pré-processa uma imagem para o modelo
    
    Args:
        image_path (str): Caminho para a imagem
        target_size (tuple): Tamanho alvo (altura, largura)
        
    Returns:
        Imagem pré-processada
    """
                     
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Não foi possível carregar imagem: {image_path}")
    
                            
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
                   
    img = cv2.resize(img, target_size)
    
                            
    img = img.astype(np.float32) / 255.0
    
                                 
    img = np.expand_dims(img, axis=0)
    
    return img


def load_test_images_from_folders(test_dir: str, labels: list):
    """
    Carrega imagens de teste organizadas em pastas por classe
    
    Args:
        test_dir (str): Diretório com subpastas por classe
        labels (list): Lista de labels (classes)
        
    Returns:
        X_test, y_test: Imagens e labels
    """
    test_path = Path(test_dir)
    
    if not test_path.exists():
        raise FileNotFoundError(f"Diretório de teste não encontrado: {test_dir}")
    
    images = []
    labels_list = []
    
    print(f"\n📂 Procurando imagens em: {test_dir}")
    
                                                  
    found_classes = {}
    for label in labels:
        label_path = test_path / label
        if label_path.exists() and label_path.is_dir():
            found_classes[label] = label_path
            print(f"   ✅ Encontrada pasta: {label}/")
    
    if not found_classes:
                                                                        
        for idx, label in enumerate(labels):
            num_path = test_path / str(idx)
            if num_path.exists() and num_path.is_dir():
                found_classes[label] = num_path
                print(f"   ✅ Encontrada pasta numerada: {idx}/ -> {label}")
    
    if not found_classes:
                                                                         
        print(f"   ⚠️  Nenhuma subpasta encontrada. Procurando imagens diretamente...")
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        for ext in image_extensions:
            for img_path in test_path.glob(f"*{ext}"):
                                                          
                img_name = img_path.stem.lower()
                for label in labels:
                    if label.lower() in img_name:
                        if label not in found_classes:
                            found_classes[label] = []
                        found_classes[label].append(img_path)
                        break
    
    if not found_classes:
        raise FileNotFoundError(
            f"Nenhuma imagem encontrada em {test_dir}.\n"
            f"Estrutura esperada:\n"
            f"  {test_dir}/\n"
            f"    A/\n"
            f"      imagem1.jpg\n"
            f"      imagem2.jpg\n"
            f"    E/\n"
            f"      ...\n"
        )
    
                      
                                                                           
    print(f"\n📸 Carregando imagens...")
    print(f"   Ordem esperada das classes: {labels}")
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
    
                                                                                                  
    for label_idx, label in enumerate(labels):
        if label not in found_classes:
            print(f"   ⚠️  Pasta {label}/ não encontrada - pulando")
            continue
            
        path = found_classes[label]
        
        if isinstance(path, list):
                               
            image_files = path
        else:
                                                     
            image_files_set = set()
            for ext in image_extensions:
                for img_file in path.glob(f"*{ext}"):
                                                                         
                    image_files_set.add(img_file.resolve())
            image_files = list(image_files_set)
                                                
            image_files.sort(key=lambda x: x.name)
        
                                                               
        print(f"   📁 {label} (índice {label_idx}): {len(image_files)} imagens")
        
        for img_path in image_files:
            try:
                img = preprocess_image(img_path)
                images.append(img)
                labels_list.append(label_idx)                                   
            except Exception as e:
                print(f"      ⚠️  Erro ao carregar {img_path.name}: {e}")
                continue
    
    if len(images) == 0:
        raise ValueError("Nenhuma imagem foi carregada com sucesso!")
    
    X_test = np.vstack(images)
    y_test = np.array(labels_list)
    
    print(f"\n✅ Total de imagens carregadas: {len(images)}")
    print(f"   Distribuição por classe:")
    for label in labels:
        count = np.sum(y_test == labels.index(label))
        print(f"      {label}: {count} imagens")
    
    return X_test, y_test


                                                                              
                                                                                 


def evaluate_model(model, X_test, y_test, class_names=None):
    """
    Avalia o modelo e retorna predições
    
    Args:
        model: Modelo Keras
        X_test: Dados de teste
        y_test: Labels verdadeiros
        class_names: Nomes das classes para debug
        
    Returns:
        y_pred: Predições do modelo
        y_true: Labels verdadeiros
    """
    print("\n🔍 Avaliando modelo no conjunto de teste...")
    print(f"   Assumindo mapeamento direto do labels.txt:")
    if class_names:
        for i, name in enumerate(class_names):
            print(f"      Índice {i} do modelo = {name}")
    
                     
    print("   Fazendo predições...")
    y_pred_proba = model.predict(X_test, verbose=1, batch_size=32)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = y_test
    
                                            
    if class_names and len(y_true) > 0:
                                                                     
        print(f"\n   🔍 Verificando mapeamento (primeiras 10 predições vs labels verdadeiros):")
        for i in range(min(10, len(y_true))):
            true_label = class_names[y_true[i]] if y_true[i] < len(class_names) else f"índice {y_true[i]}"
            pred_label = class_names[y_pred[i]] if y_pred[i] < len(class_names) else f"índice {y_pred[i]}"
            match = "✅" if y_true[i] == y_pred[i] else "❌"
            print(f"      {match} Imagem {i+1}: verdadeiro={true_label} (índice {y_true[i]}), predito={pred_label} (índice {y_pred[i]})")
    
                                              
    if class_names:
        print("\n   📊 Distribuição de predições:")
        unique, counts = np.unique(y_pred, return_counts=True)
        for idx, count in zip(unique, counts):
            if idx < len(class_names):
                print(f"      {class_names[idx]} (índice {idx}): {count} predições")
        
        print("\n   📊 Distribuição de labels verdadeiros:")
        unique_true, counts_true = np.unique(y_true, return_counts=True)
        for idx, count in zip(unique_true, counts_true):
            if idx < len(class_names):
                print(f"      {class_names[idx]} (índice {idx}): {count} amostras")
        
                                                                       
        if len(class_names) > 0:
            class_a_idx = 0                              
            a_true_indices = np.where(y_true == class_a_idx)[0]
            if len(a_true_indices) > 0:
                print(f"\n   🔍 DEBUG - Análise da classe A (índice {class_a_idx}):")
                print(f"      Total de imagens de A: {len(a_true_indices)}")
                a_predictions = y_pred[a_true_indices]
                unique_a_preds, counts_a_preds = np.unique(a_predictions, return_counts=True)
                print(f"      Predições do modelo para imagens de A:")
                for pred_idx, pred_count in zip(unique_a_preds, counts_a_preds):
                    if pred_idx < len(class_names):
                        print(f"         → Predito como {class_names[pred_idx]} (índice {pred_idx}): {pred_count} vezes")
                    else:
                        print(f"         → Predito como índice {pred_idx}: {pred_count} vezes")
                
                                                                   
                print(f"\n      Exemplo de probabilidades para primeira imagem de A:")
                first_a_idx = a_true_indices[0]
                probs = y_pred_proba[first_a_idx]
                for i, prob in enumerate(probs):
                    if i < len(class_names):
                        print(f"         {class_names[i]}: {prob:.4f}")
    
    return y_pred, y_true


def calculate_metrics(y_true, y_pred, class_names):
    """
    Calcula métricas de precisão por classe
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Predições
        class_names: Nomes das classes
        
    Returns:
        DataFrame com métricas por classe
    """
    print("\n📊 Calculando métricas de precisão...")
    
                                                      
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    support = cm.sum(axis=1)
    
                                       
    print("\n   📋 Matriz de Confusão (linhas = verdadeiro, colunas = predito):")
    header = "      " + " ".join([f"{name:>6}" for name in class_names])
    print(header)
    print("      " + "-" * (len(class_names) * 7))
    for i, row in enumerate(cm):
        row_str = f"   {class_names[i] if i < len(class_names) else str(i):>3} "
        row_str += " ".join([f"{val:>6}" for val in row])
        print(row_str)
    
                                         
    print("\n   🔍 Análise detalhada por classe:")
    for i, class_name in enumerate(class_names):
        if i < len(cm):
            true_positives = cm[i, i]
            false_positives = cm[:, i].sum() - true_positives
            false_negatives = cm[i, :].sum() - true_positives
            total_true = support[i]
            
            print(f"\n      {class_name} (índice {i}):")
            print(f"         Verdadeiros Positivos (TP): {true_positives}")
            print(f"         Falsos Positivos (FP): {false_positives}")
            print(f"         Falsos Negativos (FN): {false_negatives}")
            print(f"         Total de amostras reais: {total_true}")
            
            if (true_positives + false_positives) > 0:
                precision = true_positives / (true_positives + false_positives)
                print(f"         Precisão calculada: {precision:.4f}")
            else:
                print(f"         Precisão: 0.0000 (nenhum TP ou FP)")
            
            if total_true > 0:
                recall = true_positives / total_true
                print(f"         Recall calculado: {recall:.4f}")
            else:
                print(f"         Recall: 0.0000 (nenhuma amostra real)")
    
                                  
                                                                              
    precision = precision_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(len(class_names))))
    recall = recall_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(len(class_names))))
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(len(class_names))))
    
                                              
    if len(support) < len(class_names):
                                                       
        support_full = np.zeros(len(class_names), dtype=int)
        for i in range(len(support)):
            support_full[i] = support[i]
        support = support_full
    elif len(support) > len(class_names):
        support = support[:len(class_names)]
    
                                                      
    n_classes = len(class_names)
    if len(precision) != n_classes:
        precision = np.pad(precision, (0, n_classes - len(precision)), mode='constant', constant_values=0.0)
    if len(recall) != n_classes:
        recall = np.pad(recall, (0, n_classes - len(recall)), mode='constant', constant_values=0.0)
    if len(f1) != n_classes:
        f1 = np.pad(f1, (0, n_classes - len(f1)), mode='constant', constant_values=0.0)
    
                     
    metrics_df = pd.DataFrame({
        'Classe': [class_names[i] for i in range(len(class_names))],
        'Precisão': precision[:len(class_names)],
        'Recall': recall[:len(class_names)],
        'F1-Score': f1[:len(class_names)],
        'Suporte': support[:len(class_names)]
    })
    
                               
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    weighted_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    accuracy = np.mean(y_true == y_pred)
    
                                      
    totals_row = pd.DataFrame({
        'Classe': ['MÉDIA MACRO', 'MÉDIA PONDERADA', 'ACURÁCIA GERAL'],
        'Precisão': [macro_precision, weighted_precision, accuracy],
        'Recall': [macro_recall, weighted_recall, accuracy],
        'F1-Score': [macro_f1, weighted_f1, accuracy],
        'Suporte': [len(y_true), len(y_true), len(y_true)]
    })
    
    metrics_df = pd.concat([metrics_df, totals_row], ignore_index=True)
    
    return metrics_df, cm


def format_table_markdown(df):
    """
    Formata DataFrame como tabela Markdown
    
    Args:
        df: DataFrame
        
    Returns:
        String formatada em Markdown
    """
                                  
    df_formatted = df.copy()
    for col in ['Precisão', 'Recall', 'F1-Score']:
        if col in df_formatted.columns:
            df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
    
                                         
    try:
        markdown = df_formatted.to_markdown(index=False, tablefmt='grid')
    except ImportError:
                                                     
        lines = []
                   
        headers = list(df_formatted.columns)
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---" for _ in headers]) + "|")
               
        for _, row in df_formatted.iterrows():
            lines.append("| " + " | ".join(str(val) for val in row) + " |")
        markdown = "\n".join(lines)
    
    return markdown


def save_results(metrics_df, cm, output_dir: str = "results"):
    """
    Salva resultados em diferentes formatos
    
    Args:
        metrics_df: DataFrame com métricas
        cm: Matriz de confusão
        output_dir: Diretório de saída
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
                
    csv_path = output_path / "metricas_precisao.csv"
    metrics_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 Tabela salva em CSV: {csv_path}")
    
                     
    md_path = output_path / "metricas_precisao.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Tabela de Métricas de Precisão - Reconhecimento de Libras\n\n")
        f.write("## Métricas por Classe\n\n")
        f.write(format_table_markdown(metrics_df))
        f.write("\n\n")
        f.write("## Legenda\n\n")
        f.write("- **Precisão**: Proporção de predições positivas que são realmente positivas\n")
        f.write("- **Recall**: Proporção de casos positivos que foram corretamente identificados\n")
        f.write("- **F1-Score**: Média harmônica entre Precisão e Recall\n")
        f.write("- **Suporte**: Número de amostras reais de cada classe\n")
        f.write("- **Média Macro**: Média aritmética simples das métricas por classe\n")
        f.write("- **Média Ponderada**: Média ponderada pelo número de amostras de cada classe\n")
        f.write("- **Acurácia Geral**: Proporção de predições corretas no total\n")
    
    print(f"💾 Tabela salva em Markdown: {md_path}")
    
                               
    cm_path = output_path / "matriz_confusao.npy"
    np.save(cm_path, cm)
    print(f"💾 Matriz de confusão salva em: {cm_path}")


def print_summary(metrics_df):
    """
    Imprime resumo das métricas
    
    Args:
        metrics_df: DataFrame com métricas
    """
    print("\n" + "="*80)
    print("📊 RESUMO DAS MÉTRICAS DE PRECISÃO")
    print("="*80)
    
                                            
    class_metrics = metrics_df[metrics_df['Classe'].isin(LIBRAS_CLASSES.values())]
    summary_metrics = metrics_df[~metrics_df['Classe'].isin(LIBRAS_CLASSES.values())]
    
    print("\n📈 Métricas por Classe:")
    print("-"*80)
    print(class_metrics.to_string(index=False))
    
    print("\n📊 Métricas Gerais:")
    print("-"*80)
    print(summary_metrics.to_string(index=False))
    
    print("\n" + "="*80)
    
                             
    if len(class_metrics) > 0:
        print("\n🎯 Estatísticas Adicionais:")
        print("-"*80)
        best_precision = class_metrics.loc[class_metrics['Precisão'].idxmax()]
        worst_precision = class_metrics.loc[class_metrics['Precisão'].idxmin()]
        
        print(f"✅ Melhor Precisão: {best_precision['Classe']} ({best_precision['Precisão']:.4f})")
        print(f"❌ Pior Precisão: {worst_precision['Classe']} ({worst_precision['Precisão']:.4f})")
        
        best_recall = class_metrics.loc[class_metrics['Recall'].idxmax()]
        worst_recall = class_metrics.loc[class_metrics['Recall'].idxmin()]
        
        print(f"✅ Melhor Recall: {best_recall['Classe']} ({best_recall['Recall']:.4f})")
        print(f"❌ Pior Recall: {worst_recall['Classe']} ({worst_recall['Recall']:.4f})")
        
        best_f1 = class_metrics.loc[class_metrics['F1-Score'].idxmax()]
        worst_f1 = class_metrics.loc[class_metrics['F1-Score'].idxmin()]
        
        print(f"✅ Melhor F1-Score: {best_f1['Classe']} ({best_f1['F1-Score']:.4f})")
        print(f"❌ Pior F1-Score: {worst_f1['Classe']} ({worst_f1['F1-Score']:.4f})")
        print("="*80)


def main():
    """
    Função principal
    """
    print("🚀 Gerando Tabela de Métricas de Precisão")
    print("="*80)
    
                   
    model_path = "dataset/keras_model.h5"                               
    test_dir = "dataset/test_images"                                                     
    labels_path = "dataset/labels.txt"
    output_dir = "results"
    
    try:
                            
        if not os.path.exists(model_path):
            print(f"⚠️  Modelo não encontrado em: {model_path}")
            print("💡 Tentando carregar modelo alternativo...")
            
            alternative_paths = [
                "models/libras_brasileiro_best.h5",
                "models/libras_model.h5"
            ]
            
            model_loaded = False
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    model_loaded = True
                    break
            
            if not model_loaded:
                raise FileNotFoundError(
                    "Nenhum modelo encontrado. Por favor, verifique o caminho do modelo."
                )
        
        model = load_model(model_path)
        
                            
        labels = load_labels(labels_path)
        print(f"✅ Labels carregados: {labels}")
        
                                      
        print("\n📊 Carregando imagens de teste...")
        print("="*80)
        
                                  
        possible_test_dirs = [
            test_dir,
            "dataset/test",
            "test_images",
            "data/test_images",
            "dataset/images"
        ]
        
        images_loaded = False
        for test_dir_path in possible_test_dirs:
            try:
                X_test, y_test = load_test_images_from_folders(test_dir_path, labels)
                images_loaded = True
                break
            except FileNotFoundError as e:
                print(f"   ❌ {test_dir_path}: {str(e)[:80]}")
                continue
            except Exception as e:
                print(f"   ❌ {test_dir_path}: {str(e)[:80]}")
                continue
        
        if not images_loaded:
            print("\n" + "="*80)
            print("⚠️  IMAGENS DE TESTE NÃO ENCONTRADAS")
            print("="*80)
            print("\n📁 Para calcular métricas reais, organize suas imagens de teste assim:")
            print(f"\n   {test_dir}/")
            print("     A/")
            print("       imagem1.jpg")
            print("       imagem2.jpg")
            print("       ...")
            print("     E/")
            print("       imagem1.jpg")
            print("       ...")
            print("     I/")
            print("     O/")
            print("     U/")
            print("\n   Ou use pastas numeradas:")
            print("     0/  (para A)")
            print("     1/  (para E)")
            print("     2/  (para I)")
            print("     3/  (para O)")
            print("     4/  (para U)")
            print("\n   Formatos suportados: .jpg, .jpeg, .png, .bmp")
            print("\n" + "="*80)
            return
        
                                                                        
        y_pred, y_true = evaluate_model(model, X_test, y_test, labels)

                              
        metrics_df, cm = calculate_metrics(y_true, y_pred, labels)
        
                              
        print_summary(metrics_df)
        
                              
        save_results(metrics_df, cm, output_dir)
        
        print("\n✅ Tabela de métricas gerada com sucesso!")
        print(f"📁 Arquivos salvos em: {output_dir}/")
        
    except Exception as e:
        print(f"\n❌ Erro ao gerar tabela de métricas: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

