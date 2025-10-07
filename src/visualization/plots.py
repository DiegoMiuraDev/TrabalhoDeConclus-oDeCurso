"""
Módulo para visualização de dados e resultados do projeto de reconhecimento de Libras
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from configs.config import VISUALIZATION_CONFIG, LIBRAS_CLASSES


class LibrasVisualizer:
    """
    Classe para visualização de dados do projeto de reconhecimento de Libras
    """
    
    def __init__(self, style: str = "seaborn-v0_8", palette: str = "husl"):
        """
        Inicializa o visualizador
        
        Args:
            style (str): Estilo do matplotlib
            palette (str): Paleta de cores do seaborn
        """
        plt.style.use(style)
        sns.set_palette(palette)
        self.class_names = list(LIBRAS_CLASSES.values())
        
    def plot_class_distribution(self, class_counts: Dict[int, int], 
                               save_path: Optional[str] = None) -> None:
        """
        Plota a distribuição das classes
        
        Args:
            class_counts (Dict[int, int]): Contagem por classe
            save_path (str): Caminho para salvar (opcional)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfico de barras
        classes = sorted(class_counts.keys())
        counts = [class_counts[c] for c in classes]
        class_labels = [self.class_names[c] for c in classes]
        
        ax1.bar(class_labels, counts, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.set_title('Distribuição das Classes de Libras', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Classe (Letra)', fontsize=12)
        ax1.set_ylabel('Quantidade de Amostras', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Gráfico de pizza
        ax2.pie(counts, labels=class_labels, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Proporção das Classes', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_sample_images(self, images: np.ndarray, labels: np.ndarray,
                          n_samples: int = 8, n_cols: int = 4,
                          save_path: Optional[str] = None) -> None:
        """
        Plota amostras de imagens do dataset
        
        Args:
            images (np.ndarray): Array de imagens
            labels (np.ndarray): Labels correspondentes
            n_samples (int): Número de amostras para mostrar
            n_cols (int): Número de colunas
            save_path (str): Caminho para salvar (opcional)
        """
        n_rows = (n_samples + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            row = i // n_cols
            col = i % n_cols
            
            if i < len(images):
                axes[row, col].imshow(images[i], cmap='gray')
                axes[row, col].set_title(f'Classe: {self.class_names[labels[i]]}', 
                                       fontsize=12, fontweight='bold')
                axes[row, col].axis('off')
            else:
                axes[row, col].axis('off')
        
        plt.suptitle('Amostras do Dataset Libras MNIST', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_training_history(self, history: Dict[str, List[float]],
                             save_path: Optional[str] = None) -> None:
        """
        Plota o histórico de treinamento
        
        Args:
            history (Dict[str, List[float]]): Histórico de treinamento
            save_path (str): Caminho para salvar (opcional)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        epochs = range(1, len(history['accuracy']) + 1)
        
        # Acurácia
        ax1.plot(epochs, history['accuracy'], 'b-', label='Treino', linewidth=2)
        if 'val_accuracy' in history:
            ax1.plot(epochs, history['val_accuracy'], 'r-', label='Validação', linewidth=2)
        ax1.set_title('Acurácia do Modelo', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Época', fontsize=12)
        ax1.set_ylabel('Acurácia', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Perda
        ax2.plot(epochs, history['loss'], 'b-', label='Treino', linewidth=2)
        if 'val_loss' in history:
            ax2.plot(epochs, history['val_loss'], 'r-', label='Validação', linewidth=2)
        ax2.set_title('Perda do Modelo', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Época', fontsize=12)
        ax2.set_ylabel('Perda', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrix(self, cm: np.ndarray, 
                             save_path: Optional[str] = None) -> None:
        """
        Plota matriz de confusão
        
        Args:
            cm (np.ndarray): Matriz de confusão
            save_path (str): Caminho para salvar (opcional)
        """
        plt.figure(figsize=(12, 10))
        
        # Normalizar matriz para percentuais
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Criar heatmap
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Proporção'})
        
        plt.title('Matriz de Confusão Normalizada', fontsize=16, fontweight='bold')
        plt.xlabel('Predição', fontsize=12)
        plt.ylabel('Verdadeiro', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_class_accuracy(self, class_accuracy: Dict[int, float],
                           save_path: Optional[str] = None) -> None:
        """
        Plota acurácia por classe
        
        Args:
            class_accuracy (Dict[int, float]): Acurácia por classe
            save_path (str): Caminho para salvar (opcional)
        """
        classes = sorted(class_accuracy.keys())
        accuracies = [class_accuracy[c] for c in classes]
        class_labels = [self.class_names[c] for c in classes]
        
        plt.figure(figsize=(15, 8))
        bars = plt.bar(class_labels, accuracies, color='lightgreen', 
                      edgecolor='black', alpha=0.7)
        
        # Adicionar valores nas barras
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Acurácia por Classe', fontsize=16, fontweight='bold')
        plt.xlabel('Classe (Letra)', fontsize=12)
        plt.ylabel('Acurácia', fontsize=12)
        plt.xticks(rotation=45)
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_prediction_samples(self, images: np.ndarray, true_labels: np.ndarray,
                               predictions: np.ndarray, probabilities: np.ndarray,
                               n_samples: int = 8, n_cols: int = 4,
                               save_path: Optional[str] = None) -> None:
        """
        Plota amostras com predições
        
        Args:
            images (np.ndarray): Imagens
            true_labels (np.ndarray): Labels verdadeiros
            predictions (np.ndarray): Predições
            probabilities (np.ndarray): Probabilidades
            n_samples (int): Número de amostras
            n_cols (int): Número de colunas
            save_path (str): Caminho para salvar (opcional)
        """
        n_rows = (n_samples + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            row = i // n_cols
            col = i % n_cols
            
            if i < len(images):
                axes[row, col].imshow(images[i], cmap='gray')
                
                # Determinar cor baseada na acurácia
                is_correct = true_labels[i] == predictions[i]
                color = 'green' if is_correct else 'red'
                
                title = f'Verdadeiro: {self.class_names[true_labels[i]]}\n'
                title += f'Predito: {self.class_names[predictions[i]]}\n'
                title += f'Confiança: {probabilities[i]:.3f}'
                
                axes[row, col].set_title(title, fontsize=10, fontweight='bold', color=color)
                axes[row, col].axis('off')
            else:
                axes[row, col].axis('off')
        
        plt.suptitle('Amostras com Predições', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_architecture(self, model, save_path: Optional[str] = None) -> None:
        """
        Plota a arquitetura do modelo
        
        Args:
            model: Modelo Keras
            save_path (str): Caminho para salvar (opcional)
        """
        try:
            from tensorflow.keras.utils import plot_model
            
            plot_model(model, to_file=save_path, show_shapes=True, 
                      show_layer_names=True, rankdir='TB')
            print(f"✅ Arquitetura salva em: {save_path}")
        except ImportError:
            print("❌ TensorFlow não disponível para plotar arquitetura")
        except Exception as e:
            print(f"❌ Erro ao plotar arquitetura: {e}")
    
    def create_summary_plot(self, results: Dict[str, Any],
                           save_path: Optional[str] = None) -> None:
        """
        Cria um plot resumo com múltiplas métricas
        
        Args:
            results (Dict[str, Any]): Resultados do modelo
            save_path (str): Caminho para salvar (opcional)
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Acurácia geral
        ax1.bar(['Treino', 'Validação', 'Teste'], 
               [results.get('train_acc', 0), results.get('val_acc', 0), results.get('test_acc', 0)],
               color=['blue', 'orange', 'green'], alpha=0.7)
        ax1.set_title('Acurácia Geral', fontweight='bold')
        ax1.set_ylabel('Acurácia')
        ax1.set_ylim(0, 1)
        
        # 2. Perda geral
        ax2.bar(['Treino', 'Validação', 'Teste'],
               [results.get('train_loss', 0), results.get('val_loss', 0), results.get('test_loss', 0)],
               color=['blue', 'orange', 'green'], alpha=0.7)
        ax2.set_title('Perda Geral', fontweight='bold')
        ax2.set_ylabel('Perda')
        
        # 3. Top-5 classes com maior acurácia
        if 'class_accuracy' in results:
            class_acc = results['class_accuracy']
            top_classes = sorted(class_acc.items(), key=lambda x: x[1], reverse=True)[:5]
            classes, accs = zip(*top_classes)
            class_labels = [self.class_names[c] for c in classes]
            
            ax3.bar(class_labels, accs, color='lightblue', alpha=0.7)
            ax3.set_title('Top-5 Classes (Acurácia)', fontweight='bold')
            ax3.set_ylabel('Acurácia')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Top-5 classes com menor acurácia
        if 'class_accuracy' in results:
            bottom_classes = sorted(class_acc.items(), key=lambda x: x[1])[:5]
            classes, accs = zip(*bottom_classes)
            class_labels = [self.class_names[c] for c in classes]
            
            ax4.bar(class_labels, accs, color='lightcoral', alpha=0.7)
            ax4.set_title('Bottom-5 Classes (Acurácia)', fontweight='bold')
            ax4.set_ylabel('Acurácia')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.suptitle('Resumo dos Resultados do Modelo', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def create_visualization_report(results: Dict[str, Any], 
                               output_dir: str = "results/plots") -> None:
    """
    Cria um relatório completo de visualização
    
    Args:
        results (Dict[str, Any]): Resultados do modelo
        output_dir (str): Diretório de saída
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    visualizer = LibrasVisualizer()
    
    # 1. Histórico de treinamento
    if 'history' in results:
        visualizer.plot_training_history(
            results['history'], 
            save_path=str(output_path / "training_history.png")
        )
    
    # 2. Matriz de confusão
    if 'confusion_matrix' in results:
        visualizer.plot_confusion_matrix(
            results['confusion_matrix'],
            save_path=str(output_path / "confusion_matrix.png")
        )
    
    # 3. Acurácia por classe
    if 'class_accuracy' in results:
        visualizer.plot_class_accuracy(
            results['class_accuracy'],
            save_path=str(output_path / "class_accuracy.png")
        )
    
    # 4. Resumo geral
    visualizer.create_summary_plot(
        results,
        save_path=str(output_path / "summary.png")
    )
    
    print(f"✅ Relatório de visualização salvo em: {output_path}")


if __name__ == "__main__":
    # Exemplo de uso
    print("🧪 Testando visualizador...")
    
    # Criar dados sintéticos
    n_classes = 24
    n_samples = 100
    
    # Simular contagem de classes
    class_counts = {i: np.random.randint(50, 200) for i in range(n_classes)}
    
    # Simular imagens
    images = np.random.randint(0, 256, (n_samples, 28, 28), dtype=np.uint8)
    labels = np.random.randint(0, n_classes, n_samples)
    
    # Testar visualizador
    visualizer = LibrasVisualizer()
    
    # Plotar distribuição de classes
    visualizer.plot_class_distribution(class_counts)
    
    # Plotar amostras
    visualizer.plot_sample_images(images, labels, n_samples=8)
    
    print("✅ Teste concluído com sucesso!")
