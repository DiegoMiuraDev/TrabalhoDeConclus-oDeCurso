# 📚 Resumo do Projeto: Reconhecimento Automático de Libras com IA

## 🎯 Visão Geral

Este projeto desenvolve um **sistema de reconhecimento automático de sinais estáticos da Língua Brasileira de Sinais (Libras)** utilizando Inteligência Artificial e Visão Computacional. O objetivo é criar uma aplicação que reconheça, em tempo real através de uma webcam, os sinais correspondentes às **24 letras do alfabeto de Libras**.

## 🏗️ Arquitetura do Projeto

### Stack Tecnológico
- **Linguagem**: Python 3.8+
- **Framework de IA**: TensorFlow/Keras
- **Modelo Base**: MobileNetV2 (Transfer Learning)
- **Visão Computacional**: OpenCV
- **Dataset**: Libras MNIST (Kaggle)
- **Ambiente**: Google Colab (GPU gratuita)

### Estrutura Modular
```
tcc/
├── 📁 configs/          # Configurações centralizadas
├── 📁 src/              # Código fonte modular
│   ├── data/           # Manipulação de dados
│   ├── utils/          # Utilitários
│   └── visualization/  # Visualizações
├── 📁 scripts/         # Scripts executáveis
├── 📁 notebooks/       # Jupyter notebooks
└── 📁 docs/            # Documentação
```

## 📊 Dataset: Libras MNIST

- **Fonte**: Kaggle (https://www.kaggle.com/datasets/datamoon/libras-mnist)
- **Classes**: 24 letras (A-X) do alfabeto de Libras
- **Amostras**: ~2.000 por classe (total ~48.000)
- **Formato Original**: Imagens 28x28 em escala de cinza
- **Formato Final**: Imagens 224x224 RGB (para MobileNetV2)

## 🚀 Fases do Projeto

### ✅ Fase 1: Análise e Exploração dos Dados
**Status**: COMPLETA

**Objetivos**:
- Carregar e explorar o dataset Libras MNIST
- Visualizar amostras de cada classe
- Preparar dados para o modelo
- Pré-processar para MobileNetV2

**Arquivos**:
- `notebooks/01_data_exploration.ipynb`
- `src/data/dataset_loader.py`
- `src/data/preprocessing.py`
- `src/visualization/plots.py`

**Resultados**:
- Dataset carregado e analisado
- 24 classes identificadas e visualizadas
- Dados normalizados e divididos (70% treino, 10% validação, 20% teste)
- Imagens convertidas para formato MobileNetV2 (224x224 RGB)

### 🔄 Fase 2: Treinamento do Modelo
**Status**: PRONTA PARA EXECUÇÃO

**Objetivos**:
- Implementar modelo MobileNetV2 com Transfer Learning
- Treinar o modelo com os dados preparados
- Avaliar performance e gerar métricas
- Salvar modelo treinado

**Arquivos**:
- `scripts/train_model.py`
- `src/models/mobilenet_model.py` (a criar)
- `src/models/training.py` (a criar)

**Comando**:
```bash
python scripts/train_model.py
```

### 🎥 Fase 3: Aplicação em Tempo Real
**Status**: PRONTA PARA EXECUÇÃO

**Objetivos**:
- Carregar modelo treinado
- Implementar captura de webcam
- Fazer predições em tempo real
- Exibir resultados na tela

**Arquivos**:
- `scripts/real_time_demo.py`

**Comando**:
```bash
python scripts/real_time_demo.py
```

## 🛠️ Módulos Principais

### 1. Carregamento de Dados (`src/data/dataset_loader.py`)
```python
class LibrasDatasetLoader:
    - load_dataset()      # Carrega CSV do Kaggle
    - explore_dataset()   # Analisa estrutura dos dados
    - prepare_data()      # Separa features e labels
    - reshape_images()    # Converte para formato de imagem
    - normalize_images()  # Normaliza pixels
```

### 2. Pré-processamento (`src/data/preprocessing.py`)
```python
class ImagePreprocessor:
    - grayscale_to_rgb()     # Converte para RGB
    - resize_images()        # Redimensiona para 224x224
    - normalize_images()     # Normaliza para [0,1]
    - preprocess_for_mobilenet()  # Pipeline completo
    - prepare_training_data()     # Divisão train/val/test
```

### 3. Visualização (`src/visualization/plots.py`)
```python
class LibrasVisualizer:
    - plot_class_distribution()    # Distribuição das classes
    - plot_sample_images()         # Amostras do dataset
    - plot_training_history()      # Histórico de treinamento
    - plot_confusion_matrix()      # Matriz de confusão
    - plot_class_accuracy()        # Acurácia por classe
```

### 4. Utilitários (`src/utils/`)
```python
# helpers.py
- check_gpu_availability()  # Verifica GPU
- print_system_info()       # Informações do sistema
- resize_image()            # Redimensiona imagens
- normalize_image()         # Normaliza imagens

# kaggle_setup.py
- setup_kaggle_api()        # Configura Kaggle
- download_dataset()        # Baixa dataset
- extract_dataset()         # Extrai arquivos
```

## 📈 Resultados Esperados

### Métricas de Performance
- **Acurácia**: >90% no conjunto de teste
- **Tempo de inferência**: <100ms por frame
- **Classes reconhecidas**: 24 letras do alfabeto de Libras

### Saídas do Sistema
- **Modelo treinado**: `models/libras_model.h5`
- **Métricas**: `results/metrics.npy`
- **Visualizações**: `results/plots/`
- **Logs**: `logs/libras_recognition.log`

## 🎯 Como Usar o Projeto

### 1. Configuração Inicial
```bash
# Clone o repositório
git clone <seu-repositorio>
cd tcc

# Instale dependências
pip install -r requirements.txt
```

### 2. Configuração do Kaggle
1. Crie conta no Kaggle
2. Baixe `kaggle.json`
3. Execute: `python src/utils/kaggle_setup.py`

### 3. Execução das Fases

#### Fase 1: Exploração
```bash
# Abrir no Google Colab
jupyter notebook notebooks/01_data_exploration.ipynb
```

#### Fase 2: Treinamento
```bash
python scripts/train_model.py
```

#### Fase 3: Demo
```bash
python scripts/real_time_demo.py
```

## 🔧 Configurações Principais

### Dataset
```python
DATASET_CONFIG = {
    "n_classes": 24,
    "original_size": (28, 28),
    "target_size": (224, 224),
    "channels": 3,
    "batch_size": 32,
    "test_split": 0.2,
    "validation_split": 0.1
}
```

### Modelo
```python
MODEL_CONFIG = {
    "base_model": "MobileNetV2",
    "input_shape": (224, 224, 3),
    "include_top": False,
    "weights": "imagenet",
    "dropout_rate": 0.5,
    "dense_units": 128
}
```

### Treinamento
```python
TRAINING_CONFIG = {
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "loss": "categorical_crossentropy"
}
```

## 📚 Documentação

- **`README.md`**: Visão geral do projeto
- **`docs/FASE1_GUIA.md`**: Guia detalhado da Fase 1
- **`docs/ESTRUTURA_PROJETO.md`**: Estrutura dos arquivos
- **`requirements.txt`**: Dependências Python

## 🎉 Vantagens da Estrutura

### ✅ Organização
- **Separação clara** de responsabilidades
- **Módulos específicos** para cada função
- **Configurações centralizadas**

### ✅ Reutilização
- **Funções modulares** reutilizáveis
- **Classes bem definidas** com métodos específicos
- **Configurações fáceis de modificar**

### ✅ Manutenção
- **Código limpo** e organizado
- **Fácil depuração** de problemas
- **Documentação clara** de cada módulo

### ✅ Escalabilidade
- **Fácil adicionar** novas funcionalidades
- **Estrutura preparada** para crescimento
- **Padrões consistentes** em todo o projeto

## 🚀 Próximos Passos

1. **Executar Fase 1** no Google Colab
2. **Implementar Fase 2** (treinamento)
3. **Desenvolver Fase 3** (tempo real)
4. **Otimizar performance** do modelo
5. **Adicionar funcionalidades** extras

---

**🎯 Objetivo Final**: Criar uma aplicação funcional que reconheça sinais de Libras em tempo real, contribuindo para a inclusão e acessibilidade da comunidade surda.
