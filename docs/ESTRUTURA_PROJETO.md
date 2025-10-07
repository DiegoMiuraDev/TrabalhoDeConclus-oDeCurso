# Estrutura do Projeto - Reconhecimento de Libras

## 📁 Organização dos Arquivos

```
tcc/
├── README.md                    # Documentação principal do projeto
├── requirements.txt             # Dependências Python
├── configs/                     # Configurações
│   └── config.py               # Configurações centralizadas
├── src/                        # Código fonte principal
│   ├── data/                   # Manipulação de dados
│   │   ├── dataset_loader.py   # Carregamento do dataset
│   │   └── preprocessing.py    # Pré-processamento de imagens
│   ├── utils/                  # Utilitários
│   │   ├── kaggle_setup.py     # Configuração do Kaggle
│   │   └── helpers.py          # Funções auxiliares
│   └── visualization/          # Visualização
│       └── plots.py            # Gráficos e plots
├── scripts/                    # Scripts executáveis
│   ├── train_model.py          # Treinar modelo
│   └── real_time_demo.py       # Demo em tempo real
├── notebooks/                  # Jupyter notebooks
│   └── 01_data_exploration.ipynb  # Fase 1: Exploração
└── docs/                       # Documentação
    └── ESTRUTURA_PROJETO.md    # Este arquivo
```

## 🎯 Como Usar Cada Módulo

### 1. Configurações (`configs/`)
- **`config.py`**: Todas as configurações centralizadas
  - Parâmetros do dataset
  - Configurações do modelo
  - Parâmetros de treinamento
  - Configurações de visualização

### 2. Manipulação de Dados (`src/data/`)
- **`dataset_loader.py`**: Classe `LibrasDatasetLoader`
  - Carregamento do dataset CSV
  - Exploração e análise dos dados
  - Preparação para treinamento
  
- **`preprocessing.py`**: Classe `ImagePreprocessor`
  - Conversão grayscale → RGB
  - Redimensionamento para MobileNetV2
  - Normalização de imagens
  - Data augmentation

### 3. Utilitários (`src/utils/`)
- **`kaggle_setup.py`**: Configuração do Kaggle API
  - Download de datasets
  - Configuração de credenciais
  
- **`helpers.py`**: Funções auxiliares
  - Verificação de GPU
  - Informações do sistema
  - Funções de imagem
  - Estatísticas

### 4. Visualização (`src/visualization/`)
- **`plots.py`**: Classe `LibrasVisualizer`
  - Distribuição de classes
  - Amostras de imagens
  - Histórico de treinamento
  - Matriz de confusão
  - Acurácia por classe

### 5. Scripts Executáveis (`scripts/`)
- **`train_model.py`**: Treinamento completo
  - Carregamento de dados
  - Pré-processamento
  - Treinamento do modelo
  - Avaliação e salvamento
  
- **`real_time_demo.py`**: Demo em tempo real
  - Carregamento do modelo
  - Captura de webcam
  - Predição em tempo real
  - Interface visual

### 6. Notebooks (`notebooks/`)
- **`01_data_exploration.ipynb`**: Fase 1
  - Análise exploratória
  - Visualização dos dados
  - Pré-processamento
  - Preparação para treinamento

## 🚀 Fluxo de Trabalho

### Fase 1: Análise de Dados
1. Abrir `notebooks/01_data_exploration.ipynb`
2. Configurar Kaggle API
3. Baixar e explorar dataset
4. Visualizar amostras
5. Preparar dados

### Fase 2: Treinamento
1. Executar `scripts/train_model.py`
2. Modelo será treinado automaticamente
3. Resultados salvos em `models/` e `results/`

### Fase 3: Demo em Tempo Real
1. Executar `scripts/real_time_demo.py`
2. Usar webcam para testar
3. Ver predições em tempo real

## 📋 Vantagens da Nova Estrutura

### ✅ Organização
- **Separação clara** de responsabilidades
- **Módulos específicos** para cada função
- **Configurações centralizadas**

### ✅ Reutilização
- **Funções modulares** podem ser reutilizadas
- **Classes bem definidas** com métodos específicos
- **Configurações fáceis de modificar**

### ✅ Manutenção
- **Código mais limpo** e organizado
- **Fácil de depurar** problemas específicos
- **Documentação clara** de cada módulo

### ✅ Escalabilidade
- **Fácil adicionar** novas funcionalidades
- **Estrutura preparada** para crescimento
- **Padrões consistentes** em todo o projeto

## 🔧 Como Adicionar Novas Funcionalidades

### 1. Novos Módulos
- Criar arquivo em `src/` com classe/funções
- Adicionar `__init__.py` se necessário
- Importar em outros módulos

### 2. Novas Configurações
- Adicionar em `configs/config.py`
- Usar em outros módulos

### 3. Novos Scripts
- Criar em `scripts/`
- Seguir padrão dos existentes
- Adicionar documentação

### 4. Novos Notebooks
- Criar em `notebooks/`
- Seguir estrutura do existente
- Usar módulos do `src/`

## 📚 Exemplos de Uso

### Carregar Dataset
```python
from src.data.dataset_loader import LibrasDatasetLoader

loader = LibrasDatasetLoader("data")
df = loader.load_dataset()
X, y = loader.prepare_data()
```

### Pré-processar Imagens
```python
from src.data.preprocessing import ImagePreprocessor

preprocessor = ImagePreprocessor()
processed = preprocessor.preprocess_for_mobilenet(images)
```

### Visualizar Dados
```python
from src.visualization.plots import LibrasVisualizer

visualizer = LibrasVisualizer()
visualizer.plot_class_distribution(class_counts)
```

### Treinar Modelo
```bash
python scripts/train_model.py
```

### Demo em Tempo Real
```bash
python scripts/real_time_demo.py
```

---

**💡 Dica:** Esta estrutura torna o projeto mais profissional, organizado e fácil de manter!
