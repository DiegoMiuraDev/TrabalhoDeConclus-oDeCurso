# Projeto TCC: Reconhecimento Automático de Libras com IA

## 📋 Visão Geral

Este projeto desenvolve um sistema de reconhecimento automático de sinais estáticos da Língua Brasileira de Sinais (Libras) utilizando Inteligência Artificial e Visão Computacional.

### 🎯 Objetivo
Criar uma aplicação capaz de reconhecer, em tempo real através de uma webcam, os sinais correspondentes às letras do alfabeto de Libras.

## 🏗️ Estrutura do Projeto

```
tcc/
├── README.md                 # Este arquivo
├── requirements.txt          # Dependências do projeto
├── configs/                  # Arquivos de configuração
│   └── config.py            # Configurações gerais
├── src/                     # Código fonte principal
│   ├── data/               # Módulos para manipulação de dados
│   │   ├── __init__.py
│   │   ├── dataset_loader.py    # Carregamento do dataset
│   │   └── preprocessing.py     # Pré-processamento de imagens
│   ├── models/             # Modelos de IA
│   │   ├── __init__.py
│   │   ├── mobilenet_model.py   # Arquitetura MobileNetV2
│   │   └── training.py          # Treinamento do modelo
│   ├── utils/              # Utilitários
│   │   ├── __init__.py
│   │   ├── kaggle_setup.py      # Configuração do Kaggle
│   │   └── helpers.py           # Funções auxiliares
│   └── visualization/      # Visualização de dados
│       ├── __init__.py
│       └── plots.py             # Gráficos e visualizações
├── notebooks/              # Jupyter notebooks
│   ├── 01_data_exploration.ipynb    # Fase 1: Exploração de dados
│   ├── 02_model_training.ipynb      # Fase 2: Treinamento
│   └── 03_evaluation.ipynb          # Fase 3: Avaliação
├── scripts/                # Scripts executáveis
│   ├── train_model.py      # Treinar modelo
│   ├── evaluate_model.py   # Avaliar modelo
│   └── real_time_demo.py   # Demo em tempo real
└── docs/                   # Documentação
    ├── dataset_info.md     # Informações sobre o dataset
    └── model_architecture.md # Arquitetura do modelo
```

## 🚀 Tecnologias Utilizadas

- **Linguagem:** Python 3.8+
- **Framework de IA:** TensorFlow/Keras
- **Modelo Base:** MobileNetV2 (Transfer Learning)
- **Visão Computacional:** OpenCV
- **Dataset:** Libras MNIST (Kaggle)
- **Ambiente:** Google Colab (GPU gratuita)

## 📊 Dataset

- **Nome:** Libras MNIST
- **Fonte:** Kaggle (https://www.kaggle.com/datasets/datamoon/libras-mnist)
- **Classes:** 24 letras do alfabeto de Libras
- **Formato:** Imagens 28x28 pixels em escala de cinza
- **Tamanho:** ~2.000 amostras por classe

## 🎯 Fases do Projeto

### Fase 1: Análise e Exploração dos Dados
- [x] Configuração do ambiente
- [x] Download e carregamento do dataset
- [x] Análise exploratória
- [x] Visualização de amostras

### Fase 2: Pré-processamento e Treinamento
- [ ] Adaptação para RGB (3 canais)
- [ ] Redimensionamento para 224x224
- [ ] Implementação do modelo MobileNetV2
- [ ] Treinamento com Transfer Learning

### Fase 3: Avaliação e Aplicação
- [ ] Avaliação do modelo
- [ ] Matriz de confusão
- [ ] Aplicação em tempo real com webcam

## 🛠️ Como Usar

### 1. Configuração Inicial
```bash
# Clone o repositório
git clone <seu-repositorio>
cd tcc

# Instale as dependências
pip install -r requirements.txt
```

### 2. Configuração do Kaggle
1. Crie uma conta no Kaggle
2. Baixe seu arquivo `kaggle.json`
3. Execute: `python src/utils/kaggle_setup.py`

### 3. Execução das Fases

#### Fase 1: Exploração de Dados
```bash
# Abra o notebook no Google Colab
jupyter notebook notebooks/01_data_exploration.ipynb
```

#### Fase 2: Treinamento
```bash
# Treinar o modelo
python scripts/train_model.py
```

#### Fase 3: Demo em Tempo Real
```bash
# Executar aplicação com webcam
python scripts/real_time_demo.py
```

## 📈 Resultados Esperados

- **Acurácia:** >90% no conjunto de teste
- **Tempo de inferência:** <100ms por frame
- **Classes reconhecidas:** 24 letras do alfabeto de Libras

## 🤝 Contribuição

Este é um projeto de TCC. Para sugestões ou melhorias, entre em contato.

## 📄 Licença

Este projeto é para fins educacionais e de pesquisa.

---

**Desenvolvido por:** [Seu Nome]  
**Orientador:** [Nome do Orientador]  
**Instituição:** [Nome da Instituição]  
**Ano:** 2024
