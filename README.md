# Projeto TCC: Reconhecimento Automático de Libras com IA

## 📋 Visão Geral

Este projeto desenvolve um **sistema de reconhecimento automático de sinais estáticos da Língua Brasileira de Sinais (Libras)** utilizando Inteligência Artificial e Visão Computacional.

O foco atual está no reconhecimento de um subconjunto de sinais (principalmente **vogais A, E, I, O, U**), com infraestrutura preparada para escalar para as **24 letras do alfabeto de Libras**.

### 🎯 Objetivo Geral
Criar uma aplicação capaz de reconhecer, em tempo real através de uma webcam, os sinais manuais correspondentes às letras do alfabeto de Libras, contribuindo para inclusão e acessibilidade da comunidade surda.

---

## 🏗️ Estrutura do Projeto

Estrutura real do repositório (atualizada):

```
tcc/
├── README.md                      # Este arquivo (visão geral)
├── RESUMO_PROJETO.md             # Resumo técnico detalhado do TCC
├── requirements.txt              # Dependências Python
├── configs/
│   └── config.py                 # Configurações centralizadas (dataset, modelo, treino, realtime)
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset_loader.py     # Carregamento e organização do dataset
│   │   └── preprocessing.py      # Pré-processamento e preparação p/ MobileNetV2
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── helpers.py            # Funções auxiliares (GPU, infos de sistema, imagens)
│   │   └── kaggle_setup.py       # Automação da configuração/download do Kaggle
│   └── visualization/
│       ├── __init__.py
│       └── plots.py              # Gráficos (distribuição, histórico, matriz de confusão etc.)
├── scripts/
│   ├── train_model.py            # Pipeline completo de treinamento + avaliação
│   └── real_time_demo.py         # Aplicação de reconhecimento em tempo real via webcam
├── notebooks/
│   ├── 01_data_exploration.ipynb # Exploração detalhada do dataset
│   └── 01_data_exploration_simples.ipynb # Versão simplificada da exploração
├── libras_recognition_phase1.ipynb # Notebook geral da fase 1
├── docs/
│   ├── ESTRUTURA_PROJETO.md      # Detalhes da organização de módulos
│   └── FASE1_GUIA.md             # Guia completo da fase 1 (passo a passo)
├── models/
│   ├── libras_brasileiro_best.h5 # Modelo treinado (melhor versão atual)
│   └── libras_classes.bak        # Backup/classes do modelo
├── dataset/
│   ├── keras_model.h5            # Modelo em formato Keras (versão exportada)
│   ├── model_unquant.tflite      # Modelo convertido para TFLite (uso embarcado)
│   └── test_images/              # Conjunto de teste com imagens reais (A, E, I, O, U)
├── results/
│   ├── matriz_confusao.npy       # Matriz de confusão gerada pelo treinamento
│   ├── metricas_precisao.csv     # Métricas em CSV
│   └── metricas_precisao.md      # Resumo das métricas por classe
├── logs/                         # Logs de execução/treinamento
└── venv*, venv312_mp/            # Ambientes virtuais (não necessários para uso em Colab)
```

Para detalhes finos de cada módulo, consulte também `docs/ESTRUTURA_PROJETO.md`.

---

## 🚀 Tecnologias Utilizadas

- **Linguagem:** Python 3.8+
- **Framework de IA:** TensorFlow / Keras
- **Modelo Base:** MobileNetV2 (Transfer Learning)
- **Visão Computacional:** OpenCV
- **Análise de Dados:** NumPy, Pandas
- **Visualização:** Matplotlib, Seaborn, Plotly
- **Dataset Base:** Libras MNIST (Kaggle)
- **Ambiente de Desenvolvimento:** Google Colab (GPU) e ambiente local com virtualenv

---

## 📊 Dataset

- **Nome:** Libras MNIST  
- **Fonte:** Kaggle (`https://www.kaggle.com/datasets/datamoon/libras-mnist`)  
- **Classes (objetivo final):** 24 letras (A–X) do alfabeto de Libras  
- **Formato original:** Imagens 28×28 em escala de cinza  
- **Formato para o modelo:** Imagens 224×224 RGB (3 canais), compatíveis com MobileNetV2  
- **Configuração do dataset no código:** ver `DATASET_CONFIG` em `configs/config.py`.

O diretório `dataset/test_images/` contém um **conjunto de teste prático** com imagens reais das letras **A, E, I, O, U**, usado na avaliação manual e na validação da demo em tempo real.

---

## 🎯 Fases do Projeto

### ✅ Fase 1 – Análise e Exploração dos Dados
- Configuração do ambiente (Colab + GPU / ambiente local)
- Download e carregamento do dataset Libras MNIST via Kaggle
- Análise exploratória e visualização de amostras
- Pré-processamento das imagens para MobileNetV2
- Divisão em treino / validação / teste

**Documentação e materiais:**
- Notebook: `notebooks/01_data_exploration.ipynb`
- Guia: `docs/FASE1_GUIA.md`
- Estrutura: `docs/ESTRUTURA_PROJETO.md`

---

### 🔄 Fase 2 – Pré-processamento, Treinamento e Avaliação

Status: **implementada e com resultados iniciais gerados**.

Passos principais (automatizados em `scripts/train_model.py`):
- Carregamento e preparação dos dados (`src/data/dataset_loader.py` e `preprocessing.py`)
- Construção do modelo MobileNetV2 com Transfer Learning (`create_mobilenet_model`)
- Treinamento com callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
- Avaliação em treino/validação/teste
- Geração de histórico de treinamento e matriz de confusão
- Salvamento de métricas em `results/`

**Resumo das métricas atuais (vogais A, E, I, O, U – ver `results/metricas_precisao.md`):**
- **Acurácia geral:** ~50,5% no conjunto de teste de 5 classes  
- **Precisão por classe (exemplos):**
  - O: precisão ≈ 0,96, F1 ≈ 0,94
  - E: precisão ≈ 0,83
  - U: recall alto (≈ 0,86), mas precisão moderada

Esses resultados mostram desempenho já sólido para algumas classes (como **O**), porém com espaço para melhorar a separação entre todas as vogais e, futuramente, escalar para as 24 letras.

---

### 🎥 Fase 3 – Aplicação em Tempo Real com Webcam

Status: **script implementado e pronto para uso com modelo treinado**.

Funcionalidades principais (`scripts/real_time_demo.py`):
- Captura de vídeo em tempo real via OpenCV
- Pré-processamento do frame (grayscale → resize 224×224 → RGB → normalização)
- Predição com o modelo treinado (`models/libras_model.h5` ou similares)
- Exibição da letra prevista, confiança e status (ALTO/BAIXO) sobre o vídeo
- Controle via teclado: `q` para sair, `s` para salvar frames

Configurações como índice da câmera, resolução, limiar de confiança e intervalo de predição são definidas em `REALTIME_CONFIG` (`configs/config.py`).

---

## 🛠️ Como Usar o Projeto

### 1. Clonar o repositório e instalar dependências

```bash
git clone <seu-repositorio>
cd tcc

# (opcional) criar ambiente virtual
python -m venv venv
venv\Scripts\activate  # Windows

# instalar dependências principais e de desenvolvimento
pip install -r requirements.txt
```

> Em Google Colab, normalmente basta copiar os trechos de instalação do `requirements.txt` ou usar as versões já disponíveis no ambiente.

### 2. Configurar Kaggle (para baixar o Libras MNIST)

1. Crie uma conta no Kaggle.  
2. Em *Account* → *Create New API Token*, baixe o arquivo `kaggle.json`.  
3. No ambiente local ou Colab, coloque o `kaggle.json` no local esperado (ver `COLAB_CONFIG` em `configs/config.py`).  
4. Execute o script de setup (se estiver usando o fluxo automatizado):

```bash
python -m src.utils.kaggle_setup
```

### 3. Executar a Fase 1 (exploração de dados)

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

Ou, em Colab, faça upload do repositório/notebook e execute célula a célula conforme `docs/FASE1_GUIA.md`.

### 4. Treinar o modelo (Fase 2)

```bash
python scripts/train_model.py
```

Saídas esperadas:
- Modelo salvo em `models/libras_model.h5` (ou nome equivalente configurado)
- Histórico de treinamento: `results/training_history.npy` e PNG correspondente
- Métricas em `results/metrics.npy` e arquivos auxiliares em `results/`
- Matriz de confusão salva em `results/confusion_matrix.png` e `matriz_confusao.npy`

### 5. Rodar a demo em tempo real (Fase 3)

Certifique-se de que existe um modelo treinado em `models/` compatível com o script.

```bash
python scripts/real_time_demo.py
```

Instruções na tela:
- Posicione a mão em frente à câmera
- Faça o sinal da letra desejada
- Pressione **`q`** para encerrar, **`s`** para salvar um frame

---

## 📈 Resultados e Próximos Passos

### Resultados atuais
- **Acurácia geral (5 classes – A, E, I, O, U):** ~50,5%  
- **Boas métricas individuais** para a letra **O** (F1 ≈ 0,94) e desempenho intermediário para E e U.  
- **Matriz de confusão** e métricas detalhadas disponíveis em `results/metricas_precisao.md` e `matriz_confusao.npy`.

### Próximos passos sugeridos
- Refinar o pré-processamento e *data augmentation* para reduzir confusões entre letras semelhantes.
- Ajustar hiperparâmetros (learning rate, batch_size, epochs) e experimentar *fine-tuning* de camadas da MobileNetV2.
- Ampliar o dataset para incluir todas as 24 letras e novos fundos/iluminações.
- Otimizar o modelo para execução embarcada (uso da versão `.tflite` em dispositivos móveis ou edge).

---

## 📚 Documentação Complementar

- `RESUMO_PROJETO.md` – resumo técnico completo do TCC.
- `docs/ESTRUTURA_PROJETO.md` – detalhes da organização de pastas e módulos.
- `docs/FASE1_GUIA.md` – passo a passo detalhado da fase 1.
- `results/metricas_precisao.md` – métricas atuais por classe.

---

## 🤝 Contribuição

Este é um projeto de **Trabalho de Conclusão de Curso (TCC)**. Sugestões de melhoria, comentários e contribuições acadêmicas são bem-vindas.

Para contato ou dúvidas, utilize os canais definidos no texto do TCC (e-mail institucional, orientador etc.).

---

## 📄 Licença

Este projeto é destinado a **fins educacionais e de pesquisa**, podendo ser reutilizado e adaptado para estudos semelhantes, desde que citada a autoria original.

---

**Desenvolvido por:** [Seu Nome]  
**Orientador:** [Nome do Orientador]  
**Instituição:** [Nome da Instituição]  
**Ano:** 2024
