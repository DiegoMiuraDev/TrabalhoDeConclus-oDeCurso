# 🚀 Guia Rápido: Como Testar o Projeto

## 📋 Status Atual do Projeto

### ✅ COMPLETO (Código pronto)
- Fase 1: Exploração de dados
- Fase 2: Módulos de treinamento
- Documentação completa

### ⏳ FALTA EXECUTAR
- Instalar dependências
- Baixar dataset do Kaggle
- Treinar o modelo
- Fase 3: Aplicação em tempo real

---

## 🛠️ SETUP INICIAL (Primeira vez)

### **Passo 1: Instalar Dependências**

```bash
cd /root/tcc

# Instalar todas as bibliotecas necessárias
pip install -r requirements.txt

# OU instalar manualmente as principais:
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python jupyter
```

⏱️ **Tempo:** ~10-15 minutos

### **Passo 2: Configurar Kaggle (Para baixar dataset)**

```bash
# 1. Criar conta no Kaggle (se não tiver): https://www.kaggle.com

# 2. Baixar suas credenciais:
#    - Ir em: kaggle.com → Sua foto → Account → API → Create New Token
#    - Isso baixa o arquivo: kaggle.json

# 3. Configurar no Linux:
mkdir -p ~/.kaggle
mv /caminho/para/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 4. Instalar Kaggle CLI:
pip install kaggle
```

⏱️ **Tempo:** ~5 minutos

---

## 🧪 TESTES RÁPIDOS

### **Teste 1: Verificar Instalação** ⚡

```bash
cd /root/tcc
python3 test_fase2.py
```

**O que verifica:**
- ✅ Bibliotecas instaladas
- ✅ Módulos do projeto funcionando
- ✅ GPU disponível (ou CPU)
- ✅ Modelo pode ser criado

⏱️ **Tempo:** ~1-2 minutos

**Resultado esperado:**
```
✅ TODOS OS TESTES PASSARAM!
🚀 Sistema pronto para treinamento!
```

---

### **Teste 2: Explorar Dados (Fase 1)** 📊

```bash
# Opção A: Jupyter Notebook (Interativo)
cd /root/tcc
jupyter notebook notebooks/01_data_exploration_simples.ipynb

# Opção B: Google Colab
# 1. Upload o notebook para Google Drive
# 2. Abrir com Google Colab
# 3. Executar células
```

**O que faz:**
- Baixa dataset do Kaggle
- Carrega as imagens
- Mostra estatísticas
- Visualiza as 24 letras

⏱️ **Tempo:** ~5-10 minutos

**Resultado esperado:**
- 📊 Gráficos das 24 letras de Libras
- 📈 Distribuição das classes
- 🖼️ Amostras de cada letra

---

### **Teste 3: Criar Modelo (Sem Treinar)** 🏗️

```bash
cd /root/tcc
python3 -c "
import sys
sys.path.append('src')
from models.mobilenet_model import create_mobilenet_model

print('Criando modelo...')
model = create_mobilenet_model()
print(f'✅ Modelo criado: {model.count_params():,} parâmetros')
print('O modelo está pronto para ser treinado!')
"
```

⏱️ **Tempo:** ~30 segundos

**Resultado esperado:**
```
✅ Modelo criado: 2,345,678 parâmetros
O modelo está pronto para ser treinado!
```

---

## 🎯 EXECUTAR FASE 2 (Treinamento Real)

### **Opção A: Script Automático** 🤖

```bash
cd /root/tcc
python3 scripts/train_model.py
```

**O que faz:**
- Carrega dataset
- Pré-processa imagens
- Treina modelo por 50 épocas
- Gera métricas e gráficos
- Salva modelo treinado

⏱️ **Tempo:** 
- Com GPU: 30-45 minutos
- Sem GPU: 2-3 horas

**Arquivos gerados:**
```
models/libras_model.h5          # Modelo treinado
results/training_history.png    # Gráfico de treino
results/confusion_matrix.png    # Matriz de confusão
results/metrics.npy             # Métricas salvas
```

---

### **Opção B: Notebook Interativo** 📓

```bash
cd /root/tcc
jupyter notebook notebooks/02_model_training.ipynb
```

**Vantagens:**
- Vê cada etapa em tempo real
- Pode pausar/continuar
- Visualizações inline
- Melhor para aprendizado

⏱️ **Tempo:** Mesmo que Opção A

---

### **Opção C: Google Colab (RECOMENDADO)** ☁️

**Por que Colab?**
- ✅ GPU gratuita (15x mais rápido)
- ✅ Sem instalar nada no PC
- ✅ Jupyter já configurado
- ✅ Bibliotecas pré-instaladas

**Como usar:**
1. Abrir: https://colab.research.google.com
2. File → Upload notebook → `02_model_training.ipynb`
3. Runtime → Change runtime type → **GPU**
4. Run All

⏱️ **Tempo:** ~30 minutos com GPU

---

## 📊 VERIFICAR RESULTADOS

Após o treinamento, verifique:

```bash
cd /root/tcc

# Ver arquivos gerados
ls -lh models/
ls -lh results/

# Carregar e usar o modelo
python3 -c "
from tensorflow import keras
model = keras.models.load_model('models/libras_model.h5')
print('✅ Modelo carregado com sucesso!')
print(f'Pronto para reconhecer {model.output_shape[-1]} letras!')
"
```

---

## 🎨 VISUALIZAR RESULTADOS

```bash
cd /root/tcc/results

# Ver imagens geradas:
# - training_history.png     (gráficos de treino)
# - confusion_matrix.png     (matriz de confusão)
# - class_accuracy.png       (acurácia por letra)
# - prediction_samples.png   (exemplos)

# No Linux, use:
xdg-open training_history.png

# Ou copie para visualizar no Windows/navegador
```

---

## 🚀 PRÓXIMA FASE (Após treinar)

### **Fase 3: Aplicação em Tempo Real**

```bash
# Usar o modelo com webcam
python3 scripts/real_time_demo.py
```

**O que faz:**
- Abre sua webcam
- Você faz sinais de Libras
- Modelo reconhece em tempo real
- Mostra resultado na tela

---

## 🆘 PROBLEMAS COMUNS

### **Erro: "No module named 'tensorflow'"**
```bash
pip install tensorflow
```

### **Erro: "No module named 'cv2'"**
```bash
pip install opencv-python
```

### **Erro: "kaggle.json not found"**
- Configure as credenciais do Kaggle (Passo 2 acima)

### **Erro: "Out of Memory"**
- Reduza batch_size em `configs/config.py`
- Use Google Colab com GPU

### **Treinamento muito lento**
- Use GPU (Google Colab)
- Reduza número de épocas
- Reduza tamanho do dataset

---

## 📚 ESTRUTURA DOS TESTES

```
🧪 TESTES DISPONÍVEIS
│
├── 🔧 test_fase2.py
│   └── Verifica se tudo está instalado
│   └── Tempo: 1-2 min
│
├── 📊 Fase 1: Exploração
│   └── notebooks/01_data_exploration_simples.ipynb
│   └── Tempo: 5-10 min
│   └── Resultado: Gráficos e estatísticas
│
├── 🎯 Fase 2: Treinamento
│   ├── scripts/train_model.py (automático)
│   └── notebooks/02_model_training.ipynb (interativo)
│   └── Tempo: 30-180 min
│   └── Resultado: Modelo treinado (.h5)
│
└── 🎥 Fase 3: Tempo Real
    └── scripts/real_time_demo.py
    └── Tempo: Instantâneo
    └── Resultado: Reconhecimento ao vivo
```

---

## ✅ CHECKLIST DE TESTE

Use este checklist para testar o projeto:

### **Setup Inicial**
- [ ] Python 3.8+ instalado
- [ ] pip atualizado
- [ ] Dependências instaladas (`requirements.txt`)
- [ ] Conta Kaggle criada
- [ ] `kaggle.json` configurado

### **Teste Básico**
- [ ] `test_fase2.py` passou todos os testes
- [ ] Modelo pode ser criado
- [ ] GPU detectada (opcional)

### **Fase 1**
- [ ] Dataset baixado do Kaggle
- [ ] Imagens carregadas (48.000)
- [ ] Visualizações funcionam
- [ ] 24 classes identificadas

### **Fase 2**
- [ ] Treinamento iniciado sem erros
- [ ] Modelo converge (acurácia aumenta)
- [ ] Modelo salvo em `models/`
- [ ] Gráficos gerados em `results/`
- [ ] Acurácia final >85%

### **Fase 3**
- [ ] Webcam funciona
- [ ] Modelo carrega
- [ ] Reconhecimento em tempo real
- [ ] Resultados mostrados na tela

---

## 🎯 RESUMO RÁPIDO

**Para testar AGORA (sem treinar):**
```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Testar módulos
python3 test_fase2.py

# 3. Ver os dados
jupyter notebook notebooks/01_data_exploration_simples.ipynb
```

**Para treinar o modelo:**
```bash
# Opção mais rápida: Google Colab com GPU
# Upload: notebooks/02_model_training.ipynb
# Ativar GPU e Run All
```

**Para usar o modelo (após treinar):**
```bash
python3 scripts/real_time_demo.py
```

---

## 💡 DICAS FINAIS

1. **Use Google Colab** para treinamento (GPU grátis)
2. **Comece pela Fase 1** (explorar dados é rápido)
3. **Não pule etapas** (teste cada fase antes de avançar)
4. **Monitore o treinamento** (veja acurácia aumentando)
5. **Salve o modelo** (não perder o trabalho)

---

## 📞 SUPORTE

Se encontrar problemas:
1. Verifique o README.md
2. Leia os guias em `docs/`
3. Consulte `FASE2_GUIA.md` para detalhes

---

**Última atualização:** 2025-10-11  
**Status:** ✅ Pronto para teste  
**Próximo passo:** Instalar dependências e executar `test_fase2.py`

