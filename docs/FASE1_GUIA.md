# Fase 1: Análise e Exploração dos Dados - Guia Completo

## 📋 Resumo da Fase 1

A **Fase 1** é a etapa inicial do projeto de reconhecimento de Libras, onde carregamos, exploramos e preparamos o dataset Libras MNIST para o treinamento do modelo.

### 🎯 Objetivos
- ✅ Carregar o dataset Libras MNIST do Kaggle
- ✅ Explorar e entender a estrutura dos dados
- ✅ Visualizar amostras de cada classe
- ✅ Preparar os dados para o modelo MobileNetV2
- ✅ Dividir os dados em treino/validação/teste

## 🚀 Como Executar a Fase 1

### 1. Preparação do Ambiente

#### No Google Colab:
1. **Abrir o notebook**: `notebooks/01_data_exploration.ipynb`
2. **Executar a primeira célula** para importar bibliotecas
3. **Verificar GPU** (deve aparecer "✅ GPU detectada!")

#### Configuração do Kaggle:
1. **Criar conta no Kaggle**: https://www.kaggle.com
2. **Baixar credenciais**: Account → Create New API Token
3. **Fazer upload** do arquivo `kaggle.json` para o Colab
4. **Descomentar e executar** as linhas de configuração do Kaggle

### 2. Download do Dataset

```python
# Download automático do dataset
!kaggle datasets download -d datamoon/libras-mnist

# Extração automática
with zipfile.ZipFile('libras-mnist.zip', 'r') as zip_ref:
    zip_ref.extractall('.')
```

### 3. Carregamento e Análise

O notebook usa os módulos do projeto para:

```python
# Carregar dataset
loader = LibrasDatasetLoader(".")
df = loader.load_dataset()

# Explorar dados
info = loader.explore_dataset()
print(loader.get_dataset_info())

# Preparar dados
X, y = loader.prepare_data()
X_images = loader.reshape_images()
X_normalized = loader.normalize_images()
```

## 📊 O que a Fase 1 Produz

### 1. Análise do Dataset
- **24 classes** (letras A-X do alfabeto de Libras)
- **~2.000 amostras** por classe
- **Imagens 28x28** em escala de cinza
- **Distribuição balanceada** entre classes

### 2. Visualizações Geradas
- **Distribuição das classes** (gráfico de barras e pizza)
- **Amostras de imagens** de cada classe
- **Múltiplas amostras** da mesma classe
- **Estatísticas** do dataset

### 3. Dados Preparados
- **Imagens normalizadas** para [0, 1]
- **Labels em one-hot encoding**
- **Divisão estratificada**:
  - 70% treino
  - 10% validação  
  - 20% teste

### 4. Pré-processamento para MobileNetV2
- **Conversão** grayscale → RGB
- **Redimensionamento** 28x28 → 224x224
- **Normalização** adequada
- **Formato final**: (N, 224, 224, 3)

## 🔍 Exemplo de Saída

```
📊 Informações do Dataset Libras MNIST:
   Dimensões: (48000, 785)
   Classes: 24
   Amostras por classe: 2000.0 ± 0.0
   Range: 2000 - 2000
   Memória: 287.8 MB

🖼️  Imagens redimensionadas: (48000, 28, 28)
✅ Pixels normalizados para o range [0, 1]
   Range atual: [0.000, 1.000]

📚 Divisão dos dados:
   Treino: 33600 amostras
   Validação: 4800 amostras
   Teste: 9600 amostras
   Proporção: 3.5:1
```

## 📈 Visualizações Incluídas

### 1. Distribuição das Classes
- Gráfico de barras com contagem por classe
- Gráfico de pizza com proporções
- Estatísticas (média, desvio padrão, etc.)

### 2. Amostras de Imagens
- Grid 4x8 com uma amostra de cada classe
- Grid 2x4 com múltiplas amostras da mesma classe
- Títulos com nomes das classes

### 3. Pré-processamento
- Comparação antes/depois do pré-processamento
- Visualização das imagens convertidas para RGB
- Verificação das dimensões finais

## ⚙️ Configurações Importantes

### Dataset
```python
DATASET_CONFIG = {
    "n_classes": 24,
    "original_size": (28, 28),
    "target_size": (224, 224),
    "channels": 3,
    "test_split": 0.2,
    "validation_split": 0.1
}
```

### Classes de Libras
```python
LIBRAS_CLASSES = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F",
    6: "G", 7: "H", 8: "I", 9: "J", 10: "K", 11: "L",
    12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R",
    18: "S", 19: "T", 20: "U", 21: "V", 22: "W", 23: "X"
}
```

## 🚨 Possíveis Problemas e Soluções

### 1. Erro no Kaggle API
```
❌ Arquivo kaggle.json não encontrado
```
**Solução**: Fazer upload do arquivo kaggle.json e descomentar as linhas de configuração

### 2. GPU não detectada
```
⚠️  Nenhuma GPU detectada
```
**Solução**: No Colab, ir em Runtime → Change runtime type → GPU

### 3. Dataset não encontrado
```
❌ Arquivo CSV não encontrado
```
**Solução**: Verificar se o download do Kaggle foi concluído

### 4. Erro de memória
```
❌ Erro de memória
```
**Solução**: Reduzir o batch_size ou usar menos amostras para teste

## 📋 Checklist da Fase 1

- [ ] ✅ Ambiente configurado (Colab + GPU)
- [ ] ✅ Kaggle API configurada
- [ ] ✅ Dataset baixado e extraído
- [ ] ✅ Dados carregados e explorados
- [ ] ✅ Visualizações geradas
- [ ] ✅ Dados preparados para treinamento
- [ ] ✅ Pré-processamento para MobileNetV2
- [ ] ✅ Divisão treino/validação/teste
- [ ] ✅ Notebook salvo com resultados

## 🎯 Próximos Passos (Fase 2)

Após completar a Fase 1, você estará pronto para:

1. **Implementar o modelo MobileNetV2**
2. **Treinar com Transfer Learning**
3. **Avaliar a performance**
4. **Salvar o modelo treinado**

### Comando para Fase 2:
```bash
python scripts/train_model.py
```

## 💡 Dicas Importantes

### Para Melhor Performance:
- **Use GPU** no Google Colab
- **Mantenha o notebook salvo** regularmente
- **Execute as células em ordem**
- **Verifique os outputs** de cada célula

### Para Debugging:
- **Verifique as dimensões** dos arrays
- **Confirme a normalização** dos pixels
- **Valide a divisão** dos dados
- **Teste com poucas amostras** primeiro

### Para Visualização:
- **Ajuste o tamanho** das figuras se necessário
- **Salve as visualizações** importantes
- **Compare diferentes classes** para entender o dataset

---

**🎉 Parabéns!** Ao completar a Fase 1, você terá uma base sólida de dados preparados para treinar seu modelo de reconhecimento de Libras!
