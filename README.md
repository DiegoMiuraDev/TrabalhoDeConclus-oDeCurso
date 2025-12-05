# Projeto TCC: Reconhecimento de Libras com IA

## 🎯 Objetivo
Sistema de reconhecimento automático de sinais de Libras em tempo real usando webcam.

## 🚀 Como Usar

### 1. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 2. Executar Aplicação Web
```bash
python app_web.py
```
Acesse: http://localhost:5000

### 3. Executar Aplicação Melhorada (com detecção de problemas)
```bash
python app_web_improved_v2.py
```
Acesse: http://localhost:5000

## 📁 Estrutura do Projeto

```
tcc/
├── app_web.py                    # Aplicação web principal
├── app_web_improved_v2.py       # Aplicação web melhorada
├── models/                      # Modelos treinados
│   ├── libras_brasileiro_best.h5
│   └── libras_classes.npy
├── configs/                     # Configurações
│   └── config.py
├── src/                         # Código fonte
│   ├── data/                    # Manipulação de dados
│   ├── models/                  # Modelos de IA
│   ├── utils/                   # Utilitários
│   └── visualization/           # Visualizações
├── scripts/                     # Scripts executáveis
│   ├── train_model.py          # Treinar modelo
│   ├── real_time_demo.py       # Demo webcam
│   ├── collect_test_data.py    # Coletar dados de teste
│   └── generate_metrics_table.py # Gerar tabela de métricas
└── requirements.txt             # Dependências
```

## 🎯 Funcionalidades

- ✅ Reconhecimento de 24 letras de Libras
- ✅ Interface web com webcam
- ✅ Detecção automática de problemas no modelo
- ✅ Correções em tempo real
- ✅ Histórico de predições

## 🔧 Tecnologias

- **Python 3.8+**
- **TensorFlow/Keras**
- **Flask** (aplicação web)
- **OpenCV** (processamento de imagem)
- **MobileNetV2** (modelo base)

## 📊 Status do Projeto

- ✅ Modelo treinado e funcional
- ✅ Aplicação web operacional
- ✅ Sistema de detecção de problemas
- ⚠️ Modelo atual tem viés (classifica tudo como letra A)
- 🔄 Necessário retreinamento para melhorar precisão

## 🚀 Próximos Passos

1. **Retreinar modelo** com técnicas anti-overfitting
2. **Coletar mais dados** de treinamento
3. **Implementar validação cruzada**
4. **Otimizar hiperparâmetros**
