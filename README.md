# Análise e predição do posicionamento político do eleitor brasileiro

> **Projeto final da disciplina de Ciência de Dados** | Universidade de São Paulo - Ciência da Computação

## 📧 Autor

[Hiago Soares de Araujo](https://www.linkedin.com/in/hiago-soares-96840a271/)

## 🌐 Deploy

Acesse a aplicação em: https://espectro-politico-brasil.streamlit.app

## 📊 Visão geral

Este projeto realiza uma análise exploratória e constrói um modelo preditivo capaz de classificar o posicionamento político dos eleitores brasileiros (Esquerda, Centro ou Direita) com base em suas opiniões sobre temas políticos e sociais.

O trabalho utiliza dados de pesquisa de opinião coletados pelo Instituto DataSenado em **2024**, contendo respostas sobre temas polêmicos como direitos humanos, políticas econômicas e confiança institucional.

## 🎯 Objetivos

### Fase 1: Análise exploratória de dados
- Investigar a distribuição das variáveis
- Investigar relações entre variáveis sociodemográficas/políticas e o posicionamento político
- Identificar padrões e tendências nos dados
- Gerar visualizações que revelem insights sobre o perfil ideológico dos eleitores

### Fase 2: Modelagem preditiva
- Desenvolver um modelo de machine learning capaz de prever o posicionamento político
- Otimizar hiperparâmetros para melhor desempenho
- Obter e analisar métricas do modelo
- Avaliar a importância das variáveis para o modelo

## 📁 Estrutura do projeto

```
ds-espectro-politico-brasil/
├── app/                         # Aplicação Streamlit
│   ├── app.py                   # Código principal da aplicação Streamlit
│   ├── requirements.txt         # Dependências do projeto
├── data/                        # Dados usados no projeto
│   ├── raw/                     # Dados brutos (pesquisas originais)
│   │   ├── panorama_politico_01_2021/
│   │   ├── panorama_politico_06_2024/  # Dados principais
│   │   ├── panorama_politico_11_2022/
│   │   └── panorama_politico_12_2021/
│   └── processed/               # Dados processados
│       ├── data2024.csv         # Dados processados com as variáveis selecionadas, em formato numérico
│       └── data_ml2024.csv      # Dados com as variáveis selecionadas para gerar o modelo, em formato textual
├── images/                      # Gráficos e visualizações geradas
├── models/                      # Modelos de machine learning gerados
│   └── catboost_posicionamento_politico.cbm  # Modelo treinado com o algoritmo CatBoost
├── notebooks/                   # Notebooks do projeto
│   ├── data_analysis.ipynb      # Exploração e visualização dos dados
│   └── machine_learning.ipynb   # Treinamento do modelo com CatBoost
├── utils/                       # Códigos úteis utilizados em várias partes do projeto
│   ├── __init__.py
│   └── data_processing.py       # Funções de processamento de dados
└── README.md                    # Arquivo README para descrever o projeto
```

## 🗂️ Dados utilizados

* **Principal:** Pesquisa Panorama Político: Instituto DataSenado, 2024, *MICRODADOS* - [Pesquisa traça perfil ideológico dos eleitores brasileiros](https://www12.senado.leg.br/institucional/datasenado/publicacaodatasenado?id=pesquisa-traca-perfil-ideologico-dos-eleitores-brasileiros)

* Pesquisas Panorama Político: Instituto DataSenado, 2016 - 2024, *MICRODADOS* [Pesquisa traça perfil ideológico dos eleitores brasileiros: 2016 a 2024](https://www.senado.leg.br/institucional/datasenado/paineis_dados/#/?pesquisa=panorama_politico)

## 📊 Visualizações geradas

O projeto gera visualizações automáticas:
- 📈 Análises univariadas das variáveis
- 🗺️ Análises bivariadas das variáveis sociodemográficas e políticas e suas relações com a variável alvo **posicionamento político**
- 📉 Matriz de confusão normalizada do modelo de predição
- 📊 Gráfico de importância de variáveis para o modelo de predição

Armazenadas em `images/`

## 🤖 Modelo de predição

### Algoritmo: CatBoost Classifier

O projeto utiliza **CatBoost (Categorical Boosting)** pela sua eficiência em lidar com variáveis categóricas nativamente, sem necessidade de pré-processamento dos dados (como One-Hot Encoding).

A métrica principal utilizada será o F1-Score, devido ao desbalanceamento das classes.

**Características principais:**
- ✅ Trabalha diretamente com dados categóricos
- ✅ Tratamento nativo do desbalanceamento da variável alvo
- ✅ Sem necessidade de muito pré-processamento dos dados categóricos
- ✅ Gradient boosting com árvores de decisão
- ✅ Suporte a GPU para treinamento acelerado

### Variável alvo
**Posicionamento político** - O usuário será classificado em uma dessas posições políticas:
- 🔴 Esquerda
- ⚪ Centro
- 🟢 Direita

#### Insights
- O modelo **distingue bem espectros politicamente bem definidos** (Esquerda vs. Direita)
- **Dificuldades** em classificar o posicionamento de Centro
- **Confusões principais**: Centro é frequentemente confundido com Direita ou Esquerda
- Questões polêmicas/polarizantes estão entre as mais importantes para o modelo

## 🚀 Como usar

### Via aplicação web
Acesse a aplicação em: https://espectro-politico-brasil.streamlit.app/

**Funcionalidades:**
- 📊 Visualizações interativas dos dados
- 🔮 Preditor: Responda a questões e descubra seu posicionamento político
- 📈 Análise comparativa entre regiões, idade, educação e questões polêmicas, considerando os espectros políticos

### Localmente

#### Pré-requisitos
- Python 3.8+
- Pip ou Conda

#### Instalação

```bash
# Clonar repositório
git clone https://github.com/h-Soares/ds-espectro-politico-brasil.git

# Mudar para o repositório
cd ds-espectro-politico-brasil

# Instalar dependências
pip install -r requirements.txt
```

#### Executar a aplicação

```bash
streamlit run app/app.py
```

A aplicação abrirá em `http://localhost:8501`

## 📓 Notebooks de análise

### 1️⃣ [data_analysis.ipynb](../notebooks/data_analysis.ipynb)
Exploração completa dos dados com:
- Estatísticas descritivas
- Análise de distribuições univariadas e bivariadas
- Correlações com a variável alvo
- Visualizações gráficas
- Detecção de padrões e insights

### 2️⃣ [machine_learning.ipynb](../notebooks/machine_learning.ipynb)
Construção e otimização do modelo com:
- Preparação dos dados categóricos
- Otimização de hiperparâmetros com Randomized Search
- Treinamento do modelo com CatBoost Classifier
- Avaliação com múltiplas métricas
- Análise de importância de variáveis

## 🔍 Metodologia

### Processamento dos dados
1. **Carregamento**: Dados brutos do Instituto DataSenado
2. **Limpeza**: Remoção de valores nulos e tratamento de inconsistências como valores duplicados
3. **Seleção**: Escolha de variáveis mais relevantes para o modelo
4. **Transformação**: Conversão dos dados do formato numérico para o textual
5. **Validação**: Verificação final de integridade e distribuição dos dados

### Divisão dos dados para o modelo de predição
- **Treino**: 70%
- **Teste**: 15%
- **Validação**: 15%

Em cada divisão foram preservadas as proporções de classes desbalanceadas.

## ⚠️ Considerações importantes

1. **Não representa a população brasileira**: Parcela significativa dos dados originais não foi utilizada
2. **Desbalanceamento de classes**: Influencia o desempenho do modelo, especialmente para a classe "Centro"
3. **Escopo temporal**: Dados coletados em 2024, podem não representar tendências futuras
4. **Variáveis categóricas**: Modelo trabalha com opiniões políticas e dados sociodemográficos. Não contém variáveis numéricas

## 🎓 Contexto Acadêmico

**Instituição**: Universidade de São Paulo (USP)  
**Curso**: Bacharelado em Ciência da Computação  
**Disciplina**: Ciência de Dados  
**Tipo**: Projeto final da disciplina

## 📚 Referências

- [CatBoost Documentation](https://catboost.ai/)
- [Instituto DataSenado](https://www12.senado.leg.br/institucional/datasenado/)