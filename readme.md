# Analisador de Sentimentos com Python

Este projeto implementa um analisador de sentimentos utilizando Python. Ele foi desenvolvido para identificar o sentimento de textos (por exemplo, tweets) classificando-os como **positivo**, **negativo** ou **neutro**. Inspirado em uma solução similar em PHP, este projeto utiliza a mesma estratégia de processamento:  
- **Limpeza dos dados**  
- **Treinamento por lotes** (ensemble de modelos)  
- **Predição com medição de uso de memória**

## Introdução

A **análise de sentimentos** é uma técnica de processamento de linguagem natural (NLP) que visa extrair informações subjetivas de textos, permitindo identificar o tom emocional (positivo, negativo ou neutro). Essa abordagem é muito utilizada em:
- Monitoramento de redes sociais e reputação online
- Análise de feedback de clientes
- Pesquisas de mercado

Este projeto emprega algoritmos clássicos de machine learning (usando o [scikit-learn](https://scikit-learn.org/)) para construir um pipeline de análise de sentimentos que transforma textos em vetores de características com `CountVectorizer` e `TfidfTransformer`, e em seguida utiliza o classificador `Multinomial Naive Bayes`.

## Tecnologias e Bibliotecas

- **Python 3** – Linguagem de programação.
- **Pandas** – Manipulação e limpeza de dados.
- **scikit-learn** – Criação do pipeline de pré-processamento e do classificador.
- **psutil** – Medição do uso de memória durante o treinamento e predição.
- **Flask** – Criação de uma interface web simples para predição.
- **Bootstrap** – Estilização da interface web.

### Por que Python usa menos memória que PHP?

Python, com bibliotecas otimizadas como o pandas e o scikit-learn, gerencia os dados de forma vetorizada e eficiente. Essa abordagem, combinada com o uso de estruturas de dados otimizadas (como os DataFrames do pandas), geralmente consome menos memória em comparação a implementações em PHP que podem criar grandes estruturas de arrays ou objetos em memória. Além disso, o uso de algoritmos implementados em C (como os do scikit-learn) contribui para uma melhor eficiência.

## Estrutura do Projeto

```plaintext
.
├── datasets
│   ├── Tweets.csv             # Dataset original com múltiplas colunas
│   └── clean_tweets.csv       # Dataset limpo (colunas: texto e sentimento)
├── models                     # Modelos treinados (salvos como .pkl)
├── templates                  # Templates HTML (para a interface web com Flask)
│   └── index.html
├── app.py                     # Aplicação web (Flask)
├── limpar_dados.py            # Script para limpeza do dataset usando pandas
├── sentiment_pipeline.py      # Definição do pipeline (CountVectorizer, TfidfTransformer, MultinomialNB)
├── treinar_lotes.py           # Treinamento dos modelos por lotes com medição de memória
├── predicao.py                # Predição com ensemble (avaliação CLI) e medição de memória
├── requirements.txt           # Lista de dependências do projeto
└── README.md                  # Este arquivo
```
## Configurando o Ambiente Virtual
1. Crie um ambiente virtual
No terminal, execute:
```
python3 -m venv venv

```
2. Ative o ambiente virtual
- No Linux/macOS
```
source venv/bin/activate

```
- No Windows
```
venv\Scripts\activate

```
3. Instale as dependências

```
pip install -r requirements.txt

```

## Como Rodar o Projeto
1. Limpeza dos Dados
Execute o script de limpeza para gerar o dataset limpo:
```
python limpar_dados.py
```

2. Treinamento por Lotes
Execute o script de treinamento que:

Lê o dataset limpo
Divide os dados em 10 lotes
Treina um pipeline para cada lote e salva os modelos em ./models
Exibe o uso de memória durante o treinamento
```
python treinar_lotes.py
```

3. Predição via CLI
Para avaliar o ensemble dos modelos treinados, use:
```
python predicao.py
```
Esse script:

Carrega os submodelos (excetuando o 10º lote)
Utiliza a 10ª parte do dataset como conjunto de teste
Combina as predições via votação majoritária
Exibe a acurácia e o uso de memória durante a predição


4. Interface Web
Para rodar a interface web, execute:
```
python app.py
```

## Informações Técnicas
- Dataset:
O arquivo original Tweets.csv é limpo para gerar clean_tweets.csv contendo apenas as colunas de texto e sentimento.

- Modelos:
Cada lote de treinamento gera um modelo (pipeline) que é salvo como um arquivo .pkl usando o pickle. Na predição, os modelos são carregados e combinados via ensemble (votação majoritária).

- Uso de Memória:
Utilizamos a biblioteca psutil para medir o uso de memória (RSS) antes e depois dos processos de treinamento e predição.
- Exemplo de saída durante o treinamento:
Treinamento por lotes concluído.
Memória utilizada durante o treinamento: 7868416 bytes (7.50 MB)


## Conclusão
Este projeto demonstra uma abordagem prática e eficiente para análise de sentimentos utilizando Python e bibliotecas modernas como pandas e scikit-learn. A estratégia de treinamento por lotes e a combinação de modelos via ensemble permitem que o sistema lide com datasets grandes de maneira eficiente, reduzindo o uso de memória e otimizando o desempenho. A interface web construída com Flask e Bootstrap facilita a interação e o uso do sistema.