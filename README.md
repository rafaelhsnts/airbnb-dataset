# Airbnb Open Data Analysis

O projeto analisa dados de anúncios de Airbnb nos Estados Unidos para identificar padrões de preço, distribuição geográfica, tipos de imóveis e fatores que influenciam o valor do aluguel e a satisfação dos usuários.

## Dataset  

O dataset utilizado foi Airbnb_Open_Data.csv obtido do link "https://www.kaggle.com/datasets/arianazmoudeh/airbnbopendata?select=Airbnb_Open_Data.csv". Ele possui 102599 linhas e 26 colunas.

Principais colunas:

Id: Identificação do anúncio.
Name: Nome/título do anúncio.
Host Id: Identificação do anfitrião.
Neighbourhood Group: Região/zona do imóvel.
Room Type: Tipo de quarto.
Price: Valor de locação do imóvel.
Number Of Reviews: Quantidade de avaliações.
Review Rate Number: Nota média recebida.

## Tecnologias

* folium 0.20.0
* matplotlib 3.10.7
* numpy 2.3.5
* pandas 2.3.3
* python 3.12.3
* pywafle 1.1.1
* seaborn 0.13.2
* wordcloud 1.9.4

## Estrutura do projeto

airbnb_dataset
|-- notebooks/          # Análizes e visualizações dos dados
|-- outputs/            # Resultados e arquivos gerados
|-- .gitignore          # Arquivos e pastas não incluídos no repositório
|-- README.md           # Informações do projeto
|-- requirementes.txt   # Dependências do projeto

## Como executar

1. Clone o repositorio:

    git clone https://github.com/rafaelhsnts/airbnb-dataset.git

2. Instale as dependências:

    pip install -r requirements.txt

3. Abra o notebook no Jupter ou Colab:

    notebooks/airbnb_open_data.ipynb