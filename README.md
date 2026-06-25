# Guri-Teimoso
O projeto do time **Guri Teimoso** foi desenvolvido como trabalho da cadeira de Ciência de Dados (INF01090) em 2026/1 e analisou a correlação entre as condições ambientais (clima e qualidade do ar) e problemas de saúde pública (internações respiratórias) na cidade de **Canoas, RS**.

 Os dados consolidados no dashboard provêm das seguintes fontes oficiais:
            
* **SUS (SIH - Sistema de Informações Hospitalares) - coleta por servidor FTP do DATASUS**
    * Contém os registros de internações e óbitos em hospitais do município de Canoas.
    * **Website Oficial**: [Acessar site do SUS](https://datasus.saude.gov.br/informacoes-de-saude-tabnet/)
    
* **INMET (Instituto Nacional de Meteorologia)**
    * Fornece dados climáticos horários (temperatura, umidade, pressão e precipitação). Foi utilizada a estação meteorológica de Porto Alegre (cidade vizinha), por ser a mais próxima com dados completos.
    * **Website Oficial**: [Acessar site do INMET](https://portal.inmet.gov.br/dadoshistoricos)
    
* **FEPAM (Fundação Estadual de Proteção Ambiental)**
    * Fornece medições horárias da qualidade do ar e poluentes atmosféricos (PM10, SO2, NO2, O3, CO).
    * **Website Oficial**: [Acessar site da FEPAM](https://www.fepam.rs.gov.br/dados-do-monitoramento)

  

## Estrutura de Diretórios (Visualização de dados)
```

src
├── data-visualization
    ├── dataVisualization.ipynb   # Notebook com esboços e análises iniciais
    ├── datacleaning.py          # Script para tratamento e agregação dos dados
    ├── plots.py                 # Funções auxiliares para plots de séries temporais
    ├── df_agg.csv               # Dataset diário consolidado (SUS, INMET, FEPAM)
    └── streamlit-app            # Pasta com o aplicativo Streamlit
        └── app.py               # Arquivo principal do dashboard

└── model
	└── dataset
		├── data_cleaning.py # Arquivo que padroniza, remove colunas iniciais e junta datasets, gera dataset.csv que é o utilizado.
		├── dataset.csv # Arquivo utilizado como dataset, é junção de dados de múltiplas bases de dados
	├── main.ipynb # Notebook principal para a execução do modelo final
	├── feature_engineering # .py que contém funções auxiliares de tratamento de dados
	├── model.py # .py que contém funções auxiliares de construção e treinamento do modelo
	├── spot_checking.ipynb # Notebook que contém análises preeliminar de dados e modelos

```

## Como Instalar e Executar o Streamlit
### 1. Instalar as Dependências
Utilize o `pipenv` para instalar todos os pacotes necessários do ambiente virtual:
```bash
pipenv install
```

Como instalar o pipenv:
```bash
pip install pipenv
```

  
### 2. Executar o Aplicativo Streamlit
Para inicializar o dashboard interativo, execute o comando abaixo na raiz do repositório:
```bash
pipenv run streamlit run src/data-visualization/streamlit-app/app.py
```
Isso abrirá automaticamente a aplicação em seu navegador padrão no endereço `http://localhost:8501`.


## Estrutura de Diretórios (Visualização de dados)
```
src
├── model # Pasta para construção do modelo de predição

    ├── feature_engineering.py # Arquivo com o código para feature engineering do modelo
    ├── lstm_model.pt # Modelo gerado e usado na visualização da predição
    ├── main.ipynb # Notebook para construção e avaliação do modelo
    └── train_and_save.py # Script para treinar o modelo e gerar o salvá-lo
	
└── data-visualization # Pasta para visualização de dados

    ├── dataVisualization.ipynb   # Notebook com esboços e análises iniciais
    ├── datacleaning.py          # Script para tratamento e agregação dos dados
    ├── plots.py                 # Funções auxiliares para plots de séries temporais
    ├── df_agg.csv               # Dataset diário consolidado (SUS, INMET, FEPAM)
    └── streamlit-app            # Pasta com o aplicativo Streamlit
        └── app.py               # Arquivo principal do dashboard
```
