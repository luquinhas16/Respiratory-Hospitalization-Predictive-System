# Guri-Teimoso

  

## Estrutura de Diretórios (Visualização de dados)

  

```

src

└── data-visualization

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

  

### 2. Executar o Aplicativo Streamlit

Para inicializar o dashboard interativo, execute o comando abaixo na raiz do repositório:

```bash

pipenv run streamlit run src/data-visualization/streamlit-app/app.py

```

Isso abrirá automaticamente a aplicação em seu navegador padrão no endereço `http://localhost:8501`.

  

### 3. Executar o Notebook

Se desejar rodar o Jupyter Notebook localmente:

```bash

pipenv run jupyter notebook

```

E selecione o kernel criado pelo ambiente virtual.