# Guri-Teimoso

## Estrutura de Diretórios (Visualização de dados)

```
src
└── data-visualization
    ├── dataVisualization.ipynb   # Notebook com esboços e análises iniciais
    ├── datacleaning.py          # Script para tratamento e agregação dos dados
    ├── plots.py                 # Funções auxiliares para plots de séries temporais
    ├── df_agg.csv               # Dataset diário consolidado (SUS, INMET, FEPAM)
    └── streamlit-app            # Pasta com o aplicativo Streamlit
        └── app.py               # Arquivo principal do dashboard
```

## Como Instalar e Executar

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

