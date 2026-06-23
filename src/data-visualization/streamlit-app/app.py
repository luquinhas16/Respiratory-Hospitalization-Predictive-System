import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Configuração da página - DEVE SER O PRIMEIRO COMANDO STREAMLIT
st.set_page_config(
    page_title="Guri Teimoso: Saúde, Clima & Poluição em Canoas",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Adiciona o diretório atual e o diretório pai ao sys.path para poder importar datacleaning e os tabs
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

try:
    import datacleaning
except ImportError:
    st.error("Erro ao importar o módulo datacleaning. Certifique-se de que o arquivo datacleaning.py está no diretório correto.")

# Importa as configurações do config.py no mesmo diretório
from config import CNES_MAP, WEATHER_VARS, POLLUTION_VARS, CSS_STYLE

# Importa os módulos dos tabs
from tab1_geral import render_tab_geral
from tab2_clima import render_tab_clima
from tab3_demografia import render_tab_demografia
from tab4_base import render_tab_base
from tab5_previsoes import render_tab_previsoes
from tab6_infos import render_tab_infos

# CSS personalizado para design premium adaptável a temas claro/escuro
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# Funções de carregamento de dados com cache do Streamlit
@st.cache_data
def load_data():
    # Caminho absoluto para o df_agg.csv
    agg_path = Path(__file__).parent.parent / 'df_agg.csv'
    df_agg = pd.read_csv(agg_path)
    df_agg['data'] = pd.to_datetime(df_agg['data'])
    
    # Caminho absoluto para o sus.csv
    sus_path = Path(__file__).parent.parent.parent.parent / 'dataset' / 'sus.csv'
    df_sus_raw = pd.read_csv(sus_path)
    df_sus = datacleaning.dataCleaningSus(df_sus_raw)
    
    return df_agg, df_sus

try:
    df_agg, df_sus = load_data()
except Exception as e:
    st.error(f"Erro ao carregar os dados: {e}")
    st.info("Verifique se as pastas 'dataset' e 'src/data-visualization' contêm os arquivos corretos.")
    st.stop()

# Configuração da Barra Lateral (Filtros Globais)
st.sidebar.image("https://img.icons8.com/?size=100&id=kdZKqGGvU5ib&format=png&color=000000", width=80)
st.sidebar.title("Configurações e Filtros")

# Filtro de Data
min_date = df_agg['data'].min().date()
max_date = df_agg['data'].max().date()

st.sidebar.subheader("Período temporal")
date_range = st.sidebar.date_input(
    "Selecione o intervalo de datas:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

# Filtros de Variáveis
st.sidebar.subheader("Seleção de Variáveis")
meteorology_choice = st.sidebar.selectbox(
    "Métrica Climática (INMET):",
    options=list(WEATHER_VARS.keys()),
    format_func=lambda x: WEATHER_VARS[x]['label']
)

pollution_choice = st.sidebar.selectbox(
    "Poluente Atmosférico (FEPAM):",
    options=list(POLLUTION_VARS.keys()),
    format_func=lambda x: POLLUTION_VARS[x]['label']
)

# Filtro de Suavização (Média Móvel)
st.sidebar.subheader("Suavização de Tendências")
window_size = st.sidebar.slider(
    "Média Móvel (dias) para gráficos de linha:",
    min_value=1,
    max_value=30,
    value=7,
    step=1
)

# Filtragem dos DataFrames baseados na data selecionada
df_agg_filtered = df_agg[
    (df_agg['data'].dt.date >= start_date) & 
    (df_agg['data'].dt.date <= end_date)
].copy()

df_sus_filtered = df_sus[
    (df_sus['data'].dt.date >= start_date) & 
    (df_sus['data'].dt.date <= end_date)
].copy()

# Header Principal
st.markdown(f"""
<div class="main-header">
    <h1>Análise de Internações Respiratórias em Canoas</h1>
    <p>Impactos das Condições Climáticas (INMET) e da Poluição do Ar (FEPAM) na Saúde Pública (SUS/SIH) | {start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}</p>
</div>
""", unsafe_allow_html=True)

# Tabs Principais do Dashboard (com vírgulas corrigidas)
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Visão Geral & Séries Temporais",
    "🌡️ Clima, Poluição e Saúde",
    "👤 Perfil Demográfico dos Pacientes",
    "📈 Previsões",
    "💾 Base de Dados & Exportação",
    "ℹ️ Infos"
])

# Renderização de cada Tab
render_tab_geral(tab1, df_agg_filtered, meteorology_choice, pollution_choice, window_size, WEATHER_VARS, POLLUTION_VARS)
render_tab_clima(tab2, df_agg_filtered)
render_tab_demografia(tab3, df_sus_filtered)
render_tab_previsoes(tab4)
render_tab_base(tab5, df_agg_filtered)
render_tab_infos(tab6)
