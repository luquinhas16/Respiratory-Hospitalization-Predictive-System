import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Configuração da página - DEVE SER O PRIMEIRO COMANDO STREAMLIT
st.set_page_config(
    page_title="Guri Teimoso: Saúde, Clima & Poluição em Canoas",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Adiciona o diretório pai ao sys.path para poder importar datacleaning
sys.path.append(str(Path(__file__).parent.parent))
try:
    import datacleaning
except ImportError:
    st.error("Erro ao importar o módulo datacleaning. Certifique-se de que o arquivo datacleaning.py está no diretório correto.")

# Dicionário de mapeamento dos códigos CNES para nomes de hospitais reais
CNES_MAP = {
    3508528: "Hospital Universitário de Canoas (HU)",
    2232014: "Hospital Nossa Senhora das Graças (HNSG)",
    3626245: "Hospital de Pronto Socorro de Canoas (HPSC)",
    2237601: "Hospital de Clínicas de Porto Alegre (HCPA)",
    2237571: "Hospital Santa Casa (Porto Alegre)",
    2237822: "Hospital São Lucas da PUCRS",
    2237253: "Hospital Conceição (Porto Alegre)"
}

# Configurações de Variáveis Meteorológicas (INMET)
WEATHER_VARS = {
    'temperatura_ar_mean': {
        'label': 'Temperatura Média (°C)',
        'color': '#1D3557',
        'format': '.1f',
        'unit': ' °C'
    },
    'umidade_rel_mean': {
        'label': 'Umidade Relativa Média (%)',
        'color': '#457B9D',
        'format': '.1f',
        'unit': '%'
    },
    'precipitacao_mean': {
        'label': 'Precipitação Média (mm)',
        'color': '#A8DADC',
        'format': '.2f',
        'unit': ' mm'
    },
    'pressao_atm_mean': {
        'label': 'Pressão Atmosférica Média (hPa)',
        'color': '#E63946',
        'format': '.1f',
        'unit': ' hPa'
    }
}

# Configurações de Variáveis de Poluição (FEPAM)
POLLUTION_VARS = {
    'pm10_mean': {
        'label': 'Partículas Inaláveis - PM10 (µg/m³)',
        'color': '#2A9D8F',
        'format': '.1f',
        'unit': ' µg/m³'
    },
    'so2_mean': {
        'label': 'Dióxido de Enxofre - SO2 (µg/m³)',
        'color': '#E76F51',
        'format': '.1f',
        'unit': ' µg/m³'
    },
    'no2_mean': {
        'label': 'Dióxido de Nitrogênio - NO2 (µg/m³)',
        'color': '#F4A261',
        'format': '.1f',
        'unit': ' µg/m³'
    },
    'o3_mean': {
        'label': 'Ozônio - O3 (µg/m³)',
        'color': '#E9C46A',
        'format': '.1f',
        'unit': ' µg/m³'
    },
    'co_mean': {
        'label': 'Monóxido de Carbono - CO (ppm)',
        'color': '#264653',
        'format': '.2f',
        'unit': ' ppm'
    },
    'IQAr': {
        'label': 'Índice de Qualidade do Ar (IQAr)',
        'color': '#A855F7',
        'format': '.0f',
        'unit': ''
    }
}

# CSS personalizado para design premium adaptável a temas claro/escuro
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .kpi-container {
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        margin-bottom: 25px;
    }
    .kpi-card {
        flex: 1;
        min-width: 180px;
        background-color: var(--background-color);
        border: 1px solid var(--border-color, rgba(226, 232, 240, 0.8));
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }
    .kpi-title {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--text-color);
    }
    .kpi-subtitle {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 4px;
    }
    .main-header {
        background: linear-gradient(135deg, #1d3557 0%, #457b9d 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(29, 53, 87, 0.15);
    }
    .main-header h1 {
        margin: 0;
        font-weight: 700;
        font-size: 2.5rem;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-weight: 300;
        font-size: 1.1rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

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
st.sidebar.image("https://img.icons8.com/clouds/100/hospital.png", width=80)
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

# Tabs Principais do Dashboard
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Visão Geral & Séries Temporais",
    "🌡️ Clima, Poluição e Saúde",
    "👤 Perfil Demográfico dos Pacientes",
    "💾 Base de Dados & Exportação"
])

# ==========================================
# TAB 1: VISÃO GERAL & SÉRIES TEMPORAIS
# ==========================================
with tab1:
    # Cálculo das Métricas (KPIs)
    total_internacoes = df_agg_filtered['volume_internacoes'].sum()
    total_mortes = df_agg_filtered['volume_mortes'].sum()
    taxa_letalidade = (total_mortes / total_internacoes * 100) if total_internacoes > 0 else 0.0
    temp_media = df_agg_filtered['temperatura_ar_mean'].mean()
    
    # IQAr modal / predominante
    iqar_valid = df_agg_filtered['qualidade_do_ar'].dropna()
    class_predominante = iqar_valid.mode()[0] if not iqar_valid.empty else "Sem Dados"
    
    # Exibição dos KPI cards premium usando HTML adaptável ao tema
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-title">Total de Internações</div>
            <div class="kpi-value">{total_internacoes:,}</div>
            <div class="kpi-subtitle">Casos no período</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Total de Óbitos</div>
            <div class="kpi-value">{total_mortes:,}</div>
            <div class="kpi-subtitle">Óbitos no hospital</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Letalidade Hospitalar</div>
            <div class="kpi-value">{taxa_letalidade:.2f}%</div>
            <div class="kpi-subtitle">Óbitos / Internações</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Temperatura Média</div>
            <div class="kpi-value">{temp_media:.1f}°C</div>
            <div class="kpi-subtitle">INMET Porto Alegre</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Qualidade do Ar Comum</div>
            <div class="kpi-value">{class_predominante}</div>
            <div class="kpi-subtitle">Predominância IQAr</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("---")
    
    # Série Temporal: Internações vs Métrica Selecionada (Clima ou Poluição)
    st.subheader("📈 Correlação Temporal Dinâmica")
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.markdown("### Série Temporal")
        st.write("Esta seção plota o volume de internações (eixo esquerdo vermelho) contra uma segunda variável de sua escolha (eixo direito tracejado).")
        source_selection = st.radio(
            "Escolha a variável para comparar:",
            options=["Clima (INMET)", "Poluição (FEPAM)"],
            index=0
        )
        
        if source_selection == "Clima (INMET)":
            selected_var = meteorology_choice
            var_metadata = WEATHER_VARS[meteorology_choice]
        else:
            selected_var = pollution_choice
            var_metadata = POLLUTION_VARS[pollution_choice]
            
    with col2:
        # Preparação dos dados para a série temporal
        df_ts = df_agg_filtered.copy().sort_values('data')
        df_ts['volume_suavizado'] = df_ts['volume_internacoes'].rolling(window=window_size, center=True).mean()
        df_ts['var_suavizada'] = df_ts[selected_var].rolling(window=window_size, center=True).mean()
        
        # Remover nans da série
        df_ts_clean = df_ts[['data', 'volume_suavizado', 'var_suavizada']].dropna()
        
        if df_ts_clean.empty:
            st.warning("Dados insuficientes no período selecionado para plotar com a média móvel configurada.")
        else:
            # Seleção para zoom interactivo
            zoom_interativo = alt.selection_interval(bind='scales', encodings=['x'])
            
            # Base comum do gráfico
            base = alt.Chart(df_ts_clean).encode(
                x=alt.X('data:T', title='Linha do Tempo (Dia)')
            )
            
            # Linha de internações (Vermelho)
            linha_sus = base.mark_line(color='#E63946', strokeWidth=2.5).encode(
                y=alt.Y('volume_suavizado:Q',
                    title=f'Volume de Internações (Média Móvel {window_size}d)',
                    axis=alt.Axis(titleColor='#E63946', grid=True)
                ),
                tooltip=[
                    alt.Tooltip('data:T', title='Data'),
                    alt.Tooltip('volume_suavizado:Q', title='Internações (Média)', format='.1f')
                ]
            )
            
            # Linha da variável selecionada
            linha_var = base.mark_line(color=var_metadata['color'], strokeWidth=2.5, strokeDash=[5, 5]).encode(
                y=alt.Y('var_suavizada:Q',
                    title=f"{var_metadata['label']} (Média Móvel {window_size}d)",
                    axis=alt.Axis(titleColor=var_metadata['color'], grid=False)
                ),
                tooltip=[
                    alt.Tooltip('data:T', title='Data'),
                    alt.Tooltip('var_suavizada:Q', title=var_metadata['label'], format=var_metadata['format'])
                ]
            )
            
            # Gráfico combinado
            grafico_final = alt.layer(linha_sus, linha_var).resolve_scale(
                y='independent'
            ).properties(
                width='container',
                height=450,
                title={
                    "text": f"Comparativo: Internações SUS vs. {var_metadata['label']}",
                    "subtitle": f"Valores suavizados por média móvel de {window_size} dias. Use a rolagem do mouse para zoom horizontal.",
                    "fontSize": 15,
                    "subtitleFontSize": 11
                }
            ).add_params(
                zoom_interativo
            )
            
            st.altair_chart(grafico_final, use_container_width=True)

# ==========================================
# TAB 2: CLIMA, POLUIÇÃO E SAÚDE
# ==========================================
with tab2:
    st.subheader("📊 Distribuição e Correlações Estatísticas")
    st.write("Explore as correlações diretas entre as variáveis ambientais e o impacto imediato na saúde.")
    
    col_box1, col_box2 = st.columns(2)
    
    with col_box1:
        # Boxplot de Faixa de Temperatura (do notebook)
        temp_order = ['Frio (<15°C)', 'Ameno (15-22°C)', 'Quente (22-28°C)', 'Muito Quente (>28°C)']
        
        # Garante que as categorias válidas estejam presentes
        df_box_temp = df_agg_filtered[df_agg_filtered['faixa_temperatura'].notna()].copy()
        
        chart_temp_box = alt.Chart(df_box_temp).mark_boxplot(extent='min-max', size=35).encode(
            x=alt.X('faixa_temperatura:N', sort=temp_order, title='Faixa de Temperatura Média'),
            y=alt.Y('volume_internacoes:Q', title='Volume de Internações Diárias'),
            color=alt.Color('faixa_temperatura:N', sort=temp_order, scale=alt.Scale(domain=temp_order, range=['#457B9D', '#A8DADC', '#F4A261', '#E63946']), legend=None),
            tooltip=[
                alt.Tooltip('faixa_temperatura:N', title='Faixa de Temp.'),
                alt.Tooltip('mean(volume_internacoes):Q', title='Média de Internações', format='.1f'),
                alt.Tooltip('median(volume_internacoes):Q', title='Mediana de Internações', format='.1f')
            ]
        ).properties(
            height=350,
            title={
                "text": "Internações Diárias por Faixa de Temperatura",
                "subtitle": "Baseado na temperatura média diária (INMET)",
                "fontSize": 14
            }
        )
        st.altair_chart(chart_temp_box, use_container_width=True)
        
    with col_box2:
        # Boxplot de Qualidade do Ar (do notebook)
        ordem_categorias = ["Boa", "Moderada", "Ruim", "Muito Ruim", "Pessima", "Péssima"]
        df_box_ar = df_agg_filtered[df_agg_filtered['qualidade_do_ar'].isin(ordem_categorias)].copy()
        df_box_ar['qualidade_do_ar'] = df_box_ar['qualidade_do_ar'].replace('Pessima', 'Péssima')
        ordem_categorias_std = ["Boa", "Moderada", "Ruim", "Muito Ruim", "Péssima"]
        cores_iqar = ["#2A9D8F", "#E9C46A", "#F4A261", "#E63946", "#A855F7"]
        
        chart_ar_box = alt.Chart(df_box_ar).mark_boxplot(extent='min-max', size=35).encode(
            x=alt.X('qualidade_do_ar:N', sort=ordem_categorias_std, title='Classificação do IQAr'),
            y=alt.Y('volume_internacoes:Q', title='Volume de Internações Diárias'),
            color=alt.Color('qualidade_do_ar:N', sort=ordem_categorias_std, scale=alt.Scale(domain=ordem_categorias_std, range=cores_iqar), legend=None),
            tooltip=[
                alt.Tooltip('qualidade_do_ar:N', title='Qualidade do Ar'),
                alt.Tooltip('mean(volume_internacoes):Q', title='Média de Internações', format='.1f'),
                alt.Tooltip('median(volume_internacoes):Q', title='Mediana de Internações', format='.1f')
            ]
        ).properties(
            height=350,
            title={
                "text": "Internações Diárias por Qualidade do Ar (IQAr)",
                "subtitle": "Cores representativas das faixas de poluição oficial",
                "fontSize": 14
            }
        )
        st.altair_chart(chart_ar_box, use_container_width=True)
        
    st.write("---")
    
    col_scat, col_heat = st.columns([3, 2])
    
    with col_scat:
        st.subheader("🔍 Dispersão e Linha de Tendência")
        
        # Seleção de variáveis para o Scatter Plot
        x_var_options = {
            'temperatura_ar_mean': 'Temperatura Média (°C)',
            'umidade_rel_mean': 'Umidade Relativa (%)',
            'precipitacao_mean': 'Precipitação (mm)',
            'pressao_atm_mean': 'Pressão Atmosférica (hPa)',
            'pm10_mean': 'Poluente PM10 (µg/m³)',
            'so2_mean': 'Poluente SO2 (µg/m³)',
            'no2_mean': 'Poluente NO2 (µg/m³)',
            'o3_mean': 'Poluente O3 (µg/m³)',
            'co_mean': 'Poluente CO (ppm)',
            'IQAr': 'Índice de Qualidade do Ar (IQAr)'
        }
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            scatter_x = st.selectbox(
                "Escolha a variável Ambiental (Eixo X):",
                options=list(x_var_options.keys()),
                format_func=lambda x: x_var_options[x],
                index=0
            )
        with col_s2:
            scatter_y = st.selectbox(
                "Escolha a variável Hospitalar (Eixo Y):",
                options=['volume_internacoes', 'volume_mortes'],
                format_func=lambda y: 'Internações Diárias' if y == 'volume_internacoes' else 'Óbitos Diários',
                index=0
            )
            
        df_scatter = df_agg_filtered[[scatter_x, scatter_y, 'data']].dropna()
        
        if df_scatter.empty:
            st.warning("Não há dados válidos para plotar com as variáveis selecionadas.")
        else:
            # Coeficiente de correlação
            correlation_val = df_scatter[scatter_x].corr(df_scatter[scatter_y])
            
            # Gráfico de Dispersão
            scatter_points = alt.Chart(df_scatter).mark_circle(size=50, opacity=0.5, color='#457B9D').encode(
                x=alt.X(f"{scatter_x}:Q", title=x_var_options[scatter_x]),
                y=alt.Y(f"{scatter_y}:Q", title="Volume Diário"),
                tooltip=[
                    alt.Tooltip('data:T', title='Data'),
                    alt.Tooltip(f'{scatter_x}:Q', title=x_var_options[scatter_x], format='.2f'),
                    alt.Tooltip(f'{scatter_y}:Q', title='Volume')
                ]
            )
            
            # Adiciona linha de regressão linear
            regression_line = scatter_points.transform_regression(
                scatter_x, scatter_y
            ).mark_line(color='#E63946', strokeWidth=3)
            
            final_scatter = (scatter_points + regression_line).properties(
                height=350,
                title={
                    "text": f"Relação entre {x_var_options[scatter_x]} e {scatter_y.replace('_', ' ').title()}",
                    "subtitle": f"Correlação de Pearson (r): {correlation_val:.2f} | R²: {correlation_val**2:.2f}",
                    "fontSize": 14
                }
            )
            st.altair_chart(final_scatter, use_container_width=True)
            
    with col_heat:
        st.subheader("🌡️ Matriz de Correlação Geral")
        
        # Filtra colunas para calcular correlação
        corr_cols = [
            'volume_internacoes', 'volume_mortes',
            'temperatura_ar_mean', 'umidade_rel_mean', 'precipitacao_mean', 'pressao_atm_mean',
            'pm10_mean', 'so2_mean', 'no2_mean', 'o3_mean', 'co_mean', 'IQAr'
        ]
        
        # Nomes amigáveis em português
        friendly_names = {
            'volume_internacoes': 'Internações',
            'volume_mortes': 'Óbitos',
            'temperatura_ar_mean': 'Temp. Ar',
            'umidade_rel_mean': 'Umid. Relativa',
            'precipitacao_mean': 'Precipitação',
            'pressao_atm_mean': 'Pressão Atm.',
            'pm10_mean': 'Poluente PM10',
            'so2_mean': 'Poluente SO2',
            'no2_mean': 'Poluente NO2',
            'o3_mean': 'Poluente O3',
            'co_mean': 'Poluente CO',
            'IQAr': 'IQAr'
        }
        
        corr_data = df_agg_filtered[corr_cols].dropna().corr()
        # Renomeia colunas e linhas
        corr_data = corr_data.rename(columns=friendly_names, index=friendly_names)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Paleta estilizada divergente
        sns.heatmap(
            corr_data, 
            annot=True, 
            cmap='coolwarm', 
            fmt=".2f", 
            ax=ax, 
            square=True, 
            cbar=False,
            annot_kws={"size": 7},
            linewidths=0.5
        )
        plt.xticks(fontsize=8, rotation=90)
        plt.yticks(fontsize=8, rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

# ==========================================
# TAB 3: PERFIL DEMOGRÁFICO DOS PACIENTES
# ==========================================
with tab3:
    st.subheader("👤 Perfil Sociodemográfico das Internações")
    st.write("Análise detalhada a nível de paciente a partir dos registros brutos do SUS em Canoas.")
    
    if df_sus_filtered.empty:
        st.warning("Não há dados de pacientes SUS disponíveis no período selecionado.")
    else:
        # Classificação de Faixas Etárias
        bins = [0, 4, 14, 59, 120]
        labels = ['0-4 (Bebês/Crianças)', '5-14 (Crianças/Adolescentes)', '15-59 (Adultos)', '60+ (Idosos)']
        df_sus_filtered['faixa_etaria'] = pd.cut(df_sus_filtered['IDADE'], bins=bins, labels=labels, right=True)
        
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            # Gráfico de Internações por Faixa Etária
            df_age = df_sus_filtered['faixa_etaria'].value_counts().reset_index()
            df_age.columns = ['faixa_etaria', 'internacoes']
            
            chart_age = alt.Chart(df_age).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color='#1D3557').encode(
                x=alt.X('faixa_etaria:N', title='Faixa Etária', sort=labels),
                y=alt.Y('internacoes:Q', title='Quantidade de Internações'),
                tooltip=[
                    alt.Tooltip('faixa_etaria:N', title='Faixa Etária'),
                    alt.Tooltip('internacoes:Q', title='Internações')
                ]
            ).properties(
                height=300,
                title="Distribuição por Faixa Etária"
            )
            st.altair_chart(chart_age, use_container_width=True)
            
        with col_d2:
            # Gráfico de Gênero (Donut Chart)
            df_gender = df_sus_filtered['SEXO'].value_counts().reset_index()
            df_gender.columns = ['genero', 'internacoes']
            
            chart_gender = alt.Chart(df_gender).mark_arc(innerRadius=60).encode(
                theta=alt.Theta(field="internacoes", type="quantitative"),
                color=alt.Color(
                    field="genero", 
                    type="nominal", 
                    scale=alt.Scale(domain=['Feminino', 'Masculino'], range=['#E63946', '#457B9D']), 
                    title="Gênero"
                ),
                tooltip=[
                    alt.Tooltip('genero:N', title='Gênero'),
                    alt.Tooltip('internacoes:Q', title='Internações')
                ]
            ).properties(
                height=300,
                title="Distribuição por Gênero"
            )
            st.altair_chart(chart_gender, use_container_width=True)
            
        st.write("---")
        
        col_d3, col_d4 = st.columns(2)
        
        with col_d3:
            # Taxa de Letalidade por Faixa Etária
            df_sus_filtered['morte_num'] = (df_sus_filtered['MORTE'] == 'Sim').astype(int)
            
            df_leth_age = df_sus_filtered.groupby('faixa_etaria', observed=False)['morte_num'].agg(['mean', 'count']).reset_index()
            df_leth_age['letalidade_pct'] = df_leth_age['mean'] * 100
            
            chart_leth = alt.Chart(df_leth_age).mark_bar(color='#E63946', cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X('faixa_etaria:N', title='Faixa Etária', sort=labels),
                y=alt.Y('letalidade_pct:Q', title='Taxa de Letalidade (%)'),
                tooltip=[
                    alt.Tooltip('faixa_etaria:N', title='Faixa Etária'),
                    alt.Tooltip('letalidade_pct:Q', title='Letalidade (%)', format='.2f'),
                    alt.Tooltip('count:Q', title='Total Casos')
                ]
            ).properties(
                height=300,
                title="Taxa de Letalidade Hospitalar por Idade"
            )
            st.altair_chart(chart_leth, use_container_width=True)
            
        with col_d4:
            # Top Hospitais (CNES)
            df_hosp = df_sus_filtered['CNES'].value_counts().reset_index()
            df_hosp.columns = ['CNES', 'internacoes']
            df_hosp['Hospital'] = df_hosp['CNES'].map(CNES_MAP).fillna(df_hosp['CNES'].astype(str))
            
            chart_hosp = alt.Chart(df_hosp.head(5)).mark_bar(color='#457B9D', cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
                y=alt.Y('Hospital:N', sort='-x', title='Hospital/Estabelecimento'),
                x=alt.X('internacoes:Q', title='Quantidade de Internações'),
                tooltip=[
                    alt.Tooltip('Hospital:N', title='Hospital'),
                    alt.Tooltip('internacoes:Q', title='Internações')
                ]
            ).properties(
                height=300,
                title="Distribuição por Hospital Executante"
            )
            st.altair_chart(chart_hosp, use_container_width=True)

# ==========================================
# TAB 4: BASE DE DADOS & EXPORTAÇÃO
# ==========================================
with tab4:
    st.subheader("💾 Visualização da Base Consolidada")
    st.write("Aqui você pode inspecionar a base de dados agregada diária e fazer o download para uso acadêmico.")
    
    # Exibe tabela principal
    st.dataframe(df_agg_filtered, use_container_width=True)
    
    # Botão de download
    csv_data = df_agg_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Baixar Dados Filtrados (CSV)",
        data=csv_data,
        file_name="respiratorios_canoas_filtrado.csv",
        mime="text/csv"
    )
    
    st.write("---")
    
    st.subheader("📊 Estatísticas Descritivas")
    st.write("Estatísticas básicas das variáveis climáticas, de poluição e hospitalares selecionadas no período:")
    
    # Tabela de estatísticas
    desc_stats = df_agg_filtered[corr_cols].describe().T
    desc_stats = desc_stats.rename(index=friendly_names)
    st.dataframe(desc_stats, use_container_width=True)
