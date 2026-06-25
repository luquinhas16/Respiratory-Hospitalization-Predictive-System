# Dicionário de mapeamento dos códigos CNES para nomes de hospitais reais
CNES_MAP = {
    3508528: {
        "name":"Hospital Universitário de Canoas (HU)",
        "coordinates":"-29.8862511736772, -51.16626057386339"
    },
    2232014: {
        "name":"Hospital Nossa Senhora das Graças (HNSG)",
        "coordinates":"-29.92672469537319, -51.16134284791391"
    },
    3626245: {
        "name":"Hospital de Pronto Socorro de Canoas (HPSC)",
        "coordinates":"-29.906363272943505, -51.197530946068234"
    }
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

# Colunas para a análise de correlação e estatísticas descritivas
CORR_COLS = [
    'volume_internacoes', 'volume_mortes',
    'temperatura_ar_mean', 'umidade_rel_mean', 'precipitacao_mean', 'pressao_atm_mean',
    'pm10_mean', 'so2_mean', 'no2_mean', 'o3_mean', 'co_mean', 'IQAr'
]

# Tradução amigável dos nomes de colunas
FRIENDLY_NAMES = {
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

# CSS personalizado para design premium adaptável a temas claro/escuro
CSS_STYLE = """
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
"""
