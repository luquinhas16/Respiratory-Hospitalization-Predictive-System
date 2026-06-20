import pandas as pd
import altair as alt

def plot_internations_vs_temperature_evolution(df):
    # 1. Preparar e ordenar os dados
    df_plot = df.copy()
    df_plot['data'] = pd.to_datetime(df_plot['data'])
    df_plot = df_plot.sort_values('data')
    
    # Calcular as médias móveis de 7 dias para suavizar as linhas
    df_plot['volume_suavizado'] = df_plot['volume_internacoes'].rolling(window=7, center=True).mean()
    df_plot['temp_suavizada'] = df_plot['temperatura_ar_mean'].rolling(window=7, center=True).mean()
    
    # Selecionar apenas as colunas necessárias para deixar o gráfico mais leve
    df_plot = df_plot[['data', 'volume_suavizado', 'temp_suavizada']].dropna()

    # Criar uma seleção para permitir zoom e pan (interatividade)
    zoom_interativo = alt.selection_interval(bind='scales', encodings=['x'])

    # 2. Base comum (Eixo X compartilhado)
    base = alt.Chart(df_plot).encode(
        x=alt.X('data:T', title='Data (Linha do Tempo)')
    )

    # 3. Camada do SUS: Internações (Eixo Y Esquerdo - Vermelho)
    linha_sus = base.mark_line(color='#E63946', strokeWidth=2.5).encode(
        y=alt.Y('volume_suavizado:Q',
            title='Volume de Internações (Média Móvel 7d)',
            axis=alt.Axis(titleColor='#E63946', grid=True)
        ),
        tooltip=[alt.Tooltip('data:T', title='Data'), alt.Tooltip('volume_suavizado:Q', title='Internações (Média)', format='.1f')]
    )

    # 4. Camada do INMET: Temperatura (Eixo Y Direito - Azul)
    linha_temp = base.mark_line(color='#1D3557', strokeWidth=2.5, strokeDash=[5, 5]).encode(
        y=alt.Y('temp_suavizada:Q',
            title='Temperatura Média do Ar (°C - Média Móvel 7d)',
            axis=alt.Axis(titleColor='#1D3557', grid=False)
        ),
        tooltip=[alt.Tooltip('data:T', title='Data'), alt.Tooltip('temp_suavizada:Q', title='Temperatura', format='.1f')]
    )

    # 5. Combinar as camadas e resolver as escalas de forma independente
    grafico_final = alt.layer(linha_sus, linha_temp).resolve_scale(
        y='independent'
    ).properties(
        width=800,
        height=400,
        title={
            "text": "Série Temporal Interativa: Internações vs. Temperatura",
            "subtitle": "Dados suavizados por média móvel de 7 dias. Use o scroll do mouse para zoom no eixo X.",
            "fontSize": 16,
            "subtitleFontSize": 12
        }
    ).add_params(
        zoom_interativo
    )

    return grafico_final

