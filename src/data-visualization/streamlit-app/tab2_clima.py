import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from config import CORR_COLS, FRIENDLY_NAMES

def render_tab_clima(tab, df_agg_filtered):
    with tab:
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
            
            corr_data = df_agg_filtered[CORR_COLS].dropna().corr()
            # Renomeia colunas e linhas
            corr_data = corr_data.rename(columns=FRIENDLY_NAMES, index=FRIENDLY_NAMES)
            
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
