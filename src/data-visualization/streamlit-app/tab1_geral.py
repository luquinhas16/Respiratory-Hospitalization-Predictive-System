import streamlit as st
import pandas as pd
import altair as alt

def render_tab_geral(tab, df_agg_filtered, meteorology_choice, pollution_choice, window_size, WEATHER_VARS, POLLUTION_VARS):
    with tab:
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
            
            # Separando os dados para evitar que NaNs de uma variável dropem dados da outra
            df_sus_ts = df_ts[['data', 'volume_suavizado']].dropna()
            df_var_ts = df_ts[['data', 'var_suavizada']].dropna()
            
            if df_sus_ts.empty or df_var_ts.empty:
                st.warning("Dados insuficientes no período selecionado para plotar com a média móvel configurada.")
            else:
                # Seleção para zoom interactivo
                zoom_interativo = alt.selection_interval(bind='scales', encodings=['x'])
                
                # Linha de internações (Vermelho)
                linha_sus = alt.Chart(df_sus_ts).encode(
                    x=alt.X('data:T', title='Linha do Tempo (Dia)')
                ).mark_line(color='#E63946', strokeWidth=2.5).encode(
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
                linha_var = alt.Chart(df_var_ts).encode(
                    x=alt.X('data:T', title='Linha do Tempo (Dia)')
                ).mark_line(color=var_metadata['color'], strokeWidth=2.5, strokeDash=[5, 5]).encode(
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
