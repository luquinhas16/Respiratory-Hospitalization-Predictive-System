import streamlit as st
import pandas as pd
import altair as alt
from config import CNES_MAP

def render_tab_demografia(tab, df_sus_filtered):
    with tab:
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
                top_cnes = df_sus_filtered['CNES'].value_counts().head(5).index
                df_sus_top = df_sus_filtered[df_sus_filtered['CNES'].isin(top_cnes)].copy()
                
                df_hosp = df_sus_top.groupby('CNES').agg(
                    Obitos=('MORTE', lambda x: (x == 'Sim').sum()),
                    Sobreviventes=('MORTE', lambda x: (x != 'Sim').sum())
                ).reset_index()
                
                df_hosp['Hospital'] = df_hosp['CNES'].apply(lambda x: CNES_MAP[x]['name'] if x in CNES_MAP else str(x))
                
                df_hosp_melted = pd.melt(
                    df_hosp,
                    id_vars=['Hospital'],
                    value_vars=['Sobreviventes', 'Obitos'],
                    var_name='Status',
                    value_name='Casos'
                )
                
                status_map = {
                    'Sobreviventes': 'Sobreviventes (Altas)',
                    'Obitos': 'Óbitos'
                }
                df_hosp_melted['Status'] = df_hosp_melted['Status'].map(status_map)
                
                chart_hosp = alt.Chart(df_hosp_melted).mark_bar().encode(
                    y=alt.Y('Hospital:N', 
                            sort=alt.EncodingSortField(field='Casos', op='sum', order='descending'),
                            title='Hospital/Estabelecimento'),
                    x=alt.X('Casos:Q', title='Quantidade de Casos'),
                    color=alt.Color('Status:N', 
                                    scale=alt.Scale(domain=['Sobreviventes (Altas)', 'Óbitos'], range=['#457B9D', '#E63946']),
                                    title='Categoria'),
                    tooltip=[
                        alt.Tooltip('Hospital:N', title='Hospital'),
                        alt.Tooltip('Status:N', title='Categoria'),
                        alt.Tooltip('Casos:Q', title='Quantidade')
                    ]
                ).properties(
                    height=300,
                    title="Distribuição por Hospital Executante"
                )
                st.altair_chart(chart_hosp, use_container_width=True)
                
        # Nova Linha: Mapa de Geolocalização
        st.write("---")
        st.subheader("📍 Geolocalização & Impacto por Hospital em Canoas")
        
        # Preparação dos dados para o mapa e painel
        hosp_data = []
        for cnes, info in CNES_MAP.items():
            coords_str = info.get("coordinates", "")
            if coords_str:
                try:
                    lat_str, lon_str = coords_str.split(",")
                    lat = float(lat_str.strip())
                    lon = float(lon_str.strip())
                    
                    # Contagem de internações e mortes para este hospital
                    hosp_cases = df_sus_filtered[df_sus_filtered['CNES'] == cnes]
                    count_internacoes = hosp_cases.shape[0]
                    count_mortes = hosp_cases[hosp_cases['MORTE'] == 'Sim'].shape[0]
                    letalidade = (count_mortes / count_internacoes * 100) if count_internacoes > 0 else 0.0
                    
                    hosp_data.append({
                        "Hospital": info["name"],
                        "latitude": lat,
                        "longitude": lon,
                        "internacoes": count_internacoes,
                        "mortes": count_mortes,
                        "letalidade": letalidade
                    })
                except Exception:
                    pass
        
        df_map = pd.DataFrame(hosp_data)
        
        if not df_map.empty and df_map['internacoes'].sum() > 0:
            col_map, col_info = st.columns([2, 1])
            
            with col_map:
                import pydeck as pdk
                
                # Normalizar os tamanhos dos círculos (raio em metros)
                max_cases = df_map['internacoes'].max()
                df_map['radius'] = df_map['internacoes'].apply(lambda x: (x / max_cases) * 600 + 150 if max_cases > 0 else 150)
                
                # Coordenadas médias para centralizar o mapa em Canoas
                avg_lat = df_map['latitude'].mean()
                avg_lon = df_map['longitude'].mean()
                
                view_state = pdk.ViewState(
                    latitude=avg_lat,
                    longitude=avg_lon,
                    zoom=11.5,
                    pitch=0
                )
                
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    df_map,
                    get_position="[longitude, latitude]",
                    get_color="[29, 53, 87, 180]", # Azul escuro
                    get_radius="radius",
                    pickable=True,
                    filled=True,
                    stroked=True,
                    line_width_min_pixels=1.5,
                    get_line_color=[230, 57, 70, 255] # Borda vermelha
                )
                
                tooltip = {
                    "html": "<b>{Hospital}</b><br/>Internações: <b>{internacoes}</b><br/>Óbitos: <b>{mortes}</b>",
                    "style": {"backgroundColor": "#1d3557", "color": "white", "fontSize": "12px"}
                }
                
                st.pydeck_chart(pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    tooltip=tooltip,
                    map_style="light"
                ))
                
            with col_info:
                st.markdown("### Distribuição Geográfica")
                st.write("O mapa ao lado destaca a distribuição espacial das internações. Círculos maiores indicam maior volume de admissões no período selecionado.")
                
                # Tabela estilizada de resumo
                total_hosp_period = df_map['internacoes'].sum()
                
                for idx, row in df_map.sort_values('internacoes', ascending=False).iterrows():
                    pct = (row['internacoes'] / total_hosp_period * 100) if total_hosp_period > 0 else 0.0
                    st.markdown(f"""
                    **{row['Hospital']}**  
                    * 📊 Internações: `{row['internacoes']:,}` ({pct:.1f}%)  
                    * 💀 Letalidade: `{row['letalidade']:.1f}%` ({row['mortes']} óbitos)  
                    ---
                    """)
        else:
            st.info("Nenhuma internação registrada para os hospitais de Canoas no período selecionado.")
