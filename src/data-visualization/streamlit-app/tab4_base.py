import streamlit as st
import pandas as pd
from config import CORR_COLS, FRIENDLY_NAMES

def render_tab_base(tab, df_agg_filtered):
    with tab:
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
        desc_stats = df_agg_filtered[CORR_COLS].describe().T
        desc_stats = desc_stats.rename(index=FRIENDLY_NAMES)
        st.dataframe(desc_stats, use_container_width=True)
