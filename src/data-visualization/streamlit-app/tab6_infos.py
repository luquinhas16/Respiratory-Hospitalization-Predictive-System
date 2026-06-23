import streamlit as st

def render_tab_infos(tab):
    with tab:
        st.subheader("ℹ️ Sobre o Projeto")
        st.write("Projeto desenvolvido pela equipe Guri Teimoso para a disciplina de Data Science.")
        st.write("Objetivo: Analisar os impactos da poluição do ar e condições meteorológicas nas hospitalizações respiratórias em Canoas, RS.")
