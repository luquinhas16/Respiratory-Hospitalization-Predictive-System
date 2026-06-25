import streamlit as st


def render_tab_infos(tab):
    with tab:
        st.subheader("ℹ️ Sobre o Projeto & Fontes de Dados")
        
        st.markdown("""
        O projeto do time **Guri Teimoso** foi desenvolvido como trabalho da cadeira de Ciência de Dados (INF01090) em 2026/1 e analisou a correlação entre as condições ambientais (clima e qualidade do ar) e problemas de saúde pública (internações respiratórias) na cidade de **Canoas, RS**.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🛠️ O que foi feito")
            st.markdown("""
            * **Coleta e Consolidação de Dados**: Integração de séries temporais diárias abrangendo o período do início de **2020** ao final de **2024** das bases do SUS, INMET e FEPAM.
            * **Análise Exploratória e Demográfica**: Mapeamento georreferenciado e dinâmico dos principais hospitais executantes de Canoas, integrando dados socioeconômicos e faixas etárias dos pacientes internados.
            * **Modelagem Preditiva**: Desenvolvimento de uma rede neural recorrente **LSTM (Long Short-Term Memory)** em PyTorch para prever o volume diário de internações respiratórias futuras baseado nas variáveis climáticas e de poluição dos 7 dias anteriores.
            * **Dashboard Streamlit**: Modularização das análises em abas dinâmicas, permitindo filtros interativos de datas, suavização por média móvel e exportação de dados.
            """)
            
        with col2:
            st.markdown("### 📊 Fontes de Dados")
            st.markdown("""
            Os dados consolidados no dashboard provêm das seguintes fontes oficiais:
            
            * **SUS (SIH - Sistema de Informações Hospitalares) - coleta por servidor FTP do DATASUS**
              * Contém os registros de internações e óbitos em hospitais do município de Canoas.
              * **Website Oficial**: [Acessar site do SUS](https://datasus.saude.gov.br/informacoes-de-saude-tabnet/)
              
            * **INMET (Instituto Nacional de Meteorologia)**
              * Fornece dados climáticos horários (temperatura, umidade, pressão e precipitação). Foi utilizada a estação meteorológica de Porto Alegre (cidade vizinha), por ser a mais próxima com dados completos.
              * **Website Oficial**: [Acessar site do INMET](https://portal.inmet.gov.br/dadoshistoricos)
              
            * **FEPAM (Fundação Estadual de Proteção Ambiental)**
              * Fornece medições horárias da qualidade do ar e poluentes atmosféricos (PM10, SO2, NO2, O3, CO).
              * **Website Oficial**: [Acessar site da FEPAM](https://www.fepam.rs.gov.br/dados-do-monitoramento)
            """)
            
        st.write("---")
        
        with st.expander("🔬 Metodologia: Cálculo do Índice de Qualidade do Ar (IQAr)"):
            st.markdown(r"""
            O **IQAr (Índice de Qualidade do Ar)** foi calculado seguindo as diretrizes do **Guia Técnico de Orientação sobre Qualidade do Ar** do Ministério do Meio Ambiente e Mudança do Clima e a Resolução CONAMA 491/2018. 
            
            O índice consolida as concentrações de cinco poluentes principais monitorados pela FEPAM:
            * **Material Particulado (PM10)**: Média aritmética de 24 horas.
            * **Dióxido de Enxofre (SO2)**: Média aritmética de 24 horas.
            * **Dióxido de Nitrogênio (NO2)**: Valor máximo de 1 hora no dia.
            * **Ozônio (O3)**: Valor máximo da média móvel de 8 horas.
            * **Monóxido de Carbono (CO)**: Valor máximo da média móvel de 8 horas.
            
            #### 🧮 Fórmula de Interpolação Linear
            Para cada poluente, calcula-se o seu índice individual ($I$) a partir de sua concentração medida ($c$) correspondente à faixa de transição da tabela técnica:
            
            $$I = I_{ini} + \frac{I_{fin} - I_{ini}}{c_{fin} - c_{ini}} \cdot (c - c_{ini})$$
            
            Onde:
            * $c_{ini}$ e $c_{fin}$: limites de concentração inferior e superior da faixa onde se encontra $c$.
            * $I_{ini}$ e $I_{fin}$: limites de índice inferior e superior da faixa de transição.
            
            #### 🏆 Índice Global e Classificação
            O IQAr final diário corresponde ao **maior índice individual** entre todos os poluentes calculados para aquele dia. A classificação da qualidade segue a seguinte escala:
            * **Boa** ($0 \le IQAr \le 40$)
            * **Moderada** ($41 \le IQAr \le 80$)
            * **Ruim** ($81 \le IQAr \le 120$)
            * **Muito Ruim** ($121 \le IQAr \le 200$)
            * **Péssima** ($IQAr > 200$)
            """)
            
        st.write("---")

