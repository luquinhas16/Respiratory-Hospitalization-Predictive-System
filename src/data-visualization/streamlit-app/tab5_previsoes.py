import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pickle
import torch
import sys
from pathlib import Path

# Ensure paths are added correctly
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / "src"))
sys.path.append(str(root_dir / "src" / "model"))


import feature_engineering
from model import LSTMModel, device
from dataset.dataset_construction import TimeSeriesDataset

@st.cache_resource
def load_prediction_model():
    model_dir = root_dir / "src" / "model"
    preprocessor_path = model_dir / "preprocessor.pkl"
    model_path = model_dir / "lstm_model.pt"
    
    # 1. Load preprocessor
    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)
        
    # 2. Initialize LSTMModel
    # The preprocessor fits 28 features, and the dataset has 1 target = 29 columns
    # TimeSeriesDataset now automatically excludes the target column from the inputs, leaving 28 features
    model = LSTMModel(
        input_size=28,
        hidden_size=64,
        num_layers=2,
        output_size=1,
        label_width=1,
        dropout=0.1
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, preprocessor

@st.cache_data
def get_predictions():
    # Load model and preprocessor
    model, preprocessor = load_prediction_model()
    
    # Load dataset.csv
    dataset_path = root_dir / "src" / "model" / "dataset" / "dataset.csv"
    df = pd.read_csv(dataset_path)
    
    # Run feature engineering
    df_hybrid = feature_engineering.featureEng(df)
    
    # Split into train and test sets matching the notebook split
    X_train, y_train, X_test, y_test = feature_engineering.createSplit(df_hybrid)
    datas_test = df_hybrid['data'].loc[X_test.index]
    
    # Preprocess the test set separately, matching the notebook pipeline behavior
    X_test_processed = preprocessor.transform(X_test)
    
    # Drop the NaN rows created by lag shifts
    X_test_processed = X_test_processed.dropna()
    y_test = y_test.loc[X_test_processed.index]
    datas_test = datas_test.loc[X_test_processed.index]
    
    test_df = pd.concat([X_test_processed, y_test], axis=1)
    
    # TimeSeriesDataset
    dataset = TimeSeriesDataset(test_df, input_width=7, label_width=1, shift=1, label_columns=['internacoes'])
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Predict
    previsoes_lstm = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            preds = model(inputs)
            previsoes_lstm.extend(preds.view(-1).cpu().numpy())
            
    # Align predictions (length len(test_df) - 7) with datas_test.iloc[7:]
    df_preds = pd.DataFrame({
        'data': pd.to_datetime(datas_test.iloc[7:]),
        'Real': y_test.iloc[7:].values,
        'Previsto': previsoes_lstm
    })
    
    return df_preds

def render_tab_previsoes(tab, start_date, end_date, window_size):
    with tab:
        st.subheader("📈 Previsões de Internações (LSTM)")
        
        st.markdown("""
        > [!NOTE]
        > O modelo de rede neural **LSTM (Long Short-Term Memory)** foi treinado utilizando os dados históricos de **2020 a 2023**. 
        > As previsões abaixo representam a validação temporal do modelo para o ano de **2024** (período não visto no treinamento).
        """)
        
        # Load predictions DataFrame
        try:
            df_preds = get_predictions()
        except Exception as e:
            st.error(f"Erro ao carregar o modelo de previsão: {e}")
            st.info("Verifique se executou o treinamento do modelo e se os arquivos lstm_model.pt e preprocessor.pkl estão em src/model/.")
            return
            
        # Filter predictions based on dates in 2024
        # Since predictions are only for the test set (2024), we filter to 2024
        df_preds_2024 = df_preds[(df_preds['data'].dt.date >= pd.to_datetime('2024-01-01').date()) & 
                                 (df_preds['data'].dt.date <= pd.to_datetime('2024-12-31').date())].copy()
        
        # Further filter using user's sidebar selection
        df_chart = df_preds_2024[(df_preds_2024['data'].dt.date >= start_date) & 
                                 (df_preds_2024['data'].dt.date <= end_date)].copy()
        
        if df_chart.empty:
            st.warning("Selecione um período temporal em **2024** na barra lateral para visualizar as previsões do modelo.")
            return
            
        # Compute metrics for the selected visible period
        real_vals = df_chart['Real'].values
        pred_vals = df_chart['Previsto'].values
        
        mae = np.mean(np.abs(real_vals - pred_vals))
        rmse = np.sqrt(np.mean((real_vals - pred_vals)**2))
        
        # Simple R2 calculation
        ss_res = np.sum((real_vals - pred_vals)**2)
        ss_tot = np.sum((real_vals - np.mean(real_vals))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Render Metrics KPIs
        st.markdown("### Métricas de Desempenho no Período Selecionado")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric(label="Erro Médio Absoluto (MAE)", value=f"{mae:.2f} internações")
        with col_m2:
            st.metric(label="Raiz do Erro Quadrático Médio (RMSE)", value=f"{rmse:.2f} internações")
        with col_m3:
            st.metric(label="Coeficiente de Determinação (R²)", value=f"{r2:.2f}")
            
        st.write("---")
        
        # Apply smoothing (rolling average) based on window_size, matching the notebook's rolling behavior
        df_smooth = df_chart.copy().sort_values('data')
        df_smooth['Real_Smooth'] = df_smooth['Real'].rolling(window=window_size, min_periods=1).mean()
        df_smooth['Previsto_Smooth'] = df_smooth['Previsto'].rolling(window=window_size, min_periods=1).mean()
        
        if df_smooth.empty:
            st.warning("Dados insuficientes no período selecionado para plotar com a média móvel configurada.")
            return
            
        # Add a consecutive day index starting at 1, matching the notebook plot X-axis behavior
        df_smooth['Dia'] = np.arange(len(df_smooth)) + 1
        
        st.subheader(f"📊 Comparativo de Ocupação Hospitalar: Real vs. Previsto (Média Móvel de {window_size} dias)")
        
        # Melt data for Altair charting, including 'Dia' in id_vars
        df_melted = pd.melt(
            df_smooth,
            id_vars=['data', 'Dia'],
            value_vars=['Real_Smooth', 'Previsto_Smooth'],
            var_name='Tipo',
            value_name='Internacoes'
        )
        
        # Translate legend names
        df_melted['Tipo'] = df_melted['Tipo'].replace({
            'Real_Smooth': 'Real (Fato)',
            'Previsto_Smooth': 'Previsto pelo Modelo LSTM'
        })
        
        zoom_interativo = alt.selection_interval(bind='scales', encodings=['x'])
        
        # Chart - plotting against the consecutive day 'Dia' index to avoid gap spacing issues
        chart = alt.Chart(df_melted).mark_line(strokeWidth=2.5).encode(
            x=alt.X('Dia:Q', title='Linha do Tempo (Dia do Período - Consecutivo)'),
            y=alt.Y('Internacoes:Q', title=f'Volume Diário de Internações (Média Móvel {window_size}d)'),
            color=alt.Color('Tipo:N', scale=alt.Scale(domain=['Real (Fato)', 'Previsto pelo Modelo LSTM'], range=['#1f77b4', '#ff7f0e']), title='Legenda'),
            strokeDash=alt.condition(
                alt.datum.Tipo == 'Previsto pelo Modelo LSTM',
                alt.value([5, 5]), # dashed line for prediction
                alt.value([0]) # solid line for real
            ),
            tooltip=[
                alt.Tooltip('Dia:Q', title='Dia do Período'),
                alt.Tooltip('data:T', title='Data', format='%d/%m/%Y'),
                alt.Tooltip('Tipo:N', title='Tipo'),
                alt.Tooltip('Internacoes:Q', title='Internações (Média)', format='.2f')
            ]
        ).properties(
            height=400,
            title={
                "text": "Série Temporal de Validação: Ocupação Hospitalar Real vs. LSTM",
                "fontSize": 15,
                "subtitleFontSize": 11
            }
        ).add_params(
            zoom_interativo
        )
        
        st.altair_chart(chart, use_container_width=True)
