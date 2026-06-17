import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import RobustScaler, MinMaxScaler

def featureEng(df):
    df_hybrid = df.copy()
    cols_poluicao = ['pm10', 'so2', 'no2', 'o3', 'co']
    df_hybrid[cols_poluicao] = df_hybrid[cols_poluicao].interpolate(method='linear', limit=3)
    
    df_hybrid['data'] = pd.to_datetime(df_hybrid['data'], format='%Y-%m-%d')
    # Criando features temporais essenciais
    df_hybrid['mes'] = df_hybrid['data'].dt.month
    df_hybrid['dia_semana'] = df_hybrid['data'].dt.dayofweek
    df_hybrid['fim_de_semana'] = df_hybrid['dia_semana'].apply(lambda x: 1 if x >= 5 else 0)

    # Opcional (mas recomendado): Estações do ano
    df_hybrid['inverno'] = df_hybrid['mes'].apply(lambda x: 1 if x in [6, 7, 8] else 0)

    # Certifique-se de que o dataframe está ordenado por data antes de fazer o shift!
    df_hybrid = df_hybrid.sort_values('data')

    # Vamos criar 'lags' (atrasos) de 1 a 3 dias para as variáveis mais importantes
    features_para_lag = ['temperatura_min', 'umidade_rel', 'pm10']

    for col in features_para_lag:
        for dias_atras in [1, 2, 3]:
            df_hybrid[f'{col}_lag{dias_atras}'] = df_hybrid[col].shift(dias_atras)

    # O comando shift cria valores nulos nas primeiras 3 linhas do dataset (pois não existe "anteontem" no dia 1)
    # Vamos dropar apenas essas 3 linhas para não quebrar o pipeline
    df_hybrid = df_hybrid.dropna(subset=[f'{col}_lag3' for col in features_para_lag])
    
    return df_hybrid
  
def scaleNsplitData(df):

    X = df.drop(columns=['internacoes', 'data', 'Unnamed: 0'])
    y = df['internacoes']
    datas = df['data']

    # ── Three-way temporal split ─────────────────────────────────────────────
    limite_train = '2023-01-01'
    limite_val   = '2024-01-01'

    X_train = X[datas < limite_train]
    y_train = y[datas < limite_train]

    X_val   = X[(datas >= limite_train) & (datas < limite_val)]
    y_val   = y[(datas >= limite_train) & (datas < limite_val)]

    X_test  = X[datas >= limite_val]
    y_test  = y[datas >= limite_val]

    # ── Scaling pipeline (unchanged) ─────────────────────────────────────────
    cols_minmax = [
        'temperatura_ar', 'temperatura_orvalho', 'temperatura_min',
        'temperatura_max', 'umidade_rel', 'vento_direcao'
    ]
    cols_robust = [col for col in X_train.columns if col not in cols_minmax]

    preprocessor = ColumnTransformer(
        transformers=[
            ('minmax', MinMaxScaler(), cols_minmax),
            ('robust', RobustScaler(), cols_robust)
        ],
        remainder='drop'
    )

    imputation_pipeline = Pipeline([
        ('scaling', preprocessor),
        ('imputer', IterativeImputer(random_state=42))
    ])
    imputation_pipeline.set_output(transform="pandas")

    # ── Fit ONLY on train, transform all three ───────────────────────────────
    X_train = imputation_pipeline.fit_transform(X_train)
    X_val   = imputation_pipeline.transform(X_val)
    X_test  = imputation_pipeline.transform(X_test)

    # ── Clean column names ───────────────────────────────────────────────────
    for X in [X_train, X_val, X_test]:
        X.columns = [col.split('__')[-1] for col in X.columns]

    # ── Reattach dates ───────────────────────────────────────────────────────
    datas_train = datas[datas < limite_train]
    datas_val   = datas[(datas >= limite_train) & (datas < limite_val)]
    datas_test  = datas[datas >= limite_val]

    X_train = pd.concat([datas_train, X_train], axis=1)
    X_val   = pd.concat([datas_val,   X_val],   axis=1)
    X_test  = pd.concat([datas_test,  X_test],  axis=1)

    return X_train, y_train, X_val, y_val, X_test, y_test