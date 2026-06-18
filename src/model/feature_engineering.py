import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import RobustScaler, MinMaxScaler

def featureEng(df):
    df_hybrid = df.copy()
    cols_poluicao = ['pm10', 'so2', 'no2', 'o3', 'co']
    df_hybrid[cols_poluicao] = df_hybrid[cols_poluicao].interpolate(method='linear', limit=5)
    
    df_hybrid['data'] = pd.to_datetime(df_hybrid['data'], format='%Y-%m-%d')
    # Criando features temporais essenciais
    df_hybrid['mes'] = df_hybrid['data'].dt.month
    df_hybrid['dia_semana'] = df_hybrid['data'].dt.dayofweek
    df_hybrid['fim_de_semana'] = df_hybrid['dia_semana'].apply(lambda x: 1 if x >= 5 else 0)

    # Opcional (mas recomendado): Estações do ano
    df_hybrid['inverno'] = df_hybrid['mes'].apply(lambda x: 1 if x in [6, 7, 8] else 0)

    # Certifique-se de que o dataframe está ordenado por data antes de fazer o shift!
    df_hybrid = df_hybrid.sort_values('data')
    
    return df_hybrid


def createSplit(df):
    X = df.drop(columns=['internacoes', 'data', 'Unnamed: 0',
                            'pm10_min', 'so2_min', 'no2_min', 'o3_min', 'co_min', 'pm10_max', 'so2_max', 'no2_max', 'o3_max', 'co_max', 'co'])
    y = df['internacoes']
    datas = df['data'] # Guardamos as datas para depois

    # 2. O CORTE TEMPORAL (Garantindo que não há Data Leakage)
    # Treino: 2020 até 2023 | Teste: 2024
    limite_tempo = '2024-01-01'

    X_train = X[datas < limite_tempo]
    y_train = y[datas < limite_tempo]

    X_test = X[datas >= limite_tempo]
    y_test = y[datas >= limite_tempo]
    
    return X_train, y_train, X_test, y_test


def createPipeline(train_columns, model):
    cols_minmax = [
        'temperatura_ar', 'temperatura_orvalho', 'temperatura_min',
        'temperatura_max', 'umidade_rel', 'vento_direcao'
    ]

    # Group 2: Variables with massive spikes or heavy outliers (Robust)
    # We use a list comprehension to easily grab all the pollutants and remaining columns
    cols_robust = [col for col in train_columns if col not in cols_minmax]
    print(cols_robust)

    # --- STEP 3: Build the Preprocessor ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('minmax', MinMaxScaler(), cols_minmax),
            ('robust', RobustScaler(), cols_robust)
        ],
        remainder='drop' # Drops anything we forgot (though we caught everything)
    )
    
    cria_lags_transformer = FunctionTransformer(gerador_de_lags)
    
    pipeline_completo = Pipeline([
        ('scaling', preprocessor),
        ('imputer', IterativeImputer(max_iter=30, random_state=42)),
        ('lags', cria_lags_transformer),
        ('model', model)
    ])
    
    return pipeline_completo


def createPreprocessingPipeline(train_columns):
    cols_minmax = [
        'temperatura_ar', 'temperatura_orvalho', 'temperatura_min',
        'temperatura_max', 'umidade_rel', 'vento_direcao'
    ]
    cols_robust = [col for col in train_columns if col not in cols_minmax]

    preprocessor = ColumnTransformer(
        transformers=[
            ('minmax', MinMaxScaler(), cols_minmax),
            ('robust', RobustScaler(), cols_robust)
        ],
        remainder='drop',
        verbose_feature_names_out=False  # ← avoids 'minmax__temperatura_ar' prefixes
    )

    pipeline = Pipeline([
        ('scaling', preprocessor),
        ('imputer', IterativeImputer(max_iter=30, random_state=42)),
        ('lags', FunctionTransformer(gerador_de_lags)),
    ])
    pipeline.set_output(transform="pandas")

    return pipeline


def gerador_de_lags(df_out, features_para_lag=None, n_lags=3):    
    # ← Handle numpy arrays passed by sklearn pipeline
    if isinstance(df_out, np.ndarray):
        raise ValueError(
            "gerador_de_lags received a numpy array. "
            "Call pipeline.set_output(transform='pandas') to fix this."
        )

    if features_para_lag is None:
        features_para_lag = ['temperatura_min', 'umidade_rel', 'pm10']

    for col in features_para_lag:
        if col in df_out.columns:
            for dias_atras in range(1, n_lags + 1):
                df_out[f'{col}_lag{dias_atras}'] = df_out[col].shift(dias_atras)

    return df_out