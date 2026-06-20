import pandas as pd
import numpy as np
def dataCleaningSus(df):
    '''
    Função de data cleaning para dados do SUS. Recebe tabela crua e retorna tabela pronta para analise de dados.
    - Filtra dados apenas pelo municipio de canoas
    - Seleciona colunas de interesse
    - Transforma data em datetime
    - Cria coluna MES_ANO para possiveis gráficos
    '''
    df_sus = df.copy()
    # 430460 = Canoas
    df_sus = df_sus[df_sus['MUNIC_MOV'] == 430460] 

    # Apenas colunas de interesse
    df_sus = df_sus[['DT_INTER', 'MES_CMPT', 'ANO_CMPT', 'IDADE', 'SEXO', 'MORTE', 'PROC_REA', 'QT_DIARIAS', 'CNES']]
    
    # Correção: convertendo usando a variável correta (df_sus)
    df_sus['data'] = pd.to_datetime(df_sus['DT_INTER'], format="%Y-%m-%d")
    df_sus.drop(columns=['DT_INTER'], inplace=True)

    df_sus['MES_ANO']=df_sus['MES_CMPT'].astype(str) + '/' + df_sus['ANO_CMPT'].astype(str)
    df_sus['MES_ANO']=pd.to_datetime(df_sus['MES_ANO'], format="%m/%Y")
    return df_sus


def dataCleaningFepam(df):
    '''
    Função de data cleaning para dados do FEPAM. Recebe tabela crua e retorna tabela pronta para analise de dados.
    - Transforma data em datetime
    '''
    df_fepam = df.copy()
    df_fepam['data'] = pd.to_datetime(df_fepam['data'])
    return df_fepam

def dataCleaningInmet(df):
    '''
    Função de data cleaning para dados do INMET. Recebe tabela crua e retorna tabela pronta para analise de dados.
    - Transforma data em datetime convertendo para UTC-3 (era UTC)
    - Seleciona colunas de interesse
    '''
    df_inmet = df.copy()

    # Cria coluna de data com UTC-3
    df_inmet['hora_brasil'] = df_inmet.hora.str.slice(0,2).astype(int) - 3
    df_inmet['data'] = pd.to_datetime(df_inmet['data'], format='%Y/%m/%d')
    df_inmet['data'] = df_inmet['data'] + pd.to_timedelta(df_inmet['hora_brasil'], unit='h')

    df_inmet.drop(columns=['hora_brasil', 'hora'], inplace=True)

    df_inmet = df_inmet[df_inmet['data'] >= '2020-01-01'] # Remove as primeiras tres entradas (primeiras 3 horas que ficaram no ano passado)
    df_inmet = df_inmet[['data', 'precipitacao', 'pressao_atm', 'temperatura_ar', 'umidade_rel']] # Apenas colunas de interesse
    return df_inmet

def aggregateSusByDay(df_sus):
    '''
    Função que agrupa os dados do SUS por dia para obter o VOLUME diário de internações e mortes.
    '''
    # Conta o número de linhas (internações) por dia

    df_sus_day = df_sus.groupby('data').size().reset_index(name='volume_internacoes')
    df_mortes = df_sus[df_sus['MORTE']=='Sim']
    df_sus_mortes_day = df_mortes.groupby('data').size().reset_index(name='volume_mortes')
    df_sus_day = pd.merge(df_sus_day, df_sus_mortes_day, on='data')
    return df_sus_day

def aggregateFepamByDay(df):
    '''
    Função que agrupa os dados do FEPAM por dia calculando a média das métricas.
    '''
    df_fepam = df.copy()
    # Correção: Mantendo o tipo como datetime64 normalizando para o início do dia
    df_fepam['data_dia'] = df_fepam['data'].dt.normalize()

    metrics_to_aggregate = ['pm10', 'so2', 'no2', 'o3', 'co']
    agg_dict = {metric: ['mean'] for metric in metrics_to_aggregate}

    df_fepam_day = df_fepam.groupby('data_dia').agg(agg_dict)
    df_fepam_day.columns = ['_'.join(col).strip() for col in df_fepam_day.columns.values]
    df_fepam_day.reset_index(inplace=True)
    df_fepam_day.rename(columns={'data_dia': 'data'}, inplace=True)

    df_fepam_day = generateAirIndexFepam(df_fepam)
    return df_fepam_day.round(2)

def aggregateInmetByDay(df):
    '''
    Função que agrupa os dados do INMET por dia usando a média das métricas.
    '''
    df_inmet = df.copy()
    # Correção: Mantendo o tipo como datetime64
    df_inmet['data_dia'] = df_inmet['data'].dt.normalize()
    
    metrics_to_aggregate = ['precipitacao', 'pressao_atm', 'temperatura_ar', 'umidade_rel']
    agg_dict = {metric: ['mean'] for metric in metrics_to_aggregate}

    df_inmet_day = df_inmet.groupby('data_dia').agg(agg_dict)
    df_inmet_day.columns = ['_'.join(col).strip() for col in df_inmet_day.columns.values]
    df_inmet_day.reset_index(inplace=True)
    df_inmet_day.rename(columns={'data_dia': 'data'}, inplace=True)

    df_inmet_day = generateTemperatureClassificationInmet(df_inmet_day)
    return df_inmet_day.round(2)

def joinTablesByDate(df_sus_day, df_fepam_day, df_inmet_day):
    # Usando how='left' garante que manteremos os dias mesmo se não houver internações (volume = NaN, que podemos tratar como 0)
    df_aggr = pd.merge(df_inmet_day, df_fepam_day, on='data', how='inner')
    df_aggr = pd.merge(df_aggr, df_sus_day, on='data', how='left')
    
    # Dias sem registro de internação passam a ser volume 0
    df_aggr['volume_internacoes'] = df_aggr['volume_internacoes'].fillna(0).astype(int)
    df_aggr['volume_mortes'] = df_aggr['volume_mortes'].fillna(0).astype(int)
    return df_aggr


def generateAirIndexFepam(df_input):
    '''
    Função que gera o índice de qualidade do ar baseado no 
    '''
    df_clean = df_input.copy()
    df_clean['data'] = pd.to_datetime(df_clean['data'])
    
    # Remove duplicatas de data/hora e ordena cronologicamente
    df_clean = df_clean.drop_duplicates(subset=['data']).sort_values('data').set_index('data')
    
    # 1. Calcular médias móveis de 8h para O3 e CO (conforme a norma)
    df_clean['o3_roll8h'] = df_clean['o3'].rolling('8h', min_periods=1).mean()
    df_clean['co_roll8h'] = df_clean['co'].rolling('8h', min_periods=1).mean()
    
    # 2. Agregação diária necessária para o cálculo do índice
    df_daily_metrics = pd.DataFrame()
    df_daily_metrics['pm10_val'] = df_clean['pm10'].resample('D').mean()
    df_daily_metrics['so2_val'] = df_clean['so2'].resample('D').mean()
    df_daily_metrics['no2_val'] = df_clean['no2'].resample('D').max()
    df_daily_metrics['o3_val'] = df_clean['o3_roll8h'].resample('D').max()
    df_daily_metrics['co_val'] = df_clean['co_roll8h'].resample('D').max()
    
    # 3. Valores médios tradicionais pedidos (com round(2))
    df_daily_means = pd.DataFrame()
    df_daily_means['pm10_mean'] = df_clean['pm10'].resample('D').mean().round(2)
    df_daily_means['so2_mean'] = df_clean['so2'].resample('D').mean().round(2)
    df_daily_means['no2_mean'] = df_clean['no2'].resample('D').mean().round(2)
    df_daily_means['o3_mean'] = df_clean['o3'].resample('D').mean().round(2)
    df_daily_means['co_mean'] = df_clean['co'].resample('D').mean().round(2)
    
    # Definição das faixas de transição e índices (C_ini, C_fin, I_ini, I_fin)
    pm10_ranges = [(0, 45, 0, 40), (46, 100, 41, 80), (101, 150, 81, 120), (151, 250, 121, 200), (251, 600, 201, 400)]
    o3_ranges = [(0, 100, 0, 40), (101, 130, 41, 80), (131, 160, 81, 120), (161, 200, 121, 200), (201, 800, 201, 400)]
    co_ranges = [(0.0, 9.0, 0, 40), (9.1, 11.0, 41, 80), (11.1, 13.0, 81, 120), (13.1, 15.0, 121, 200), (15.1, 50.0, 201, 400)]
    no2_ranges = [(0, 200, 0, 40), (201, 240, 41, 80), (241, 320, 81, 120), (321, 1130, 121, 200), (1131, 3750, 201, 400)]
    so2_ranges = [(0, 40, 0, 40), (41, 50, 41, 80), (51, 125, 81, 120), (126, 800, 121, 200), (801, 2620, 201, 400)]
    
    def calc_iqar_pollutant(c, ranges):
        if pd.isna(c):
            return np.nan
        for r in ranges:
            c_ini, c_fin, i_ini, i_fin = r
            if c_ini <= c <= c_fin:
                if c_fin == c_ini:
                    return i_ini
                return i_ini + ((i_fin - i_ini) / (c_fin - c_ini)) * (c - c_ini)
        # Extrapolação para casos de poluição extrema fora da tabela básica
        c_ini, c_fin, i_ini, i_fin = ranges[-1]
        if c > c_fin:
            return i_ini + ((i_fin - i_ini) / (c_fin - c_ini)) * (c - c_ini)
        return np.nan

    def classificar_qualidade(iqar):
        if pd.isna(iqar):
            return "Sem Dados"
        elif iqar <= 40:
            return "Boa"
        elif iqar <= 80:
            return "Moderada"
        elif iqar <= 120:
            return "Ruim"
        elif iqar <= 200:
            return "Muito Ruim"
        else:
            return "Péssima"

    iqar_list = []
    for idx, row in df_daily_metrics.iterrows():
        # Arredondamento banker's (padrão do Python) conforme a nota técnica
        c_pm10 = round(row['pm10_val']) if not pd.isna(row['pm10_val']) else np.nan
        c_so2 = round(row['so2_val']) if not pd.isna(row['so2_val']) else np.nan
        c_no2 = round(row['no2_val']) if not pd.isna(row['no2_val']) else np.nan
        c_o3 = round(row['o3_val']) if not pd.isna(row['o3_val']) else np.nan
        c_co = round(row['co_val'], 1) if not pd.isna(row['co_val']) else np.nan
        
        # Calcula o índice individual
        i_pm10 = calc_iqar_pollutant(c_pm10, pm10_ranges)
        i_so2 = calc_iqar_pollutant(c_so2, so2_ranges)
        i_no2 = calc_iqar_pollutant(c_no2, no2_ranges)
        i_o3 = calc_iqar_pollutant(c_o3, o3_ranges)
        i_co = calc_iqar_pollutant(c_co, co_ranges)
        
        valid_indices = [i for i in [i_pm10, i_so2, i_no2, i_o3, i_co] if not pd.isna(i)]
        
        if valid_indices:
            # O IQAr final é o maior índice arredondado para o inteiro mais próximo
            iqar_list.append(int(round(max(valid_indices))))
        else:
            iqar_list.append(np.nan)
            
    df_final = df_daily_means.copy()
    df_final['IQAr'] = iqar_list
    
    # Nova coluna mapeando a classificação com base no valor do IQAr final
    df_final['qualidade_do_ar'] = df_final['IQAr'].apply(classificar_qualidade)
    
    df_final = df_final.reset_index()
    return df_final


def generateTemperatureClassificationInmet(df):
    """
    Cria uma coluna categórica baseada na temperatura média do ar
    para facilitar análises de distribuição (como Boxplots).
    """
    df_temp = df.copy()
    
    # Definição dos limites das faixas (bins) e seus respectivos nomes (labels)
    # Usamos -inf e inf para garantir que nenhum valor fora dos limites seja perdido
    bins = [-float('inf'), 15, 22, 28, float('inf')]
    labels = ['Frio (<15°C)', 'Ameno (15-22°C)', 'Quente (22-28°C)', 'Muito Quente (>28°C)']
    
    # Criação da nova coluna usando pd.cut
    df_temp['faixa_temperatura'] = pd.cut(
        df_temp['temperatura_ar_mean'], 
        bins=bins, 
        labels=labels, 
        right=False  # O limite inferior é inclusivo e o superior é exclusivo (ex: [15, 22))
    )
    
    return df_temp