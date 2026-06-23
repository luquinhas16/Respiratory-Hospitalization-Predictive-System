import pandas as pd
from tqdm import tqdm
from src import config

def cleanInmet():
    raw = pd.read_csv(f'{config.raw_data_dir}/inmet.csv')
    
    cols_to_remove = ["hora", "pressao_max_hora_ant", "pressao_min_hora_ant", "temperatura_orvalho_max_hora_ant", "temperatura_orvalho_min_hora_ant", "umidade_rel_max_hora_ant", "umidade_rel_min_hora_ant"]

    raw = raw.drop(columns=cols_to_remove)

    raw["data"] = pd.to_datetime(raw["data"])
    
    clean_cols = raw.columns.to_list() + ["temperatura_min", "temperatura_max"]
    
    dataset = pd.DataFrame(columns=clean_cols)

    dates = list(dict.fromkeys(raw['data'].to_list()))

    for date in tqdm(dates, desc="Cleaning INMET data"):
        
        day_df = raw[raw['data'] == date]
        
        cols_avg = day_df.mean(numeric_only=True)
        min_temp = day_df['temperatura_max_hora_ant'].min()
        max_temp = day_df['temperatura_min_hora_ant'].max()
        
        new_row = [date] + cols_avg.to_list() + [min_temp, max_temp]
        
        dataset.loc[len(dataset)] = new_row
    
    dataset = dataset.drop(columns=["temperatura_max_hora_ant", "temperatura_min_hora_ant"])
    print(dataset)
    
    dataset.to_csv(f'{config.model_dataset_dir}/inmet.csv', index=False)

def cleanFepam():
    raw = pd.read_csv(f'{config.raw_data_dir}/fepam.csv')
    
    raw['data'] = pd.to_datetime(raw['data'])
    raw['data'] = raw['data'].dt.date
    
    min_cols = [f'{col}_min' for col in raw.columns[1:]]
    max_cols = [f'{col}_max' for col in raw.columns[1:]]
    
    clean_cols = raw.columns.to_list() + min_cols + max_cols
    
    dataset = pd.DataFrame(columns=clean_cols)
    
    dates = list(dict.fromkeys(raw['data'].to_list()))
    
    for date in tqdm(dates, desc="Cleaning FEPAM data"):
        day_df = raw[raw['data'] == date]
        
        cols_avg = day_df.mean(numeric_only=True).to_list()
        cols_min = day_df.min(numeric_only=True).to_list()
        cols_max = day_df.max(numeric_only=True).to_list()
        
        new_row = [date] + cols_avg + cols_min + cols_max
        
        dataset.loc[len(dataset)] = new_row
    
    print(dataset)
    
    dataset.to_csv(f'{config.model_dataset_dir}/fepam.csv', index=False)

def cleanSus():
    raw = pd.read_csv(f'{config.raw_data_dir}/sus.csv')
    raw = raw.rename(columns={"DT_INTER": "data"})
    raw['data'] = pd.to_datetime(raw['data'])
    
    dataset = pd.DataFrame(columns=["data","internacoes"])
    
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D').to_list()
    
    for date in tqdm(dates, desc="Cleaning SUS data"):
        count = (raw["data"] == date).sum()
        
        dataset.loc[len(dataset)] = [date, count]
    
    print(dataset)
    
    dataset.to_csv(f'{config.model_dataset_dir}/sus.csv', index=False)
    
def joinCleanDatasets():
    inmet = pd.read_csv(f'{config.model_dataset_dir}/inmet.csv')
    fepam = pd.read_csv(f'{config.model_dataset_dir}/fepam.csv')
    sus = pd.read_csv(f'{config.model_dataset_dir}/sus.csv')
    
    #verificação de datas concordantes
    
    full_dataset = pd.merge(inmet, fepam, on='data', how='inner')
    full_dataset = pd.merge(full_dataset, sus, on='data', how='inner')
    
    print(full_dataset)
    
    full_dataset.to_csv(f'{config.model_dataset_dir}/dataset.csv')
    

if __name__ == "__main__":
    cleanInmet()
    cleanFepam()
    cleanSus()
    
    joinCleanDatasets()