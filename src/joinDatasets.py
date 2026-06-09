import pandas as pd
from pathlib import Path
from . import config

def joinInmet():

    main_df = pd.DataFrame()

    for y in config.years:
        dir = f'{config.raw_old_dir}/INMET/{y}'

        file_name = [f.name for f in Path(dir).iterdir() if f.is_file()]
        file_name = file_name[0]

        new_df = pd.read_csv(f'{dir}/{file_name}', decimal=',', encoding='latin-1', sep=';')

        main_df = pd.concat([main_df, new_df], ignore_index=True)

    main_df = main_df.drop(columns=['Unnamed: 19'])

    main_df.columns = renameInmet()

    main_df.to_csv(f'{config.raw_data_dir}/inmet.csv', index=False)

def joinFepam():
    main_df = pd.DataFrame()

    for y in config.years:
        new_df = pd.read_excel(f'{config.raw_old_dir}/FEPAM/{y}.xlsx', sheet_name='005A-Canoas P Universitario')

        new_df = new_df.rename(columns={'co ': 'co'})

        new_df = new_df.drop(new_df.index[0])

        main_df = pd.concat([main_df, new_df], ignore_index=True)

    main_df.to_csv(f'{config.raw_data_dir}/fepam.csv', index=False)

def joinSus():
    main_df = pd.DataFrame()

    for y in config.years:
        new_df = pd.read_csv(f'{config.raw_old_dir}/SUS/sih_respiratorio_canoas_{y}.csv')

        main_df = pd.concat([main_df, new_df], ignore_index=True)

    main_df.to_csv(f'{config.raw_data_dir}/sus.csv', index=False)

def renameInmet():
    rename = [
        "data",
        "hora",
        "precipitacao", 
        "pressao_atm",
        "pressao_max_hora_ant",
        "pressao_min_hora_ant",
        "radiacao",
        "temperatura_ar",
        "temperatura_orvalho",
        "temperatura_max_hora_ant",
        "temperatura_min_hora_ant",
        "temperatura_orvalho_max_hora_ant",
        "temperatura_orvalho_min_hora_ant",
        "umidade_rel_max_hora_ant",
        "umidade_rel_min_hora_ant",
        "umidade_rel",
        "vento_direcao",
        "vento_rajada_max",
        "vento_velocidade"
    ]
    return rename

if __name__ == "__main__":
    joinInmet()
    joinFepam()
    joinSus()