from . import config
import pandas as pd

def joinInmet():

    main_df = pd.DataFrame()

    for y in config.years:
        new_df = pd.read_csv(f'{config.raw_old_dir}/INMET/{y}/INMET_S_RS_A801_PORTO ALEGRE - JARDIM BOTANICO_01-01-{y}_A_31-12-{y}.CSV', decimal=',', encoding='latin-1', sep=';')

        main_df = pd.concat([main_df, new_df], ignore_index=True)

    main_df = main_df.drop(columns=['Unnamed: 19'])

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

if __name__ == "__main__":
    joinInmet()
    joinFepam()
    joinSus()