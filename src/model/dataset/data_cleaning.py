import pandas as pd
from .. import config

def clean():
    raw = pd.read_csv(f'{config.raw_data_dir}/inmet.csv', decimal=',', encoding='latin-1')
    raw = raw.drop(columns=["Hora UTC"])

    print(raw)

    dataset = pd.DataFrame(columns=raw.columns)

    avg = min_ = max_ = []

    for date in raw['Data'].to_list():
        avg = raw[raw['Data'] == date].mean()
        dataset.loc[len(dataset)] = avg
        print(dataset)
        raise

if __name__ == "__main__":
    clean()