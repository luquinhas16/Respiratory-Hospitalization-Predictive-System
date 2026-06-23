import torch
import pandas as pd
from pathlib import Path
from . import feature_engineering
from .dataset import dataset_construction
from .model import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_dir = Path(__file__).parent

def loadDataset():
    data_path = script_dir / "dataset" / "dataset.csv"
    df = pd.read_csv(data_path)
    
    return df

if __name__ == "__main__":
    df = loadDataset()
    df = feature_engineering.featureEng(df)
    
    df_hybrid = df.drop(columns=['data', 'Unnamed: 0'])
    dates = df_hybrid['data']
    time_limit = '2024-01-01'
    train_df, test_df = df_hybrid[dates < time_limit], df_hybrid[dates >= time_limit]
    
    complete_pipeline = feature_engineering.pipeline(train_df.columns.list())
    
    X_train, y_train, X_test, y_test = feature_engineering.scaleNsplitData(df)
    
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_dataset = dataset_construction.TimeSeriesDataset(train_df, input_width=14, label_width=1, shift=1, label_columns=['internacoes'])
    train_loader, test_loader = dataset_construction.make_loaders(train_df, test_df)
        
    # Infer input_size from the loader
    inputs, labels = next(iter(train_loader))
    n_features = inputs.shape[2]
    
    hyperparameters = {
        'input_size': n_features,
        'hidden_size': 64,
        'num_layers': 2,
        'output_size': 1, # predicting 'internacoes' only
        'label_width': 1, # next day prediction
        'dropout': 0.1
    }

    model = LSTMModel(input_size=hyperparameters['input_size'], hidden_size=hyperparameters['hidden_size'], num_layers=hyperparameters['num_layers'], output_size=hyperparameters['output_size'], label_width=hyperparameters['label_width'], dropout=hyperparameters['dropout'])

    model, cv_history, final_history = train(
        model,
        dataset=train_dataset,
        hyperparameters=hyperparameters,
        n_splits=5,
        epochs=50,
        device=device
    )
    
    #result = evaluate(model, test_loader, criterion = nn.MSELoss(), device=device)
    #print(result)