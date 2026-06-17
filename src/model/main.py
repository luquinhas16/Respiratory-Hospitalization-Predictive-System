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
    X_train, y_train, X_val, y_val, X_test, y_test = feature_engineering.scaleNsplitData(df)
    
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val,   y_val],   axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_loader, val_loader, test_loader = dataset_construction.make_loaders(train_df, val_df, test_df)
        
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

    model, history = train(model, train_loader, val_loader, hyperparameters, device=device)
    
    result = evaluate(model, test_loader, criterion = nn.MSELoss(), device=device)
    print(result)