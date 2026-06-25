import torch
import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path

# Add root and model directories to sys.path to allow imports
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent))

import feature_engineering
from model import LSTMModel, train, evaluate, preprocessFold, device
from dataset.dataset_construction import TimeSeriesDataset

def run_training():
    print(f"Using device: {device}")
    
    # 1. Load dataset
    data_path = Path(__file__).parent / "dataset" / "dataset.csv"
    print(f"Loading data from {data_path}...")
    raw_df = pd.read_csv(data_path)
    
    # 2. Feature engineering
    print("Running feature engineering...")
    df_hybrid = feature_engineering.featureEng(raw_df)
    
    # 3. Create train/test split
    print("Splitting data...")
    X_train, y_train, X_test, y_test = feature_engineering.createSplit(df_hybrid)
    
    # 4. Create preprocessing pipeline
    print("Fitting preprocessing pipeline...")
    preprocessor = feature_engineering.createPreprocessingPipeline(X_train.columns)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save the fitted preprocessor using pickle
    preprocessor_path = Path(__file__).parent / "preprocessor.pkl"
    with open(preprocessor_path, "wb") as f:
        pickle.dump(preprocessor, f)
    print(f"Saved preprocessor to {preprocessor_path}")
    
    # 5. Clean processed data
    X_train_processed = X_train_processed.dropna()
    y_train = y_train.loc[X_train_processed.index]
    
    X_test_processed = X_test_processed.dropna()
    y_test = y_test.loc[X_test_processed.index]
    
    train_df = pd.concat([X_train_processed, y_train], axis=1)
    test_df = pd.concat([X_test_processed, y_test], axis=1)
    
    # 6. Create Datasets and Loaders
    print("Creating datasets...")
    train_dataset = TimeSeriesDataset(train_df, input_width=7, label_width=1, shift=1, label_columns=['internacoes'])
    test_dataset = TimeSeriesDataset(test_df, input_width=7, label_width=1, shift=1, label_columns=['internacoes'])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 7. Initialize model with best hyperparameters from Optuna
    n_features = train_df.shape[1]
    print(f"Number of input features: {n_features}")
    
    hyperparameters = {
        'input_size': n_features,
        'hidden_size': 256,
        'num_layers': 2,
        'output_size': 1,
        'label_width': 1,
        'dropout': 0.4
    }
    
    train_params = {
        'lr': 0.000307,
        'epochs': 30
    }
    
    model = LSTMModel(
        input_size=hyperparameters['input_size'],
        hidden_size=hyperparameters['hidden_size'],
        num_layers=hyperparameters['num_layers'],
        output_size=hyperparameters['output_size'],
        label_width=hyperparameters['label_width'],
        dropout=hyperparameters['dropout']
    )
    model = model.to(device)
    
    # 8. Train the model
    print("Starting training...")
    model, history = train(model, train_loader, train_params)
    print(f"Finished training. Final epoch loss: {history['train_loss'][-1]:.4f}")
    
    # 9. Save trained model weights
    model_path = Path(__file__).parent / "lstm_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model weights to {model_path}")
    
    # 10. Evaluate on test set
    metrics = evaluate(model, test_loader, device)
    print(f"Test Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    run_training()
