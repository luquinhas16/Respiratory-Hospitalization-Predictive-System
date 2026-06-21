import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torchmetrics.functional import r2_score
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import TimeSeriesSplit
from dataset.dataset_construction import TimeSeriesDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def crossValidate(X, y, hyperparameters, preprocessor, train_params, n_splits=5, patience=10):
    criterion = nn.MSELoss()
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    cv_history = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        
        train_loader, val_loader = preprocessFold(preprocessor, train_params['batch_size'], X_train_fold, y_train_fold, X_val_fold, y_val_fold)

        model = LSTMModel(
            input_size=hyperparameters['input_size'],
            hidden_size=hyperparameters['hidden_size'],
            num_layers=hyperparameters['num_layers'],
            output_size=hyperparameters['output_size'],
            label_width=hyperparameters['label_width'],
            dropout=hyperparameters['dropout']
        )
        model = model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=train_params['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        fold_history = {'train_loss': [], 'val_mse': [], 'val_mae': [], 'val_rmse': [], 'val_r2': []}
        best_val_loss = np.inf
        epochs_without_improvement = 0

        for epoch in range(train_params['epochs']):
            model.train()
            train_losses = []

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            metrics    = evaluate(model, val_loader, device)
            val_loss   = metrics['mse']

            scheduler.step(val_loss)
            fold_history['train_loss'].append(train_loss)
            fold_history['val_mse'].append(metrics['mse'])
            fold_history['val_mae'].append(metrics['mae'])
            fold_history['val_rmse'].append(metrics['rmse'])
            fold_history['val_r2'].append(metrics['r2'])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break

        cv_history.append(fold_history)

    return cv_history


def preprocessFold(preprocessor, batch_size, X_train, y_train, X_val, y_val):
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed  = preprocessor.transform(X_val)
    
    X_train_processed = X_train_processed.dropna()
    y_train = y_train.loc[X_train_processed.index]

    X_val_processed = X_val_processed.dropna()
    y_val = y_val.loc[X_val_processed.index]

    train_df = pd.concat([X_train_processed, y_train], axis=1)
    val_df  = pd.concat([X_val_processed,  y_val],  axis=1)

    train_dataset = TimeSeriesDataset(train_df, input_width=7, label_width=1, shift=1, label_columns=['internacoes'])
    val_dataset = TimeSeriesDataset(val_df, input_width=7, label_width=1, shift=1, label_columns=['internacoes'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def train(model, train_loader, hyperparameters):
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['lr'])
    criterion = nn.MSELoss()

    history = {'train_loss': []}
    for epoch in range(hyperparameters['epochs']):
        model.train()
        train_losses = []
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
            
        train_loss = np.mean(train_losses)
        history['train_loss'].append(train_loss)

    return model, history


def evaluate(model, loader, device):
    model.eval()
    losses = []
    all_preds = []
    all_labels = []

    criterion = nn.MSELoss()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            preds  = model(inputs)
            loss   = criterion(preds, labels)
            losses.append(loss.item())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds  = torch.cat(all_preds,  dim=0)  # (n_samples, label_width, output_size)
    all_labels = torch.cat(all_labels, dim=0)

    mse = nn.MSELoss()(all_preds, all_labels).item()
    mae = nn.L1Loss()(all_preds, all_labels).item()
    rmse = np.sqrt(mse)
    loss = np.mean(losses)
    
    preds_flat = all_preds.view(-1)
    labels_flat = all_labels.view(-1)
    
    r2 = r2_score(preds_flat, labels_flat).item()
    
    return {'loss': loss, 'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2}

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, label_width, dropout=0.2):
        """
        Args:
            input_size  : number of input features (n_features in your DataFrame)
            hidden_size : number of LSTM units per layer (tunable hyperparameter)
            num_layers  : how many stacked LSTM layers (tunable hyperparameter)
            output_size : number of target columns (1 for 'internacoes')
            label_width : how many future timesteps to predict (1 for next-day)
            dropout     : dropout rate between LSTM layers (regularization)
        """
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.label_width = label_width
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,       # input shape: (batch, seq_len, features)
            dropout=dropout if num_layers > 1 else 0.0  # dropout only between layers
        )

        self.dropout = nn.Dropout(dropout)

        # Maps LSTM output to prediction
        self.fc = nn.Linear(hidden_size, label_width * output_size)

    def forward(self, x):
        # x shape: (batch, input_width, input_size)

        # Initialize hidden and cell states to zero
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        # out shape: (batch, input_width, hidden_size)

        # Take only the last timestep's output
        out = out[:, -1, :]
        # out shape: (batch, hidden_size)

        out = self.dropout(out)
        out = self.fc(out)
        # out shape: (batch, label_width * output_size)

        out = out.view(-1, self.label_width, self.output_size)
        # out shape: (batch, label_width, output_size)

        return out