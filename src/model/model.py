import torch
import numpy as np
import torch.nn as nn
from torchmetrics.functional import r2_score
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import TimeSeriesSplit

def train(model, dataset, hyperparameters, n_splits=5, epochs=50, lr=1e-3, patience=10, device='cpu'):
    model = model.to(device)
    criterion = nn.MSELoss()
    batch_size = hyperparameters.get('batch_size', 32)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_history = []

    print(f"Starting {n_splits}-fold time series cross-validation...\n")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(dataset)):
        print(f"{'─'*55}")
        print(f"Fold {fold+1}/{n_splits} | train: {len(train_idx)} samples | val: {len(val_idx)} samples")
        print(f"{'─'*55}")

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=False)
        val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=batch_size, shuffle=False)

        model.apply(reset_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        fold_history = {'train_loss': [], 'val_mse': [], 'val_mae': [], 'val_rmse': [], 'val_r2': []}
        best_val_loss = np.inf
        best_model_state = None
        epochs_without_improvement = 0

        for epoch in range(epochs):
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

            print(f"  Epoch {epoch+1:3d}/{epochs} | train_loss: {train_loss:.4f} | val_MSE: {metrics['mse']:.4f} | val_MAE: {metrics['mae']:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"\n  Early stopping at epoch {epoch+1} — best val_MSE: {best_val_loss:.4f}")
                    break

        cv_history.append(fold_history)

    # ── Generalization summary ────────────────────────────────────────────────
    fold_mse  = [np.min(h['val_mse']) for h in cv_history]   # best MSE per fold
    fold_mae  = [np.min(h['val_mae']) for h in cv_history]   # best MAE per fold

    print(f"\n{'='*55}")
    print(f"Cross-Validation Generalization Summary")
    print(f"{'='*55}")
    print(f"{'Fold':<8} {'Best MSE':>10} {'Best MAE':>10}")
    print(f"{'─'*30}")
    for i, (mse, mae) in enumerate(zip(fold_mse, fold_mae)):
        print(f"  {i+1:<6} {mse:>10.4f} {mae:>10.4f}")
    print(f"{'─'*30}")
    print(f"  {'Mean':<6} {np.mean(fold_mse):>10.4f} {np.mean(fold_mae):>10.4f}")
    print(f"  {'Std':<6} {np.std(fold_mse):>10.4f} {np.std(fold_mae):>10.4f}")
    print(f"{'='*55}\n")

    # ── Final training on full dataset ────────────────────────────────────────
    avg_epochs = int(np.mean([len(h['train_loss']) for h in cv_history]))
    print(f"Training final model on full dataset for {avg_epochs} epochs...\n")

    full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.apply(reset_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    final_history = {'train_loss': []}
    for epoch in range(avg_epochs):
        model.train()
        train_losses = []
        for inputs, labels in full_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = np.mean(train_losses)
        final_history['train_loss'].append(train_loss)
        print(f"  Epoch {epoch+1:3d}/{avg_epochs} | train_loss: {train_loss:.4f}")

    return model, cv_history, final_history


def reset_weights(layer):
    """Resets layer weights to avoid leaking information between folds."""
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()

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