import torch
import torch.nn as nn
import numpy as np

def train(model, train_loader, val_loader, hyperparameters, epochs=50, lr=1e-3, patience=10, device='cpu'):
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()                          # good default for regression
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )                                                 # halves lr if val loss plateaus

    history = {'train_loss': [], 'val_loss': []}

    best_val_loss = np.inf
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(epochs):

        # ── Training phase ──────────────────────────────────────────────────
        model.train()
        train_losses = []

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            preds = model(inputs)
            loss  = criterion(preds, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevents exploding gradients
            optimizer.step()
            train_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)

        # ── Validation phase ─────────────────────────────────────────────────
        val_loss = evaluate(model, val_loader, criterion, device)['loss']
        
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1:3d}/{epochs} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")

        # ── Early stopping ───────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} — best val_loss: {best_val_loss:.4f}")
                break

    # Restore best weights before returning
    model.load_state_dict(best_model_state)
    return model, history

def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    all_preds = []
    all_labels = []

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
    loss   = np.mean(losses)
    
    return {'loss': loss, 'mse': mse, 'mae': mae}

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