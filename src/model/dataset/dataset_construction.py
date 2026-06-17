import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, df, input_width, label_width, shift, label_columns):
        """
        Args:
            df           : scaled DataFrame (train, val, or test)
            input_width  : how many past timesteps the model sees
            label_width  : how many future timesteps to predict
            shift        : offset between end of input and end of label (usually 1)
            label_columns: list of column names to predict, e.g. ['internacoes']
        """
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift

        df = df.drop(columns=['data'], errors='ignore')

        self.label_col_indices = [df.columns.get_loc(c) for c in label_columns]
        
        self.data = df.values.astype(np.float32)
        self.n_samples = len(self.data) - self.total_window_size + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        window = self.data[idx : idx + self.total_window_size]

        # Input: first `input_width` steps, all features
        inputs = window[:self.input_width, :]                          # (input_width, n_features)

        # Label: last `label_width` steps, only target column(s)
        label_start = self.total_window_size - self.label_width
        labels = window[label_start:, self.label_col_indices]          # (label_width, n_targets)

        return torch.tensor(inputs), torch.tensor(labels)


def make_loaders(train_df, val_df, test_df, input_width=7, label_width=1, shift=1, label_columns=['internacoes'], batch_size=32):

    train_ds = TimeSeriesDataset(train_df, input_width, label_width, shift, label_columns)
    val_ds = TimeSeriesDataset(val_df, input_width, label_width, shift, label_columns)
    test_ds  = TimeSeriesDataset(test_df,  input_width, label_width, shift, label_columns)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)  # ⚠️ shuffle=False for time series
    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader