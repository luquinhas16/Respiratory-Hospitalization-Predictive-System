import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, df, input_width, label_width, shift, label_columns):
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift

        df = df.drop(columns=['data'], errors='ignore')

        self.label_col_indices = [df.columns.get_loc(c) for c in label_columns]

        # ── NEW: track which columns are input features (everything except labels)
        all_indices = list(range(len(df.columns)))
        self.feature_col_indices = [i for i in all_indices if i not in self.label_col_indices]

        self.data = df.values.astype(np.float32)
        self.n_samples = len(self.data) - self.total_window_size + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        window = self.data[idx : idx + self.total_window_size]

        # ── FIXED: slice only feature columns, excluding internacoes
        inputs = window[:self.input_width, :][:, self.feature_col_indices]  # (input_width, n_features)

        label_start = self.total_window_size - self.label_width
        labels = window[label_start:, self.label_col_indices]               # (label_width, n_targets)

        return torch.tensor(inputs), torch.tensor(labels)


def make_loaders(train_df, test_df, input_width=7, label_width=1, shift=1, label_columns=['internacoes'], batch_size=32):

    train_ds = TimeSeriesDataset(train_df, input_width, label_width, shift, label_columns)
    test_ds  = TimeSeriesDataset(test_df,  input_width, label_width, shift, label_columns)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)  # ⚠️ shuffle=False for time series
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader