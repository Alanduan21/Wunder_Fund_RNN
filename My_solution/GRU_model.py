import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# Model class

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.gru(x)          # (batch, seq, hidden)
        x = out[:, -1, :]             # take last time step
        x = self.dropout(x)           # apply dropout feature-wise
        x = self.relu(self.fc1(x))    # FC → ReLU
        x = self.fc2(x)               # FC → 1 output
        return x.squeeze(-1)