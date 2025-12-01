import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# Model class

class GRUModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512, num_layers=4, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        x = out[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        return x.squeeze(-1)

    