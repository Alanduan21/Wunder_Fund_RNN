import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# Load data
val_X, val_y = torch.load("val.pt", map_location="cpu")

# Load model
from GRU_model import GRUModel  # or copy the class here
model = GRUModel(input_size=val_X.shape[2], hidden_size=64, num_layers=2)
model.load_state_dict(torch.load("gru_model.pth", map_location="cpu", weights_only=True))
model.eval()

# load validation data
val_ds = TensorDataset(val_X, val_y)
val_loader = DataLoader(val_ds, batch_size=64)


preds, targets = [], []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        out = model(X_batch)
        preds.append(out.squeeze().numpy())
        targets.append(y_batch.numpy())

preds = np.concatenate(preds)
targets = np.concatenate(targets)

# compute metrics

rmse = np.sqrt(np.mean((preds - targets)**2))
print("Validation RMSE:", rmse)

