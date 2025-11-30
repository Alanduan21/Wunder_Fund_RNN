import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


device = torch.device("cpu")

print("=== Loading Validation Data ===")
# Load data (already normalized from dataSplit.py)
val_X, val_y = torch.load("val.pt", map_location=device, weights_only=False)
print(f"Validation samples: {val_X.shape[0]}")
print(f"Input features: {val_X.shape[2]}")

# Load scaler for denormalization
scaler = torch.load("scaler.pt", map_location=device, weights_only=False)
mean_y = scaler["mean"].values[0]  # Target is first feature
std_y = scaler["std"].values[0]

print(f"Target mean: {mean_y:.6f}, std: {std_y:.6f}")

# load validation data
val_ds = TensorDataset(val_X, val_y)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

# Load model
from GRU_model import GRUModel  # Import from training file
model = GRUModel(input_size=val_X.shape[2], hidden_size=128, num_layers=2, dropout=0.2)
model.load_state_dict(torch.load("gru_model.pth", map_location=device, weights_only=False))
model.to(device)
model.eval()
print("Model loaded successfully")


print("\n=== Running Validation ===")
all_preds = []
all_targets = []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device)
        out = model(X_batch)   # expected shape: (batch,) or (batch,1)
        out = out.view(-1).cpu().numpy()
        all_preds.append(out)
        all_targets.append(y_batch.cpu().numpy().flatten())

all_preds = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)

# === Denormalize predictions and targets to original scale ===
preds_denorm   = all_preds * std_y.item() + mean_y.item()
targets_denorm = all_targets * std_y.item() + mean_y.item()

# === Metrics ===
mse  = mean_squared_error(targets_denorm, preds_denorm)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(targets_denorm, preds_denorm)
r2   = r2_score(targets_denorm, preds_denorm)

print("\n=== Validation Metrics ===")
print(f"RMSE:  {rmse:.6f}")
print(f"MSE:   {mse:.6f}")
print(f"MAE:   {mae:.6f}")
print(f"R²:    {r2:.6f}")

# Residuals and ranges
residuals = targets_denorm - preds_denorm
print(f"\nMean Residual: {residuals.mean():.6f}")
print(f"Std Residual:  {residuals.std():.6f}")

print("\n=== Prediction Statistics ===")
print(f"Target range:     [{targets_denorm.min():.4f}, {targets_denorm.max():.4f}]")
print(f"Prediction range: [{preds_denorm.min():.4f}, {preds_denorm.max():.4f}]")

print("\n=== Sample Predictions (first 10) ===")
print("Target     | Prediction | Error")
print("-" * 40)
for i in range(min(10, len(targets_denorm))):
    t = targets_denorm[i]
    p = preds_denorm[i]
    print(f"{t:9.4f} | {p:10.4f} | {t-p:6.4f}")

print("\n=== Validation Complete ===")
print(f"Key metric (R²): {r2:.6f}")