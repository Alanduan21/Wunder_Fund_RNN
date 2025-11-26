import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


print("=== Loading Validation Data ===")
# Load data
val_X, val_y = torch.load("val.pt")
print(f"Validation samples: {val_X.shape[0]}")
print(f"Input features: {val_X.shape[2]}")

# Load model
from GRU_model import GRUModel  # Import from training file
model = GRUModel(input_size=val_X.shape[2], hidden_size=256, num_layers=3, dropout=0.3)
model.load_state_dict(torch.load("gru_model.pth"))
model.eval()
print("Model loaded successfully")

# load validation data
val_ds = TensorDataset(val_X, val_y)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)


print("\n=== Running Validation ===")
# Collect predictions
preds, targets = [], []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        out = model(X_batch)
        preds.append(out.squeeze().numpy())
        targets.append(y_batch.numpy())

preds = np.concatenate(preds)
targets = np.concatenate(targets)

print(f"Predictions shape: {preds.shape}")
print(f"Targets shape: {targets.shape}")

# Compute comprehensive metrics
print("\n=== Validation Metrics ===")

# RMSE (what you had)
rmse = np.sqrt(mean_squared_error(targets, preds))
print(f"RMSE:  {rmse:.6f}")

# MSE
mse = mean_squared_error(targets, preds)
print(f"MSE:   {mse:.6f}")

# MAE
mae = mean_absolute_error(targets, preds)
print(f"MAE:   {mae:.6f}")

# R² Score (what the challenge uses!)
r2 = r2_score(targets, preds)
print(f"R²:    {r2:.6f}")

# Additional useful metrics
residuals = targets - preds
mean_residual = np.mean(residuals)
std_residual = np.std(residuals)

print(f"\nMean Residual: {mean_residual:.6f}")
print(f"Std Residual:  {std_residual:.6f}")

# Prediction range info
print(f"\n=== Prediction Statistics ===")
print(f"Target range:     [{targets.min():.4f}, {targets.max():.4f}]")
print(f"Prediction range: [{preds.min():.4f}, {preds.max():.4f}]")

# Sample predictions vs targets
print(f"\n=== Sample Predictions (first 10) ===")
print("Target     | Prediction | Error")
print("-" * 40)
for i in range(min(10, len(targets))):
    error = targets[i] - preds[i]
    print(f"{targets[i]:9.4f} | {preds[i]:10.4f} | {error:6.4f}")

print("\n=== Validation Complete ===")
print(f"Key metric (R²): {r2:.6f}")