import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=== Loading Validation Data ===")
# Load data (already normalized from dataSplit.py)
val_X, val_y = torch.load("val.pt", map_location=device, weights_only=False)
print(f"Validation samples: {val_X.shape[0]}")
print(f"Input features: {val_X.shape[2]}")
print(f"Output features: {val_y.shape[1]}")

# Load scaler for denormalization
scaler = torch.load("scaler.pt", map_location=device, weights_only=False)
mean_vals = scaler["mean"].values.astype(np.float32)
std_vals = scaler["std"].values.astype(np.float32)

print(f"Target mean: {mean_vals[0]:.6f}, std: {std_vals[0]:.6f}")

# load validation data
val_ds = TensorDataset(val_X, val_y)
val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

# Load model
from GRU_model import GRUModel  # Import from training file
model = GRUModel(input_size=val_X.shape[2], output_size=val_y.shape[1], hidden_size=512, num_layers=4, dropout=0.3)
model.load_state_dict(torch.load("gru_model.pth", map_location=device, weights_only=False))
model.to(device)
model.eval()
print("Model loaded successfully")


print("\n=== Running Validation ===")
all_preds = []
all_targets = []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        out = model(X_batch)   # expected shape: (batch,) or (batch,1)
        all_preds.append(out.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())

# Concatenate all batches
preds_norm = np.concatenate(all_preds)     # Shape: (N, 32)
targets_norm = np.concatenate(all_targets)  # Shape: (N, 32)
print(f"Predictions shape: {preds_norm.shape}")
print(f"Targets shape: {targets_norm.shape}")

# Denormalize ALL features
preds_denorm = preds_norm * std_vals + mean_vals
targets_denorm = targets_norm * std_vals + mean_vals


# Compute metrics PER FEATURE (like the competition does)
print("\n=== Validation Metrics (Per Feature) ===")
feature_r2_scores = []

for i in range(targets_denorm.shape[1]):
    r2 = r2_score(targets_denorm[:, i], preds_denorm[:, i])
    feature_r2_scores.append(r2)
    if i < 5 or i >= 30:  # Print first 5 and last 2
        print(f"Feature {i:2d} - R²: {r2:.6f}")

# Overall metrics (like competition scoring)
mean_r2 = np.mean(feature_r2_scores)
print(f"\n{'='*40}")
print(f"MEAN R² (Competition Score): {mean_r2:.6f}")
print(f"{'='*40}")

# Additional metrics
all_preds_flat = preds_denorm.flatten()
all_targets_flat = targets_denorm.flatten()

rmse = np.sqrt(mean_squared_error(all_targets_flat, all_preds_flat))
mae = mean_absolute_error(all_targets_flat, all_preds_flat)

print(f"\nOverall RMSE: {rmse:.6f}")
print(f"Overall MAE:  {mae:.6f}")

# Show feature 0 specifically (for comparison with old model)
print(f"\n=== Feature 0 (Previously Only Predicted) ===")
r2_feat0 = r2_score(targets_denorm[:, 0], preds_denorm[:, 0])
rmse_feat0 = np.sqrt(mean_squared_error(targets_denorm[:, 0], preds_denorm[:, 0]))
print(f"Feature 0 R²:   {r2_feat0:.6f}")
print(f"Feature 0 RMSE: {rmse_feat0:.6f}")

print("\n=== Validation Complete ===")
print(f" Competition Score (Mean R²): {mean_r2:.6f}")