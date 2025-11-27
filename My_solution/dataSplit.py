import pandas as pd
import numpy as np
import torch

SEQ_LEN = 50  # Example sequence length


df = pd.read_parquet("../competition_package/datasets/train.parquet")

### first 3 columns are id, time, seq_ix
print(f"Original data shape: {df.shape}")
df = df.iloc[:, 3:]  # Skip first 3 metadata columns
print(f"Data shape after removing metadata: {df.shape}")
print(f"Number of features: {df.shape[1]}")

### Split data into train and validation sets

## 80% us train, 20% us validation
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
val_df = df.iloc[split_idx:]

## Normalize data
mean = train_df.mean()
std = train_df.std()

## recalculate normalized data
train_df = (train_df - mean) / std
val_df = (val_df - mean) / std

torch.save({"mean": mean, "std": std}, "scaler.pt")

### make sequences
def make_sequences(data):
    arr = data.values.astype(np.float32)
    
    num_rows, num_features = arr.shape
    num_sequences = num_rows - SEQ_LEN

    X = np.lib.stride_tricks.sliding_window_view(arr, SEQ_LEN, axis=0)
    X = X[:num_sequences].copy()
    ## transpose to (num_sequences, SEQ_LEN, num_features)
    X = X.transpose(0, 2, 1)
    y = arr[SEQ_LEN:, 0].copy()   # column 0 must be target
    
    return torch.from_numpy(X), torch.from_numpy(y)


train_X, train_y = make_sequences(train_df)
val_X, val_y = make_sequences(val_df)

torch.save((train_X, train_y), "train.pt")
torch.save((val_X, val_y), "val.pt")

print("\ndebug info:")
print(f"Training sequences: {train_X.shape[0]}")
print(f"Validation sequences: {val_X.shape[0]}")
print(f"Sequence length: {train_X.shape[1]}")
print(f"Number of features: {train_X.shape[2]}")
print(f"train_X shape: {train_X.shape}")
print(f"train_y shape: {train_y.shape}")
print(f"val_X shape: {val_X.shape}")
print(f"val_y shape: {val_y.shape}")
print(f"Train target mean: {train_df.iloc[:, 0].mean()}")
print(f"Train target std: {train_df.iloc[:, 0].std()}")
print(f"Train target range: [{train_df.iloc[:, 0].min()}, {train_df.iloc[:, 0].max()}]")