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

if __name__=="__main__":
    # training codes, put this here so validation code doesn't train again
    # Load data
    print("=== Loading Training Data ===")
    train_X, train_y = torch.load("train.pt")
    val_X, val_y = torch.load("val.pt")
    print(f"Training samples: {train_X.shape[0]}")
    print(f"Validation samples: {val_X.shape[0]}")
    print(f"Input features: {train_X.shape[2]}")

    # load normalized data
    scaler = torch.load("scaler.pt",weights_only=False)
    print(f"Input features: {len(scaler['mean'])} features")
    

    # Create DataLoaders
    train_ds = TensorDataset(train_X, train_y)
    val_ds = TensorDataset(val_X, val_y)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)


    # Initialize model, loss, optimizer
    print("\n=== Initializing Model ===")
    model = GRUModel(input_size=train_X.shape[2], hidden_size=256, num_layers=3, dropout=0.2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)

    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")


    print("\n=== Training begins... ===")
    # Training loop with validation
    best_val_loss = float('inf')
    epochs = 20
    patience = 8
    epochs_no_improve = 0
    # Training loop
    for epoch in range(epochs): 
        model.train()

        # no loss now
        train_loss=0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds.squeeze(), y_batch)
            loss.backward()

            # Add gradient clipping for stability (good practice)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_ds)
        print(f"Epoch {epoch+1} done")

        ###########################################################################
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch)
                loss = criterion(preds.squeeze(), y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss /= len(val_ds)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "gru_model.pth")
            print(f"  → Best model saved (val_loss: {val_loss:.6f})")
        else:
            epochs_no_improve += 1
            print(f"  → No improvement for {epochs_no_improve} epoch(s)")
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break

print("\n=== Training Complete ===")
print(f"Best validation loss: {best_val_loss:.6f}")
print(f"Model saved to gru_model.pth")