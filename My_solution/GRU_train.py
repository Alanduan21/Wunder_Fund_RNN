import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# Model class

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

if __name__=="__main__":
    # training codes, put this here so validation code doesn't train again
    # Load data
    train_X, train_y = torch.load("train.pt")
    val_X, val_y = torch.load("val.pt")
    print(f"Training samples: {train_X.shape[0]}")
    print(f"Validation samples: {val_X.shape[0]}")
    print(f"Input features: {train_X.shape[2]}")


    train_ds = TensorDataset(train_X, train_y)
    val_ds = TensorDataset(val_X, val_y)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    ### remember no shuffle for validation
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)


    # Initialize model, loss, optimizer
    print("\n=== Initializing Model ===")
    model = GRUModel(input_size=train_X.shape[2], hidden_size=64, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")


    print("\n=== Training begins... ===")
    # Training loop with validation
    best_val_loss = float('inf')
    epochs = 10  
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

print("\n=== Training Complete ===")
print(f"Best validation loss: {best_val_loss:.6f}")
print(f"Model saved to gru_model.pth")