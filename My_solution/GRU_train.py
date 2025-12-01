import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Lightweight model for CPU
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

if __name__ == "__main__":
    print("=== Loading Training Data ===")
    train_X, train_y = torch.load("train.pt", weights_only=False)
    val_X, val_y = torch.load("val.pt", weights_only=False)
    
    print(f"Training samples: {train_X.shape[0]}")
    print(f"Validation samples: {val_X.shape[0]}")
    print(f"Sequence length: {train_X.shape[1]}")
    print(f"Input features: {train_X.shape[2]}")
    print(f"Output features: {train_y.shape[1]}") 

    scaler = torch.load("scaler.pt", weights_only=False)
    
    # Create DataLoaders with LARGER batches for CPU efficiency
    train_ds = TensorDataset(train_X, train_y)
    val_ds = TensorDataset(val_X, val_y)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)


    ###########################################################################
    print("\n=== Initializing Model ===")
    input_size = train_X.shape[2]  # 32
    output_size = train_y.shape[1]  # 32

    model = GRUModel(
        input_size=input_size, 
        output_size=output_size,
        hidden_size=512, 
        num_layers=4, 
        dropout=0.3
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)

    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Config: input={input_size}, output={output_size}, hidden=512, layers=4")
    

    print("\n=== Training Begins ===")
    best_val_loss = float('inf')
    epochs = 40
    patience = 8
    epochs_no_improve = 0
    
    import time
    
    for epoch in range(epochs): 
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss /= len(val_ds)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f} ({epoch_time/60:.1f} min)")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.cpu().state_dict(), "gru_model.pth")
            model.to(device)
            print(f"  ✓ Best model saved!")
        else:
            epochs_no_improve += 1
            print(f"  → No improvement ({epochs_no_improve}/{patience})")
            
            if epochs_no_improve >= patience:
                print(f"\n✓ Early stopping after {epoch+1} epochs")
                break

    print("\n=== Training Complete ===")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("Model saved to gru_model.pth")