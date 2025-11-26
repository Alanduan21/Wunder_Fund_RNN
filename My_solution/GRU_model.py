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
    # training codes
    # Load data
    train_X, train_y = torch.load("train.pt")
    train_ds = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)


    # Initialize model, loss, optimizer

    model = GRUModel(input_size=train_X.shape[2])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(5):  # 5 epochs
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} done")

    torch.save(model.state_dict(), "gru_model.pth")
    print("Model saved to gru_model.pth")
