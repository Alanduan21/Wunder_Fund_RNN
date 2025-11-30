import numpy as np
from submission.utils import DataPoint
import torch
from GRU_model_submission import GRUModel

class PredictionModel:
    def __init__(self):
        # Load scaler
        scaler = torch.load("scaler.pt", weights_only=False)
        self.mean = scaler["mean"].values.astype(np.float32)
        self.std = scaler["std"].values.astype(np.float32)

        # Load model
        input_size = len(self.mean)  # 32 features
        self.model = GRUModel(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.2)
        self.model.load_state_dict(torch.load("gru_model.pth", map_location="cpu", weights_only=False))
        self.model.eval()

        # Sequence buffer per seq_ix
        self.buffers = {}
        self.seq_len = 100
        self.warmup_steps = 20

    def normalize(self, state: np.ndarray) -> np.ndarray:
        return (state - self.mean) / self.std

    def denormalize_target(self, value: float) -> float:
        return value * self.std[0] + self.mean[0]

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        if not data_point.need_prediction:
            return None
        
        seq_id = data_point.seq_ix
        if seq_id not in self.buffers:
            self.buffers[seq_id] = []
        
        # Add normalized state to buffer
        self.buffers[seq_id].append(self.normalize(data_point.state))
        if len(self.buffers[seq_id]) > self.seq_len:
            self.buffers[seq_id] = self.buffers[seq_id][-self.seq_len:]

        # WARM-UP: Use simple baseline for first few predictions
        if len(self.buffers[seq_id]) < self.warmup_steps:
            # Return current state unchanged (predict no change)
            return np.copy(data_point.state)

        # After warm-up, use model with padding if needed
        current_buffer = self.buffers[seq_id]
        
        if len(current_buffer) < self.seq_len:
            # Pad by repeating first observation
            padding_needed = self.seq_len - len(current_buffer)
            first_obs = current_buffer[0]
            padding = np.tile(first_obs, (padding_needed, 1))
            seq_array = np.vstack([padding, np.array(current_buffer, dtype=np.float32)])
        else:
            seq_array = np.array(current_buffer, dtype=np.float32)
        
        seq_array = np.expand_dims(seq_array, axis=0)  # (1, seq_len, features)
        seq_tensor = torch.from_numpy(seq_array)

        # Predict target feature
        with torch.no_grad():
            out = self.model(seq_tensor).item()

        # Build full prediction vector
        prediction = np.copy(data_point.state)
        prediction[0] = self.denormalize_target(out)

        return prediction