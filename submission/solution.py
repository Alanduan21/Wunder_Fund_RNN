import numpy as np
from utils import DataPoint
import torch
from GRU_model_submission import GRUModel

# pass PENDING and CHECK 10:09PM 30/11/2025

class PredictionModel:
    def __init__(self):
        # Load scaler
        scaler = torch.load("scaler.pt", weights_only=False)
        self.mean = scaler["mean"].values.astype(np.float32)
        self.std = scaler["std"].values.astype(np.float32)

        # Load model
        input_size = len(self.mean)   # 32
        output_size = len(self.mean)  # 32
        self.model = GRUModel(
            input_size=input_size, 
            output_size=output_size, 
            hidden_size=128, 
            num_layers=2, 
            dropout=0.2
        )
        self.model.load_state_dict(torch.load("gru_model.pth", map_location="cpu", weights_only=False))
        self.model.eval()

        # Sequence buffer per seq_ix
        self.buffers = {}
        self.seq_len = 50  # MATCH YOUR TRAINING!
        self.warmup_steps = 20

    def normalize(self, state: np.ndarray) -> np.ndarray:
        # Ensure float32
        normalized = (state.astype(np.float32) - self.mean) / self.std
        return normalized.astype(np.float32)

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        # Denormalize ALL 32 features
        denormalized = values.astype(np.float32) * self.std + self.mean
        return denormalized.astype(np.float32)

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        if not data_point.need_prediction:
            return None
        
        seq_id = data_point.seq_ix
        if seq_id not in self.buffers:
            self.buffers[seq_id] = []
        
        # Add normalized state to buffer (ensure float32)
        normalized_state = self.normalize(data_point.state)
        self.buffers[seq_id].append(normalized_state)
        
        if len(self.buffers[seq_id]) > self.seq_len:
            self.buffers[seq_id] = self.buffers[seq_id][-self.seq_len:]

        # WARM-UP: Use simple baseline for first few predictions
        if len(self.buffers[seq_id]) < self.warmup_steps:
            # Return current state unchanged (predict no change)
            return data_point.state.astype(np.float32)

        # After warm-up, use model
        current_buffer = self.buffers[seq_id]
        
        # Pad if needed
        if len(current_buffer) < self.seq_len:
            padding_needed = self.seq_len - len(current_buffer)
            first_obs = current_buffer[0]
            padding = np.tile(first_obs, (padding_needed, 1)).astype(np.float32)
            seq_array = np.vstack([padding, np.array(current_buffer, dtype=np.float32)])
        else:
            seq_array = np.array(current_buffer, dtype=np.float32)
        
        # Ensure float32 for torch
        seq_array = seq_array.astype(np.float32)
        seq_array = np.expand_dims(seq_array, axis=0)  # (1, seq_len, 32)
        seq_tensor = torch.from_numpy(seq_array)

        # Predict ALL 32 features
        with torch.no_grad():
            predictions_norm = self.model(seq_tensor)  # Shape: (1, 32)
            predictions_norm = predictions_norm.squeeze(0).numpy()  # Shape: (32,)

        # Denormalize ALL 32 features
        prediction = self.denormalize(predictions_norm)

        return prediction.astype(np.float32)