import numpy as np
from utils import DataPoint
import torch
from GRU_model import GRUModel

class PredictionModel:
    def __init__(self):
        # Initialize your model, load weights, etc.
        
        # Load scaler
        scaler = torch.load("scaler.pt", map_location="cpu",weights_only=False)
        self.mean = scaler["mean"].values.astype(np.float32)
        self.std = scaler["std"].values.astype(np.float32)

        
        # Debug: check scaler dimensions
        print(f"[DEBUG] Scaler mean length: {len(self.mean)}, std length: {len(self.std)}")


        # Load model
        
        input_size = len(self.mean)  # should be 35
        self.model = GRUModel(input_size=input_size, hidden_size=64, num_layers=2)
        state_dict = torch.load("gru_model.pth", map_location="cpu", weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Sequence buffer per seq_ix
        self.buffers = {}
        self.seq_len = 50


    def normalize(self, state: np.ndarray) -> np.ndarray:

        # Debug: check state shape
        print(f"[DEBUG] Incoming state shape: {state.shape}")
        # Pad if state is shorter than scaler length
        if state.shape[0] < len(self.mean):
            pad_len = len(self.mean) - state.shape[0]
            print(f"[DEBUG] Padding state with {pad_len} zeros")
            state = np.pad(state, (0, pad_len), mode="constant")
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


        # Prepare input tensor
        seq_array = np.array(self.buffers[seq_id], dtype=np.float32)
        seq_array = np.expand_dims(seq_array, axis=0)  # (1, T, features)
        seq_tensor = torch.from_numpy(seq_array)


        # Predict target feature
        with torch.no_grad():
            out = self.model(seq_tensor).item()  # scalar prediction


        # Build full prediction vector
        prediction = np.copy(data_point.state)
        if prediction.shape[0] < len(self.mean):
            prediction = np.pad(prediction, (0, len(self.mean) - prediction.shape[0]), mode="constant")
        prediction[0] = self.denormalize_target(out)


        return prediction
