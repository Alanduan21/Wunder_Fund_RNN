# Wunder Fund RNN Challenge – Submission by Duan Yihe

## Overview
This project implements a GRU-based sequence prediction model for the **Wunder Fund RNN Challenge**.  
The goal is to predict the next state in a multivariate market time series using historical sequences.

## Project Structure

├── solution.py # Entry point; implements PredictionModel with predict() for inference<br>
├── GRU_model.py # GRU-based neural network architecture<br>
├── gru_model.pth # Best validation checkpoint (trained model weights)<br>
├── scaler.pt # Saved normalization statistics (mean & std)<br>
├── utils.py # Helper utilities (DataPoint class, scoring functions)<br>
└── requirements.txt # Python dependencies


## Workflow

### 1. Data Preprocessing
- Load training data from `train.parquet`.
- Remove metadata columns: `id`, `time`, `seq_ix`.
- Split dataset into **80% training** and **20% validation**.
- Normalize and denormalize feature columns
- Save normalization parameters into `scaler.pt`.


### 2. Sequence Construction
- Construct sliding windows of **50 timesteps** per sequence.
- Shapes:
  - **Input:** `(num_sequences, 50, num_features)`
  - **Target:** `(num_sequences, num_features)` (next-step prediction)


### 3. Model Architecture
A 2-layer **GRU** model with:

- **Input size:** ~35 features  
- **Hidden size:** 128  
- **Layers:** 2  
- **Dropout:** 0.2  

A fully connected layer maps the final hidden state to the output feature vector.  
Total parameter count is displayed during training.



### 4. Training Strategy
- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** Adam  
  - Learning rate: `5e-4`  
  - Weight decay: `1e-6`
- **Batch size:** 256
- **Gradient clipping:** max norm = 1.0
- **Early stopping:** patience = 5 epochs  
- Best model checkpoint saved as `gru_model.pth`.



### 5. Inference
- Normalize input using `scaler.pt`.
- Predict next state:
  - First feature predicted by the GRU.
  - Remaining features copied from the last timestep (current limitation).
- Denormalize outputs before returning predictions.



## Limitations & Future Work

### Hyperparameter Exploration
More experiments needed with deeper GRUs, other hidden sizes, and alternative optimization schedules.

### Training Automation
Validation/checkpointing is manual; could be improved using Optuna, Ray Tune, or MLflow.

### Hardware Utilization
Training was CPU-only; GPU acceleration would improve training time and scalability.

### Alternative Sequence Models
Future iterations may explore LSTM, Transformers, or diffusion-based models.

### Financial Modeling Integration
Current approach is fully data-driven; integrating stochastic modeling, volatility estimation, and risk-adjusted metrics would increase robustness.

### Prediction Scope
Only the first feature is accurately predicted; future versions should jointly learn all features.

### Regularization & Generalization
Possible enhancements include LayerNorm, recurrent dropout, and learning rate schedulers.

