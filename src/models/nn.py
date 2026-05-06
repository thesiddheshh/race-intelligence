"""Neural Network implementation using PyTorch."""
from .base import F1PredictionModel
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

class F1NeuralNet(nn.Module):
    """Simple feedforward neural network for F1 prediction."""
    
    def __init__(self, input_dim: int, hidden_dims: list = [128, 64, 32], dropout_rate: float = 0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))  # Regression output
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class NeuralNetworkModel(F1PredictionModel):
    """PyTorch neural network wrapper for sklearn compatibility."""
    
    def __init__(self, hidden_dims: list = None, **kwargs):
        super().__init__(model_name="NeuralNetwork", **kwargs)
        self.hidden_dims = hidden_dims or [128, 64, 32]
        self.scaler_ = StandardScaler()
        
    def _create_model(self):
        return None  # Model created in fit() due to input_dim dependency
    
    def _get_param_grid(self) -> dict:
        return {
            'hidden_dims': [[64, 32], [128, 64, 32], [256, 128, 64]],
            'dropout_rate': [0.1, 0.2, 0.3],
            'learning_rate': [1e-3, 1e-4],
            'batch_size': [32, 64, 128],
            'epochs': [50, 100, 150]
        }
    
    def fit(self, X, y, **kwargs):
        # Preprocess
        X_scaled = self.scaler_.fit_transform(X)
        y_array = y.values if hasattr(y, 'values') else np.array(y)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y_array).unsqueeze(1)
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Model config
        input_dim = X_scaled.shape[1]
        dropout_rate = kwargs.get('dropout_rate', 0.2)
        self.model_ = F1NeuralNet(input_dim, self.hidden_dims, dropout_rate)
        
        # Training config
        lr = kwargs.get('learning_rate', 1e-3)
        batch_size = kwargs.get('batch_size', 64)
        epochs = kwargs.get('epochs', 100)
        
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=lr)
        criterion = nn.MSELoss()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        self.model_.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
        
        self.feature_names_ = X.columns.tolist() if hasattr(X, 'columns') else None
        self.is_fitted_ = True
        logger.info("Neural network training completed")
        return self
    
    def predict(self, X):
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction")
        X_scaled = self.scaler_.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        self.model_.eval()
        with torch.no_grad():
            preds = self.model_(X_tensor)
        return preds.numpy().flatten()