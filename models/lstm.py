import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.utils import train_test_split

import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-notebook")

torch.manual_seed(1)

def reshape_timeseries(values, n_seq, n_steps, n_out=1):
    n_in, cols = n_seq * n_steps, list()
    df = pd.DataFrame(values)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    data = pd.concat(cols, axis=1).dropna().values
    x, y = data[:, :-1], data[:, -1]
    x = x.reshape((x.shape[0], n_seq, n_steps, 1))
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def compute_rmse(model, data_loader):
    criterion = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            total_loss += loss.item() * batch_x.size(0)
    
    avg_loss = total_loss / len(data_loader.dataset)
    
    return np.sqrt(avg_loss)

# Define the CNN-LSTM model
class CNN_LSTM(nn.Module):
    def __init__(self, config):
        super(CNN_LSTM, self).__init__()
        n_steps, n_filters, n_kernel, n_nodes = config["n_steps"], config["n_filters"], config["n_kernel"], config["n_nodes"]
        print("CNN LSTM Parameters:")
        print("- - - - - - - - - - -")
        print(f"CNN  | Output Filters: {n_filters}")
        print(f"CNN  | Convolution Window Size: {n_kernel}")
        print(f"LSTM | Subsequence Steps: {n_steps}")
        print(f"FC   | Nodes in LSTM / FC Layers: {n_nodes}\n")

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=n_kernel)
        self.conv2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=n_kernel)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(n_filters * ((n_steps - 2 * (n_kernel - 1)) // 2), n_nodes, batch_first=True)
        self.fc1 = nn.Linear(n_nodes, n_nodes)
        self.fc2 = nn.Linear(n_nodes, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.6) # for regularization

    def forward(self, x):
        batch_size, seq_len, step_len, _ = x.size()
        x = x.view(-1, step_len, 1).transpose(1, 2)  # Reshape to (batch_size*seq_len, step_len, 1) -> (batch_size*seq_len, 1, step_len)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = x.view(batch_size, seq_len, -1)  # Reshape back to (batch_size, seq_len, flattened feature size)
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc1(x)  # Only take the last output from LSTM
        x = self.relu(x)
        x = self.fc2(x)
        return x

class CNN_LSTM_Model:
    def __init__(self, ts, config):
        self.ts = ts
        self.config = config
        self.train, self.test, self.split_date = train_test_split(self.ts, split=0.8)

        print("Instantiating CNN-LSTM model . . .\n")
        self.model = CNN_LSTM(config=config)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        # Learning Rate Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    def load(self):
        n_seq, n_steps = self.config["n_seq"], self.config["n_steps"]
        X_train, Y_train = reshape_timeseries(self.train.y, n_seq, n_steps)
        X_test, Y_test = reshape_timeseries(self.test.y, n_seq, n_steps)

        # Create DataLoader for training and validation data
        n_batch = self.config["n_batch"]
        self.train_loader = DataLoader(
            TensorDataset(X_train, Y_train), batch_size=n_batch, shuffle=True)
        self.test_loader = DataLoader(
            TensorDataset(X_test, Y_test), batch_size=n_batch, shuffle=False)
        
        print("Loading completed. . .\n")

    def train_model(self):
        print("Starting model training. . .\n")
        self.model.train()
        n_epochs = self.config["n_epochs"]

        self.train_losses, self.test_losses = [], []
        early_stopping_patience = 5
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(n_epochs):
            train_loss = 0.0
            for batch_x, batch_y in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs.squeeze(), batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * batch_x.size(0)
            
            train_loss /= len(self.train_loader.dataset)
            print(f'Epoch {epoch+1}/{n_epochs}, Training Loss: {train_loss:.4f}')
            self.train_losses.append(train_loss)

            self.model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in self.test_loader:
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs.squeeze(), batch_y)
                    test_loss += loss.item() * batch_x.size(0)
            
            test_loss /= len(self.test_loader.dataset)
            print(f'Epoch {epoch+1}/{n_epochs}, Testing Loss: {test_loss:.4f}')
            self.test_losses.append(test_loss)

            # Apply step to learning rate scheduler
            self.scheduler.step(test_loss)

            # Check for Early Stopping
            if test_loss < best_val_loss:
                best_val_loss = test_loss
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement == early_stopping_patience:
                    print("Triggering early stopping! Training has terminated")
                    break

    def loss_plot(self):
        n_epochs = len(self.train_losses)
        # Plot the training and validation losses over epochs
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, n_epochs+1), self.train_losses, color="blue", label='Training Loss')
        plt.plot(range(1, n_epochs+1), self.test_losses, color="orange", label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.show()

    def measure(self):
        train_rmse = compute_rmse(self.model, self.train_loader)
        test_rmse = compute_rmse(self.model, self.test_loader)
        print(f"Training RMSE: {train_rmse:.4f}, Testing RMSE: {test_rmse:.4f}")

    def save_state(self):
        torch.save(self.model.state_dict(), 'hollywood_cnn_lstm.pth')