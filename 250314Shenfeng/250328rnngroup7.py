import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
# Load the data
file_path = '/Users/w/Developer/250306 KYK Datamining/250314Shenfeng/combined/group_7_combined.csv'
table1 = pd.read_csv(file_path)
print(table1['Date'].head(5))
table1['Date'] = table1['Date'].apply(lambda x: pd.to_datetime(x, format='%d/%m/%Y %H:%M') if ':' in x else pd.to_datetime(x, format='%d/%m/%Y'))
table1.set_index('Date', inplace=True)
table1.sort_index(inplace=True)
# Calculate daily counts
daily_counts = table1.resample('D').size().to_frame(name='Count')


# Print date range and frequency
print("Start Date:", daily_counts.index.min())
print("End Date:", daily_counts.index.max())
print("Total Days:", (daily_counts.index.max() - daily_counts.index.min()).days)

# Check frequency
inferred_freq = pd.infer_freq(daily_counts.index)
print("Inferred Frequency:", inferred_freq)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(daily_counts)

# Prepare the training and testing data
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create sequences for the RNN model
def create_sequences(data, time_step=2):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i + time_step, 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 10
X_train, y_train = create_sequences(train_data, time_step)
X_test, y_test = create_sequences(test_data, time_step)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # Add feature dimension
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader for batching
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)  # RNN output
        out = self.fc(out[:, -1, :])  # Fully connected layer on the last time step
        return out

# Model parameters
input_size = 1
hidden_size = 50
output_size = 1
num_layers = 2

# Initialize the model, loss function, and optimizer
model = RNNModel(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate the model
model.eval()
train_predictions = model(X_train).detach().numpy()
test_predictions = model(X_test).detach().numpy()

# Inverse transform the predictions
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions.reshape(-1, 1))
y_train = scaler.inverse_transform(y_train.numpy().reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(daily_counts.index, daily_counts['Count'], label='Actual Data')
plt.plot(daily_counts.index[time_step:len(train_predictions) + time_step], train_predictions, label='Train Predictions')
plt.plot(daily_counts.index[len(train_predictions) + (time_step * 2):], test_predictions, label='Test Predictions', color='red')
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('RNN Model Predictions for Ambulance Data (PyTorch)')
plt.legend()
plt.show()

# Forecast future values (next 30 days)
last_sequence = torch.tensor(scaled_data[-time_step:], dtype=torch.float32).unsqueeze(0) #.unsqueeze(-1)
future_predictions = []
model.eval()
for _ in range(30):
    next_prediction = model(last_sequence)
    future_predictions.append(next_prediction.item())
    last_sequence = torch.cat((last_sequence[:, 1:, :], next_prediction.unsqueeze(-1)), dim=1)

# Inverse transform the future predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Create future dates
future_dates = pd.date_range(start=daily_counts.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')

# Plot the historical data and the forecastsz
plt.figure(figsize=(12, 6))
plt.plot(daily_counts.index, daily_counts['Count'], label='Historical Data')
plt.plot(future_dates, future_predictions, label='Future Forecast', color='red')
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('RNN Forecast for Ambulance Calls (Next 30 Days) - PyTorch')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error

# SSE
sse = np.sum((y_test - test_predictions) ** 2)
# MSE
mse = mean_squared_error(y_test, test_predictions)
# RMSE
rmse = np.sqrt(mse)
# MAE
mae = mean_absolute_error(y_test, test_predictions)

print(f"SSE: {sse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"rnn_model_{timestamp}.pth"
torch.save(model.state_dict(), model_filename)
print(f"Model saved as {model_filename}")


# Load model
"""
# Recreate the model architecture first
model = RNNModel(input_size=1, hidden_size=50, output_size=1, num_layers=2)

# Load model weights
model.load_state_dict(torch.load("rnn_model.pth"))
model.eval()
print("Model loaded successfully.")

"""

# Define test_dates based on the daily_counts index
test_dates = daily_counts.index[len(train_data) + time_step:]

plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test, label='Actual', linewidth=2)
plt.plot(test_dates, test_predictions, label='Predicted', linestyle='--')
plt.title('Actual vs Predicted Ambulance Calls (Test Set)')
plt.xlabel('Date')
plt.ylabel('Call Count')
plt.legend()
plt.show()
