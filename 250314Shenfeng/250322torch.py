import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load the data
file_path = '/Users/w/Developer/250306 KYK Datamining/250314Shenfeng/2023-open-data-dfb-ambulance.csv'
table1 = pd.read_csv(file_path)
table1['Date'] = pd.to_datetime(table1['Date'], format='%d/%m/%Y')
table1.set_index('Date', inplace=True)
table1.sort_index(inplace=True)

# Assuming you want to analyze the count of records per day
daily_counts = table1.resample('D').size().to_frame(name='Count')

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(daily_counts)

# Prepare the training and testing data
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 10
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train).unsqueeze(2)  # Add feature dimension
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test).unsqueeze(2)
y_test_tensor = torch.FloatTensor(y_test)

# Create PyTorch datasets and dataloaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, return_sequences=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 25)
        self.fc2 = nn.Linear(25, output_size)
        
    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc1(x[:, -1, :])
        x = self.fc2(x)
        return x

# Initialize the model, loss function and optimizer
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Train the model
epochs = 1
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

# Make predictions
model.eval()
with torch.no_grad():
    train_predictions = []
    for inputs, _ in train_loader:
        output = model(inputs)
        train_predictions.append(output.item())
    
    test_predictions = []
    for inputs, _ in test_loader:
        output = model(inputs)
        test_predictions.append(output.item())

# Convert predictions to numpy arrays and reshape
train_predict = np.array(train_predictions).reshape(-1, 1)
test_predict = np.array(test_predictions).reshape(-1, 1)

# Inverse transform to get actual values
train_predict = scaler.inverse_transform(train_predict)
y_train_np = y_train.reshape(1, -1)
y_train_actual = scaler.inverse_transform(y_train_np.T)
test_predict = scaler.inverse_transform(test_predict)
y_test_np = y_test.reshape(1, -1)
y_test_actual = scaler.inverse_transform(y_test_np.T)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(daily_counts.index, daily_counts['Count'], label='Actual Data')
plt.plot(daily_counts.index[time_step:len(train_predict) + time_step], train_predict, label='Train Predict')
plt.plot(daily_counts.index[len(train_predict) + (time_step * 2) + 1:len(daily_counts) - 1], test_predict, label='Test Predict')
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('LSTM Model Predictions (PyTorch)')
plt.legend()
plt.show()