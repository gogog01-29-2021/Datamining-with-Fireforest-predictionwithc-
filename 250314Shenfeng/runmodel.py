import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Define same model structure
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# Load and preprocess data (match original)
file_path = '/Users/w/Developer/250306 KYK Datamining/250314Shenfeng/combined/group_1_combined.csv'
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)
daily_counts = df.resample('D').size().to_frame(name='Count')
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(daily_counts)

# Prepare last sequence
time_step = 10
last_sequence = torch.tensor(scaled_data[-time_step:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

# Load model
model = RNNModel(input_size=1, hidden_size=50, output_size=1, num_layers=2)
model.load_state_dict(torch.load("rnn_model_20250430_0835.pth"))  # change to your filename
model.eval()

# Predict next 30 days
future_predictions = []
with torch.no_grad():
    for _ in range(30):
        next_pred = model(last_sequence)
        next_pred = next_pred.unsqueeze(-1)
        future_predictions.append(next_pred.item())
        last_sequence = torch.cat((last_sequence[:, 1:, :], next_pred), dim=1)

# Inverse transform and show result
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
future_dates = pd.date_range(start=daily_counts.index[-1] + pd.Timedelta(days=1), periods=30)

# Plot
import matplotlib.pyplot as plt
plt.plot(daily_counts.index, daily_counts['Count'], label='Historical')
plt.plot(future_dates, future_predictions, label='Forecast', color='red')
plt.legend()
plt.title("Future Forecast from Loaded Model")
plt.show()
