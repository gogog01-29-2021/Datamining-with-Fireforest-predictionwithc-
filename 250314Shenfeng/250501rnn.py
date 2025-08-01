import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ------------------ Dataset Configuration ------------------
dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".", "datasets"))

import os


dataset_groups = {
    "group1": [
        "dfb-fire-2023-opendata.csv",
        "df-opendata-2016-to-2017-with-stn-area.csv",
        "df-opendata-2020-to-2022-with-stn-area.csv",
        "df-opendata-2018-to-2019-with-stn-area.csv"
    ],
    "group2": [
        "2013-2015-dfb-ambulance.csv"
    ],
    "group3": [
        "2023-open-data-dfb-ambulance.csv",
        "da-opendata-2018-to-2019-with-stn-area.csv",
        "da-opendata-2016-to-2017-with-stn-area.csv"
    ],
    "group4": [
        "da-opendata-2020-to-2022-with-stn-area.csv"
    ],
    "group5": [
        "2013-2015-dfb-fire.csv"
    ],
    "group6": [
        "dccfirebrigadeambulanceincidents2011.csv",
        "dccfirebrigadeambulanceincidents2012.csv"
    ]
}



# ------------------ Load and Combine Datasets ------------------

def combine_dataset_group(group_name):
    file_list = dataset_groups[group_name]
    dfs = []
    for file in file_list:
        path = os.path.join(dataset_dir, file)
        if file.endswith(".csv"):
            dfs.append(pd.read_csv(path))
        elif file.endswith(".xlsx"):
            dfs.append(pd.read_excel(path))
        else:
            raise ValueError(f"Unsupported file format: {file}")
    return pd.concat(dfs, ignore_index=True)

#selected_files = ["dfb_ambulance_2013_2015", "dfb_ambulance_2023", "da_2020_2022"]
table1 = combine_dataset_group(list(dataset_groups.keys())[0])

# ------------------ Preprocess Date and Resample ------------------
table1.columns = [col.strip().lower() for col in table1.columns]
date_col = [col for col in table1.columns if 'date' in col][0]
table1[date_col] = pd.to_datetime(table1[date_col], errors='coerce')
table1.dropna(subset=[date_col], inplace=True)
table1.set_index(date_col, inplace=True)
table1.sort_index(inplace=True)

daily_counts = table1.resample('D').size().to_frame(name='Count')
print("Start Date:", daily_counts.index.min())
print("End Date:", daily_counts.index.max())
print("Total Days:", (daily_counts.index.max() - daily_counts.index.min()).days)
print("Inferred Frequency:", pd.infer_freq(daily_counts.index))

# ------------------ Normalize and Sequence Preparation ------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(daily_counts)
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_sequences(data, time_step=10):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i + time_step, 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 10
X_train, y_train = create_sequences(train_data, time_step)
X_test, y_test = create_sequences(test_data, time_step)

X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

# ------------------ Define RNN Model ------------------
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

model = RNNModel(input_size=1, hidden_size=50, output_size=1, num_layers=2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ------------------ Train the Model ------------------
for epoch in range(20):
    model.train()
    for X_batch, y_batch in train_loader:
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/20], Loss: {loss.item():.4f}")

# ------------------ Evaluate and Forecast ------------------
model.eval()
train_predictions = model(X_train).detach().numpy()
test_predictions = model(X_test).detach().numpy()

train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)
y_train = scaler.inverse_transform(y_train.numpy().reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))

# Forecast future
last_sequence = torch.tensor(scaled_data[-time_step:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
future_predictions = []
for _ in range(30):
    next_pred = model(last_sequence)
    next_pred = next_pred.unsqueeze(-1)
    future_predictions.append(next_pred.item())
    last_sequence = torch.cat((last_sequence[:, 1:, :], next_pred), dim=1)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
future_dates = pd.date_range(start=daily_counts.index[-1] + pd.Timedelta(days=1), periods=30)

# ------------------ Metrics ------------------
sse = np.sum((y_test - test_predictions) ** 2)
rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
mae = mean_absolute_error(y_test, test_predictions)
print(f"SSE: {sse:.2f}\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}")

# ------------------ Save the Model ------------------
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"rnn_model_{timestamp}.pth"
torch.save(model.state_dict(), model_filename)
print(f"Model saved as {model_filename}")

# ------------------ Plot Results ------------------
test_dates = daily_counts.index[len(train_data) + time_step:]

plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test, label='Actual')
plt.plot(test_dates, test_predictions, label='Predicted', linestyle='--')
plt.title('Actual vs Predicted (Test Set)')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(daily_counts.index, daily_counts['Count'], label='Historical')
plt.plot(future_dates, future_predictions, label='Forecast', color='red')
plt.title('Next 30 Days Forecast')
plt.legend()
plt.show()
