import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping

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

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)

# Train the model
model.fit(X_train, y_train, batch_size=1, epochs=1, callbacks=[early_stop])

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform to get actual values
train_predict = scaler.inverse_transform(train_predict)
y_train_reshaped = y_train.reshape(-1, 1)
y_train_actual = scaler.inverse_transform(y_train_reshaped)
test_predict = scaler.inverse_transform(test_predict)
y_test_reshaped = y_test.reshape(-1, 1)
y_test_actual = scaler.inverse_transform(y_test_reshaped)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(daily_counts.index, daily_counts['Count'], label='Actual Data')
plt.plot(daily_counts.index[time_step:len(train_predict) + time_step], train_predict, label='Train Predict')
plt.plot(daily_counts.index[len(train_predict) + (time_step * 2) + 1:len(daily_counts) - 1], test_predict, label='Test Predict')
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('LSTM Model Predictions (TensorFlow/Keras)')
plt.legend()
plt.show()