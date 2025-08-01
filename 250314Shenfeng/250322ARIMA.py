import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from pmdarima import auto_arima

# Load the data
file_path = '/Users/w/Developer/250306 KYK Datamining/250314Shenfeng/2023-open-data-dfb-ambulance.csv'
table1 = pd.read_csv(file_path)
table1['Date'] = pd.to_datetime(table1['Date'], format='%d/%m/%Y')
table1.set_index('Date', inplace=True)
table1.sort_index(inplace=True)

# Calculate daily counts
daily_counts = table1.resample('D').size().to_frame(name='Count')

# Split the data into training and testing sets
train_size = int(len(daily_counts) * 0.8)
train_data = daily_counts[:train_size]
test_data = daily_counts[train_size:]

# Find the optimal ARIMA parameters using auto_arima
print("Finding optimal ARIMA parameters...")
auto_model = auto_arima(train_data, seasonal=True, m=7,  # Using 7 for weekly seasonality
                         start_p=0, start_q=0, max_p=5, max_q=5, max_d=2,
                         trace=True, error_action='ignore', suppress_warnings=True,
                         stepwise=True)

# Get the optimal parameters
p, d, q = auto_model.order
P, D, Q, m = auto_model.seasonal_order
print(f"Optimal ARIMA parameters: ARIMA({p},{d},{q})({P},{D},{Q},{m})")

# Fit the ARIMA model with the optimal parameters
model = ARIMA(train_data, order=(p, d, q), 
              seasonal_order=(P, D, Q, m) if m > 0 else None)
model_fit = model.fit()

# Summary of the model
print(model_fit.summary())

# Make predictions on the test set
predictions = model_fit.forecast(steps=len(test_data))

# If predictions is a Series, convert to DataFrame with the same column name
if isinstance(predictions, pd.Series):
    predictions = predictions.to_frame(name='Count')

# Calculate RMSE
rmse = sqrt(mean_squared_error(test_data['Count'], predictions['Count']))
print(f'Test RMSE: {rmse}')

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(daily_counts.index, daily_counts['Count'], label='Actual Data')
plt.plot(train_data.index, train_data['Count'], label='Training Data')
plt.plot(test_data.index, test_data['Count'], label='Test Data')
plt.plot(test_data.index, predictions['Count'], label='ARIMA Predictions', color='red')
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('ARIMA Model Predictions for Ambulance Data')
plt.legend()
plt.show()

# To make the model comparable to the LSTM version, let's also forecast on the training set
train_predictions = model_fit.predict(start=train_data.index[0], end=train_data.index[-1])

# Plot comparing the LSTM style (both training and test predictions)
plt.figure(figsize=(12, 6))
plt.plot(daily_counts.index, daily_counts['Count'], label='Actual Data')
plt.plot(train_data.index, train_predictions, label='Train Predictions', color='green')
plt.plot(test_data.index, predictions, label='Test Predictions', color='red')
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('ARIMA Model Predictions (Like LSTM Plot)')
plt.legend()
plt.show()

# Plot residuals to check model quality
residuals = pd.DataFrame(model_fit.resid)
plt.figure(figsize=(12, 6))
plt.plot(residuals)
plt.title('Residuals from ARIMA Model')
plt.show()

# Plot ACF of residuals to check if there's any remaining pattern
from statsmodels.graphics.tsaplots import plot_acf
plt.figure(figsize=(12, 6))
plot_acf(residuals, lags=40)
plt.title('ACF of Residuals')
plt.show()

# Create a forecast for future days (next 30 days)
future_forecast = model_fit.forecast(steps=30)
future_dates = pd.date_range(start=daily_counts.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')

# Plot the historical data and the forecasts
plt.figure(figsize=(12, 6))
plt.plot(daily_counts.index, daily_counts['Count'], label='Historical Data')
plt.plot(future_dates, future_forecast, label='Future Forecast', color='red')
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('ARIMA Forecast for Ambulance Calls (Next 30 Days)')
plt.legend()
plt.show()