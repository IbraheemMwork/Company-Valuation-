import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA

# 1. Download historical stock data
ticker = input("Enter stock ticker: ")
df = yf.download(ticker, period="6mo", interval="1d")

# 2. Prepare the data
df = df[["Close"]].dropna()
df.reset_index(inplace=True)
df['Date'] = pd.to_datetime(df['Date'])

# 3. Train the ARIMA model
# You can tune the (p,d,q) parameters as needed
model = ARIMA(df['Close'], order=(5,1,1))
model_fit = model.fit()

# 4. Predict the next 7 weekdays
future_dates = []
date = df['Date'].max()
while len(future_dates) < 7:
    date += timedelta(days=1)
    if date.weekday() < 5:
        future_dates.append(date)

# Forecast next 7 steps
forecast = model_fit.forecast(steps=7)
forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted Close': forecast.values})

print(f'{ticker} Price Forecast for Next Week (ARIMA):')
print(forecast_df)

# 5. Plot
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Close'], label="Historical Close")
plt.plot(forecast_df['Date'], forecast_df['Forecasted Close'], label="Forecasted Close (ARIMA)", linestyle='--')
plt.legend()
plt.title(f"{ticker} Price Forecast for Next Week (ARIMA)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid()
plt.show()
