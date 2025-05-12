# Bitcoin Price Analysis: Data Fetching and Preprocessing

import yfinance as yf
import pandas as pd
import numpy as np

# Define the ticker symbol and the date range
TICKE_R = "BTC-USD"
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"

# Fetch the data
print(f"Fetching Bitcoin (BTC-USD) data from {START_DATE} to {END_DATE}...")
data = yf.download(TICKE_R, start=START_DATE, end=END_DATE)

if data.empty:
    print("Could not download data. Please check the ticker and date range.")
else:
    print("Data downloaded successfully.")
    print(data.head())
    print(f"\nData shape: {data.shape}")
    print(f"\nMissing values:\n{data.isnull().sum()}")

    # Save the raw data
    raw_data_path = "/home/ubuntu/Bitcoin_Price_Analysis_Project/data/BTC-USD_2015_2024_raw.csv"
    data.to_csv(raw_data_path)
    print(f"\nRaw data saved to {raw_data_path}")

    # Preprocessing for models
    # Use Closing price for ARIMA and LSTM
    df_close = data[["Close"]].copy()

    # Calculate log returns for GARCH
    df_close["Log_Return"] = np.log(df_close["Close"] / df_close["Close"].shift(1))
    df_close = df_close.dropna() # Remove first NaN row due to shift

    processed_data_path = "/home/ubuntu/Bitcoin_Price_Analysis_Project/data/BTC-USD_2015_2024_processed.csv"
    df_close.to_csv(processed_data_path)
    print(f"Processed data (Close price and Log Returns) saved to {processed_data_path}")

    print("\n--- Data Fetching and Preprocessing Complete ---")

