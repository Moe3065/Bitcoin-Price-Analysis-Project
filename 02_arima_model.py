# Bitcoin Price Analysis: ARIMA Model

import pandas as pd
import numpy as np
import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Load processed data
processed_data_path = "/home/ubuntu/Bitcoin_Price_Analysis_Project/data/BTC-USD_2015_2024_processed.csv"
df = pd.read_csv(processed_data_path, index_col=0, parse_dates=True)

if df.empty:
    print("Processed data not found. Please run 01_data_preparation.py first.")
else:
    print("Processed data loaded successfully.")
    series = df["Close"]

    # Split data into training and testing sets (80% train, 20% test)
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]

    print(f"Training set size: {len(train)}")
    print(f"Test set size: {len(test)}")

    # Fit auto_arima model
    # The report mentioned ARIMA(1,1,0) with drift. 
    # pmdarima.auto_arima can find this or a similar model.
    # We will allow it to search, but guide it based on the report's findings if needed.
    print("\nFitting ARIMA model...")
    # Note: The report mentions R's auto.arima. pmdarima is a Python equivalent.
    # The report found ARIMA(1,1,0) with drift. `with_intercept=True` can act like drift for non-stationary series after differencing.
    model = pm.auto_arima(train, 
                          start_p=1, start_q=0, 
                          max_p=3, max_q=3, # Keep search space reasonable
                          d=1,             # Based on report's (1,1,0)
                          start_P=0, seasonal=False, # No seasonality mentioned for daily Bitcoin prices in this context
                          stepwise=True, suppress_warnings=True,
                          trace=True, error_action="ignore",
                          with_intercept="auto") # auto can determine if drift/intercept is needed

    print("\nARIMA Model Summary:")
    print(model.summary())

    # Make predictions
    predictions, conf_int = model.predict(n_periods=len(test), return_conf_int=True)
    forecast_series = pd.Series(predictions, index=test.index)

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(test, forecast_series))
    mae = mean_absolute_error(test, forecast_series)

    print(f"\nARIMA Model Performance (Test Set):")
    print(f"RMSE: {rmse:.2f}") # Report: approx. 2419
    print(f"MAE: {mae:.2f}")   # Report: approx. 1639
    # The values might differ due to Python vs R implementation, auto_arima parameters, and exact data split.

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label="Training Data")
    plt.plot(test.index, test, label="Actual Prices (Test)")
    plt.plot(forecast_series.index, forecast_series, label="ARIMA Forecast")
    plt.fill_between(forecast_series.index, 
                     conf_int[:, 0], 
                     conf_int[:, 1], 
                     color='k', alpha=.15)
    plt.title("Bitcoin Price: ARIMA Forecast vs Actual (Test Set)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    arima_plot_path = "/home/ubuntu/Bitcoin_Price_Analysis_Project/report/arima_forecast_plot.png"
    plt.savefig(arima_plot_path)
    print(f"\nARIMA forecast plot saved to {arima_plot_path}")
    # plt.show() # Uncomment to display plot if running in an interactive environment

    print("\n--- ARIMA Model Script Complete ---")

