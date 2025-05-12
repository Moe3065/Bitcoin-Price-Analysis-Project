# Bitcoin Price Analysis: GARCH Model

import pandas as pd
import numpy as np
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Load processed data
processed_data_path = "/home/ubuntu/Bitcoin_Price_Analysis_Project/data/BTC-USD_2015_2024_processed.csv"
df = pd.read_csv(processed_data_path, index_col=0, parse_dates=True)

if df.empty or "Log_Return" not in df.columns:
    print("Processed data with Log_Return not found. Please run 01_data_preparation.py first.")
else:
    print("Processed data loaded successfully.")
    # GARCH model uses log returns
    log_returns = df["Log_Return"] * 100 # Multiply by 100 for better model convergence, common practice
    log_returns = log_returns.dropna()

    # Split data into training and testing sets (80% train, 20% test)
    train_size = int(len(log_returns) * 0.8)
    train_returns, test_returns = log_returns[:train_size], log_returns[train_size:]

    print(f"Training set size (returns): {len(train_returns)}")
    print(f"Test set size (returns): {len(test_returns)}")

    # Fit GARCH(1,1) model
    # Report: GARCH(1,1) with ARFIMA(1,0,0) mean model. 
    # Here, we will use a simpler constant mean or AR(1) for demonstration as ARFIMA can be complex to fit directly with `arch` package without more specific parameters.
    # The primary focus of GARCH is on the variance.
    print("\nFitting GARCH(1,1) model...")
    # Using `ConstantMean` for simplicity, or `ARX` if an AR component is strongly suggested by ACF/PACF of returns.
    # The report mentions ARFIMA(1,0,0) for the mean model. We can use an AR(1) for the mean model as an approximation.
    garch_model = arch_model(train_returns, vol="Garch", p=1, q=1, mean="AR", lags=1, dist="Normal")
    # Alternative: Constant mean: garch_model = arch_model(train_returns, vol=	ext"Garch", p=1, q=1, mean="Constant", dist="Normal")
    
    results = garch_model.fit(update_freq=5, disp="off")

    print("\nGARCH Model Summary:")
    print(results.summary())

    # Forecast volatility and returns
    # Forecasting one step ahead iteratively for the test period
    print("\nForecasting with GARCH model...")
    forecasts = results.forecast(horizon=1, start=test_returns.index[0], method=	ext'simulation	ext', simulations=1000) # Use simulation for multi-step ahead if needed, or 'analytic'
    
    # For a rolling forecast (more robust for multi-step ahead if not using simulation for full horizon):
    rolling_predictions_mean = []
    rolling_predictions_variance = []
    current_data = train_returns.copy()

    for i in range(len(test_returns)):
        train_for_step = log_returns[:train_size+i]
        model_step = arch_model(train_for_step, vol="Garch", p=1, q=1, mean="AR", lags=1, dist="Normal")
        res_step = model_step.fit(disp="off")
        temp_forecast = res_step.forecast(horizon=1)
        rolling_predictions_mean.append(temp_forecast.mean.iloc[-1,0])
        rolling_predictions_variance.append(temp_forecast.variance.iloc[-1,0])

    forecast_mean_series = pd.Series(rolling_predictions_mean, index=test_returns.index)
    forecast_variance_series = pd.Series(rolling_predictions_variance, index=test_returns.index)

    # Evaluate mean forecast (log returns)
    # Report values: RMSE approx 0.039, MAE approx 0.027 (for log returns, not scaled by 100)
    # Our log_returns were scaled by 100, so we compare accordingly or scale back.
    rmse_returns = np.sqrt(mean_squared_error(test_returns, forecast_mean_series))
    mae_returns = mean_absolute_error(test_returns, forecast_mean_series)

    print(f"\nGARCH Model Mean Forecast Performance (Test Set, on returns*100):")
    print(f"RMSE: {rmse_returns:.4f}") 
    print(f"MAE: {mae_returns:.4f}")
    print(f"(Report values were for unscaled log returns: RMSE ~0.039, MAE ~0.027)")
    print(f"To compare: RMSE_unscaled: {rmse_returns/100:.4f}, MAE_unscaled: {mae_returns/100:.4f}")


    # Plot actual squared returns vs conditional variance
    # The report shows "estimated volatility (blue line) against the actual squared returns (grey)"
    plt.figure(figsize=(12, 6))
    plt.plot(test_returns.index, test_returns**2, label="Actual Squared Log Returns (Test)", color="grey", alpha=0.7)
    plt.plot(forecast_variance_series.index, forecast_variance_series, label="Forecasted Conditional Variance (Test)", color="blue")
    plt.title("GARCH: Forecasted Conditional Variance vs Actual Squared Log Returns (Test Set)")
    plt.xlabel("Date")
    plt.ylabel("Variance / Squared Log Returns (*100)^2")
    plt.legend()
    plt.grid(True)
    garch_plot_path = "/home/ubuntu/Bitcoin_Price_Analysis_Project/report/garch_volatility_forecast_plot.png"
    plt.savefig(garch_plot_path)
    print(f"\nGARCH volatility forecast plot saved to {garch_plot_path}")
    # plt.show()

    print("\n--- GARCH Model Script Complete ---")

