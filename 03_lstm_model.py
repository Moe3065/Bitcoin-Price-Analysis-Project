# Bitcoin Price Analysis: LSTM Model

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Load processed data
processed_data_path = "/home/ubuntu/Bitcoin_Price_Analysis_Project/data/BTC-USD_2015_2024_processed.csv"
df = pd.read_csv(processed_data_path, index_col=0, parse_dates=True)

if df.empty:
    print("Processed data not found. Please run 01_data_preparation.py first.")
else:
    print("Processed data loaded successfully.")
    series = df[["Close"]].values # Use numpy array for LSTM

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series)

    # Split data into training and testing sets (80% train, 20% test)
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[0:train_size, :]
    test_data = scaled_data[train_size - 60:, :] # Include 60 days prior to test set for first prediction

    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape (with look_back): {test_data.shape}")

    # Create dataset with look_back period
    def create_dataset(dataset, look_back=60):
        X, Y = [], []
        for i in range(look_back, len(dataset)):
            X.append(dataset[i-look_back:i, 0])
            Y.append(dataset[i, 0])
        return np.array(X), np.array(Y)

    look_back = 60 # Report: used past 60 days
    X_train, y_train = create_dataset(train_data, look_back)
    X_test, y_test = create_dataset(test_data, look_back)

    # Reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    print(f"\nX_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Build the LSTM model
    # Report: 2 layers, 50 units each, 10 epochs
    print("\nBuilding LSTM model...")
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
    # model.add(Dropout(0.2)) # Optional: Dropout for regularization
    model.add(LSTM(units=50, return_sequences=False))
    # model.add(Dropout(0.2)) # Optional: Dropout for regularization
    model.add(Dense(units=1))

    model.compile(optimizer=	f.keras.optimizers.Adam(learning_rate=0.001), loss="mean_squared_error")
    print("\nLSTM Model Summary:")
    model.summary()

    # Train the model
    print("\nTraining LSTM model...")
    # Report: 10 epochs. Batch size can be tuned.
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, validation_data=(X_test, y_test))

    # Make predictions
    print("\nMaking predictions with LSTM model...")
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverse transform predictions to original scale
    train_predict = scaler.inverse_transform(train_predict)
    y_train_actual = scaler.inverse_transform([y_train]).T # Reshape y_train for inverse_transform
    test_predict = scaler.inverse_transform(test_predict)
    y_test_actual = scaler.inverse_transform([y_test]).T   # Reshape y_test for inverse_transform

    # Evaluate the model
    # Note: Report RMSE/MAE are on the original scale of prices
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))
    test_mae = mean_absolute_error(y_test_actual, test_predict)

    print(f"\nLSTM Model Performance (Test Set):")
    print(f"RMSE: {test_rmse:.2f}") # Report: approx. 1462
    print(f"MAE: {test_mae:.2f}")   # Report: approx. 1059
    # Values might differ due to Python vs R implementation, specific Keras/TF versions, batch size, optimizer details etc.

    # Plot the results
    plt.figure(figsize=(14, 7))
    
    # Plot training data
    plt.plot(df.index[:train_size], scaler.inverse_transform(scaled_data[:train_size]), label="Original Training Data", color="blue")
    
    # Plot actual test data
    actual_test_plot_index = df.index[train_size:]
    plt.plot(actual_test_plot_index, y_test_actual, label="Actual Prices (Test)", color="green")

    # Plot LSTM forecast on test data
    # Ensure the forecast_series aligns with the test data index
    forecast_plot_index = actual_test_plot_index[:len(test_predict)] # Adjust if lengths differ slightly
    plt.plot(forecast_plot_index, test_predict, label="LSTM Forecast (Test)", color="red", linestyle="--")
    
    plt.title("Bitcoin Price: LSTM Forecast vs Actual (Test Set)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    lstm_plot_path = "/home/ubuntu/Bitcoin_Price_Analysis_Project/report/lstm_forecast_plot.png"
    plt.savefig(lstm_plot_path)
    print(f"\nLSTM forecast plot saved to {lstm_plot_path}")
    # plt.show()

    print("\n--- LSTM Model Script Complete ---")

