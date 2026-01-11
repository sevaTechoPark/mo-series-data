import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from yahoo_api import get_ticket_filepath, get_ticket_plot_filepath

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_lag_features(series, n_lags=5):
    X, y = [], []
    for i in range(n_lags, len(series)):
        X.append(series[i-n_lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)

def create_lstm_features(series, n_steps=10):
    X, y = [], []
    for i in range(n_steps, len(series)):
        X.append(series[i-n_steps:i])
        y.append(series[i])
    return np.array(X), np.array(y)

def forecasting_pipeline(ticker, test_size=30, n_lags=5, n_steps=10, epochs=20):
    filename = get_ticket_filepath(ticker)
    df = pd.read_csv(filename, parse_dates=['Date'])
    df = df.sort_values('Date')
    df = df.dropna()
    series = df['close'].values
    dates = df['Date'].values

    train, test = series[:-test_size], series[-test_size:]
    train_dates, test_dates = dates[:-test_size], dates[-test_size:]

    X_train, y_train = create_lag_features(train, n_lags)
    X_test, y_test = create_lag_features(np.concatenate([train[-n_lags:], test]), n_lags)

    ridge = Ridge()
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)

    df_prophet = pd.DataFrame({'ds': df['Date'], 'y': df['close']})
    train_prophet = df_prophet.iloc[:-test_size]
    model_prophet = Prophet()
    model_prophet.fit(train_prophet)
    future = model_prophet.make_future_dataframe(periods=test_size)
    forecast = model_prophet.predict(future)
    prophet_pred = forecast['yhat'].iloc[-test_size:].values

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train.reshape(-1, 1))
    test_scaled = scaler.transform(test.reshape(-1, 1))

    X_train_lstm, y_train_lstm = create_lstm_features(train_scaled, n_steps)
    X_test_lstm, y_test_lstm = create_lstm_features(np.concatenate([train_scaled[-n_steps:], test_scaled]), n_steps)

    model = Sequential([
        LSTM(32, input_shape=(n_steps, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train_lstm, y_train_lstm, epochs=epochs, batch_size=16, verbose=0)

    lstm_pred_scaled = model.predict(X_test_lstm).flatten()
    lstm_pred = scaler.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).flatten()

    ridge_rmse = mean_squared_error(y_test, ridge_pred, squared=False)
    ridge_mape = mean_absolute_percentage_error(y_test, ridge_pred)
    prophet_rmse = mean_squared_error(test, prophet_pred, squared=False)
    prophet_mape = mean_absolute_percentage_error(test, prophet_pred)
    lstm_rmse = mean_squared_error(y_test, lstm_pred, squared=False)
    lstm_mape = mean_absolute_percentage_error(y_test, lstm_pred)

    metrics = pd.DataFrame({
        'model': ['Ridge', 'Prophet', 'LSTM'],
        'RMSE': [ridge_rmse, prophet_rmse, lstm_rmse],
        'MAPE': [ridge_mape, prophet_mape, lstm_mape]
    })
    print("Метрики на тесте:\n", metrics)

    result_len = min(len(y_test), len(ridge_pred), len(prophet_pred[-len(y_test):]), len(lstm_pred), len(test_dates[n_lags:]))
    result = pd.DataFrame({
        'date': test_dates[n_lags:][:result_len],
        'actual': y_test[:result_len],
        'ridge_pred': ridge_pred[:result_len],
        'prophet_pred': prophet_pred[-result_len:],
        'lstm_pred': lstm_pred[:result_len]
    })

    best_idx = metrics['RMSE'].idxmin()
    best_model = metrics.loc[best_idx, 'model']
    print(f"Лучшая модель: {best_model}")

    if best_model == 'Ridge':
        X_full, y_full = create_lag_features(series, n_lags)
        ridge.fit(X_full, y_full)
        last_vals = series[-n_lags:].tolist()
        ridge_forecast = []
        for _ in range(30):
            x_input = np.array(last_vals[-n_lags:]).reshape(1, -1)
            pred = ridge.predict(x_input)[0]
            ridge_forecast.append(pred)
            last_vals.append(pred)
        forecast_dates = pd.date_range(df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=30)
        forecast_df = pd.DataFrame({'date': forecast_dates, 'best_pred': ridge_forecast})
        best_metric = metrics.loc[metrics['model'] == 'Ridge', 'RMSE'].values[0]
    elif best_model == 'Prophet':
        model_prophet = Prophet()
        model_prophet.fit(df_prophet)
        future = model_prophet.make_future_dataframe(periods=30)
        forecast = model_prophet.predict(future)
        forecast_dates = forecast['ds'].iloc[-30:].values
        forecast_df = pd.DataFrame({'date': forecast_dates, 'best_pred': forecast['yhat'].iloc[-30:].values})
        best_metric = metrics.loc[metrics['model'] == 'Prophet', 'RMSE'].values[0]
    elif best_model == 'LSTM':
        scaler_full = StandardScaler()
        series_scaled = scaler_full.fit_transform(series.reshape(-1, 1))
        X_full_lstm, y_full_lstm = create_lstm_features(series_scaled, n_steps)
        model = Sequential([
            LSTM(32, input_shape=(n_steps, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_full_lstm, y_full_lstm, epochs=epochs, batch_size=16, verbose=0)
        last_vals = series_scaled[-n_steps:].tolist()
        lstm_forecast = []
        for _ in range(30):
            x_input = np.array(last_vals[-n_steps:]).reshape(1, n_steps, 1)
            pred_scaled = model.predict(x_input, verbose=0)[0, 0]
            pred = scaler_full.inverse_transform([[pred_scaled]])[0, 0]
            lstm_forecast.append(pred)
            last_vals.append([pred_scaled])
        forecast_dates = pd.date_range(df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=30)
        forecast_df = pd.DataFrame({'date': forecast_dates, 'best_pred': lstm_forecast})
        best_metric = metrics.loc[metrics['model'] == 'LSTM', 'RMSE'].values[0]
    else:
        best_metric = ""
        raise ValueError("Unknown model")
    
    return result, metrics, best_model, best_metric, forecast_df, df

def plot_forecast(df, forecast_df, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['close'], label='Исторические данные', color='blue')
    plt.plot(forecast_df['date'], forecast_df['best_pred'], label='Прогноз на 30 дней', color='red')
    plt.xlabel('Дата')
    plt.ylabel('Цена акции')
    plt.title(f'Прогноз цены акций {ticker.upper()} на 30 дней')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = get_ticket_plot_filepath(ticker)
    plt.savefig(filename)
    plt.close()
