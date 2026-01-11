import yfinance as yf
import os
import pandas as pd
from datetime import datetime, timedelta

tickets_path = os.path.join('./', 'yahoo-cache')

def get_ticket_filepath(ticker):
    return os.path.join(tickets_path, f"{ticker}.csv")

def get_ticket_plot_filepath(ticker):
    return os.path.join(tickets_path, f"{ticker}.png")

def is_valid_ticker(ticker):
    try:
        hist = yf.download(ticker, period="1mo")
        return not hist.empty
    except Exception as e:
        return False

def clean_ticker_csv(filename):
    df = pd.read_csv(filename, skiprows=2)
    df.columns = ['Date', 'close', 'high', 'low', 'open', 'volume']
    df.to_csv(filename, index=False)
    return df

def save_ticker_history(ticker):
    end_date = datetime.today()
     # 2 года
    start_date = end_date - timedelta(days=730)

    ticker_obj = yf.Ticker(ticker)
    hist = yf.download(
        ticker,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
    )

    hist.to_csv(get_ticket_filepath(ticker))
    clean_ticker_csv(get_ticket_filepath(ticker))

def file_exists(ticker):
    return os.path.isfile(get_ticket_filepath(ticker))