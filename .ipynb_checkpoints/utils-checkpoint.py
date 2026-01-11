from scipy.signal import find_peaks
import numpy as np

def find_trade_points(prices):
    peaks, _ = find_peaks(prices, prominence=0.05)
    valleys, _ = find_peaks(-prices, prominence=0.05)
    return valleys, peaks

def simulate_trading(prices, amount, dates=None):
    valleys, peaks = find_trade_points(prices)
    trade_points = sorted(list(valleys) + list(peaks))
    trade_points.sort()
    profit = 0
    actions = []
    i = 0
    while i < len(trade_points) - 1:
        buy_idx = trade_points[i]
        sell_idx = trade_points[i+1]
        if buy_idx in valleys and sell_idx in peaks and sell_idx > buy_idx:
            buy_price = prices[buy_idx]
            sell_price = prices[sell_idx]
            shares = amount / buy_price
            profit += (sell_price - buy_price) * shares
            actions.append({
                'buy_date': dates[buy_idx] if dates is not None else buy_idx,
                'buy_price': buy_price,
                'sell_date': dates[sell_idx] if dates is not None else sell_idx,
                'sell_price': sell_price,
                'profit': (sell_price - buy_price) * shares
            })
            i += 2  # переходим к следующей паре
        else:
            i += 1  # если не пара "минимум-максимум", двигаемся дальше

    if len(actions) == 0 and len(prices) > 1:
        buy_idx = 0
        sell_idx = len(prices) - 1
        buy_price = prices[buy_idx]
        sell_price = prices[sell_idx]
        shares = amount / buy_price
        profit = (sell_price - buy_price) * shares
        actions.append({
            'buy_date': dates[buy_idx] if dates is not None else buy_idx,
            'buy_price': buy_price,
            'sell_date': dates[sell_idx] if dates is not None else sell_idx,
            'sell_price': sell_price,
            'profit': profit
        })
    
    return profit, actions