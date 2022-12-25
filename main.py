import pandas as pd
import numpy as np
import yfinance as yf
from pprint import pprint
import matplotlib.pyplot as plt
import warnings


def download_ticker_data(ticker, period='max'):
    return yf.download(tickers=ticker, period=period)


def create_strategy_ticker_series(ticker, signal, strategy_type='long short', plot_results=False):
    
    if strategy_type == 'long short':
        signal.replace(0, -1, inplace=True)
    
    ticker_return = ticker.pct_change()
    ticker_return.iloc[0] = 0
    signal_ticker_return = ticker_return * signal
    strategy_index = signal_ticker_return.add(1).cumprod()
    ticker_index = ticker_return.add(1).cumprod()

    if plot_results:
        plot_series(strategy_index, ticker_index)

    return strategy_index


def smac(ticker_df, short_lb=50, long_lb=200):
     ## SMAC strategy

    if short_lb > long_lb:
        warnings.warn("Warning: Short moving average time window ({}) longer than long time window ({}).".format(short_lb, long_lb))

    signal_df = pd.DataFrame(index=ticker_df.index)
    signal_df['signal'] = 0.0
    signal_df['short_mav'] = ticker_df['Adj Close'].rolling(window=short_lb, min_periods=1, center=False).mean()
    signal_df['long_mav'] = ticker_df['Adj Close'].rolling(window=long_lb, min_periods=1, center=False).mean()
    signal_df['signal'] = np.where(signal_df['short_mav'] > signal_df['long_mav'], 1.0, 0.0)   
    signal_df['positions'] = signal_df['signal'].diff()

    # fig = plt.figure()
    # plt1 = fig.add_subplot(111, ylabel='Price')
    # ticker_df['Adj Close'].plot(ax=plt1, color='r', lw=2.)
    # signal_df[['short_mav', 'long_mav']].plot(ax=plt1, lw=2., figsize=(12,8))
    # plt1.plot(signal_df.loc[signal_df.positions == -1.0].index, signal_df.short_mav[signal_df.positions == -1.0],'v', markersize=10, color='k')
    # plt1.plot(signal_df.loc[signal_df.positions == 1.0].index, signal_df.short_mav[signal_df.positions == 1.0],'^', markersize=10, color='m')
    # plt.show()

    return signal_df['signal']


## RSI
def rsi(ticker, period=14):
    rsi_df = pd.DataFrame(index=ticker.index)
    rsi_df['returns'] = ticker.pct_change()
    rsi_df['returns_gains'] = np.where(rsi_df['returns']>0, rsi_df['returns'], 0)
    rsi_df['returns_losses'] = np.where(rsi_df['returns']<0, rsi_df['returns'], 0)
    rsi_df['returns_gains_avg'] = rsi_df['returns_gains'].rolling(period).mean()
    rsi_df['returns_losses_avg'] = rsi_df['returns_losses'].rolling(period).mean().abs()
    rsi_df['rsi'] = 100 - (100 / (1 + rsi_df['returns_gains_avg']/rsi_df['returns_losses_avg']))
    return rsi_df['rsi']

def average_trading_range(ticker_df, period=14):
    ticker_df['H-L'] = ticker_df['High'] - ticker_df['Low']
    ticker_df['H-Cp'] = abs(ticker_df['High'] - ticker_df['Close'].shift())
    ticker_df['L-Cp'] = abs(ticker_df['Low'] - ticker_df['Close'].shift())
    ticker_df['TR'] = ticker_df[['H-L', 'H-Cp', 'L-Cp']].max(axis=1)
    ticker_df['ATR'] = ticker_df['TR'].rolling(period).mean()
    return ticker_df['ATR']


def plot_series(*args):
    fig = plt.figure()
    plt1 = fig.add_subplot(111)

    for arg in args:
        arg.plot(ax=plt1, lw=2.)
    
    plt.show()

    return None


def calculate_annualized_volatility(ticker, start=None, end=None, frequency='daily'):
    
    if start is None:
        start = ticker.index.min()
    if end is None:
        end = ticker.index.max()
    
    ticker = ticker.loc[(ticker.index >= start) & (ticker.index <=end)]
    ticker_return = ticker.pct_change()
    vol = ticker_return.std()

    if frequency == 'daily':
        vol *= np.sqrt(52*5)
    elif frequency == 'monthly':
        vol *= np.sqrt(12)
    
    return vol

def calculate_annualized_return(ticker, start=None, end=None):

    if start is None:
        start = ticker.index.min()
    if end is None:
        end = ticker.index.max()
    ticker = ticker.loc[(ticker.index >= start) & (ticker.index <=end)]

    tot_return = ticker.loc[ticker.index == end][0] / ticker.loc[ticker.index == start][0]
    ann_return = np.power(1+tot_return, 1/((end-start).days/365)) - 1

    return ann_return


if __name__ == '__main__':
    ticker_df = download_ticker_data('ES=F')
    average_trading_range(ticker_df)
    print(ticker_df)
    signal = smac(ticker_df)
    return_series = create_strategy_ticker_series(ticker_df['Adj Close'], signal, plot_results=False)
    print(calculate_annualized_volatility(return_series))
    print(calculate_annualized_return(return_series))
    rsi = rsi(ticker_df['Adj Close'])
    plot_series(rsi)
