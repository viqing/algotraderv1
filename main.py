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
    signal_ticker_return = ticker_return * signal
    strategy_index = signal_ticker_return.add(1).cumprod()
    ticker_index = ticker_return.add(1).cumprod()

    if plot_results:
        fig = plt.figure()
        plt1 = fig.add_subplot(111, ylabel='Price')
        ticker_index.plot(ax=plt1, color='r', lw=2.)
        strategy_index.plot(ax=plt1, color='b', lw=2., figsize=(12,8))
        plt.show()

    #ticker_return = np.diff(ticker) / ticker[:-1]
    print(ticker)
    print(ticker_return)
    print(signal_ticker_return)
    print(strategy_index)
    print('nothing')
    pass

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

    fig = plt.figure()
    plt1 = fig.add_subplot(111, ylabel='Price')
    ticker_df['Adj Close'].plot(ax=plt1, color='r', lw=2.)
    signal_df[['short_mav', 'long_mav']].plot(ax=plt1, lw=2., figsize=(12,8))

    plt1.plot(signal_df.loc[signal_df.positions == -1.0].index, signal_df.short_mav[signal_df.positions == -1.0],'v', markersize=10, color='k')
    plt1.plot(signal_df.loc[signal_df.positions == 1.0].index, signal_df.short_mav[signal_df.positions == 1.0],'^', markersize=10, color='m')

    plt.show()

    return signal_df['signal']

## RSI
def rsi():
    rsi_period = 14
    rsi_df = pd.DataFrame(index=ticker_df.index)
    rsi_df['returns'] = ticker_df['Adj Close'].pct_change()
    rsi_df['returns_gains'] = np.where(rsi_df['returns']>0, rsi_df['returns'], 0)
    rsi_df['returns_losses'] = np.where(rsi_df['returns']<0, rsi_df['returns'], 0)
    rsi_df['returns_gains_avg'] = rsi_df['returns_gains'].rolling(rsi_period).mean()
    rsi_df['returns_losses_avg'] = rsi_df['returns_losses'].rolling(rsi_period).mean().abs()
    rsi_df['rsi'] = 100 - (100 / (1 + rsi_df['returns_gains_avg']/rsi_df['returns_losses_avg']))
    
    fig = plt.figure()
    plt1 = fig.add_subplot(111, ylabel='RSI')
    rsi_df['rsi'].plot(ax=plt1, color='r', lw=2.)
    plt.show()



if __name__ == '__main__':
    ticker_df = download_ticker_data('ES=F')
    signal = smac(ticker_df)
    return_series = create_strategy_ticker_series(ticker_df['Adj Close'], signal, plot_results=True)
    #rsi()
