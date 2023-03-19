import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import warnings

from strategies import simple_moving_average_strategy

import seaborn as sns
import plotly.express as px

# PyPortfolioOpt
# from pyfopt.expected_returns import mean_historical_return
# from pypfopt.risk_models import CovarianceShrinkage


def download_ticker_data(ticker, period='max'):
    return yf.download(tickers=ticker, period=period)


def download_ticker_adj_close_data(ticker, period='max'):
    return yf.download(tickers=ticker, period=period)['Adj Close']


def combine_tickers(ticker_dict):
    ticker_list = []
    for key in ticker_dict:
        return_df = ticker_dict[key]
        series = return_df['Close']
        series.name = key
        ticker_list.append(series)
        
    combined_tickers = pd.concat(ticker_list, axis=1)
    return combined_tickers


def create_strategy_ticker_series(ticker, signal, plot_results=False):
    
    ticker_return = ticker.pct_change()
    ticker_return.iloc[0] = 0
    signal_ticker_return = ticker_return * signal
    strategy_index = signal_ticker_return.add(1).cumprod()
    ticker_index = ticker_return.add(1).cumprod()

    if plot_results:
        plot_series(strategy_index, ticker_index)

    return strategy_index


def smac_strategy(ticker, short_lb=50, long_lb=200):
    ## SMAC strategy

    signal_df = pd.DataFrame(index=ticker.index)
    signal_df['signal'] = 0.0
    signal_df['short_mav'] = sma(ticker, short_lb)
    signal_df['long_mav'] = sma(ticker, long_lb)
    signal_df['signal'] = np.where(signal_df['short_mav'] > signal_df['long_mav'], 1.0, 0.0)
    signal_df['positions'] = signal_df['signal'].diff()
    signal_df['signal'] = np.where(signal_df['short_mav'] < signal_df['long_mav'], -1.0, signal_df['signal'])

    return signal_df['signal']


def sma(ticker, period):
    return ticker.rolling(window=period).mean()

def ema(ticker, period):
    return ticker.ewm(span=period, adjust=False).mean()


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
    ticker_df['ATR'] = ticker_df['TR'].rolling(window=period, min_periods=None, center=False).mean()
    return ticker_df['ATR']


def macd(ticker, l_period=26, m_period=12, s_period=9):
    macd_df = pd.DataFrame(index=ticker.index)
    macd_df['l_EMA'] = ticker.ewm(span=l_period, min_periods=l_period, adjust=False).mean()
    macd_df['m_EMA'] = ticker.ewm(span=m_period, min_periods=m_period, adjust=False).mean()
    macd_df['macd'] = macd_df['m_EMA'] - macd_df['l_EMA']
    macd_df['macd_s'] = macd_df['macd'].ewm(span=s_period, min_periods=s_period, adjust=False).mean()
    macd_df['macd_d'] = macd_df['macd'] - macd_df['macd_s']
    print(macd_df)
    return macd_df['macd_d'], macd_df['macd'], macd_df['macd_s']


def bollinger_bands(ticker, period=20, m=2):
    bol_df = pd.DataFrame(index=ticker.index)
    bol_df['SMA'] = sma(ticker, period=period)
    bol_df['std'] = ticker.rolling(window=period, min_periods=period, center=False).std()
    bol_df['bol_u'] = bol_df['SMA'] + m * bol_df['std']
    bol_df['bol_d'] = bol_df['SMA'] - m * bol_df['std']
    return bol_df['bol_u'], bol_df['bol_d']


def plot_series(*args):
    fig = plt.figure()
    plt1 = fig.add_subplot(111)

    for arg in args:
        arg.plot(ax=plt1, linewidth=2.)
    
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
    ann_return = np.power(1 + tot_return, 1 / np.max(1, ((end-start).days/365))) - 1

    return ann_return


def calculate_returns_from_prices(prices_df):
    return prices_df.pct_change()


def calculate_returns_from_prices_and_signals(prices_df, signals_df):
    returns_df = calculate_returns_from_prices(prices_df)
    return returns_df * signals_df.shift(1).values


def calculate_ticker_index_from_returns(returns_df):
    ticker_index = returns_df.add(1).cumprod().sub(1) * 100
    return ticker_index

def plot_strategy_return_ticker_and_signals(index_df, signals_df):
    fig = plt.figure()
    plt1 = fig.add_subplot(111, ylabel='Price')
    index_df.plot(ax=plt1, color='rb', lw=2.)

    buy_sell_signals = signals_df.diff()

    # signal_df[['short_mav', 'long_mav']].plot(ax=plt1, lw=2., figsize=(12,8))
    plt1.plot(buy_sell_signals.loc[buy_sell_signals == -1.0].index, buy_sell_signals.short_mav[buy_sell_signals == -1.0], 'v', markersize=10, color='k')
    plt1.plot(buy_sell_signals.loc[buy_sell_signals == 1.0].index, buy_sell_signals.short_mav[buy_sell_signals == 1.0], '^', markersize=10, color='m')
    plt.show()

    pass




if __name__ == '__main__':
    
    ticker_names = [
        'ES=F', #S&P 500
        'RTY=F', #Russell Mini 2000
        'ZN=F', #10 year note
        'ZT=F', #2 year note
        'GC=F', #Gold
        'SI=F', #Silver
        'CL=F', #Crude oil
        'OJ=F', #Orange juice
    ]

    ticker_names = [
        'ES=F', #S&P 500
        'CL=F', #Crude oil
    ]

    prices_df = download_ticker_adj_close_data(ticker_names)
    returns_df = calculate_returns_from_prices(prices_df)
    indices_df = calculate_ticker_index_from_returns(returns_df)

    indices_df.plot()

    ma_signals = simple_moving_average_strategy(prices_df)
    strategy_returns = calculate_returns_from_prices_and_signals(prices_df, ma_signals)
    indices_strategy_df = calculate_ticker_index_from_returns(strategy_returns)

    indices_strategy_df.plot()

    plot_strategy_return_ticker_and_signals(indices_strategy_df, ma_signals)


    ticker_dict = {}
    for ticker in ticker_names:
        ticker_dict[ticker] = download_ticker_data(ticker)

    combined_portfolio = combine_tickers(ticker_dict)
    combined_portfolio_rets = calc_returns(combined_portfolio)

    fig = px.line(
        combined_portfolio_rets, 
        x=combined_portfolio_rets.index, 
        y=combined_portfolio_rets.columns, 
        title='Cumulative Returns of Indices (2010-2020)'
        )
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Cumulative Return in %')
    fig.show()

    # print(combined_portfolio)

    signal_portfolio = combined_portfolio.apply(lambda x: smac_strategy(x))

    print('a')
    
    # ticker_dict['ES=F']['sma'] = sma(ticker_dict['ES=F']['Adj Close'], period=10)
    # plot_series(ticker_dict['ES=F']['sma'], ticker_dict['ES=F']['Adj Close'])

    print("Finish")