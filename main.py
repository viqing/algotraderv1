import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
import math

from strategies import simple_moving_average_strategy

import seaborn as sns
import plotly.express as px



# PyPortfolioOpt
# from pyfopt.expected_returns import mean_historical_return
# from pypfopt.risk_models import CovarianceShrinkage


def download_ticker_data(ticker, period='max'):
    return yf.download(tickers=ticker, period=period)


def download_ticker_adj_close_data(ticker, period='max'):
    df = yf.download(tickers=ticker, period=period)['Adj Close']

    if len(ticker) == 1:
        df = pd.DataFrame(df)
        df = df.rename(columns={'Adj Close': ticker[0]})
        return df

    return df


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

def plot_strategy_return_ticker_and_signals(index_df, signals_df, n_cols=3):
    
    n_rows = math.ceil(len(index_df.columns) / n_cols)
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols)

    buy_sell_signals = signals_df.diff()

    fig.tight_layout()
    for index, col in enumerate(index_df.columns):
        i = math.floor(index / 3)
        j = index % n_cols

        # handle the fact that the ax array becomes 1-dimensional when shape is (1, z)
        if len(index_df.columns)>n_cols:
            index_df[col].plot(ax=ax[i, j], lw=2.)
            b_s_signal = buy_sell_signals.iloc[:, index]
            ax[i, j].plot(b_s_signal.loc[b_s_signal == -1.0].index, b_s_signal.loc[b_s_signal == -1.0], 'v', markersize=10, color='k')
            ax[i, j].plot(b_s_signal.loc[b_s_signal ==  1.0].index, b_s_signal.loc[b_s_signal ==  1.0], '^', markersize=10, color='k')
            ax[i, j].set_ylabel(col)
        else:
            index_df[col].plot(ax=ax[j], lw=2.)
            ax[j].plot(b_s_signal.loc[b_s_signal == -1.0].index, b_s_signal.loc[b_s_signal == -1.0], 'v', markersize=10, color='k')
            ax[j].plot(b_s_signal.loc[b_s_signal ==  1.0].index, b_s_signal.loc[b_s_signal ==  1.0], '^', markersize=10, color='k')
            ax[j].set_ylabel(col)

    #TODO Show buy/sell signals
    buy_sell_signals = signals_df.diff()

    # signal_df[['short_mav', 'long_mav']].plot(ax=plt1, lw=2., figsize=(12,8))
    # plt1.plot(buy_sell_signals.lo[buy_sell_signals == -1.0].index, buy_sell_signals.short_mav[buy_sell_signals == -1.0], 'v', markersize=10, color='k')
    # plt1.plot(buy_sell_signals.loc[buy_sell_signals == 1.0].index, buy_sell_signals.short_mav[buy_sell_signals == 1.0], '^', markersize=10, color='m')
    #bla bla bla
    plt.show()


def calculate_portfolio_weights(signals_df):

    weights_df = pd.DataFrame(columns=signals_df.columns,
                              index=signals_df.index)

    weights_df = signals_df.div(signals_df.abs().sum(axis=1), axis=0)

    #for index, row in signals_df.iterrows():

        #total = row.abs().sum()
        #weights = row.div(total)
        #weights_df[index] = weights

    print(weights_df.tail())
    pass

class Portfolio:
    def __init__(self):
        self.portfolio_tickers = []

    def add_tickers(self, ticker):
        if isinstance(ticker, list):
            self.portfolio_tickers.extend(ticker)
        else:
            self.portfolio_tickers.append(ticker)

    
if __name__ == '__main__':
    
    ticker_names = [
        'CL=F', #Crude oil
        'ES=F', #S&P 500
    ]

    portfolio_1 = Portfolio()
    portfolio_1.add_tickers(ticker_names)
    print(portfolio_1.portfolio_tickers)

    prices = download_ticker_adj_close_data(portfolio_1.portfolio_tickers)
    signals = simple_moving_average_strategy(prices)
    calculate_portfolio_weights(signals)

    
    print(signals.head())

    ma_signals = simple_moving_average_strategy(prices)
    strategy_returns = calculate_returns_from_prices_and_signals(prices, signals)
    indices_strategy_df = calculate_ticker_index_from_returns(strategy_returns)

    indices_strategy_df.plot()

    #plot_strategy_return_ticker_and_signals(indices_strategy_df.iloc[:, 0], ma_signals.iloc[:, 0], n_cols=3)

