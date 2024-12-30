"""
Student Name: Waleed Elsakka
GT User ID: welsakka3
GT ID: 904053428
"""

import numpy as np
import pandas as pd
from util import get_data, plot_data
import matplotlib.pyplot as plt


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "welsakka3"


def study_group():
    """
    :return: A comma separated string of GT_Name of each member of your study group
    :rtype: str
    """
    return "welsakka3"

"""
INDICATORS
    EMA
    Momentum
    MACD
    RSI
    CCI
"""


def ema(prices, window=20):
    """
    :return: Exponential moving average of the prices
    :rtype: Dataframe
    """
    ema = prices.ewm(span=window, min_periods=0).mean()

    figure, axis = plt.subplots()
    ema.plot(ax=axis, label="EMA", color="red")
    prices.plot(ax=axis, label="JPM Price", color="green")
    plt.title('Exponential Moving Average (EMA)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(["EMA", "JPM"])
    plt.savefig('images/ema.png')

    return ema


def rsi(prices, window=20):
    """
    :return: Relative Strength Index
    :rtype: Dataframe
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Plot
    figure, axis = plt.subplots()
    rsi.plot(ax=axis, label='RSI', color='Purple')
    plt.title('Relative Strength Index (RSI)')
    plt.xlabel('Date')
    plt.ylabel('RSI Value')
    plt.ylim([0, 100])
    plt.legend(["RSI"])
    plt.grid(True)
    plt.savefig('images/rsi.png')

    return rsi

def macd(prices, short_window=10, long_window=20):
    """
    :return: Moving Average Convergence Divergence
    :rtype: Dataframe
    """
    macd = prices.ewm(span=short_window, min_periods=0).mean() - prices.ewm(span=long_window, min_periods=0).mean()
    signal = macd.ewm(span=9, min_periods=0).mean()

    # Plot
    figure, axis = plt.subplots()
    macd.plot(ax=axis, label='MACD', color='red')
    signal.plot(ax=axis, label='Signal', color='green')
    plt.title('Moving Average Convergence Divergence (MACD)')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend(["MACD", "Signal"])
    plt.grid(True)
    plt.savefig('images/macd.png')

    return macd

def momentum(prices):
    """Calculate and plot Momentum Indicator"""
    momentum = prices['JPM'] - prices['JPM'].shift(15)

    # Plot
    figure, axis = plt.subplots()
    momentum.plot(ax=axis, label='Momentum', color='green')
    plt.title('Momentum Indicator')
    plt.xlabel('Date')
    plt.ylabel('Momentum Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('images/momentum.png')

    return momentum


def cci(prices, period=20):
    """
    Commodity Channel Index (CCI)
    :param period: Time period to calculate CCI
    :return: CCI values
    """
    # Calculate the Moving Average of the Typical Price
    SMA_average = prices.rolling(window=period).mean()
    # Calculate the Mean Deviation
    mean_deviation = prices.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    # Calculate the Commodity Channel Index (CCI)
    cci = (prices - SMA_average) / (0.015 * mean_deviation)

    # Plot
    figure, axis = plt.subplots()
    cci.plot(ax=axis, label='CCI', color='black')
    plt.title('Commodity Channel Index (CCI)')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend(["CCI"])
    plt.grid(True)
    plt.savefig('images/cci.png')

    return cci


if __name__ == "__main__":
    symbol = ['JPM']
    sd = pd.Timestamp('2008-01-01')
    ed = pd.Timestamp('2009-12-31')
    dates = pd.date_range(sd, ed)
    prices_all = get_data(symbol, dates)
    prices = prices_all[symbol]  # only portfolio symbols

    # Calculate indicators
    ema(prices)
    rsi(prices)
    macd(prices)
    momentum(prices)
    cci(prices)

