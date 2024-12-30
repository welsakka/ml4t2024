"""
Student Name: Waleed Elsakka
GT User ID: welsakka3
GT ID: 904053428
"""

import datetime as dt
import os
import numpy as np
import pandas as pd
from util import get_data, plot_data


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

def testPolicy(
    symbol="AAPL",
    sd=dt.datetime(2010, 1, 1),
    ed=dt.datetime(2011,12,31),
    sv = 100000
):
    """
    PSEUDOCODE

    Create two pointers for TOS: current day price and next day price
    iterate twice: once to create long positions and once to create short positions
    long:
        if no position, check if next day is higher than current. if yes, create buy order. else, iterate until true
        if next day price is higher than current, continue holding position, create NOTHING order
        if next day price is cheaper, create sell order
        terminate at end of prices list
    short:
        opposite of long position strategy
    generate portfolio value dataframe from marketsimcode and return
    """

    columns = ["Date", "Symbol", "Order", "Shares"]

    #Get prices
    dates = pd.date_range(sd, ed)
    prices_all = get_data([symbol], dates)
    price = pd.DataFrame(index=prices_all.index, data=prices_all[symbol])
    price.index = pd.to_datetime(prices_all.index)
    price.index = price.index.strftime('%Y-%m-%d')

    #longs
    trades = pd.DataFrame(None,None, columns)
    position = False
    current = 0
    next = 1
    while next < len(price):
        next_price = price.iloc[next, 0]
        if price.iloc[next,0] > price.iloc[current,0]:
            if position is False:
                trades.loc[len(trades)] = [price.index[current], symbol, "BUY", 1000]
                position = True
            else:
                trades.loc[len(trades)] = [price.index[current], symbol, "NOTHING", 0]
        elif price.iloc[next,0] < price.iloc[current,0]:
            if position is True:
                trades.loc[len(trades)] = [price.index[current], symbol, "SELL", 1000]
                position = False
            else:
                trades.loc[len(trades)] = [price.index[current], symbol, "NOTHING", 0]
        current = current + 1
        next = next + 1

    #shorts
    position = False
    current = 0
    next = 1
    while next < len(price):
        if price.iloc[next,0] > price.iloc[current,0]:
            if position is True:
                trades.loc[len(trades)] = [price.index[current], symbol, "BUY", 1000]
                position = False
            else:
                trades.loc[len(trades)] = [price.index[current], symbol, "NOTHING", 0]
        elif price.iloc[next,0] < price.iloc[current,0]:
            if position is False:
                trades.loc[len(trades)] = [price.index[current], symbol, "SELL", 1000]
                position = True
            else:
                trades.loc[len(trades)] = [price.index[current], symbol, "NOTHING", 0]
        current = current + 1
        next = next + 1
    trades.loc[len(trades)] = [price.index[current], symbol, "NOTHING", 0]

    return trades

if __name__ == "__main__":
    testPolicy("JPM", dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), 100000)