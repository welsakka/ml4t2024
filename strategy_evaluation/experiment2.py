"""
Student Name: Waleed Elsakka
GT User ID: welsakka3
GT ID: 904053428
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from marketsimcode import compute_portvals, daily_rets
from util import get_data, plot_data
import StrategyLearner as sl
import ManualStrategy as ms
import datetime as dt
import StrategyLearner as sl

def calulate_learner_portvals(symbol, sd, ed, impact):
    learner = sl.StrategyLearner(impact=impact)
    learner.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=100000)
    res = learner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=100000)

    #get price index to construct trades
    columns = ["Date", "Symbol", "Order", "Shares"]
    dates = pd.date_range(sd, ed)
    prices_all = get_data([symbol], dates)
    price = pd.DataFrame(index=prices_all.index, data=prices_all[symbol])
    price.index = pd.to_datetime(prices_all.index)
    price.index = price.index.strftime('%Y-%m-%d')

    # Create trades
    trades = pd.DataFrame(None, range(len(price.index)), columns)
    for x in range(len(price.index)):
        if res.iloc[x,0] == 0:
            trades.iloc[x] = [price.index[x],symbol,"NOTHING", 0]
        if res.iloc[x,0] == 1000:
            trades.iloc[x] = [price.index[x],symbol,"BUY", 1000]
        if res.iloc[x,0] == 2000:
            trades.iloc[x] = [price.index[x],symbol,"BUY", 2000]
        if res.iloc[x,0] == -1000:
            trades.iloc[x] = [price.index[x],symbol,"SELL", -1000]
        if res.iloc[x,0] == -2000:
            trades.iloc[x] = [price.index[x],symbol,"SELL", -2000]

    portvals = compute_portvals(trades, start_val=100000, commission=0, impact=impact)
    portvals = portvals / portvals.iloc[0]

    # Calculate statistics
    cr = (portvals.iloc[-1]/portvals.iloc[0]) - 1
    adr = np.mean(daily_rets(portvals))
    sddr = np.std(daily_rets(portvals))

    # Compare portfolio against $SPX
    print(f"Date Range: {sd} to {ed}")
    print()
    print(f"Cumulative Return of Strategy with impact {impact}: {cr}")
    print()
    print(f"Standard Deviation of Strategy with impact {impact}: {sddr}")
    print()
    print(f"Average Daily Return of Strategy with impact {impact}: {adr}")
    print()

    return portvals

if __name__ == "__main__":
    print("One does not simply think up a strategy")

    # Training
    impacts = [0.001, 0.005, 0.05, 0.5, 1]
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)

    portvals = []
    for impact in impacts:
        portvals.append(calulate_learner_portvals("JPM", sd, ed, impact))

    # plot strategy vs benchmark
    plt.figure(figsize=(12, 6))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=9, prune=None))
    i = 0
    for portval in portvals:
        portval.index = pd.to_datetime(portval.index)
        plt.plot(portval, label=f"Impact {impacts[i]}")
        i += 1
    plt.legend()
    plt.grid(True)
    plt.xlabel('Date')  # Label for the x-axis
    plt.ylabel('Portfolio Value Normalized')  # Label for the y-axis
    plt.title('Strategy Learner Impact Experiment')  # Title of the plot
    filename = "images/experiment_2.png"
    plt.savefig(filename, format='png')

    plt.close()

