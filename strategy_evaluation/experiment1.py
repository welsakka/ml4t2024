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

if __name__ == "__main__":
    print("One does not simply think up a strategy")

    # Training
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)

    strategyLearner = sl.StrategyLearner(commission=9.95, impact=0.005)
    strategyLearner.add_evidence("JPM", sd, ed, 100000)

    manualStrategy = ms.ManualStrategy()

    # In-Sample Testing
    res_sl = strategyLearner.testPolicy("JPM", sd, ed, 100000)
    res_ms = manualStrategy.testPolicy("JPM", sd, ed, 100000)

    # Generate Benchmark
    columns = ["Date", "Symbol", "Order", "Shares"]
    dates = pd.date_range(sd, ed)
    prices_all = get_data(["JPM"], dates)
    price = prices_all["JPM"]
    benchmark_trades = pd.DataFrame(None, range(len(price.index)), columns)
    for x in range(len(price.index)):
        benchmark_trades.iloc[x] = [price.index[x], "JPM", "NOTHING", 0]
    benchmark_trades.loc[0] = [price.index[0], "JPM", "BUY", 1000]

    # Create trades
    trades_sl = pd.DataFrame(None, range(len(price.index)), columns)
    for x in range(len(price.index)):
        if res_sl.iloc[x,0] == 0:
            trades_sl.iloc[x] = [price.index[x],"JPM","NOTHING", 0]
        if res_sl.iloc[x,0] == 1000:
            trades_sl.iloc[x] = [price.index[x],"JPM","BUY", 1000]
        if res_sl.iloc[x,0] == 2000:
            trades_sl.iloc[x] = [price.index[x],"JPM","BUY", 2000]
        if res_sl.iloc[x,0] == -1000:
            trades_sl.iloc[x] = [price.index[x],"JPM","SELL", -1000]
        if res_sl.iloc[x,0] == -2000:
            trades_sl.iloc[x] = [price.index[x],"JPM","SELL", -2000]

    # Create trades
    trades_ms = pd.DataFrame(None, range(len(price.index)), columns)
    for x in range(len(price.index)):
        if res_ms.iloc[x,0] == 0:
            trades_ms.iloc[x] = [price.index[x],"JPM","NOTHING", 0]
        if res_ms.iloc[x,0] == 1000:
            trades_ms.iloc[x] = [price.index[x],"JPM","BUY", 1000]
        if res_ms.iloc[x,0] == 2000:
            trades_ms.iloc[x] = [price.index[x],"JPM","BUY", 2000]
        if res_ms.iloc[x,0] == -1000:
            trades_ms.iloc[x] = [price.index[x],"JPM","SELL", -1000]
        if res_ms.iloc[x,0] == -2000:
            trades_ms.iloc[x] = [price.index[x],"JPM","SELL", -2000]

    # Get portfolio values for strategy vs benchmark
    strategy_sl = compute_portvals(trades_sl, start_val=100000, commission=9.95, impact=0.005)
    strategy_sl = strategy_sl / strategy_sl.iloc[0]
    strategy_ms = compute_portvals(trades_ms, start_val=100000, commission=9.95, impact=0.005)
    strategy_ms = strategy_ms / strategy_ms.iloc[0]
    benchmark = compute_portvals(benchmark_trades, start_val=100000, commission=9.95, impact=0.005)
    benchmark = benchmark / benchmark.iloc[0]

    # plot strategy vs benchmark
    plt.figure(figsize=(12, 6))
    benchmark.index = pd.to_datetime(benchmark.index)
    strategy_sl.index = pd.to_datetime(strategy_sl.index)
    strategy_ms.index = pd.to_datetime(strategy_ms.index)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=9, prune=None))
    plt.plot(strategy_sl, label="Strategy Learner", color="red")
    plt.plot(strategy_ms, label="Manual Strategy", color="green")
    plt.plot(benchmark, label="Benchmark", color="purple")
    plt.legend(["Strategy Learner", "Manual Strategy", "Benchmark"])
    plt.grid(True)
    plt.xlabel('Date')  # Label for the x-axis
    plt.ylabel('Portfolio Value Normalized')  # Label for the y-axis
    plt.title('In-Sample: Strategy Learner / Manual Strategy / Benchmark')  # Title of the plot

    filename = "images/experiment_1_in_sample.png"
    plt.savefig(filename, format='png')

    plt.close()


    # Out-Sample Testing
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    res_sl = strategyLearner.testPolicy("JPM", sd, ed, 100000)
    res_ms = manualStrategy.testPolicy("JPM", sd, ed, 100000)

    # Generate Benchmark
    columns = ["Date", "Symbol", "Order", "Shares"]
    dates = pd.date_range(sd, ed)
    prices_all = get_data(["JPM"], dates)
    price = pd.DataFrame(index=prices_all.index, data=prices_all["JPM"])
    price.index = pd.to_datetime(prices_all.index)
    price.index = price.index.strftime('%Y-%m-%d')
    benchmark_trades = pd.DataFrame(None, range(len(price.index)), columns)
    for x in range(len(price.index)):
        benchmark_trades.iloc[x] = [price.index[x], "JPM", "NOTHING", 0]
    benchmark_trades.loc[0] = [price.index[0], "JPM", "BUY", 1000]

    # Create trades
    trades_sl = pd.DataFrame(None, range(len(price.index)), columns)
    for x in range(len(price.index)):
        if res_sl.iloc[x, 0] == 0:
            trades_sl.iloc[x] = [price.index[x], "JPM", "NOTHING", 0]
        if res_sl.iloc[x, 0] == 1000:
            trades_sl.iloc[x] = [price.index[x], "JPM", "BUY", 1000]
        if res_sl.iloc[x, 0] == 2000:
            trades_sl.iloc[x] = [price.index[x], "JPM", "BUY", 2000]
        if res_sl.iloc[x, 0] == -1000:
            trades_sl.iloc[x] = [price.index[x], "JPM", "SELL", -1000]
        if res_sl.iloc[x, 0] == -2000:
            trades_sl.iloc[x] = [price.index[x], "JPM", "SELL", -2000]

    # Create trades
    trades_ms = pd.DataFrame(None, range(len(price.index)), columns)
    for x in range(len(price.index)):
        if res_ms.iloc[x, 0] == 0:
            trades_ms.iloc[x] = [price.index[x], "JPM", "NOTHING", 0]
        if res_ms.iloc[x, 0] == 1000:
            trades_ms.iloc[x] = [price.index[x], "JPM", "BUY", 1000]
        if res_ms.iloc[x, 0] == 2000:
            trades_ms.iloc[x] = [price.index[x], "JPM", "BUY", 2000]
        if res_ms.iloc[x, 0] == -1000:
            trades_ms.iloc[x] = [price.index[x], "JPM", "SELL", -1000]
        if res_ms.iloc[x, 0] == -2000:
            trades_ms.iloc[x] = [price.index[x], "JPM", "SELL", -2000]

    # Get portfolio values for strategy vs benchmark
    strategy_sl = compute_portvals(trades_sl, start_val=100000, commission=9.95, impact=0.005)
    strategy_sl = strategy_sl / strategy_sl.iloc[0]
    strategy_ms = compute_portvals(trades_ms, start_val=100000, commission=9.95, impact=0.005)
    strategy_ms = strategy_ms / strategy_ms.iloc[0]
    benchmark = compute_portvals(benchmark_trades, start_val=100000, commission=9.95, impact=0.005)
    benchmark = benchmark / benchmark.iloc[0]

    # plot strategy vs benchmark
    plt.figure(figsize=(12, 6))
    benchmark.index = pd.to_datetime(benchmark.index)
    strategy_sl.index = pd.to_datetime(strategy_sl.index)
    strategy_ms.index = pd.to_datetime(strategy_ms.index)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=9, prune=None))
    plt.plot(strategy_sl, label="Strategy Learner", color="red")
    plt.plot(strategy_ms, label="Manual Strategy", color="green")
    plt.plot(benchmark, label="Benchmark", color="purple")
    plt.legend(["Strategy Learner", "Manual Strategy", "Benchmark"])
    plt.grid(True)
    plt.xlabel('Date')  # Label for the x-axis
    plt.ylabel('Portfolio Value Normalized')  # Label for the y-axis
    plt.title('Out-Sample: Strategy Learner / Manual Strategy / Benchmark')  # Title of the plot

    filename = "images/experiment_1_out_sample.png"
    plt.savefig(filename, format='png')

    plt.close()
