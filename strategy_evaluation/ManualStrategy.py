"""
Student Name: Waleed Elsakka
GT User ID: welsakka3
GT ID: 904053428
"""
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from marketsimcode import compute_portvals, daily_rets
from util import get_data, plot_data
from indicators import rsi, ema, cci

class ManualStrategy(object):

    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """
        Constructor method
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

    def addEvidence(
        self,
        symbol="IBM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 1, 1),
        sv=100000,
    ):
        pass



    def testPolicy(
        self,
        symbol="AAPL",
        sd=dt.datetime(2010, 1, 1),
        ed=dt.datetime(2011, 12, 31),
        sv=100000,
    ):
        """
        :return: All trades to execute based on indicators
        :rtype: Dataframe
        """

        # Get prices
        dates = pd.date_range(sd, ed)
        prices_all = get_data([symbol], dates)
        price = pd.DataFrame(index=prices_all.index, data=prices_all[symbol])

        # Create all long and short trades based on indicators
        trades = pd.DataFrame(None, price.index, ["Signal"])
        rsi_symbol = rsi(price, 20)
        ema_symbol = ema(price, 20)
        cci_symbol = cci(price, 20)
        position_long = False
        position_short = False
        rsi_long = False
        ema_long = False
        cci_long = False
        rsi_short = False
        ema_short = False
        cci_short = False
        current = 0
        while current < len(price):
            """
            Buy signals to create a long position:
                if RSI is below 40,
                if EMA is greater than current price,
                if CCI is below -100
            """
            rsi_long = True if rsi_symbol.iloc[current, 0] < 30 else False
            ema_long = True if ema_symbol.iloc[current, 0] > price.iloc[current, 0] else False
            cci_long = True if cci_symbol.iloc[current, 0] < -120 else False

            """
            Sell signals to create a short position:
                if RSI is above 60,
                if EMA is less than current price,
                if CCI is above 100
            """
            rsi_short = True if rsi_symbol.iloc[current, 0] > 70 else False
            ema_short = True if ema_symbol.iloc[current, 0] < price.iloc[current, 0] else False
            cci_short = True if cci_symbol.iloc[current, 0] > 120 else False

            # Create a long postion
            if rsi_long and ema_long and cci_long:
                if position_short is True:
                    trades.iloc[current] = 2000
                    position_short = False
                    position_long = True
                elif position_long is False:
                    trades.iloc[current] = 1000
                    position_long = True
                else:
                    trades.iloc[current] = 0

            # Create a short position
            elif rsi_short and ema_short and cci_short:
                if position_long is True:
                    trades.iloc[current] = -2000
                    position_long = False
                    position_short = True
                elif position_short is False:
                    trades.iloc[current] = -1000
                    position_short = True
                else:
                    trades.iloc[current] = 0

            # Else, hold
            else:
                trades.iloc[current] = 0

            current = current + 1

        return trades



    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "welsakka3"

    def study_group(self):
        """
        :return: A comma separated string of GT_Name of each member of your study group
        :rtype: str
        """
        return "welsakka3"

if __name__ == "__main__":

    # In-Sample Testing
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    columns = ["Date", "Symbol", "Order", "Shares"]

    ms = ManualStrategy(False, 0.0, 0.0)
    res = ms.testPolicy("JPM", sd, ed, 100000)

    #Generate Benchmark
    dates = pd.date_range(sd, ed)
    prices_all = get_data(["JPM"], dates)
    price = pd.DataFrame(index=prices_all.index, data=prices_all["JPM"])
    benchmark_trades = pd.DataFrame(None, range(len(price.index)), columns)
    for x in range(len(price.index)):
        benchmark_trades.iloc[x] = [price.index[x],"JPM","NOTHING", 0]
    benchmark_trades.loc[0] = [price.index[0], "JPM", "BUY", 1000]

    # Create trades
    trades = pd.DataFrame(None, range(len(price.index)), columns)
    for x in range(len(price.index)):
        if res.iloc[x,0] == 0:
            trades.iloc[x] = [price.index[x],"JPM","NOTHING", 0]
        if res.iloc[x,0] == 1000:
            trades.iloc[x] = [price.index[x],"JPM","BUY", 1000]
        if res.iloc[x,0] == 2000:
            trades.iloc[x] = [price.index[x],"JPM","BUY", 2000]
        if res.iloc[x,0] == -1000:
            trades.iloc[x] = [price.index[x],"JPM","SELL", -1000]
        if res.iloc[x,0] == -2000:
            trades.iloc[x] = [price.index[x],"JPM","SELL", -2000]


    # Get portfolio values for strategy vs benchmark
    strategy = compute_portvals(trades, start_val=100000, commission=9.95, impact=0.005)
    strategy = strategy / strategy.iloc[0]
    benchmark = compute_portvals(benchmark_trades, start_val=100000, commission=9.95, impact=0.005)
    benchmark = benchmark / benchmark.iloc[0]

    # plot strategy vs benchmark
    plt.figure(figsize=(12, 6))
    benchmark.index = pd.to_datetime(benchmark.index)
    strategy.index = pd.to_datetime(strategy.index)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=9, prune=None))
    plt.plot(strategy, label="Manual Strategy", color="red")
    plt.plot(benchmark, label="Benchmark", color="purple")
    plt.legend(["Manual Strategy", "Benchmark"])
    plt.grid(True)
    plt.xlabel('Date')  # Label for the x-axis
    plt.ylabel('Portfolio Value Normalized')  # Label for the y-axis
    plt.title('In-Sample: Manual Strategy vs. Benchmark (JPM)')  # Title of the plot

    # plot long orders and short orders entry points
    long_orders = trades[trades["Order"].isin(["BUY"])]
    short_orders = trades[trades["Order"].isin(["SELL"])]
    long_orders['Date'] = pd.to_datetime(long_orders['Date'])
    short_orders['Date'] = pd.to_datetime(short_orders['Date'])
    for _, order in long_orders.iterrows():
        plt.axvline(x=order["Date"], color='blue')
    for _, order in short_orders.iterrows():
        plt.axvline(x=order["Date"], color='black')

    filename = "images/ms_1.png"
    plt.savefig(filename, format='png')

    # Calculate statistics
    cr = (strategy.iloc[-1]/strategy.iloc[0]) - 1
    adr = np.mean(daily_rets(strategy))
    sddr = np.std(daily_rets(strategy))

    cr_b = (benchmark.iloc[-1] / benchmark.iloc[0]) - 1
    adr_b = np.mean(daily_rets(benchmark))
    sddr_b = np.std(daily_rets(benchmark))

    # Compare portfolio against $SPX
    print(f"Date Range: {sd} to {ed}")
    print()
    print(f"Cumulative Return of Strategy: {cr}")
    print(f"Cumulative Return of Benchmark : {cr_b}")
    print()
    print(f"Standard Deviation of Strategy: {sddr}")
    print(f"Standard Deviation of Benchmark : {sddr_b}")
    print()
    print(f"Average Daily Return of Strategy: {adr}")
    print(f"Average Daily Return of Benchmark : {adr_b}")
    print()

    plt.close()


    # Out of Sample Testing
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)

    columns = ["Date", "Symbol", "Order", "Shares"]
    ms = ManualStrategy(False, 0.0, 0.0)
    res = ms.testPolicy("JPM", sd, ed, 100000)

    # Generate Benchmark
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
    trades = pd.DataFrame(None, range(len(price.index)), columns)
    for x in range(len(price.index)):
        if res.iloc[x,0] == 0:
            trades.iloc[x] = [price.index[x],"JPM","NOTHING", 0]
        if res.iloc[x,0] == 1000:
            trades.iloc[x] = [price.index[x],"JPM","BUY", 1000]
        if res.iloc[x,0] == 2000:
            trades.iloc[x] = [price.index[x],"JPM","BUY", 2000]
        if res.iloc[x,0] == -1000:
            trades.iloc[x] = [price.index[x],"JPM","SELL", -1000]
        if res.iloc[x,0] == -2000:
            trades.iloc[x] = [price.index[x],"JPM","SELL", -2000]

    # Get portfolio values for strategy vs benchmark
    strategy = compute_portvals(trades, start_val=100000, commission=9.95, impact=0.005)
    strategy = strategy / strategy.iloc[0]
    benchmark = compute_portvals(benchmark_trades, start_val=100000, commission=9.95, impact=0.005)
    benchmark = benchmark / benchmark.iloc[0]

    # plot strategy vs benchmark
    plt.figure(figsize=(12, 6))
    benchmark.index = pd.to_datetime(benchmark.index)
    strategy.index = pd.to_datetime(strategy.index)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=9, prune=None))
    plt.plot(strategy, label="Manual Strategy", color="red")
    plt.plot(benchmark, label="Benchmark", color="purple")
    plt.legend(["Manual Strategy", "Benchmark"])
    plt.grid(True)
    plt.xlabel('Date')  # Label for the x-axis
    plt.ylabel('Portfolio Value Normalized')  # Label for the y-axis
    plt.title('Out-Sample: Manual Strategy vs. Benchmark (JPM)')  # Title of the plot

    # plot long orders and short orders entry points
    long_orders = trades[trades["Order"].isin(["BUY"])]
    short_orders = trades[trades["Order"].isin(["SELL"])]
    long_orders['Date'] = pd.to_datetime(long_orders['Date'])
    short_orders['Date'] = pd.to_datetime(short_orders['Date'])
    for _, order in long_orders.iterrows():
        plt.axvline(x=order["Date"], color='blue')
    for _, order in short_orders.iterrows():
        plt.axvline(x=order["Date"], color='black')

    filename = "images/ms_2.png"
    plt.savefig(filename, format='png')


    # Calculate statistics
    cr = (strategy.iloc[-1] / strategy.iloc[0]) - 1
    adr = np.mean(daily_rets(strategy))
    sddr = np.std(daily_rets(strategy))

    cr_b = (benchmark.iloc[-1] / benchmark.iloc[0]) - 1
    adr_b = np.mean(daily_rets(benchmark))
    sddr_b = np.std(daily_rets(benchmark))

    # Compare portfolio against $SPX
    print(f"Date Range: {sd} to {ed}")
    print()
    print(f"Cumulative Return of Strategy: {cr}")
    print(f"Cumulative Return of Benchmark : {cr_b}")
    print()
    print(f"Standard Deviation of Strategy: {sddr}")
    print(f"Standard Deviation of Benchmark : {sddr_b}")
    print()
    print(f"Average Daily Return of Strategy: {adr}")
    print(f"Average Daily Return of Benchmark : {adr_b}")
    print()

    plt.close()
