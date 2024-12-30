"""
Student Name: Waleed Elsakka
GT User ID: welsakka3
GT ID: 904053428
"""

import datetime as dt
import TheoreticallyOptimalStrategy as tos
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt
from indicators import *

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

def daily_rets(port_val):
    #Convert portfolio value to daily returns
    daily_ret = [(port_val.iloc[x] - port_val.iloc[x - 1]) / port_val.iloc[x] for x in range(len(port_val))]
    daily_ret = daily_ret[1:]
    return daily_ret


if __name__ == "__main__":
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    df_trades = tos.testPolicy(symbol="JPM", sd=sd, ed=ed, sv=100000)

    #Generate Benchmark
    columns = ["Date","Symbol","Order","Shares"]
    dates = pd.date_range(sd, ed)
    prices_all = get_data(["JPM"], dates)
    price = pd.DataFrame(index=prices_all.index, data=prices_all["JPM"])
    price.index = pd.to_datetime(prices_all.index)
    price.index = price.index.strftime('%Y-%m-%d')
    benchmark_trades = pd.DataFrame(None, range(len(price.index)), columns)
    for x in range(len(price.index)):
        benchmark_trades.iloc[x] = [price.index[x],"JPM","NOTHING", 0]
    benchmark_trades.loc[0] = [price.index[0], "JPM", "BUY", 1000]

    strategy = compute_portvals(df_trades, start_val=100000, commission=0, impact=0)
    strategy = strategy / strategy.iloc[0]
    benchmark = compute_portvals(benchmark_trades, start_val=100000, commission=0, impact=0)
    benchmark = benchmark / benchmark.iloc[0]

    #plot TOS vs benchmark
    figure, axis = plt.subplots()
    strategy.plot(ax=axis, label="Theoretically Optimal Portfolio", color="red")
    benchmark.plot(ax=axis, label="Benchmark", color="purple")
    plt.legend(["Theoretically Optimal Portfolio", "Benchmark"])
    ymin = 0
    ymax = 7
    plt.ylim([ymin, ymax])
    plt.grid(True)
    plt.xlabel('Date')  # Label for the x-axis
    plt.ylabel('Portfolio Value Normalized')  # Label for the y-axis
    plt.title('Theoretically Optimal Portfolio Performance vs. Benchmark (JPM)')  # Title of the plot
    plt.savefig('images/tos.png', format='png')

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
    print(f"Cumulative Return of TOS: {cr}")
    print(f"Cumulative Return of Benchmark : {cr_b}")
    print()
    print(f"Standard Deviation of TOS: {sddr}")
    print(f"Standard Deviation of Benchmark : {sddr_b}")
    print()
    print(f"Average Daily Return of TOS: {adr}")
    print(f"Average Daily Return of Benchmark : {adr_b}")
    print()

    # call indicators code
    # Calculate indicators
    ema(price)
    rsi(price)
    macd(price)
    momentum(price)
    cci(price)
