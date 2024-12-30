""""""
"""  		  	   		 	   		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	   		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		 	   		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	   		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		 	   		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		 	   		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		 	   		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	   		  		  		    	 		 		   		 		  
or edited.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		 	   		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		 	   		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	   		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
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
from indicators import rsi, ema, cci
import RTLearner as rtl
import BagLearner as bl
import datetime as dt

  		  	   		 	   		  		  		    	 		 		   		 		  
class StrategyLearner(object):  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	   		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output.  		  	   		 	   		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		 	   		  		  		    	 		 		   		 		  
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		 	   		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		 	   		  		  		    	 		 		   		 		  
    :param commission: The commission amount charged, defaults to 0.0  		  	   		 	   		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    # constructor  		  	   		 	   		  		  		    	 		 		   		 		  
    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		 	   		  		  		    	 		 		   		 		  
        """  		  	   		 	   		  		  		    	 		 		   		 		  
        Constructor method  		  	   		 	   		  		  		    	 		 		   		 		  
        """  		  	   		 	   		  		  		    	 		 		   		 		  
        self.verbose = verbose  		  	   		 	   		  		  		    	 		 		   		 		  
        self.impact = impact  		  	   		 	   		  		  		    	 		 		   		 		  
        self.commission = commission
        self.learner = None
  		  	   		 	   		  		  		    	 		 		   		 		  
    # this method will create a Random Tree Forest using a BagLearner
    def add_evidence(  		  	   		 	   		  		  		    	 		 		   		 		  
        self,  		  	   		 	   		  		  		    	 		 		   		 		  
        symbol="IBM",  		  	   		 	   		  		  		    	 		 		   		 		  
        sd=dt.datetime(2008, 1, 1),  		  	   		 	   		  		  		    	 		 		   		 		  
        ed=dt.datetime(2009, 1, 1),  		  	   		 	   		  		  		    	 		 		   		 		  
        sv=10000,  		  	   		 	   		  		  		    	 		 		   		 		  
    ):  		  	   		 	   		  		  		    	 		 		   		 		  
        """  		  	   		 	   		  		  		    	 		 		   		 		  
        Trains your strategy learner over a given time frame.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol to train on  		  	   		 	   		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		 	   		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	   		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		 	   		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	   		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		 	   		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		 	   		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		 	   		  		  		    	 		 		   		 		  
        """  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
        # PSEUDOCODE
        # Import RTLearner
        # Import BagLearner
        # Create a BagLearner instance with x number of RTLearners
        # Train the learner with Data
            # Data X has factors:
                # Every indicator value for a date
            # Data Y is the signal to buy or sell
                # Y is determined through configurable variables YBUY and YSELL that represent a return value
                # from todays price to N day's price.
                # Retrieve the price of JPM between sd and ed, calculate returns for each date, and construct Y
                # based on if the returns are high enough
        # Use X and Y to train the RTLearners.
        # Return tree

        # Get prices
        dates = pd.date_range(sd, ed)
        prices_all = get_data([symbol], dates)
        price = pd.DataFrame(index=prices_all.index, data=prices_all[symbol])
        # price.index = pd.to_datetime(prices_all.index)
        # price.index = price.index.strftime('%Y-%m-%d')

        # Get Indicator values for the price of the symbol within start date and end date
        rsi_jpm = rsi(price, 20)
        ema_jpm = ema(price, 20)
        cci_jpm = cci(price, 20)

        # Construct data X
        # columns = ["RSI", "EMA", "CCI"]
        train_x = pd.DataFrame()
        train_x = pd.concat([train_x, rsi_jpm], axis=1)
        train_x = pd.concat([train_x, ema_jpm], axis=1)
        train_x = pd.concat([train_x, cci_jpm], axis=1)
        train_x = train_x.to_numpy()

        # Creating Y data, where +1 = Buy, -1 = Sell, 0 = Nothing
        # N = 20
        # Calculate returns between both dates and determine which signal to make
        Y = np.empty([len(price),1])
        ybuy = .04
        ysell = -.04

        # Factor in impact on ybuy and ysell
        ybuy = ybuy + (self.impact * ybuy)
        ysell = ysell + (self.impact * ysell)

        current = 0
        n = 20
        while n < len(price):
            daily_return = (price.iloc[n,0] / price.iloc[current,0]) - 1.0
            if daily_return > ybuy:
                Y[current] = 1
            elif daily_return < ysell:
                Y[current] = -1
            else: Y[current] = 0
            current += 1
            n += 1
        # Fill the rest with Nothing
        Y[current:n] = 0

        strategy_bl = bl.BagLearner(rtl.RTLearner, kwargs={"leaf_size": 35}, bags=90, boost=False, verbose=False)
        strategy_bl.add_evidence(train_x, Y)
        self.learner = strategy_bl
  		  	   		 	   		  		  		    	 		 		   		 		  
    # this method should use the existing policy and test it against new data  		  	   		 	   		  		  		    	 		 		   		 		  
    def testPolicy(  		  	   		 	   		  		  		    	 		 		   		 		  
        self,  		  	   		 	   		  		  		    	 		 		   		 		  
        symbol="IBM",  		  	   		 	   		  		  		    	 		 		   		 		  
        sd=dt.datetime(2009, 1, 1),  		  	   		 	   		  		  		    	 		 		   		 		  
        ed=dt.datetime(2010, 1, 1),  		  	   		 	   		  		  		    	 		 		   		 		  
        sv=10000,  		  	   		 	   		  		  		    	 		 		   		 		  
    ):  		  	   		 	   		  		  		    	 		 		   		 		  
        """  		  	   		 	   		  		  		    	 		 		   		 		  
        Tests your learner using data outside of the training data  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol that you trained on on  		  	   		 	   		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		 	   		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	   		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		 	   		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	   		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		 	   		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		 	   		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		 	   		  		  		    	 		 		   		 		  
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		 	   		  		  		    	 		 		   		 		  
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		 	   		  		  		    	 		 		   		 		  
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		 	   		  		  		    	 		 		   		 		  
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		 	   		  		  		    	 		 		   		 		  
        :rtype: pandas.DataFrame  		  	   		 	   		  		  		    	 		 		   		 		  
        """

        # Get prices
        dates = pd.date_range(sd, ed)
        prices_all = get_data([symbol], dates)
        price = pd.DataFrame(index=prices_all.index, data=prices_all[symbol])
        # price.index = pd.to_datetime(prices_all.index, format='%Y-%m-%d')
        # price.index = price.index.strftime('%Y-%m-%d')

        # Get Indicator values for the price of the symbol within start date and end date
        rsi_jpm = rsi(price, 20)
        ema_jpm = ema(price, 20)
        cci_jpm = cci(price, 20)

        # Construct data X
        # columns_x = ["RSI", "EMA", "CCI"]
        test_x = pd.DataFrame()
        test_x = pd.concat([test_x, rsi_jpm], axis=1)
        test_x = pd.concat([test_x, ema_jpm], axis=1)
        test_x = pd.concat([test_x, cci_jpm], axis=1)
        test_x = test_x.to_numpy()

        # Get data Y
        Y = self.learner.query(test_x)

        # Construct trades
        position_long = False
        position_short = False
        trades = pd.DataFrame(None, price.index, ["Signal"])
        current = 0
        while current < len(Y):
            # Create a long position
            if Y[current,0] == 1:
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
            elif Y[current,0] == -1:
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
            current += 1

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
    print("One does not simply think up a strategy")
    #
    # # Training
    # sd = dt.datetime(2008, 1, 1)
    # ed = dt.datetime(2009, 12, 31)
    #
    # test = StrategyLearner()
    # test.add_evidence("JPM", sd, ed,100000)
    #
    # # In-Sample Testing
    # trades = test.testPolicy("JPM", sd, ed,100000)
    #
    # # Generate Benchmark
    # columns = ["Date", "Symbol", "Order", "Shares"]
    # dates = pd.date_range(sd, ed)
    # prices_all = get_data(["JPM"], dates)
    # price = pd.DataFrame(index=prices_all.index, data=prices_all["JPM"])
    # price.index = pd.to_datetime(prices_all.index)
    # price.index = price.index.strftime('%Y-%m-%d')
    # benchmark_trades = pd.DataFrame(None, range(len(price.index)), columns)
    # for x in range(len(price.index)):
    #     benchmark_trades.iloc[x] = [price.index[x], "JPM", "NOTHING", 0]
    # benchmark_trades.loc[0] = [price.index[0], "JPM", "BUY", 1000]
    #
    # # Get portfolio values for strategy vs benchmark
    # strategy = compute_portvals(trades, start_val=100000, commission=9.95, impact=0.005)
    # strategy = strategy / strategy.iloc[0]
    # benchmark = compute_portvals(benchmark_trades, start_val=100000, commission=9.95, impact=0.005)
    # benchmark = benchmark / benchmark.iloc[0]
    #
    # # plot strategy vs benchmark
    # plt.figure(figsize=(12, 6))
    # benchmark.index = pd.to_datetime(benchmark.index)
    # strategy.index = pd.to_datetime(strategy.index)
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    # plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=9, prune=None))
    # plt.plot(strategy, label="Strategy Learner", color="red")
    # plt.plot(benchmark, label="Benchmark", color="purple")
    # plt.legend(["Strategy Learner", "Benchmark"])
    # ymin = 0.5
    # ymax = 1.50
    # plt.ylim([ymin, ymax])
    # plt.grid(True)
    # plt.xlabel('Date')  # Label for the x-axis
    # plt.ylabel('Portfolio Value Normalized')  # Label for the y-axis
    # plt.title('In-Sample: Strategy Learner vs. Benchmark (JPM)')  # Title of the plot
    #
    # # plot long orders and short orders entry points
    # long_orders = trades[trades["Order"].isin(["BUY"])]
    # short_orders = trades[trades["Order"].isin(["SELL"])]
    # long_orders['Date'] = pd.to_datetime(long_orders['Date'])
    # short_orders['Date'] = pd.to_datetime(short_orders['Date'])
    # for _, order in long_orders.iterrows():
    #     plt.axvline(x=order["Date"], color='blue')
    # for _, order in short_orders.iterrows():
    #     plt.axvline(x=order["Date"], color='black')
    #
    # filename = "images/sl_1.png"
    # plt.savefig(filename, format='png')
    #
    # # Calculate statistics
    # cr = (strategy.iloc[-1] / strategy.iloc[0]) - 1
    # adr = np.mean(daily_rets(strategy))
    # sddr = np.std(daily_rets(strategy))
    #
    # cr_b = (benchmark.iloc[-1] / benchmark.iloc[0]) - 1
    # adr_b = np.mean(daily_rets(benchmark))
    # sddr_b = np.std(daily_rets(benchmark))
    #
    # # Compare portfolio against $SPX
    # print(f"Date Range: {sd} to {ed}")
    # print()
    # print(f"Cumulative Return of Strategy: {cr}")
    # print(f"Cumulative Return of Benchmark : {cr_b}")
    # print()
    # print(f"Standard Deviation of Strategy: {sddr}")
    # print(f"Standard Deviation of Benchmark : {sddr_b}")
    # print()
    # print(f"Average Daily Return of Strategy: {adr}")
    # print(f"Average Daily Return of Benchmark : {adr_b}")
    # print()
    #
    # plt.close()
    #
    # # Out-Sample Testing
    # sd = dt.datetime(2010, 1, 1)
    # ed = dt.datetime(2011, 12, 31)
    # trades = test.testPolicy("JPM", sd, ed, 100000)
    #
    # # Generate Benchmark
    # columns = ["Date", "Symbol", "Order", "Shares"]
    # dates = pd.date_range(sd, ed)
    # prices_all = get_data(["JPM"], dates)
    # price = pd.DataFrame(index=prices_all.index, data=prices_all["JPM"])
    # price.index = pd.to_datetime(prices_all.index)
    # price.index = price.index.strftime('%Y-%m-%d')
    # benchmark_trades = pd.DataFrame(None, range(len(price.index)), columns)
    # for x in range(len(price.index)):
    #     benchmark_trades.iloc[x] = [price.index[x], "JPM", "NOTHING", 0]
    # benchmark_trades.loc[0] = [price.index[0], "JPM", "BUY", 1000]
    #
    # # Get portfolio values for strategy vs benchmark
    # strategy = compute_portvals(trades, start_val=100000, commission=9.95, impact=0.005)
    # strategy = strategy / strategy.iloc[0]
    # benchmark = compute_portvals(benchmark_trades, start_val=100000, commission=9.95, impact=0.005)
    # benchmark = benchmark / benchmark.iloc[0]
    #
    # # plot strategy vs benchmark
    # plt.figure(figsize=(12, 6))
    # benchmark.index = pd.to_datetime(benchmark.index)
    # strategy.index = pd.to_datetime(strategy.index)
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    # plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=9, prune=None))
    # plt.plot(strategy, label="Strategy Learner", color="red")
    # plt.plot(benchmark, label="Benchmark", color="purple")
    # plt.legend(["Strategy Learner", "Benchmark"])
    # ymin = 0.5
    # ymax = 1.50
    # plt.ylim([ymin, ymax])
    # plt.grid(True)
    # plt.xlabel('Date')  # Label for the x-axis
    # plt.ylabel('Portfolio Value Normalized')  # Label for the y-axis
    # plt.title('Out-Sample: Strategy Learner vs. Benchmark (JPM)')  # Title of the plot
    #
    # # plot long orders and short orders entry points
    # long_orders = trades[trades["Order"].isin(["BUY"])]
    # short_orders = trades[trades["Order"].isin(["SELL"])]
    # long_orders['Date'] = pd.to_datetime(long_orders['Date'])
    # short_orders['Date'] = pd.to_datetime(short_orders['Date'])
    # for _, order in long_orders.iterrows():
    #     plt.axvline(x=order["Date"], color='blue')
    # for _, order in short_orders.iterrows():
    #     plt.axvline(x=order["Date"], color='black')
    #
    # filename = "images/sl_2.png"
    # plt.savefig(filename, format='png')
    #
    # # Calculate statistics
    # cr = (strategy.iloc[-1] / strategy.iloc[0]) - 1
    # adr = np.mean(daily_rets(strategy))
    # sddr = np.std(daily_rets(strategy))
    #
    # cr_b = (benchmark.iloc[-1] / benchmark.iloc[0]) - 1
    # adr_b = np.mean(daily_rets(benchmark))
    # sddr_b = np.std(daily_rets(benchmark))
    #
    # # Compare portfolio against $SPX
    # print(f"Date Range: {sd} to {ed}")
    # print()
    # print(f"Cumulative Return of Strategy: {cr}")
    # print(f"Cumulative Return of Benchmark : {cr_b}")
    # print()
    # print(f"Standard Deviation of Strategy: {sddr}")
    # print(f"Standard Deviation of Benchmark : {sddr_b}")
    # print()
    # print(f"Average Daily Return of Strategy: {adr}")
    # print(f"Average Daily Return of Benchmark : {adr_b}")
    # print()


