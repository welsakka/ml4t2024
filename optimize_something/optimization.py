""""""
"""MC1-P2: Optimize a portfolio.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
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
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
import numpy as np  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
import matplotlib.pyplot as plt  		  	   		 	   		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		 	   		  		  		    	 		 		   		 		  
from util import get_data, plot_data
import scipy.optimize as spo

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
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
# This is the function that will be tested by the autograder  		  	   		 	   		  		  		    	 		 		   		 		  
# The student must update this code to properly implement the functionality  		  	   		 	   		  		  		    	 		 		   		 		  
def optimize_portfolio(  		  	   		 	   		  		  		    	 		 		   		 		  
    sd=dt.datetime(2008, 1, 1),  		  	   		 	   		  		  		    	 		 		   		 		  
    ed=dt.datetime(2009, 1, 1),  		  	   		 	   		  		  		    	 		 		   		 		  
    syms=["GOOG", "AAPL", "GLD", "XOM"],  		  	   		 	   		  		  		    	 		 		   		 		  
    gen_plot=False,  		  	   		 	   		  		  		    	 		 		   		 		  
):  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		 	   		  		  		    	 		 		   		 		  
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		 	   		  		  		    	 		 		   		 		  
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		 	   		  		  		    	 		 		   		 		  
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		 	   		  		  		    	 		 		   		 		  
    statistics.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	   		  		  		    	 		 		   		 		  
    :type sd: datetime  		  	   		 	   		  		  		    	 		 		   		 		  
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	   		  		  		    	 		 		   		 		  
    :type ed: datetime  		  	   		 	   		  		  		    	 		 		   		 		  
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		 	   		  		  		    	 		 		   		 		  
        symbol in the data directory)  		  	   		 	   		  		  		    	 		 		   		 		  
    :type syms: list  		  	   		 	   		  		  		    	 		 		   		 		  
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		 	   		  		  		    	 		 		   		 		  
        code with gen_plot = False.  		  	   		 	   		  		  		    	 		 		   		 		  
    :type gen_plot: bool  		  	   		 	   		  		  		    	 		 		   		 		  
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		 	   		  		  		    	 		 		   		 		  
        standard deviation of daily returns, and Sharpe ratio  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: tuple  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    # Read in adjusted closing prices for given symbols, date range  		  	   		 	   		  		  		    	 		 		   		 		  
    dates = pd.date_range(sd, ed)  		  	   		 	   		  		  		    	 		 		   		 		  
    prices_all = get_data(syms, dates)  # automatically adds SPY  		  	   		 	   		  		  		    	 		 		   		 		  
    prices = prices_all[syms]  # only portfolio symbols  		  	   		 	   		  		  		    	 		 		   		 		  
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later

    # Create a guess allocation for the optimizer
    guess = 1 / len(syms)
    init_allocs = np.full(len(syms), guess)
    bounds = [(0,100) for _ in range(len(syms))]

    # Initialize and use an optimizer to find the allocation of stocks with the best sharpe ratio
    optimized_allocs = spo.minimize(compute_sharpe_ratio_in_portfolio,
                                    init_allocs,
                                    args=(prices),
                                    bounds=bounds,
                                    constraints = ({ 'type': 'eq', 'fun': lambda inputs: 1 - np.sum(inputs) }),
                                    method='SLSQP',
                                    options={'disp': True})

    #Use the optimized allocation to retrieve the daily returns of that portfolio
    daily_rets, portfolio_values = get_daily_returns(optimized_allocs.x, prices)

    # Compute statistics for the optimized portfolio:
    # cr: Cumulative return
    # adr: Average daily return
    # sddr: Standard deviation of daily return
    # sr: Sharpe ratio
    cr = (portfolio_values[-1]/portfolio_values[0]) - 1
    adr = np.mean(daily_rets)
    sddr = np.std(daily_rets)
    sr = sharpe_ratio(daily_rets)

    # Compare daily portfolio value with SPY using a normalized plot  		  	   		 	   		  		  		    	 		 		   		 		  
    if gen_plot:
        norm_sp = prices_SPY / prices_SPY[0]
        norm_port = portfolio_values / portfolio_values[0]
        plt.plot(norm_sp, label="SPY")
        plt.plot(norm_port, label="Portfolio")
        plt.title('Daily Portfolio Value versus SPY')
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.ylim([0.9, 1.4])
        plt.grid(True)
        plt.legend()
        plt.savefig('images/Figure1.png', format='png')
  		  	   		 	   		  		  		    	 		 		   		 		  
    return optimized_allocs.x, cr, adr, sddr, sr

def sharpe_ratio(daily_rets):
    """
    Calculates the Sharpe ratio given an array of daily returns.
    :param daily_rets: A numpy array of the daily returns
    :type daily_rets: DataFrame
    :return: The sharpe ratio
    :rtype: float
    """
    open_days = 252
    return np.sqrt(open_days) * (np.mean(daily_rets) / np.std(daily_rets))

def get_daily_returns(allocs, prices):
    """
    Given a dataframe of prices and an array of portfolio allocations, this function returns the daily returns of the portfolio
    :param prices: a DataFrame containing the prices for a list of stocks between two dates
    :type prices: DataFrame
    :param allocs: a DataFrame containing the allocations for each stock
    :type allocs: DataFrame
    :return: A DataFrame of the portfolio's daily returns and a Dataframe of the portfolio's value
    :rtype: DataFrame[]
    """
    postion_value = 10000
    # normalize prices
    normed = prices.div(prices.iloc[0], axis=1)
    #apply allocations onto the normalized prices
    allocated = normed * allocs
    #apply postion_value to the allocated df
    pos_values = allocated * postion_value
    #get the portfolio value for each date
    port_val = np.sum(pos_values, axis=1)
    #Convert portfolio value to daily returns
    daily_ret = [(port_val[x] - port_val[x - 1]) / port_val[x] for x in range(len(port_val))]
    daily_ret = daily_ret[1:]
    return daily_ret, port_val

def compute_sharpe_ratio_in_portfolio(allocs, prices):
    """
    Function to pass into the optimizer, to find the best allocation of stocks accounting for sharpe ratio
    """
    daily_rets, portfolio_value = get_daily_returns(allocs, prices)
    sr = sharpe_ratio(daily_rets)
    optimize_sr = 1 - sr
    return optimize_sr


def test_code():  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    This function WILL NOT be called by the auto grader.  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    start_date = dt.datetime(2010, 1, 1)
    end_date = dt.datetime(2010, 12, 31)
    symbols = ["GOOG", "GLD", "AAPL", "IBM"]
  		  	   		 	   		  		  		    	 		 		   		 		  
    # Assess the portfolio  		  	   		 	   		  		  		    	 		 		   		 		  
    allocations, cr, adr, sddr, sr = optimize_portfolio(  		  	   		 	   		  		  		    	 		 		   		 		  
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True
    )  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    # Print statistics  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Start Date: {start_date}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"End Date: {end_date}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Symbols: {symbols}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Allocations:{allocations}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio: {sr}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Average Daily Return: {adr}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Cumulative Return: {cr}")  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 	   		  		  		    	 		 		   		 		  
    # This code WILL NOT be called by the auto grader  		  	   		 	   		  		  		    	 		 		   		 		  
    # Do not assume that it will be called  		  	   		 	   		  		  		    	 		 		   		 		  
    test_code()  		  	   		 	   		  		  		    	 		 		   		 		  
