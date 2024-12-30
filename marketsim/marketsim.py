""""""
from future.backports.datetime import datetime
from numpy.ma.core import greater

"""MC2-P1: Market simulator.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
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
  		  	   		 	   		  		  		    	 		 		   		 		  
def compute_portvals(  		  	   		 	   		  		  		    	 		 		   		 		  
    orders_file="./orders/orders.csv",  		  	   		 	   		  		  		    	 		 		   		 		  
    start_val=1000000,  		  	   		 	   		  		  		    	 		 		   		 		  
    commission=9.95,  		  	   		 	   		  		  		    	 		 		   		 		  
    impact=0.005,  		  	   		 	   		  		  		    	 		 		   		 		  
):  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    Computes the portfolio values.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    :param orders_file: Path of the order file or the file object  		  	   		 	   		  		  		    	 		 		   		 		  
    :type orders_file: str or file object  		  	   		 	   		  		  		    	 		 		   		 		  
    :param start_val: The starting value of the portfolio  		  	   		 	   		  		  		    	 		 		   		 		  
    :type start_val: int  		  	   		 	   		  		  		    	 		 		   		 		  
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		 	   		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		 	   		  		  		    	 		 		   		 		  
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		 	   		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		 	   		  		  		    	 		 		   		 		  
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: pandas.DataFrame  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    # this is the function the autograder will call to test your code  		  	   		 	   		  		  		    	 		 		   		 		  
    # NOTE: orders_file may be a string, or it may be a file object. Your  		  	   		 	   		  		  		    	 		 		   		 		  
    # code should work correctly with either input

    ### PSEUDOCODE ###
    # Keep dictionary representing portfolio, with ticker as key and no. of shares as value
    # Keep track of cash in portfolio
    # Keep pandas dataframe representing portfolio value, date indexed matching the orders csv
    # for each date from start to end
    #   if no order on that date
    #       calculate value of all stocks in dictionary plus cash
    #       Add value to res
    #   if there is an order
    #       adjust portfolio based on buy or sell
    #       calculate portfolio value as above
    # return pandas dataframe

    # Prepare variables to track portfolio value
    orders = pd.read_csv(orders_file)
    orders = orders.sort_values(by='Date')
    orders = orders.reset_index()
    start_date = orders.iloc[0]['Date']
    end_date = orders.iloc[-1]['Date']
    dates = pd.date_range(start_date, end_date)

    syms = []
    prices = get_data(syms, dates)
    portfolio = {}
    cash = start_val
    # format date index
    res = pd.DataFrame(index=prices.index)
    res.index = pd.to_datetime(prices.index)
    res.index = res.index.strftime('%Y-%m-%d')
    order_tracker = 0

    #iterate through each date in the prices dataframe, which includes valid dates in the stock market
    for date in prices.index:
        #if order is found on date
        date = date.strftime('%Y-%m-%d')
        order_date = orders['Date'][order_tracker]
        while date == order_date:
            sym = orders['Symbol'][order_tracker]
            #update prices for symbols
            if sym not in syms:
                syms.append(sym)
                prices = get_data(syms, dates)
            #update portfolio
            if sym not in portfolio:
                portfolio[sym] = 0
            if orders["Order"][order_tracker] == 'BUY':
                price = prices[sym][date] + (impact * prices[sym][date])
                order_value_in_dollars = price * orders["Shares"][order_tracker]
                portfolio[sym] = portfolio[sym] + orders["Shares"][order_tracker]
                cash = cash - order_value_in_dollars
            else:
                price = prices[sym][date] - (impact * prices[sym][date])
                order_value_in_dollars = price * orders["Shares"][order_tracker]
                portfolio[sym] = portfolio[sym] - orders["Shares"][order_tracker]
                cash = cash + order_value_in_dollars
            #subtract commission
            cash = cash - commission
            if order_tracker < len(orders.index) - 1:
                order_tracker = order_tracker + 1
                order_date = orders['Date'][order_tracker]
            else: break

        #calculate portfolio value
        current_value = 0
        for sym in portfolio:
            stock_value = portfolio[sym] * prices[sym][date]
            current_value = current_value + stock_value
        current_value = current_value + cash
        res.loc[date, ""] = current_value
    return res
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
def test_code():  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    Helper function to test code  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    # this is a helper function you can use to test your code  		  	   		 	   		  		  		    	 		 		   		 		  
    # note that during autograding his function will not be called.  		  	   		 	   		  		  		    	 		 		   		 		  
    # Define input parameters  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    of = "./orders/orders-10.csv"
    sv = 1000000  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    # Process orders  		  	   		 	   		  		  		    	 		 		   		 		  
    portvals = compute_portvals(orders_file=of, start_val=sv)  		  	   		 	   		  		  		    	 		 		   		 		  
    if isinstance(portvals, pd.DataFrame):  		  	   		 	   		  		  		    	 		 		   		 		  
        portvals = portvals[portvals.columns[0]]  # just get the first column  		  	   		 	   		  		  		    	 		 		   		 		  
    else:  		  	   		 	   		  		  		    	 		 		   		 		  
        "warning, code did not return a DataFrame"  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    # Get portfolio stats  		  	   		 	   		  		  		    	 		 		   		 		  
    # Here we just fake the data. you should use your code from previous assignments.  		  	   		 	   		  		  		    	 		 		   		 		  
    start_date = dt.datetime(2008, 1, 1)  		  	   		 	   		  		  		    	 		 		   		 		  
    end_date = dt.datetime(2008, 6, 1)  		  	   		 	   		  		  		    	 		 		   		 		  
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [  		  	   		 	   		  		  		    	 		 		   		 		  
        0.2,  		  	   		 	   		  		  		    	 		 		   		 		  
        0.01,  		  	   		 	   		  		  		    	 		 		   		 		  
        0.02,  		  	   		 	   		  		  		    	 		 		   		 		  
        1.5,  		  	   		 	   		  		  		    	 		 		   		 		  
    ]  		  	   		 	   		  		  		    	 		 		   		 		  
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [  		  	   		 	   		  		  		    	 		 		   		 		  
        0.2,  		  	   		 	   		  		  		    	 		 		   		 		  
        0.01,  		  	   		 	   		  		  		    	 		 		   		 		  
        0.02,  		  	   		 	   		  		  		    	 		 		   		 		  
        1.5,  		  	   		 	   		  		  		    	 		 		   		 		  
    ]  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    # Compare portfolio against $SPX  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Date Range: {start_date} to {end_date}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print()  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print()  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print()  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Standard Deviation of Fund: {std_daily_ret}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print()  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Average Daily Return of Fund: {avg_daily_ret}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print()  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Final Portfolio Value: {portvals[-1]}")  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 	   		  		  		    	 		 		   		 		  
    test_code()  		  	   		 	   		  		  		    	 		 		   		 		  
