""""""
import random

"""  		  	   		 	   		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
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
"""

import math
import sys

import numpy as np
import matplotlib.pyplot as plt

import DTLearner as dtl
import RTLearner as rtl
import BagLearner as bl
import InsaneLearner as il

def random_select(data):
    # compute how much of the data is training and testing
    n_train = int(0.6 * data.shape[0])

    # separate out training and testing data
    train_data_x = np.empty((0, data.shape[1] - 1), dtype=float)
    train_data_y = np.empty((0, 1), dtype=float)

    for i in range(0, n_train):
        random_instance = random.randint(0, data.shape[0] - 1)
        train_data_x = np.append(train_data_x, data[random_instance, 0:-1].reshape(1, -1), axis=0)
        train_data_y = np.append(train_data_y, data[random_instance, -1])
        data = np.delete(data, random_instance, axis=0)

    test_data_x = data[:, :-1]
    test_data_y = data[:, -1]

    return train_data_x, train_data_y, test_data_x, test_data_y

def evaluate_sample(learner, x, y):
    pred_y = learner.query(x)  # get the predictions
    rmse = math.sqrt(((y - pred_y) ** 2).sum() / y.shape[0])
    print()
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=y)
    print(f"corr: {c[0, 1]}")
    return rmse, c

def test_rtlearner():
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    inf = open(sys.argv[1])
    data = np.array(
        [list(map(str, s.strip().split(","))) for s in inf.readlines()]
    )
    data = data[1:, 1:]
    data = data.astype(float)

    # separate out training and testing data
    train_x, train_y, test_x, test_y = random_select(data)

    print("Beginning Testing Random Tree Learner ")
    print(f"{test_x.shape}")
    print(f"{test_y.shape}")

    # create a learner and train it
    rt_learner = rtl.RTLearner(leaf_size=1, verbose=False)
    rt_learner.add_evidence(train_x, train_y)
    print(rt_learner.author())

    # evaluate in sample
    evaluate_sample(rt_learner, train_x, train_y)
    # evaluate out of sample
    evaluate_sample(rt_learner, test_x, test_y)

    # create a learner and train it
    rt_learner = rtl.RTLearner(leaf_size=50, verbose=False)
    rt_learner.add_evidence(train_x, train_y)
    print(rt_learner.author())

    # evaluate in sample
    evaluate_sample(rt_learner, train_x, train_y)
    # evaluate out of sample
    evaluate_sample(rt_learner, test_x, test_y)

def test_dtlearner():
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    inf = open(sys.argv[1])
    data = np.array(
        [list(map(str, s.strip().split(","))) for s in inf.readlines()]
    )

    #Clean data of headers and date column
    data = data[1:, 1:]
    data = data.astype(float)

    # separate out training and testing data
    train_x, train_y, test_x, test_y = random_select(data)

    print("Beginning Testing Decision Tree Learner")
    print(f"{test_x.shape}")
    print(f"{test_y.shape}")

    # create a learner and train it
    dt_learner = dtl.DTLearner(leaf_size=1, verbose=False)
    dt_learner.add_evidence(train_x, train_y)
    print(dt_learner.author())

    # evaluate in sample
    print("In Sample results")
    evaluate_sample(dt_learner, train_x, train_y)
    # evaluate out of sample
    evaluate_sample(dt_learner, test_x, test_y)

    # create a learner with leaf size 50 and train it
    dt_learner = dtl.DTLearner(leaf_size=50, verbose=False)
    dt_learner.add_evidence(train_x, train_y)
    print(dt_learner.author())

    # evaluate in sample
    print("In Sample results")
    evaluate_sample(dt_learner, train_x, train_y)
    # evaluate out of sample
    evaluate_sample(dt_learner, test_x, test_y)

def test_baglearner():
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    inf = open(sys.argv[1])
    data = np.array(
        [list(map(str, s.strip().split(","))) for s in inf.readlines()]
    )

    #Clean data of headers and date column
    data = data[1:, 1:]
    data = data.astype(float)

    print("Beginning Testing Bag Learner ")

    # separate out training and testing data
    train_x, train_y, test_x, test_y = random_select(data)

    # create a dtl learner with 1 bag and train it
    bag_learner = bl.BagLearner(learner=dtl.DTLearner, kwargs={"leaf_size": 1}, bags=1, boost=False, verbose=False)
    bag_learner.add_evidence(train_x, train_y)
    print(bag_learner.author())

    # evaluate in sample
    evaluate_sample(bag_learner, train_x, train_y)
    # evaluate out of sample
    evaluate_sample(bag_learner, test_x, test_y)

    # evaluate in sample
    evaluate_sample(bag_learner, train_x, train_y)
    # evaluate out of sample
    evaluate_sample(bag_learner, test_x, test_y)

    # create a rtl learner and train it
    bag_learner = bl.BagLearner(learner=rtl.RTLearner, kwargs={"leaf_size": 1}, bags=20, boost=False, verbose=False)
    bag_learner.add_evidence(train_x, train_y)
    print(bag_learner.author())

    # evaluate in sample
    evaluate_sample(bag_learner, train_x, train_y)
    # evaluate out of sample
    evaluate_sample(bag_learner, test_x, test_y)

    for _ in range(0,10):
        # create a dtl learner and train it
        bag_learner = bl.BagLearner(learner=dtl.DTLearner, kwargs={"leaf_size": 1}, bags=20, boost=False, verbose=False)
        bag_learner.add_evidence(train_x, train_y)
        print(bag_learner.author())

        # evaluate in sample
        evaluate_sample(bag_learner, train_x, train_y)
        # evaluate out of sample
        evaluate_sample(bag_learner, test_x, test_y)

def test_insanelearner():
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    inf = open(sys.argv[1])
    data = np.array(
        [list(map(str, s.strip().split(","))) for s in inf.readlines()]
    )

    #Clean data of headers and date column
    data = data[1:, 1:]
    data = data.astype(float)

    print("Beginning Testing INSANE Learner ")

    # separate out training and testing data
    train_x, train_y, test_x, test_y = random_select(data)

    # create a dtl learner with 1 bag and train it
    insane_learner = il.InsaneLearner(verbose=False)
    insane_learner.add_evidence(train_x, train_y)
    print(insane_learner.author())

    # evaluate in sample
    evaluate_sample(insane_learner, train_x, train_y)
    # evaluate out of sample
    evaluate_sample(insane_learner, test_x, test_y)

def chart_generator():
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    inf = open(sys.argv[1])
    data = np.array(
        [list(map(str, s.strip().split(","))) for s in inf.readlines()]
    )

    #Clean data of headers and date column
    data = data[1:, 1:]
    data = data.astype(float)

    print("Beginning Testing Bag Learner ")

    # separate out training and testing data
    train_x, train_y, test_x, test_y = random_select(data)


    """
    Experiment 1
    """

    # Produce a chart using leaf size as degrees of freedom as x axis and RMSE as y axis
    in_sample = np.empty(50)
    out_sample = np.empty(50)
    for i in range(50):
        # create a learner with leaf size 50 and train it
        dt_learner = dtl.DTLearner(leaf_size=i, verbose=False)
        dt_learner.add_evidence(train_x, train_y)

        # evaluate in sample
        print("In Sample results")
        rmse_in, c = evaluate_sample(dt_learner, train_x, train_y)
        # evaluate out of sample
        rmse_out, c = evaluate_sample(dt_learner, test_x, test_y)

        in_sample[i] = rmse_in
        out_sample[i] = rmse_out
    plt.plot(in_sample, color="blue", label="In Sample")
    plt.plot(out_sample, color="red", label="Out of Sample")
    plt.xlim([0, 50])
    plt.ylim([0, .01])
    plt.xlabel('Leaf Size Degrees of Freedom')  # Label for the x-axis
    plt.ylabel('RMSE')  # Label for the y-axis
    plt.title('Decision Tree Learner Backtesting Results: In-Sample vs Out-of-Sample')  # Title of the plot
    plt.legend()
    plt.grid()
    plt.savefig('images/figure_1.png', format='png')
    plt.close()

    """
    Experiment 2
    """

    # Produce a chart using leaf size as degrees of freedom as x axis and RMSE as y axis
    in_sample = np.empty(50)
    out_sample = np.empty(50)
    for i in range(50):
        # create a learner with leaf size 50 and train it
        bag_learner = bl.BagLearner(learner=rtl.RTLearner, kwargs={"leaf_size": i}, bags=20, boost=False, verbose=False)
        bag_learner.add_evidence(train_x, train_y)

        # evaluate in sample
        print("In Sample results")
        rmse_in, c = evaluate_sample(bag_learner, train_x, train_y)
        # evaluate out of sample
        rmse_out, c = evaluate_sample(bag_learner, test_x, test_y)

        in_sample[i] = rmse_in
        out_sample[i] = rmse_out
    plt.plot(in_sample, color="blue", label="In Sample")
    plt.plot(out_sample, color="red", label="Out of Sample")
    plt.xlim([0, 50])
    plt.ylim([0, .01])
    plt.xlabel('Leaf Size Degrees of Freedom')  # Label for the x-axis
    plt.ylabel('RMSE')  # Label for the y-axis
    plt.title('Bag Learner Backtesting Results: In-Sample vs Out-of-Sample')  # Title of the plot
    plt.legend()
    plt.grid()
    plt.savefig('images/figure_2.png', format='png')
    plt.close()

    """
    Experiment 3: Coefficient of Determination (R-Squared)
    """

    # Produce a chart using leaf size as degrees of freedom as x axis
    in_sample = np.empty(50)
    out_sample = np.empty(50)
    for i in range(50):
        # create a learner with leaf size 50 and train it
        dt_learner = dtl.DTLearner(leaf_size=i, verbose=False)
        dt_learner.add_evidence(train_x, train_y)

        # evaluate in sample
        print("In Sample results")
        pred_y = dt_learner.query(train_x)  # get the predictions
        # Calculate R Squared
        # Calculate the total sum of squares (SS_tot)
        ss_tot = np.sum((train_y - np.mean(train_y)) ** 2)
        # Calculate the sum of squares of residuals (SS_res)
        ss_res = np.sum((train_y - pred_y) ** 2)
        train_r2 = 1 - (ss_res / ss_tot)
        print(f"R2: {train_r2}")

        # evaluate out of sample sample
        print("Out of Sample results")
        pred_y = dt_learner.query(test_x)  # get the predictions
        # Calculate R Squared
        # Calculate the total sum of squares (SS_tot)
        ss_tot = np.sum((test_y - np.mean(test_y)) ** 2)
        # Calculate the sum of squares of residuals (SS_res)
        ss_res = np.sum((test_y - pred_y) ** 2)
        test_r2 = 1 - (ss_res / ss_tot)
        print(f"R2: {test_r2}")

        in_sample[i] = train_r2
        out_sample[i] = test_r2

    # Produce a chart using leaf size as degrees of freedom as x axis
    in_sample_random = np.empty(50)
    out_sample_random = np.empty(50)
    for i in range(50):
        # create a learner with leaf size 50 and train it
        rt_learner = rtl.RTLearner(leaf_size=i, verbose=False)
        rt_learner.add_evidence(train_x, train_y)

        # evaluate in sample
        print("In Sample results")
        pred_y = rt_learner.query(train_x)  # get the predictions
        # Calculate R Squared
        # Calculate the total sum of squares (SS_tot)
        ss_tot = np.sum((train_y - np.mean(train_y)) ** 2)
        # Calculate the sum of squares of residuals (SS_res)
        ss_res = np.sum((train_y - pred_y) ** 2)
        train_r2 = 1 - (ss_res / ss_tot)
        print(f"R2: {train_r2}")

        # evaluate out of sample sample
        print("Out of Sample results")
        pred_y = rt_learner.query(test_x)  # get the predictions
        # Calculate R Squared
        # Calculate the total sum of squares (SS_tot)
        ss_tot = np.sum((test_y - np.mean(test_y)) ** 2)
        # Calculate the sum of squares of residuals (SS_res)
        ss_res = np.sum((test_y - pred_y) ** 2)
        test_r2 = 1 - (ss_res / ss_tot)
        print(f"R2: {test_r2}")

        in_sample_random[i] = train_r2
        out_sample_random[i] = test_r2


    plt.plot(in_sample, color="blue", label="DT In Sample R Squared")
    plt.plot(out_sample, color="red", label="DT Out of Sample R Squared")
    plt.plot(in_sample_random, color="green", label="RT In Sample R Squared")
    plt.plot(out_sample_random, color="orange", label="RT Out of Sample R Squared")
    plt.xlim([0, 50])
    plt.ylim([0, 1])
    plt.xlabel('Leaf Size Degrees of Freedom')  # Label for the x-axis
    plt.ylabel('Coefficient of Determination (R-Squared)')  # Label for the y-axis
    plt.title('DT and RT Backtesting Results: In-Sample vs Out-of-Sample')  # Title of the plot
    plt.legend()
    plt.grid()
    plt.savefig('images/figure_3.png', format='png')
    plt.close()

    """ 
    Experiment 3: Mean Actual Percentage Error
    """

    # Produce a chart using leaf size as degrees of freedom as x axis
    in_sample = np.empty(50)
    out_sample = np.empty(50)
    for i in range(50):
        # create a learner with leaf size 50 and train it
        dt_learner = dtl.DTLearner(leaf_size=i, verbose=False)
        dt_learner.add_evidence(train_x, train_y)

        # evaluate in sample
        print("In Sample results")
        pred_y = dt_learner.query(train_x)  # get the predictions

        # Calculate MAPE
        train_mape = np.mean(np.abs((train_y - pred_y) / train_y))
        print(f"MAPE: {train_mape}")

        # evaluate out of sample sample
        print("Out of Sample results")
        pred_y = dt_learner.query(test_x)  # get the predictions
        # Calculate MAPE
        test_mape = np.mean(np.abs((test_y - pred_y) / test_y))
        print(f"MAPE: {test_mape}")

        in_sample[i] = train_mape
        out_sample[i] = test_mape

    # Produce a chart using leaf size as degrees of freedom as x axis
    in_sample_random = np.empty(50)
    out_sample_random = np.empty(50)
    for i in range(50):
        # create a learner with leaf size 50 and train it
        rt_learner = rtl.RTLearner(leaf_size=i, verbose=False)
        rt_learner.add_evidence(train_x, train_y)

        # evaluate in sample
        print("In Sample results")
        pred_y = rt_learner.query(train_x)  # get the predictions
        # Calculate MAPE
        train_mape = np.mean(np.abs((train_y - pred_y) / train_y))
        print(f"MAPE: {train_mape}")

        # evaluate out of sample sample
        print("Out of Sample results")
        pred_y = rt_learner.query(test_x)  # get the predictions
        # Calculate MAPE
        test_mape = np.mean(np.abs((test_y - pred_y) / test_y))
        print(f"MAPE: {test_mape}")

        in_sample_random[i] = train_mape
        out_sample_random[i] = test_mape


    plt.plot(in_sample, color="blue", label="DT In Sample MAPE")
    plt.plot(out_sample, color="red", label="DT Out of Sample MAPE")
    plt.plot(in_sample_random, color="green", label="RT In Sample MAPE")
    plt.plot(out_sample_random, color="orange", label="RT Out of Sample MAPE")
    plt.xlim([0, 50])
    plt.ylim([0, 10])
    plt.xlabel('Leaf Size Degrees of Freedom')  # Label for the x-axis
    plt.ylabel('Mean Actual Percentage Error')  # Label for the y-axis
    plt.title('DT and RT Backtesting Results: In-Sample vs Out-of-Sample')  # Title of the plot
    plt.legend()
    plt.grid()
    plt.savefig('images/figure_4.png', format='png')
    plt.close()


if __name__ == "__main__":

    test_dtlearner()
    test_rtlearner()
    test_baglearner()
    test_insanelearner()
    chart_generator()