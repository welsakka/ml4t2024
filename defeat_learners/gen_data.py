""""""  		  	   		 	   		  		  		    	 		 		   		 		  
"""  		  	   		 	   		  		  		    	 		 		   		 		  
template for generating data to fool learners (c) 2016 Tucker Balch  		  	   		 	   		  		  		    	 		 		   		 		  
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
  		  	   		 	   		  		  		    	 		 		   		 		  
import math  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
import numpy as np  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
# this function should return a dataset (X and Y) that will work  		  	   		 	   		  		  		    	 		 		   		 		  
# better for linear regression than decision trees  		  	   		 	   		  		  		    	 		 		   		 		  
def best_4_lin_reg(seed=1489683273):  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		 	   		  		  		    	 		 		   		 		  
    The data set should include from 2 to 10 columns in X, and one column in Y.  		  	   		 	   		  		  		    	 		 		   		 		  
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    :param seed: The random seed for your data generation.  		  	   		 	   		  		  		    	 		 		   		 		  
    :type seed: int  		  	   		 	   		  		  		    	 		 		   		 		  
    :return: Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: numpy.ndarray  		  	   		 	   		  		  		    	 		 		   		 		  
    """
    np.random.seed(seed)
    x = np.random.random(size=(1000, 4))
    y = np.zeros((100,))
    # Here's is an example of creating a Y from randomly generated  		  	   		 	   		  		  		    	 		 		   		 		  
    # X with multiple columns  		  	   		 	   		  		  		    	 		 		   		 		  
    y = x[:,0] + np.sin(x[:,1]) + x[:,2]**2 + x[:,3]**3
    return x, y  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
def best_4_dt(seed=1489683273):  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		 	   		  		  		    	 		 		   		 		  
    The data set should include from 2 to 10 columns in X, and one column in Y.  		  	   		 	   		  		  		    	 		 		   		 		  
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    :param seed: The random seed for your data generation.  		  	   		 	   		  		  		    	 		 		   		 		  
    :type seed: int  		  	   		 	   		  		  		    	 		 		   		 		  
    :return: Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: numpy.ndarray  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    np.random.seed(seed)
    x = np.random.random(size=(1000, 4))
    y = np.zeros((100,))
    # Create a piecewise function, a nonlinear function for DT Learner
    x_piecewise = np.piecewise(x, [x < .5, x >= .5], [lambda x: x * np.random.rand(), lambda x: 2 * x - np.random.rand()])
    y = x_piecewise[:,2]

    return x, y
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
def author():  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    return "welsakka3"  # Change this to your user ID

def study_group(self):
    """
    :return: A comma separated string of GT_Name of each member of your study group
    :rtype: str
    """
    return "welsakka3"
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 	   		  		  		    	 		 		   		 		  
    print("They call me Waleed.")
