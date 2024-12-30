""""""
import random
from os.path import split

from numpy import dtype

"""  		  	   		 	   		  		  		    	 		 		   		 		  
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		 	   		  		  		    	 		 		   		 		  

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

import numpy as np


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")


class BagLearner(object):

    def __init__(self, learner, kwargs, bags = 20, boost = False, verbose = False):
        """
        Constructor method
        """
        self.kwargs = kwargs
        self.learner = learner
        self.boost = boost
        self.bags = bags
        self.verbose = verbose
        self.learners = []

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

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """

        # initialize learners
        for i in range(0, self.bags):
            self.learners.append(self.learner(**self.kwargs))

        # Prep the data and train the learners
        for learner in self.learners:
            n_prime = int(0.6 * data_x.shape[0])
            train_data_x = np.empty((0, data_x.shape[1]), dtype=float)
            train_data_y = np.empty((0, 1), dtype=float)

            for i in range(0, n_prime):
                random_instance = random.randint(0, data_x.shape[0] - 1)
                train_data_x = np.append(train_data_x, data_x[random_instance, :].reshape(1,-1), axis=0)
                train_data_y = np.append(train_data_y, data_y[random_instance])

            learner.add_evidence(train_data_x, train_data_y)

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """

        """
        Call each learner's query function
        Append each result into a 2d ndarray
        Take the mean of each row in the res 2d ndarray
        return the resulting 1d ndarray of means
        """

        y_predict = np.empty((points.shape[0],0))

        for learner in self.learners:
            result = learner.query(points)
            result = result.reshape(-1, 1)
            y_predict = np.append(y_predict, result, axis=1)

        mean = np.array([np.mean(y_predict[i,:]) for i in range (0, y_predict.shape[0])])
        return mean