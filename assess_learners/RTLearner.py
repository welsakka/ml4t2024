""""""
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
import random


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")

def all_y_are_the_same(data):
    return np.mean(data[:, -1]) == data[0, -1]

class RTLearner(object):

    def __init__(self, leaf_size=1, verbose=False):
        """
        Constructor method
        """
        self.verbose = verbose
        self.leaf_size = leaf_size
        self.tree = None

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

        # merge x and y together
        data_y = data_y.reshape(-1,1)
        merge = np.append(data_x, data_y, axis=1)

        # Set tree equal to an ndarray with headers node, factor, splitVal, left, right
        self.tree = np.zeros((0,4), dtype=object)

        # function call: treeBuilderHelper ( data_x, tree) where we will do recursive calls here
        self.tree = self.tree_builder_helper(merge, self.tree)

    def tree_builder_helper(self, train, tree):
        """
        Recursive function that builds the tree structure through a NDArray

        :param train: A set of feature values and expected values used to build the tree
        :type train: numpy.ndarray
        :param tree: The tree structure
        :type tree: numpy.ndarray
        """

        # Select a Random Feature to split on
        # SplitVal is the mean of two random data points underneath the random feature
        # Columns:
        #            Factor  ,  SplitVal  , Left  ,  Right

        if train.shape[0] <= self.leaf_size:
            sample_y_median = np.median(train[:,-1])
            return np.array([[-1, sample_y_median, None, None]])
        if all_y_are_the_same(train):
            return np.array([[-1, train[0, -1], None, None]])
        else:
            # Select a random feature, exclude y train column and make range zero-indexed
            random_feature = random.randint(0, train.shape[1] - 2)

            # Determine Split Value
            split_val = np.median(train[:, random_feature])

            # Handle potential issue of recursive infinite loop by even divide
            train_left = train[train[:, random_feature] <= split_val]
            train_right = train[train[:, random_feature] > split_val]

            if train_left.shape[0] == 0 or train_right.shape[0] == 0:
                half = int(train.shape[0] / 2)
                train_left = train[0:half, :]
                train_right = train[half:, :]

            # Call left and right
            left = self.tree_builder_helper(train_left, tree)
            right = self.tree_builder_helper(train_right, tree)
            node = np.array([[random_feature, split_val, 1, left.shape[0] + 1]])
            node = np.append(node, left, axis=0)
            node = np.append(node, right, axis=0)
            return node

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """

        """
        While left column 2 is not null
        First take the factor column 0
        Determine if the y value is <= or > than the split val column 1
        If node
            If left, append 1 to i
            if right, append i to right column 3
        If leaf, return split val of tree
        """

        y_predict = np.array([])

        for point in points:
            i = int(0)
            while self.tree[i, 2] is not None:
                factor = int(self.tree[i, 0])
                split_val = self.tree[i, 1]
                y_factor = point[factor]
                if y_factor <= split_val:
                    i = i + 1
                else:
                    i = i + int(self.tree[i, 3])
            y_value = self.tree[i, 1]
            y_predict = np.append(y_predict, y_value)

        return y_predict