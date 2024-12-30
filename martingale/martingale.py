""""""
from cProfile import label

"""Assess a betting strategy.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
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
import matplotlib.pyplot as plt

"""
episode = 1000 bets

#THIS IS PSEUDOCODE
episode_winnings = $0
while episode_winnings < $80:
    won = False
    bet_amount = $1
    while not won
        wager bet_amount on black
        won = result of roulette wheel spin
        if won == True:
            episode_winnings = episode_winnings + bet_amount
        else:
            episode_winnings = episode_winnings - bet_amount
            bet_amount = bet_amount * 2

"""

def strategy(limited_bankroll):
    """
    :param limited_bankroll: determines if a bankroll is used or not
    :type limited_bankroll: boolean
    :return: The episode winnings
    :rtype: ndarray
    """

    winning_limit = 80
    max_spins = 1000
    bankroll = -256
    spin_winnings = 0
    index = 0

    if limited_bankroll:
        episode_winnings = np.full([1001], bankroll)
    else:
        episode_winnings = np.full([1001], winning_limit)
    while spin_winnings < winning_limit:
        won = False
        bet_amount = 1
        while not won:
            if index == max_spins:
                return episode_winnings
            bet = .47
            won = get_spin_result(bet)
            if won:
                spin_winnings = spin_winnings + bet_amount
            else:
                spin_winnings = spin_winnings - bet_amount
                bet_amount = bet_amount * 2
            if limited_bankroll:
                if spin_winnings <= bankroll:
                    return episode_winnings
                if bet_amount > abs(bankroll) - abs(spin_winnings):
                    bet_amount = abs(bankroll) - abs(spin_winnings)

            episode_winnings[index] = spin_winnings
            index += 1

    #forward fill if success
    if spin_winnings >= winning_limit:
        for x in range(1000):
            if episode_winnings[x] == bankroll:
                episode_winnings[x] = winning_limit


    return episode_winnings



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

def gtid():
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    :return: The GT ID of the student  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: int  		  	   		 	   		  		  		    	 		 		   		 		  
    """
    return 904053428  # replace with your GT ID number


def get_spin_result(win_prob):
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    :param win_prob: The probability of winning  		  	   		 	   		  		  		    	 		 		   		 		  
    :type win_prob: float  		  	   		 	   		  		  		    	 		 		   		 		  
    :return: The result of the spin.  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: bool  		  	   		 	   		  		  		    	 		 		   		 		  
    """
    result = False
    if np.random.random() <= win_prob:
        result = True
    return result

def figure_one():
    """
    Figure 1
    """
    for x in range(10):
        episode = strategy(False)
        plt.plot(episode, label='episode {}'.format(x))

    xmin = 0
    xmax = 300
    ymin = -256
    ymax = 100
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.xlabel('Spin Number')  # Label for the x-axis
    plt.ylabel('Total Winnings/Losses')  # Label for the y-axis
    plt.title('Professor Balch’s Roulette Strategy Results: Figure 1')  # Title of the plot
    plt.legend()
    plt.savefig('images/figure_1.png', format='png')
    plt.close()


def figure_two():
    """
    Figure 2

    Needed to create an ndarray that could hold 1000 ndarrays of shape (1001,)
    """

    nparr = np.zeros([1001,1001])

    for x in range(1000):
        episode = strategy(False)
        nparr[x] = episode

    # npmean = [np.mean(nparr[x,:]) for x in range(1001)]

    npmean = np.zeros([1001])
    npstd = np.zeros([1001])
    for x in range(1000):
        mean = np.mean(nparr[:,x])
        std = np.std(nparr[:,x])
        npmean[x] = mean
        npstd[x] = std

    plt.plot(npmean, color="blue", label='Mean')
    plt.plot(npmean + npstd * 2, color="red", label='Mean + Standard Deviation')
    plt.plot(npmean - npstd * 2, color="green", label='Mean - Standard Deviation')

    xmin = 0
    xmax = 300
    ymin = -256
    ymax = 100
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])

    plt.xlabel('Spin Number ')  # Label for the x-axis
    plt.ylabel('Mean Winnings/Losses per Spin in 1000 Episodes')  # Label for the y-axis
    plt.title('Professor Balch’s Roulette Strategy Results: Figure 2')  # Title of the plot
    plt.legend()
    plt.savefig('images/figure_2.png', format='png')
    plt.close()

def figure_three():
    """
    Figure 3

    Needed to create an ndarray that could hold 1000 ndarrays of shape (1001,)
    """

    nparr = np.zeros([1001,1001])

    for x in range(1000):
        episode = strategy(False)
        nparr[x] = episode


    npmedian = np.zeros([1001])
    npstd = np.zeros([1001])
    for x in range(1000):
        median = np.median(nparr[:,x])
        std = np.std(nparr[:,x])
        npmedian[x] = median
        npstd[x] = std

    plt.plot(npmedian, color="blue", label='Median')
    plt.plot(npmedian + npstd * 2, color="red", label='Median + Standard Deviation')
    plt.plot(npmedian - npstd * 2, color="green", label='Median - Standard Deviation')

    xmin = 0
    xmax = 300
    ymin = -256
    ymax = 100
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])

    plt.xlabel('Spin Number')  # Label for the x-axis
    plt.ylabel('Median Winnings/Losses per Spin in 1000 Episodes')  # Label for the y-axis
    plt.title('Professor Balch’s Roulette Strategy Results: Figure 3')  # Title of the plot
    plt.legend()
    plt.savefig('images/figure_3.png', format='png')
    plt.close()

def figure_four():
    """
    Figure 4

    Needed to create an ndarray that could hold 1000 ndarrays of shape (1001,)
    """

    nparr = np.zeros([1001, 1001])

    for x in range(1000):
        episode = strategy(True)
        nparr[x] = episode

    # Collect the amount of successful Episodes
    successes = 0
    for x in range(1000):
        res = nparr[x,999]
        if res == 80:
            successes += 1
    print(successes/1000)


    npmean = np.zeros([1001])
    npstd = np.zeros([1001])
    for x in range(1000):
        mean = np.mean(nparr[:, x])
        std = np.std(nparr[:, x])
        npmean[x] = mean
        npstd[x] = std

    plt.plot(npmean, color="blue", label='Mean')
    plt.plot(npmean + npstd * 2, color="red", label='Mean + Standard Deviation')
    plt.plot(npmean - npstd * 2, color="green", label='Mean - Standard Deviation')

    xmin = 0
    xmax = 300
    ymin = -256
    ymax = 100
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])

    plt.xlabel('Spin Number')  # Label for the x-axis
    plt.ylabel('Mean Winnings/Losses per Spin in 1000 Episodes')  # Label for the y-axis
    plt.title('Professor Balch’s Roulette Strategy Results: Figure 4')  # Title of the plot
    plt.legend()
    plt.savefig('images/figure_4.png', format='png')
    plt.close()


def figure_five():
    """
    Figure 5

    Needed to create an ndarray that could hold 1000 ndarrays of shape (1001,)
    """

    nparr = np.zeros([1001,1001])

    for x in range(1000):
        episode = strategy(True)
        nparr[x] = episode

    npmedian = np.zeros([1001])
    npstd = np.zeros([1001])
    for x in range(1000):
        median = np.median(nparr[:,x])
        std = np.std(nparr[:,x])
        npmedian[x] = median
        npstd[x] = std

    plt.plot(npmedian, color="blue", label='Median')
    plt.plot(npmedian + npstd * 2, color="red", label='Median + Standard Deviation')
    plt.plot(npmedian - npstd * 2, color="green", label='Median - Standard Deviation')

    xmin = 0
    xmax = 300
    ymin = -256
    ymax = 100
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])

    plt.xlabel('Spin Number')  # Label for the x-axis
    plt.ylabel('Median Winnings/Losses per Spin in 1000 Episodes')  # Label for the y-axis
    plt.title('Professor Balch’s Roulette Strategy Results: Figure 5')  # Title of the plot
    plt.legend()
    plt.savefig('images/figure_5.png', format='png')
    plt.close()

def test_code():
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    Method to test your code  		  	   		 	   		  		  		    	 		 		   		 		  
    """
    win_prob = 0.47  # set appropriately to the probability of a win of betting BLACK on an American roulette wheel
    np.random.seed(gtid())  # do this only once
    print(get_spin_result(win_prob))  # test the roulette spin
    # add your code here to implement the experiments
    figure_one()
    figure_two()
    figure_three()
    figure_four()
    figure_five()

if __name__ == "__main__":
    test_code()
