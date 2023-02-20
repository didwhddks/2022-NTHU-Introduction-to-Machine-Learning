import numpy as np
import math


def compute_BCE_cost(AL, Y):
    """
    Implement the binary cross-entropy cost function using the above formula.

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- binary cross-entropy cost
    """
    
    m = Y.shape[1]

    ### PASTE YOUR CODE HERE ###
    ### START CODE HERE ###
    cost = -np.sum([Y[0][i]*math.log(AL[0][i]+1e-5)+(1-Y[0][i])*math.log(1-AL[0][i]+1e-5) for i in range(m)])/m
    ### END CODE HERE ###
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost


