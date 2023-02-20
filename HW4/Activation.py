import numpy as np
import math

class Activation():
    def __init__(self, function):
        self.function = function
        self.name = function
        

    def forward(self, Z):
        if self.function == "sigmoid":
            """
            Implements the sigmoid activation in numpy
            
            Arguments:
            Z -- numpy array of any shape
            self.cache -- stores Z as well, useful during backpropagation
            
            Returns:
            A -- output of sigmoid(z), same shape as Z
            
            """

            ### PASTE YOUR CODE HERE ###
            ### START CODE HERE ###
            A = np.float64(Z.flatten())
            for i in range(len(A)):
              if A[i] >= 0: A[i] = 1/(1+math.exp(-A[i]))
              else: A[i] = math.exp(A[i])/(1+math.exp(A[i]))
            A = A.reshape(Z.shape)
            self.cache = Z
            ### END CODE HERE ###
            
            return A

        elif self.function == "softmax":
            """
            Implements the softmax activation in numpy
            
            Arguments:
            Z -- numpy array of any shape (dim 0: number of classes, dim 1: number of samples)
            self.cache -- stores Z as well, useful during backpropagation
            
            Returns:
            A -- output of softmax(z), same shape as Z
            """

            ### PASTE YOUR CODE HERE ###
            ### START CODE HERE ###
            A = np.zeros(Z.shape); C = np.transpose(Z)
            for i in range(A.shape[1]):
              sum = np.sum([math.exp(C[i][j]-np.max(C[i])) for j in range(A.shape[0])])
              for j in range(A.shape[0]):
                A[j][i] = math.exp(C[i][j]-np.max(C[i]))/sum
            self.cache = Z
            ### END CODE HERE ###
            
            return A

        elif self.function == "relu":
            """
            Implement the RELU function in numpy
            Arguments:
            Z -- numpy array of any shape
            self.cache -- stores Z as well, useful during backpropagation
            Returns:
            A -- output of relu(z), same shape as Z
            
            """
            
            ### PASTE YOUR CODE HERE ###
            ### START CODE HERE ###
            A = np.float64(Z.flatten())
            for i in range(len(A)):
              A[i] = max(A[i], 0)
            A = A.reshape(Z.shape)
            self.cache = Z
            ### END CODE HERE ###
            
            assert(A.shape == Z.shape)
            
            return A

    def backward(self, dA=None, Y=None):
        if self.function == "sigmoid":
            """
            Implement the backward propagation for a single SIGMOID unit.
            Arguments:
            dA -- post-activation gradient, of any shape
            self.cache -- 'Z' where we store for computing backward propagation efficiently
            Returns:
            dZ -- Gradient of the cost with respect to Z
            """
            
            ### PASTE YOUR CODE HERE ###
            ### START CODE HERE ###
            Z = self.cache
            derivative = np.float64(Z.flatten())
            for i in range(len(derivative)):
              if derivative[i] >= 0: derivative[i] = math.exp(-derivative[i])/(1+math.exp(-derivative[i]))**2
              else: derivative[i] = math.exp(derivative[i])/(1+math.exp(derivative[i]))**2
            derivative = derivative.reshape(Z.shape)
            dZ = np.multiply(dA, derivative)
            ### END CODE HERE ###
            
            assert (dZ.shape == Z.shape)
            
            return dZ

        elif self.function == "relu":
            """
            Implement the backward propagation for a single RELU unit.
            Arguments:
            dA -- post-activation gradient, of any shape
            self.cache -- 'Z' where we store for computing backward propagation efficiently
            Returns:
            dZ -- Gradient of the cost with respect to Z
            """
            
            ### PASTE YOUR CODE HERE ###
            ### START CODE HERE ### 
            Z = self.cache
            dZ = dA # just converting dz to a correct object. 
            dZ[Z <= 0] = 0 # When z <= 0, you should set dz to 0 as well.
            ### END CODE HERE ###
            
            assert (dZ.shape == Z.shape)
            
            return dZ

        elif self.function == "softmax":
            """
            Implement the backward propagation for a [SOFTMAX->CCE LOSS] unit.
            Arguments:
            Y -- true "label" vector (one hot vector, for example: [[1], [0], [0]] represents rock, [[0], [1], [0]] represents paper, [[0], [0], [1]] represents scissors 
                                      in a Rock-Paper-Scissors image classification), shape (number of classes, number of examples)
            self.cache -- 'Z' where we store for computing backward propagation efficiently
            Returns:
            dZ -- Gradient of the cost with respect to Z
            """
            
            ### PASTE YOUR CODE HERE ###
            ### START CODE HERE ### 
            Z = self.cache
            s = np.zeros(Z.shape); C = np.transpose(Z)
            for i in range(s.shape[1]):
              sum = np.sum([math.exp(C[i][j]-np.max(C[i])) for j in range(s.shape[0])])
              for j in range(s.shape[0]):
                s[j][i] = math.exp(C[i][j]-np.max(C[i]))/sum
            dZ = s-Y
            ### END CODE HERE ###
            
            assert (dZ.shape == Z.shape)
            
            return dZ
