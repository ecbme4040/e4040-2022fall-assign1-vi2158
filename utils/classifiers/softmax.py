"""
Implementation of softmax classifer.
"""

import numpy as np


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops over N samples)

    NOTE:
    In this function, you are NOT supposed to use functions like:
    - np.dot
    - np.matmul (or operator @)
    - np.linalg.norm
    You can (not necessarily) use functions like:
    - np.sum
    - np.log
    - np.exp

    Inputs have dimension D, there are K classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: a numpy array of shape (D + 1, K) containing weights.
    - X: a numpy array of shape (N, D + 1) containing a minibatch of data.
    - y: a numpy array of shape (N,) containing training labels; y[i] = k means 
        that X[i] has label k, where 0 <= k < K.
    - reg: regularization strength. For regularization, we use L2 norm.

    Returns a tuple of:
    - loss: the mean value of loss functions over N examples in minibatch.
    - gradient: gradient wrt W, an array of same shape as W
    """

    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.    #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the       #
    # regularization!                                                          #
    # NOTE: PLEASE pay attention to data types!                                #
    ############################################################################
    ############################################################################
    #                     START OF YOUR CODE                                   #
    ############################################################################
    N = X.shape[0]
    K = W.shape[1]
    M = X.shape[1]
    norm_val = 0
   
    p = onehot(y, K)

    z = [0]*N
    for i in range(N):
        for j in range(M):
            z[i]+= X[i][j]*W[j]
    
    h = np.exp(z - np.max(z))
    for i in range(len(z)):
        h[i] /= np.sum(h[i])
            
    H = np.zeros((N,1))
    h1 = -np.log(h).T
    for i in range(N):
        for j in range(K):
            H[i] += p[i][j]*h1[j][i] 
      
    for i in range(M):
        for j in range(K):
            norm_val += W[i][j]**2
            
    loss = (1/N)*np.sum(H) + (reg/2)*norm_val
    

    diff = p-h
    for i in range(M):
        for j in range(N):
            dW[i]+=X.T[i][j]*diff[j]
    dW = (-1/N)*dW
    dW += reg*W
    # raise NotImplementedError
    ############################################################################
    #                     END OF YOUR CODE                                     #
    ############################################################################

    return loss, dW


def softmax(x):
    """
    Softmax function, vectorized version

    Inputs
    - x: (float) a numpy array of shape (N, C), containing the data

    Return a numpy array
    - h: (float) a numpy array of shape (N, C), containing the softmax of x
    """

    h = np.zeros_like(x)

    ############################################################################
    # TODO:                                                                    #
    # Implement the softmax function.                                          #
    # NOTE:                                                                    #
    # Carefully deal with different input shapes.                              #
    ############################################################################
    ############################################################################
    #                     START OF YOUR CODE                                   #
    ############################################################################
    h = np.exp(x - np.max(x))
    
    for i in range(len(x)):
        h[i] /= np.sum(h[i])
    # raise NotImplementedError
    ############################################################################
    #                     END OF YOUR CODE                                     #
    ############################################################################

    return h


def onehot(x, K):
    """
    One-hot encoding function, vectorized version.

    Inputs
    - x: (uint8) a numpy array of shape (N,) containing labels; y[i] = k means 
        that X[i] has label k, where 0 <= k < K.
    - K: total number of classes

    Returns a numpy array
    - y: (float) the encoded labels of shape (N, K)
    """

    N = x.shape[0]
    y = np.zeros((N, K))

    ############################################################################
    # TODO:                                                                    #
    # Implement the one-hot encoding function.                                 #
    ############################################################################
    ############################################################################
    #                     START OF YOUR CODE                                   #
    ############################################################################
    
    y[np.arange(N), x] = 1
    
    # raise NotImplementedError
    ############################################################################
    #                     END OF YOUR CODE                                     #
    ############################################################################

    return y


def cross_entropy(p, q):
    """
    Cross entropy function, vectorized version.

    Inputs:
    - p: (float) a numpy array of shape (N, K), containing ground truth labels
    - q: (float) a numpy array of shape (N, K), containing predicted labels

    Returns:
    - h: (float) a numpy array of shape (N,), containing the cross entropy of 
        each data point
    """

    h = np.zeros(p.shape[0])

    ############################################################################
    # TODO:                                                                    #
    # Implement cross entropy function.                                        #
    ############################################################################
    ############################################################################
    #                     START OF YOUR CODE                                   #
    ############################################################################
    h = -p*np.log(q)
    # raise NotImplementedError
    ############################################################################
    #                     END OF YOUR CODE                                     #
    ############################################################################

    return h


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    NOTE:
    In this function, you CAN (not necessarily) use functions like:
    - np.dot (unrecommanded)
    - np.matmul (or operator @)
    - np.linalg.norm
    You MUST use the functions you wrote above:
    - onehot
    - softmax
    - crossentropy

    Inputs and outputs are the same as softmax_loss_naive.
    """

    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: 																   #
	# Compute the softmax loss and its gradient using no explicit loops.       #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the       #
    # regularization!                                                          #
    ############################################################################
    ############################################################################
    #                     START OF YOUR CODE                                   #
    ############################################################################
    N = X.shape[0]
    K = W.shape[1]
    
    p = onehot(y, K)
    
    fn = np.matmul(X, W)
    
    h = softmax(fn)
    
    H = cross_entropy(p, h)
    
    loss = (1/N)*np.sum(H) + (reg/2)*(np.linalg.norm(W))**2
    
    dW = (-1/N)*np.matmul(X.T, (p - h)) + reg*W
    # raise NotImplementedError
    ############################################################################
    #                     END OF YOUR CODE                                     #
    ############################################################################

    return loss, dW
