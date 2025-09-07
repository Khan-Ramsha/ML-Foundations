"""
This file implements batch gradient descent for Linear regression
passing entire dataset at once for computing gradients and updating parameters only once after an iteration
1 epoch = 1 iteration

1 complete Epoch: 1 complete pass through the entire training dataset. 
1 complete Iteration: a single update of model parameters (weights & bias) after processing a batch of data
if batchsize = 2 and 100 training samples then 50 iterations each consist of 2 samples.

Batch size = total samples → Batch GD
    
In batch Gradient descent: 1 epoch = 1 iteration as entire dataset is passed at once

"""

import numpy as np
x = np.arange(1, 50)
y = 2 * x + 5
lr = 0.001
epochs = 100
theta0, theta1 = 0,0

for i in range(epochs):
    predictions = theta0 + theta1 * x
    # calculate partial derivative
    d_theta0 = np.mean((predictions - y))
    d_theta1 = np.mean(x * (predictions - y))

    #updating parameters 
    theta0 -= lr * d_theta0
    theta1 -= lr * d_theta1
    cost = (1/(2 * len(y))) * (np.sum((predictions - y) ** 2))
    print(cost)