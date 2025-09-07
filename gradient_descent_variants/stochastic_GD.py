"""
This file implements stochastic gradient descent for Linear regression
passing one sample  once for computing gradients and updating parameters after computing gradients for that one sample.

In stochastic Gradient descent: 1 epoch = 100 iteration (if 100 samples in entire dataset)

Batch size = 1 → Stochastic GD (not mini-batch)

"""

import numpy as np
x = np.arange(1, 50)
y = 2 * x + 5
lr = 0.001
epochs = 100
theta0, theta1 = 0,0

for i in range(epochs):
    for i in range(len(x)):
        predictions = theta0 + theta1 * x[i]
        # calculate partial derivative
        d_theta0 = (predictions - y[i])
        d_theta1 = x[i] * (predictions - y[i])

        #updating parameters 
        theta0 -= lr * d_theta0
        theta1 -= lr * d_theta1
        cost = ((predictions - y[i]) ** 2)
        print(cost)