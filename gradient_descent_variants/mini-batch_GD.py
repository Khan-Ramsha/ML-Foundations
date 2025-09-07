"""
This file implements mini-batch gradient descent for Linear regression. 
passing batches (not entire dataset) for computing gradients and updating parameters after computing gradients for that mini-batch.

In mini-batch Gradient descent: 1 epoch = 25 iteration (if 50 samples in entire dataset and batchsize = 2)

Batch size > 1 but < total samples → Mini-batch GD

"""
import numpy as np
x = np.arange(1, 50)
y = 2 * x + 5
lr = 0.001
epochs = 100
theta0, theta1 = 0,0
samples_per_batch = 5
for i in range(epochs):
    #shuffling
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    x_shuffled, y_shuffled = x[indices], y[indices] #shuffled data
    for i in range(0, len(x), samples_per_batch):
        batch_x = x_shuffled[i: i + samples_per_batch]
        batch_y = y_shuffled[i: i + samples_per_batch]
        predictions = theta0 + theta1 * batch_x
        # calculate partial derivative
        d_theta0 = np.mean((predictions - batch_y))
        d_theta1 = np.mean(batch_x * (predictions - batch_y))

        #updating parameters 
        theta0 -= lr * d_theta0
        theta1 -= lr * d_theta1
        cost = (1/(2 * len(batch_y))) * (np.sum((predictions - batch_y) ** 2))
        print(cost)