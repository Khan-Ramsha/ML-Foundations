import numpy as np

#these are direct results of partial derivatives. 
def derivative(predictions, x, y):
    d_theta0 = np.mean((predictions - y))
    d_theta1 = np.mean(x * (predictions - y))
    return d_theta0, d_theta1

def gradient_descent(theta0, theta1, d_theta0, d_theta1, lr):
    theta0 -= lr * d_theta0
    theta1 -= lr * d_theta1
    return theta0, theta1

#exponentially weighted avg of current and past gradients. 

prev_theta0, prev_theta1, beta = 0, 0, 0.9
def avg_gradients(prev_theta0, prev_theta1 , d_theta0, d_theta1):
    prev_theta0 = beta * prev_theta0 + d_theta0
    prev_theta1 = beta * prev_theta1 + d_theta1

    return prev_theta0, prev_theta1 #so far gradients

theta0, theta1 = 0, 0
x = np.arange(1, 20)
y = 2 * x + 5
lr = 0.001
for i in range(10):
    predictions = theta0 + theta1 * x
    d_theta0, d_theta1 = derivative(predictions, x, y)
    prev_theta0, prev_theta1 = avg_gradients(prev_theta0, prev_theta1, d_theta0, d_theta1)
    theta0, theta1 = gradient_descent(theta0, theta1, prev_theta0, prev_theta1, lr)
    print(theta0, theta1)