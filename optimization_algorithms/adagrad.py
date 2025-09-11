"""
Adaptive Learning rate. 
-> adapts LR for each parameter based on past history of gradients
-> handles sparse features well
"""
import numpy as np

#these are direct results of partial derivatives. 
def derivative(predictions, x, y):
    d_theta0 = np.mean((predictions - y))
    d_theta1 = np.mean(x * (predictions - y))
    return d_theta0, d_theta1

def gradient_descent(theta0, theta1, d_theta0, d_theta1, prev_theta0, prev_theta1, lr, eps):
    theta0 -= (lr/(np.sqrt(prev_theta0 + eps))) * d_theta0
    theta1 -= (lr/(np.sqrt(prev_theta1 + eps))) * d_theta1
    return theta0, theta1


prev_theta0, prev_theta1, eps= 0, 0, 1e-8
def accumulated_sq_grad(prev_theta0, prev_theta1 , d_theta0, d_theta1):
    prev_theta0 = prev_theta0 + (d_theta0 ** 2)
    prev_theta1 = prev_theta1 + (d_theta1 ** 2)

    return prev_theta0, prev_theta1 #so far gradients

theta0, theta1 = 0, 0
x = np.arange(1, 20)
y = 2 * x + 5
lr = 0.1
for i in range(10):
    predictions = theta0 + theta1 * x
    d_theta0, d_theta1 = derivative(predictions, x, y)
    prev_theta0, prev_theta1 = accumulated_sq_grad(prev_theta0, prev_theta1, d_theta0, d_theta1)
    theta0, theta1 = gradient_descent(theta0, theta1, d_theta0, d_theta1, prev_theta0, prev_theta1, lr, eps)
    print(theta0, theta1)