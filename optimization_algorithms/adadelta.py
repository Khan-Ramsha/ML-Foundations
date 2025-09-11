""""
This one comes after rmsprop. Core idea is to "make the algorithm such that theres no need for initial learning rate"
so this apparently elimates whole LR by adding "sqroot of ut / sqroot of vt" where ut = accumulation of parameter updates and vt is accumulation of gradient updates
And this idea grows LR when on gentle surfaces and decreases LR on steeper ones. 

"""

import numpy as np

def derivative(predictions, x, y):
    d_theta0 = np.mean((predictions - y))
    d_theta1 = np.mean((predictions - y) * x)
    return d_theta0, d_theta1

def gradient_descent(theta0, theta1, wt0, wt1):
    theta0 += wt0
    theta1 += wt1
    return theta0, theta1

def accumulated_sq_gradient(d_theta0, d_theta1, beta, eps, ut0, ut1, vt0, vt1):
    vt0 = beta * vt0 + (1 - beta) * (d_theta0 ** 2)
    vt1 = beta * vt1 + (1 - beta) * (d_theta1 ** 2)

    #parameter updates
    wt0 = -np.sqrt(ut0 + eps)/np.sqrt(vt0 + eps) * d_theta0
    wt1 = - np.sqrt(ut1 + eps)/np.sqrt(vt1 + eps) * d_theta1

    ut0 = beta * ut0 + (1 - beta) * (wt0 ** 2)
    ut1 = beta * ut1 + (1 - beta) * (wt1 ** 2)

    return wt0, wt1, vt0, vt1, ut0, ut1

x = np.arange(1, 20)
y = 2 * x + 5
theta0 , theta1 = 0, 0 # theta0 is bias, theta1 is W
beta, eps = 0.9, 1e-8
ut0, ut1, vt0, vt1 = 0, 0, 0, 0

for i in range(10):
    predictions = theta0 + theta1 * x
    d_theta0, d_theta1 = derivative(predictions, x, y)
    wt0, wt1, vt0, vt1, ut0, ut1 = accumulated_sq_gradient(d_theta0, d_theta1, beta, eps, ut0, ut1, vt0, vt1)
    theta0, theta1 = gradient_descent(theta0, theta1, wt0, wt1)