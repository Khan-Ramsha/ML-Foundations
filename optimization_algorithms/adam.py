"""
The combination of rmsprop and momentum!!

"""
import numpy as np

#these are direct results of partial derivatives. 
def derivative(predictions, x, y):
    d_theta0 = np.mean((predictions - y))
    d_theta1 = np.mean(x * (predictions - y))
    return d_theta0, d_theta1

def gradient_descent(theta0, theta1, mt0_hat, mt1_hat, vt0_hat, vt1_hat, lr, eps = 1e-8):
    theta0 -= lr/(np.sqrt(vt0_hat) + eps) * mt0_hat
    theta1 -= lr/(np.sqrt(vt1_hat) + eps) * mt1_hat 
    return theta0, theta1

def avg_gradients(mt0, mt1, vt0, vt1, d_theta0, d_theta1, i):
    mt0 = beta1 * mt0 + (1 - beta1) * d_theta0
    mt1 = beta1 * mt1 + (1 - beta1) * d_theta1

    vt0 = beta2 * vt0 + (1 - beta2) * (d_theta0 ** 2)
    vt1 = beta2 * vt1 + (1 - beta2) * (d_theta1 ** 2)

    mt0_hat = mt0 / (1 - np.power(beta1, i + 1))
    mt1_hat = mt1 / (1 - np.power(beta1, i + 1))
    vt0_hat = vt0 / (1 - np.power(beta2, i + 1))
    vt1_hat = vt1 / (1 - np.power(beta2, i + 1))

    return mt0, mt1, vt0, vt1, mt0_hat, mt1_hat, vt0_hat, vt1_hat

theta0, theta1 = 0, 0
mt0, mt1, vt0, vt1 = 0,0,0,0
beta1, beta2 = 0.9, 0.999
x = np.arange(1, 20)
y = 2 * x + 5
lr = 0.01
for i in range(10):
    predictions = theta0 + theta1 * x
    d_theta0, d_theta1 = derivative(predictions, x, y)
    mt0, mt1, vt0, vt1,mt0_hat, mt1_hat, vt0_hat, vt1_hat = avg_gradients(mt0, mt1, vt0, vt1, d_theta0, d_theta1, i)
    theta0, theta1 = gradient_descent(theta0, theta1, mt0_hat, mt1_hat, vt0_hat, vt1_hat, lr)