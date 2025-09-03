"""linear regression: univariate
hypothesis = theta0 + theta1 * x"""

import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
         # weights and bias are scalar values
         self.theta0 = np.random.uniform() 
         self.theta1 = np.random.uniform()

    def predict(self,x):
        return self.theta0 + self.theta1 * x
    # aim is to find best theta0 and theta1

    def cost_function(self,predictions, y):
        cost = (1/(2 * len(y))) * (np.sum((predictions - y) ** 2))
        return cost

    # aim is to minimize this cost 
    def derivative(self, x, y):
        predictions = self.predict(x)
        d_theta0 = np.mean((predictions - y))
        d_theta1 = np.mean( x * (predictions - y))
        return d_theta0, d_theta1

    def gradient_descent(self, d_theta0, d_theta1,lr):
            #updating parameters
            self.theta0 -= lr * d_theta0 
            self.theta1 -= lr * d_theta1
    def fit(self,x,y,lr, epochs):
        losses = []
        for i in range(epochs):
            d_theta0, d_theta1 = self.derivative(x, y)
            #updating parameters
            self.gradient_descent(d_theta0, d_theta1, lr)
            predictions = self.predict(x)
            loss = self.cost_function(predictions, y)
            losses.append(loss)
            plt.plot(losses)
            plt.xlabel("Iterations") # using batch-gradient descent here, so 1 epoch = 1 iteration (passing whole dataset)
            plt.ylabel("Losses")
            print(f"Loss at epoch {i}: {loss}")
            # plt.show()
        return losses

x_train = np.arange(1,50)
y_train = 2 * x_train + 5
lr = 0.001
epochs = 100
model = LinearRegression()
losses = model.fit(x_train, y_train, lr, epochs)
predictions = model.predict(x_train)
print(predictions)

"""linear regression: multivariate
hypothesis = theta0 * x0 + theta1 * x1 + theta2 * x2 
x0 = 1
 """

import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionMultiVar:
    def __init__(self):
         self.theta0 = np.random.uniform(size = [1, ]) 
         self.theta1 = np.random.uniform(size = [5,  ])

    def predict(self,x):
        return np.dot(x, self.theta1) + self.theta0 
    # aim is to find best theta0 and theta1

    def cost_function(self,predictions, y):
        cost = (1/(2 * len(y))) * (np.sum((predictions - y) ** 2))
        return cost

    # aim is to minimize this cost 
    def derivative(self, x, y):
        predictions = self.predict(x)
        d_theta0 = np.mean((predictions - y))
        d_theta1 = np.mean(np.dot(x.T , (predictions - y)))
        return d_theta0, d_theta1

    def gradient_descent(self, d_theta0, d_theta1,lr):
            #updating parameters
            self.theta0 -= lr * d_theta0 
            self.theta1 -= lr * d_theta1
    def fit(self,x,y,lr, epochs):
        losses = []
        for i in range(epochs):
            d_theta0, d_theta1 = self.derivative(x, y)
            #updating parameters
            self.gradient_descent(d_theta0, d_theta1, lr)
            predictions = self.predict(x)
            loss = self.cost_function(predictions, y)
            losses.append(loss)
            plt.plot(losses)
            plt.xlabel("Iterations") # using batch-gradient descent here, so 1 epoch = 1 iteration (passing whole dataset)
            plt.ylabel("Losses")
            print(f"Loss at epoch {i}: {loss}")
            # plt.show()
        return losses

x_train = np.random.rand(100,5)
y_train = 2 * x_train @ np.array([1,2,3,4,5]) + 5
lr = 0.001
epochs = 100
model = LinearRegressionMultiVar()
losses = model.fit(x_train, y_train, lr, epochs)
predictions = model.predict(x_train)
print(predictions)