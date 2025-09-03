"""linear regression: univariate
hypothesis = theta0 + theta1 * x"""

import numpy as np
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
            print(f"Loss at epoch {i}: {loss}")
        return losses

x_train = np.arange(1,50)
y_train = 2 * x_train + 5
lr = 0.001
epochs = 100
model = LinearRegression()
losses = model.fit(x_train, y_train, lr, epochs)
predictions = model.predict(x_train)
print(predictions)