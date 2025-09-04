"""
Logistic regression
for classification hypothesis = hθ(x) = g((θT x))

-> g(z): where z is the real number. here z = hypothesis(x)
-> hypothetis(x): x = theta transpose + x
-> g(z) is the sigmoid function. 1 / 1 + e^-z (z = theta transpose + x)

Cost function is different from linear regression: 
J(theta) = 1/m [∑ 1 to m (y * log hypothesis(x) + (1 - y) log (1 - hypothesis(x)))]
by putting y = 1 formula turns out to be-> log*hypothesis(x)
so if hypothesis(x) predicts 1 then log(1) = 0 (NO PENALTY)

by putting y =  0 formula turns out to be-> (1)* log (1 - hypothesis(x))
so if hypothesis(x) predicts 0 then log(1 - 0) = log(1) = 0 (NO PENALTY)

else heavy penalty!

"""

import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self):
         # weights and bias are scalar values
         self.theta0 = np.random.uniform() 
         self.theta1 = np.random.uniform()

    def sigmoid(self,z):
         return 1 / (1 + np.exp(-z))

    def hypothesis(self,x):
        z = self.theta0 + np.dot(x, self.theta1)
        return self.sigmoid(z)
    
    def predict(self,x):
        threshold = 0.5
        z = self.theta0 + np.dot(x, self.theta1)
        probab = self.sigmoid(z)
        return (probab >= threshold).astype(int)
    # aim is to find best theta0 and theta1

    def cost_function(self, x , y):
        predictions = self.hypothesis(x)
        cost = (1 / len(y)) * np.sum(- (y * np.log(predictions)) - (1 - y) * np.log(1 - predictions))
        return cost

    # aim is to minimize this cost 
    def derivative(self, x ,y):
        predictions = self.hypothesis(x)
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
            loss = self.cost_function(x,y)
            losses.append(loss)
            plt.plot(losses)
            plt.xlabel("Iterations") # using batch-gradient descent here, so 1 epoch = 1 iteration (passing whole dataset)
            plt.ylabel("Losses")
            print(f"Loss at epoch {i}: {loss}")
            # plt.show()
        return losses

x_train = np.arange(1,50)
y_train = (x_train < 25).astype(int) #binary output
lr = 0.001
epochs = 500
model =LogisticRegression()
losses = model.fit(x_train, y_train, lr, epochs)
predictions = model.predict(x_train)
print(predictions)