# Let's create shallow NN for solving logistic regression problem
"""
The important steps to build Neural Network:
-> Decide the model structure => number of input features, number of hidden layers, number of neurons in hidden layers and output layer
Let's keep it simple, 
Layer1: with 3 input features (x1, x2, x3)
Layer2 (Hidden layer): with 3 neurons 
Layer3 (Output layer): with 1 neuron
there will be 2 weight matrices - 1st matrix (from layer1 to layer2), 2nd matrix (from layer2 to layer3)
now comes the most important part: "shapes" of weight matrices
number of neurons in the following layer is the number of rows for weights of layer1 
number of neurons in the current layer is the number of cols for weights of layer1 
shape of x = rows are number of feature , cols are number of samples
"""
import numpy as np

w1 = np.random.randn(3, 3) * 0.01
b1 = np.zeros((3, 1))

w2 = np.random.randn(1, 3) * 0.01
b2 = np.zeros((1,1))

x = np.array([[1,2, 3], [4, 5, 6], [7, 8, 9]]) # shape 3*3
y = np.array([[1, 1, 0]])

def sigmoid(z):
    return 1/(1 + np.exp(-z))

for i in range(100):
    
    m = x.shape[1]  # Number of samples

    #forward pass
    z1 = np.dot(w1, x) + b1 #linear transformation
    a1 = sigmoid(z1)

    #next layer
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)

    #cost function
    cost = - (1 / m) * np.sum(y * np.log(a2) + (1 - y) * np.log(1 - a2))

    # #backpropagation

    #compute output gradient
    dz2 = a2 - y
    # gradients wrt to parameters
    dw2 = np.dot(dz2, a1.T) / m
    db2 = np.sum(dz2, axis = 1, keepdims= True) / m

    #compute gradient wrt layer below
    da1 = np.dot(w2.T,dz2) / m
    dz1 = da1 * (a1) * (1 - a1) #by applying sigmoid derivative
    dw1 = np.dot(dz1, x.T) / m 
    db1 = np.sum(dz1, axis = 1, keepdims= True) / m

    #weight updates
    lr = 0.01
    w1 -= lr * dw1
    w2 -= lr * dw2
    b1 -= lr * db1
    b2 -= lr * db2