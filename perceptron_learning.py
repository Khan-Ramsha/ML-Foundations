"""
Perceptron learning AND gate.
Algorithm: 
1) Initialize weights randomly
2) Converge until all the points are correctly classified
3) Pick random points having different output (0 or 1)
4) If point is supposed to be classified under (output 0) but it has 1 then 
5) updates weights such as w -> w - x ===> (w -> w learning rate * (y - y^) x ) where y is the actual output, y^ is the predicted one so (0 - 1) and learning rate is 1 here.
6) If point is supposed to be classified under (output 1) but it has 0 then 
7) updates weights such as w -> w + x ===> (w -> w learning rate * (y - y^) x ) where y is the actual output, y^ is the predicted one so (1 - 0)

"""
# Implementation using numpy
import numpy as np

#Input space
x = np.array([[0,0], [1,0], [0, 1], [1,1]])
# AND output
y = np.array([0, 0, 0, 1])

# weights
w = np.zeros(x.shape[1])
b = 0
lr = 1

def step(z):
    return 1 if z >= 0 else 0

for epoch in range(5):
    for xi, yi in zip(x, y):
        z = np.dot(w, xi) + b
        y_hat = step(z)
        error = yi - y_hat
        w += lr * error * xi
        b += lr * error
for xi in x:
    print(f"Output for xi: {step(np.dot(w,xi) + b)}")

# in pytorch
import torch
import torch.nn as nn

#Input space
x = torch.tensor([[0,0], [1,0], [0, 1], [1,1]], dtype=torch.float32)
# AND output
y = torch.tensor([0, 0, 0, 1], dtype=torch.float32)

# weights
w = torch.zeros(x.shape[1], requires_grad= False)
b = 0
lr = 1

def step(z):
    return torch.where(z >= 0.0, 1.0, 0.0)

for epoch in range(5):
    for xi, yi in zip(x, y):
        z = torch.dot(w, xi) + b
        y_hat = step(z)
        error = yi - y_hat
        w += lr * error * xi
        b += lr * error
for xi in x:
    print(f"Output for xi: {step(torch.dot(w,xi) + b).item()}")