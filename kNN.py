"""
K Nearest Neighbor allows us to get an idea about basic approach of image classification
Since, images are represented as "multi-dim arrays" consist of height * weight * channels (for color images- 3 channel and grey - 1 channel) 
The idea is to predict the labels of test images by comparing the test image pixels with the training set images pixels. 
Computing the difference between the pixels. The result achieved speaks about similarity or how close the test image is to train image. 

"""

# Let's start by creating a dummy dataset first
"""
x_tr -> training images -> (num_samples, num_features)-> shape (30000, 32, 32, 3) 32 pixels wide, 32 pixels tall, 3 channels
y_tr -> training labels -> (num_samples, ) 

x_te -> test images -> (10000, 32, 32, 3)
y_te -> test labels -> ((10000,))
"""
#Nearest neighbor using L1 Distance. 
import numpy as np

x_tr = np.random.randint(0, 256, size = (30000,3072))
y_tr = np.random.randint(0, 10, size = (30000,))
x_te = np.random.randint(0, 256, size = (10000, 3072))

def predict(test_data):
    y_te = test_data.shape[0]
    y_pred = np.zeros(y_te)
    
    for i in range(y_te):
        distances = np.sum(np.abs(x_tr - test_data[i, : ]), axis = 1) # row wise summing of distance
        idx_min = np.argmin(distances)
        y_pred[i] = y_tr[idx_min] 
    
    return y_pred

# The above is the implementation of 1 nearest neighbor but problem with 1NN is that it is sensitive to outliers. 
# So we pick 'k' nearest neighbors and majority wins for voting for label of test image 

class kNN:
    def __init__(self, k):
        self.k = k
    
    def fit(self, x_tr, y_tr):
        self.x_tr = x_tr
        self.y_tr = y_tr

    def predict(self, x_te):
        # y_te = x_te.shape[0]
        y_pred = []
        for x in x_te:
            distances = []
            for x_train in self.x_tr:
                distance = np.linalg.norm(x - x_train)
                distances.append(distance)
               
            top_k = np.argsort(distances)[:self.k]
            indx = self.y_tr[top_k]
            #assuming you have an array now
            label, counts = np.unique(indx, return_counts = True)
            y_pred.append(label[np.argmax(counts)])
        return y_pred
        