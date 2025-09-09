"""
K Nearest Neighbor allows us to get an idea of about basic approach of image classification
Since, images are represented as "multi-dim arrays" consist of height * weight * channels (for color images- 3 channel and grey - 1 channel) 
The idea is to predict the labels of test images by comparing the test image pixels with the training set images pixels. 
Computing the difference between the pixels. The result achieved speaks about similarity or how close the test image is to train image. 

"""

# Let's start by creating a dummy dataset first
"""
x -> (num_samples, num_features)
y -> (num_samples, )    
"""

x_labels = 