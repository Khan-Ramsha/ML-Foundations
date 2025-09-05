import pandas as pd
import math

def gini(samples):
    sum_samples = sum(samples)
    probab = []
    scores = 0
    for sample in samples:
        proba = sample / sum_samples
        probab.append(proba)
    for p in probab:
        scores += p * (1 - p)
    return scores

def entropy(samples):
    sum_samples = sum(samples)
    probab = []
    scores = 0
    for sample in samples:
        proba = sample / sum_samples
        probab.append(proba)
    for p in probab:
        if p > 0:
            scores += p * math.log(p)
    return -1 * scores

humidity = [85, 70, 65, 90, 75, 55, 66, 78, 96, 44]
labels = [1, 1, 0, 0, 0, 1, 1, 0, 0, 1]
threshold = 55
def split_data(threshold, humidity, labels):
    left = []
    right = []
    left_labels = []
    right_labels = []
    for i in range(len(humidity)):
        if humidity[i] <= threshold:
            left.append(humidity[i])
            left_labels.append(labels[i])
        else:
            right.append(humidity[i])
            right_labels.append(labels[i])
    ones_left = left_labels.count(1)
    zeros_left = left_labels.count(0)
    ones_right = right_labels.count(1)      
    zeros_right = right_labels.count(0)
    left_gini = gini([ones_left, zeros_left]) # gini for left child ==> 0.48
    right_gini = gini([ones_right, zeros_right]) # gini for right child ==> 0.375
    return len(left), len(right), left_gini, right_gini

split_data(threshold , humidity, labels)
print(gini([labels.count(1), labels.count(0)])) # reduce the gini
"""
Algorithm:
split dataset into left and right childrens
calculate gini for each
calculate weight avg of children
calculate information gain (parenty impurity - weight children impurity)
pick threshold with high information gain
"""
total_samples = len(humidity)

def weight_avg(left_samples , right_samples, total_samples, left_gini, right_gini):
    """
    weight = (left_samples/ total)  * left_gini / (right_samples / total) * right_gini
    """
    avg = (left_samples/total_samples) * left_gini + (right_samples/ total_samples) * right_gini
    return avg

def info_gain(parent, child):
    return parent - child

left, right, left_gini, right_gini = split_data(threshold , humidity, labels)
child = weight_avg(left, right, total_samples, left_gini, right_gini)
parent = gini([labels.count(1), labels.count(0)])
print(info_gain(parent , child)) # gini = 0.125