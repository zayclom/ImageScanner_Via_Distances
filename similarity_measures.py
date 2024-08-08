import numpy as np

def euclidean_distance(feature1, feature2):
    return np.linalg.norm(feature1 - feature2)

def manhattan_distance(feature1, feature2):
    return np.sum(np.abs(feature1 - feature2))

def chebyshev_distance(feature1, feature2):
    return np.max(np.abs(feature1 - feature2))

def canberra_distance(feature1, feature2):
    return np.sum(np.abs(feature1 - feature2) / (np.abs(feature1) + np.abs(feature2) + np.finfo(float).eps))
