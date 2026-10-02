import numpy as np

def euclidean_distance(x: list, y: list) -> float:
    """
    Returns the Euclidean distance as a float.
    """
    distance = np.sqrt(np.sum((np.array(x)-np.array(y))**2))
    return  float(distance)