"""
Crossover operators for Binary Genetic Algorithm.
Based on Yarpiz implementation (www.yarpiz.com)
"""
import numpy as np
from .selection import roulette_wheel_selection


def single_point_crossover(x1, x2):
    """
    Single Point Crossover.
    
    Parameters:
    -----------
    x1, x2 : array-like
        Parent chromosomes
        
    Returns:
    --------
    tuple : Two offspring
    """
    n_var = len(x1)
    c = np.random.randint(1, n_var)
    
    y1 = np.concatenate([x1[:c], x2[c:]])
    y2 = np.concatenate([x2[:c], x1[c:]])
    
    return y1, y2


def double_point_crossover(x1, x2):
    """
    Double Point Crossover.
    
    Parameters:
    -----------
    x1, x2 : array-like
        Parent chromosomes
        
    Returns:
    --------
    tuple : Two offspring
    """
    n_var = len(x1)
    
    # Sample two different points
    cc = np.random.choice(n_var - 1, 2, replace=False)
    c1 = min(cc)
    c2 = max(cc)
    
    y1 = np.concatenate([x1[:c1], x2[c1:c2], x1[c2:]])
    y2 = np.concatenate([x2[:c1], x1[c1:c2], x2[c2:]])
    
    return y1, y2


def uniform_crossover(x1, x2):
    """
    Uniform Crossover.
    
    Parameters:
    -----------
    x1, x2 : array-like
        Parent chromosomes
        
    Returns:
    --------
    tuple : Two offspring
    """
    alpha = np.random.randint(0, 2, size=x1.shape)
    
    y1 = alpha * x1 + (1 - alpha) * x2
    y2 = alpha * x2 + (1 - alpha) * x1
    
    return y1, y2


def crossover(x1, x2, p_single_point, p_double_point):
    """
    Perform crossover using one of three methods based on probabilities.
    
    Parameters:
    -----------
    x1, x2 : array-like
        Parent chromosomes
    p_single_point : float
        Probability of single point crossover
    p_double_point : float
        Probability of double point crossover
        
    Returns:
    --------
    tuple : Two offspring
    """
    p_uniform = 1 - p_single_point - p_double_point
    
    method = roulette_wheel_selection([p_single_point, p_double_point, p_uniform])
    
    if method == 0:
        return single_point_crossover(x1, x2)
    elif method == 1:
        return double_point_crossover(x1, x2)
    else:  # method == 2
        return uniform_crossover(x1, x2)
