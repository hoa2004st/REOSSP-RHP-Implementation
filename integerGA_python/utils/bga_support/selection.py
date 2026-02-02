"""
Selection functions for Binary Genetic Algorithm.
Based on Yarpiz implementation (www.yarpiz.com)
"""
import numpy as np


def roulette_wheel_selection(P):
    """
    Roulette Wheel Selection.
    
    Parameters:
    -----------
    P : array-like
        Probability distribution
        
    Returns:
    --------
    int : Selected index
    """
    r = np.random.rand()
    c = np.cumsum(P)
    i = np.where(r <= c)[0]
    if len(i) > 0:
        return i[0]
    return len(P) - 1


def tournament_selection(pop, m):
    """
    Tournament Selection.
    
    Parameters:
    -----------
    pop : list
        Population list where each individual has a 'cost' attribute
    m : int
        Tournament size
        
    Returns:
    --------
    int : Index of selected individual
    """
    n_pop = len(pop)
    
    # Random sample of m individuals
    indices = np.random.choice(n_pop, m, replace=False)
    
    # Find the one with minimum cost (best)
    costs = [pop[i]['cost'] for i in indices]
    best_idx = np.argmin(costs)
    
    return indices[best_idx]
