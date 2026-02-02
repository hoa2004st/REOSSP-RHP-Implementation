"""
Mutation operators for Binary Genetic Algorithm.
Based on Yarpiz implementation (www.yarpiz.com)
"""
import numpy as np


def mutate(x, mu):
    """
    Perform mutation on binary chromosome.
    
    Parameters:
    -----------
    x : array-like
        Chromosome to mutate
    mu : float
        Mutation rate
        
    Returns:
    --------
    array : Mutated chromosome
    """
    n_var = len(x)
    n_mu = int(np.ceil(mu * n_var))
    
    # Random sample of positions to mutate
    j = np.random.choice(n_var, n_mu, replace=False)
    
    y = x.copy()
    y[j] = 1 - x[j]
    
    return y
