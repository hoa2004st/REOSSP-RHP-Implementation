"""
Selection operator for combining populations.
"""
import numpy as np


def selection(next_size, pop1, fit1, pop2, fit2):
    """
    Select best individuals from two populations.
    
    Parameters:
    -----------
    next_size : int
        Size of next population
    pop1 : ndarray
        First population
    fit1 : ndarray
        Fitness of first population
    pop2 : ndarray
        Second population
    fit2 : ndarray
        Fitness of second population
        
    Returns:
    --------
    tuple : (next_pop, next_fit) - Selected population and fitness
    """
    # Combine populations
    imm_pop = np.vstack([pop1, pop2])
    imm_fit = np.concatenate([fit1, fit2])
    
    # Sort by fitness (descending order - maximization)
    idx = np.argsort(imm_fit)[::-1]
    
    # Select top individuals
    next_pop = imm_pop[idx[:next_size]]
    next_fit = imm_fit[idx[:next_size]]
    
    return next_pop, next_fit
