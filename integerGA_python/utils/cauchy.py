"""
Cauchy distribution sampling function.
"""
import numpy as np


def cauchy(mu, gamma):
    """
    Generate a random sample from Cauchy distribution.
    
    Parameters:
    -----------
    mu : float
        Location parameter
    gamma : float
        Scale parameter
        
    Returns:
    --------
    float : Random value from Cauchy distribution
    """
    return mu + gamma * np.tan(np.pi * (np.random.rand() - 0.5))
