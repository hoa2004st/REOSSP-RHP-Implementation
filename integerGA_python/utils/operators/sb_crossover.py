"""
Simulated Binary (SBX) Crossover operator.
Based on: Kalyanmoy Deb - An Efficient Constraint Handling Method for Genetic Algorithms
"""
import numpy as np


def sb_crossover(p1, p2):
    """
    Simulated Binary Crossover (SBX).
    
    Parameters:
    -----------
    p1, p2 : ndarray
        Parent individuals
        
    Returns:
    --------
    ndarray : Two offspring (2 x dim)
    """
    EPS = 1e-6
    eta_c = 2
    UB = 1.0
    LB = 0.0
    
    dim = len(p1)
    o1 = np.zeros(dim)
    o2 = np.zeros(dim)
    
    for i in range(dim):
        if np.random.rand() <= 0.5 and abs(p1[i] - p2[i]) >= EPS:
            # Perform crossover
            y1 = p1[i]
            y2 = p2[i]
            if p1[i] > p2[i]:
                y1 = p2[i]
                y2 = p1[i]
            
            rand_num = np.random.rand()
            
            # First offspring
            beta = 1.0 + (2.0 * (y1 - LB) / (y2 - y1))
            alpha = 2.0 - beta ** (-(eta_c + 1.0))
            
            if rand_num <= (1.0 / alpha):
                betaq = (rand_num * alpha) ** (1.0 / (eta_c + 1.0))
            else:
                betaq = (1.0 / (2.0 - rand_num * alpha)) ** (1.0 / (eta_c + 1.0))
            
            o1[i] = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
            if o1[i] > UB:
                o1[i] = UB
            elif o1[i] < LB:
                o1[i] = LB
            
            # Second offspring
            beta = 1.0 + (2.0 * (UB - y2) / (y2 - y1))
            alpha = 2.0 - beta ** (-(eta_c + 1.0))
            
            if rand_num <= (1.0 / alpha):
                betaq = (rand_num * alpha) ** (1.0 / (eta_c + 1.0))
            else:
                betaq = (1.0 / (2.0 - rand_num * alpha)) ** (1.0 / (eta_c + 1.0))
            
            o2[i] = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
            if o2[i] > UB:
                o2[i] = UB
            elif o2[i] < LB:
                o2[i] = LB
        else:
            o1[i] = p1[i]
            o2[i] = p2[i]
    
    return np.array([o1, o2])
