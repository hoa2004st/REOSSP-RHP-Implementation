"""
Polynomial mutation operator.
"""
import numpy as np


def polynomial_mutation(p):
    """
    Polynomial mutation operator.
    
    Parameters:
    -----------
    p : ndarray
        Individual to mutate
        
    Returns:
    --------
    ndarray : Mutated individual
    """
    distribution_index = 5
    UB = 1.0
    LB = 0.0
    
    dim = len(p)
    mut_prob = 1.0 / dim
    
    p_new = p.copy()
    
    for i in range(dim):
        if np.random.rand() <= mut_prob:
            delta1 = (p[i] - LB) / (UB - LB)
            delta2 = (UB - p[i]) / (UB - LB)
            rand_num = np.random.rand()
            mut_pow = 1.0 / (distribution_index + 1.0)
            
            if rand_num <= 0.5:
                val = (2.0 * rand_num + (1.0 - 2.0 * rand_num) * 
                       ((1.0 - delta1) ** (distribution_index + 1.0)))
                deltaq = val ** mut_pow - 1.0
            else:
                val = (2.0 * (1.0 - rand_num) + 2.0 * (rand_num - 0.5) * 
                       ((1.0 - delta2) ** (distribution_index + 1.0)))
                deltaq = 1.0 - val ** mut_pow
            
            p_new[i] = p[i] + deltaq * (UB - LB)
            
            if p_new[i] > UB:
                p_new[i] = UB
            elif p_new[i] < LB:
                p_new[i] = LB
    
    return p_new
