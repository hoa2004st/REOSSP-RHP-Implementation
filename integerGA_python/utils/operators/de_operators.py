"""
Differential Evolution mutation operators.
"""
import numpy as np


def operate_pbest_1_bin(pop, target_index, best_index, cr, sf, archive, arc_count):
    """
    DE/pbest/1/bin mutation operator.
    V = X_best + F(X_rand1 - X_rand2)
    
    Parameters:
    -----------
    pop : ndarray
        Current population
    target_index : int
        Index of target individual
    best_index : int
        Index of best individual
    cr : float
        Crossover rate
    sf : float
        Scaling factor
    archive : ndarray
        Archive of inferior solutions
    arc_count : int
        Number of solutions in archive
        
    Returns:
    --------
    ndarray : Offspring individual
    """
    LB = 0.0
    UB = 1.0
    
    pop_size, dim = pop.shape
    
    r1 = np.random.randint(0, pop_size)
    
    # Use archive with probability
    if arc_count > 0 and np.random.rand() < arc_count / (arc_count + pop_size):
        arc = np.random.randint(0, arc_count)
        v = pop[best_index] + sf * (pop[r1] - archive[arc])
    else:
        r2 = np.random.randint(0, pop_size)
        while r1 == r2:
            r2 = np.random.randint(0, pop_size)
        v = pop[best_index] + sf * (pop[r1] - pop[r2])
    
    offspring = pop[target_index].copy()
    j_rand = np.random.randint(0, dim)
    
    for j in range(dim):
        if np.random.rand() <= cr or j == j_rand:
            offspring[j] = v[j]
            
            # Add small Gaussian noise with 5% probability
            if np.random.rand() <= 0.05:
                offspring[j] = offspring[j] + np.random.normal(0, 0.05)
            
            # Bound checking
            if offspring[j] > UB:
                offspring[j] = UB
            elif offspring[j] < LB:
                offspring[j] = LB
    
    return offspring


def operate_current_to_pbest_1_bin(pop, target_index, best_index, cr, sf, archive, arc_count):
    """
    DE/current-to-pbest/1/bin mutation operator.
    V = X + F(X_best - X + X_rand1 - X_rand2)
    
    Parameters:
    -----------
    pop : ndarray
        Current population
    target_index : int
        Index of target individual
    best_index : int
        Index of best individual
    cr : float
        Crossover rate
    sf : float
        Scaling factor
    archive : ndarray
        Archive of inferior solutions
    arc_count : int
        Number of solutions in archive
        
    Returns:
    --------
    ndarray : Offspring individual
    """
    LB = 0.0
    UB = 1.0
    
    pop_size, dim = pop.shape
    r1 = np.random.randint(0, pop_size)
    
    # Use archive with probability
    if arc_count > 0 and np.random.rand() < arc_count / (arc_count + pop_size):
        arc = np.random.randint(0, arc_count)
        v = (pop[target_index] + sf * (pop[best_index] - pop[target_index] + 
             pop[r1] - archive[arc]))
    else:
        r2 = np.random.randint(0, pop_size)
        while r1 == r2:
            r2 = np.random.randint(0, pop_size)
        v = (pop[target_index] + sf * (pop[best_index] - pop[target_index] + 
             pop[r1] - pop[r2]))
    
    offspring = pop[target_index].copy()
    j_rand = np.random.randint(0, dim)
    
    for j in range(dim):
        if np.random.rand() <= cr or j == j_rand:
            offspring[j] = v[j]
            
            # Add small Gaussian noise with 5% probability
            if np.random.rand() <= 0.05:
                offspring[j] = offspring[j] + sf * np.random.normal(0, 0.05)
            
            # Bound checking
            if offspring[j] > UB:
                offspring[j] = UB
            elif offspring[j] < LB:
                offspring[j] = LB
    
    return offspring
