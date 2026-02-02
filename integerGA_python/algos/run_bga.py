"""
Binary Genetic Algorithm (BGA) implementation.
Based on Yarpiz implementation (www.yarpiz.com)
"""
import numpy as np
from ..utils.bga_support.crossover import crossover
from ..utils.bga_support.mutation import mutate
from ..utils.bga_support.selection import roulette_wheel_selection, tournament_selection


def run_bga(dim, obj_func, max_fes, pop_size, verbose=True, 
            p_single_point=0.1, p_double_point=0.2):
    """
    Run Binary Genetic Algorithm for maximization.
    
    Parameters:
    -----------
    dim : int
        Problem dimension
    obj_func : callable
        Objective function to maximize
    max_fes : int
        Maximum function evaluations
    pop_size : int
        Population size
    verbose : bool
        Whether to print progress
    p_single_point : float
        Probability of single point crossover
    p_double_point : float
        Probability of double point crossover
        
    Returns:
    --------
    tuple : (best_solution, convergence, best_obj)
    """
    p_uniform = 1 - p_single_point - p_double_point
    
    if verbose:
        print(f"BGA running for maximization, generation = {max_fes//pop_size}, "
              f"n={pop_size}, pSinglePoint={p_single_point}, "
              f"pDoublePoint={p_double_point}, pUniform={p_uniform}")
    
    # GA Parameters
    max_it = max_fes // pop_size
    pc = 0.8  # Crossover percentage
    nc = 2 * round(pc * pop_size / 2)  # Number of offsprings
    pm = 0.3  # Mutation percentage
    nm = round(pm * pop_size)  # Number of mutants
    mu = 0.02  # Mutation rate
    
    # Selection method
    use_tournament_selection = True
    tournament_size = 3
    
    # Initialize population
    pop = []
    for i in range(pop_size):
        individual = {}
        if i == pop_size - 1:
            individual['position'] = np.ones(dim)
        else:
            individual['position'] = np.random.randint(0, 2, size=dim)
        individual['cost'] = obj_func(individual['position'])
        pop.append(individual)
    
    # Sort population by cost (ascending for minimization in code, but we negate for max)
    pop = sorted(pop, key=lambda x: x['cost'], reverse=False)
    
    # Store best solution
    best_sol = pop[0].copy()
    
    # Convergence tracking
    convergence = np.zeros(max_it)
    
    # Store worst cost for selection pressure
    worst_cost = pop[-1]['cost']
    
    # Main loop
    for it in range(max_it):
        costs = np.array([ind['cost'] for ind in pop])
        
        # Crossover
        popc = []
        for k in range(nc // 2):
            # Select parents
            if use_tournament_selection:
                i1 = tournament_selection(pop, tournament_size)
                i2 = tournament_selection(pop, tournament_size)
            else:
                i1 = np.random.randint(0, pop_size)
                i2 = np.random.randint(0, pop_size)
            
            p1 = pop[i1]
            p2 = pop[i2]
            
            # Perform crossover
            y1, y2 = crossover(p1['position'], p2['position'], 
                              p_single_point, p_double_point)
            
            # Create offspring
            offspring1 = {'position': y1, 'cost': obj_func(y1)}
            offspring2 = {'position': y2, 'cost': obj_func(y2)}
            
            popc.extend([offspring1, offspring2])
        
        # Mutation
        popm = []
        for k in range(nm):
            i = np.random.randint(0, pop_size)
            p = pop[i]
            
            y = mutate(p['position'], mu)
            offspring = {'position': y, 'cost': obj_func(y)}
            popm.append(offspring)
        
        # Merge populations
        pop = pop + popc + popm
        
        # Sort population
        pop = sorted(pop, key=lambda x: x['cost'], reverse=False)
        
        # Update worst cost
        worst_cost = max(worst_cost, pop[-1]['cost'])
        
        # Truncation - keep only pop_size individuals
        pop = pop[:pop_size]
        
        # Store best solution
        best_sol = pop[0].copy()
        
        # Store convergence (negate for maximization)
        convergence[it] = -best_sol['cost']
    
    best_solution = best_sol['position']
    best_obj = -best_sol['cost']
    
    return best_solution, convergence, best_obj
