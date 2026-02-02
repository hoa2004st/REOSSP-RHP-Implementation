"""
Particle Swarm Optimization (PSO) implementation.
"""
import numpy as np


def run_pso(dim, obj_func, max_fes, pop_size, w=0.7, c1=1.5, c2=1.5, verbose=True):
    """
    Run Particle Swarm Optimization for maximization.
    
    Parameters:
    -----------
    dim : int
        Problem dimension
    obj_func : callable
        Objective function to maximize
    max_fes : int
        Maximum function evaluations
    pop_size : int
        Population size (swarm size)
    w : float
        Inertia weight
    c1 : float
        Cognitive coefficient
    c2 : float
        Social coefficient
    verbose : bool
        Whether to print progress
        
    Returns:
    --------
    tuple : (best_solution, convergence, best_obj)
    """
    iters = round(max_fes / pop_size)
    
    if verbose:
        print(f"PSO running for maximization, generation = {iters}, n={pop_size} ...")
    
    # Initialize swarm
    swarm = []
    for i in range(pop_size):
        particle = {
            'gene': np.random.rand(dim),
            'velocity': np.zeros(dim),
            'fitness': -np.inf,
            'best_gene': None,
            'best_fitness': -np.inf
        }
        
        # Evaluate fitness
        particle['fitness'] = obj_func(particle['gene'])
        
        # Initialize personal best
        particle['best_gene'] = particle['gene'].copy()
        particle['best_fitness'] = particle['fitness']
        
        swarm.append(particle)
    
    # Initialize global best
    global_best = {
        'gene': None,
        'fitness': -np.inf
    }
    
    # Find initial global best
    for particle in swarm:
        if particle['fitness'] > global_best['fitness']:
            global_best['fitness'] = particle['fitness']
            global_best['gene'] = particle['gene'].copy()
    
    convergence = np.zeros(iters)
    convergence[0] = global_best['fitness']
    
    # Main PSO loop
    for iter_num in range(1, iters):
        for i in range(pop_size):
            # Generate random coefficients
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            
            # Update velocity
            # v = w*v + c1*r1*(pBest - x) + c2*r2*(gBest - x)
            swarm[i]['velocity'] = (
                w * swarm[i]['velocity'] +
                c1 * r1 * (swarm[i]['best_gene'] - swarm[i]['gene']) +
                c2 * r2 * (global_best['gene'] - swarm[i]['gene'])
            )
            
            # Update position
            swarm[i]['gene'] = swarm[i]['gene'] + swarm[i]['velocity']
            
            # Clamp position to [0, 1]
            swarm[i]['gene'] = np.clip(swarm[i]['gene'], 0, 1)
            
            # Evaluation
            swarm[i]['fitness'] = obj_func(swarm[i]['gene'])
            
            # Update personal best
            if swarm[i]['fitness'] > swarm[i]['best_fitness']:
                swarm[i]['best_gene'] = swarm[i]['gene'].copy()
                swarm[i]['best_fitness'] = swarm[i]['fitness']
            
            # Update global best
            if swarm[i]['fitness'] > global_best['fitness']:
                global_best['fitness'] = swarm[i]['fitness']
                global_best['gene'] = swarm[i]['gene'].copy()
        
        convergence[iter_num] = global_best['fitness']
        
        # Print progress
        if verbose and iter_num % 50 == 0:
            print(f"Generation {iter_num}, best objective = {global_best['fitness']:.4f}")
    
    best_solution = global_best['gene']
    best_obj = global_best['fitness']
    
    return best_solution, convergence, best_obj
