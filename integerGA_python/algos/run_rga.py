"""
Real-Coded Genetic Algorithm (RGA) implementation.
"""
import numpy as np
from ..utils.operators.sb_crossover import sb_crossover
from ..utils.operators.polynomial_mutation import polynomial_mutation
from ..utils.operators.selection import selection


def run_rga(dim, obj_func, max_fes, pop_size, verbose=True):
    """
    Run Real-Coded Genetic Algorithm for maximization.
    
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
        
    Returns:
    --------
    tuple : (best_solution, convergence, best_obj)
    """
    if verbose:
        print(f"RGA running for maximization, generation = {max_fes//pop_size}, n={pop_size} ...")
    
    # Initialize population
    indivs = np.random.rand(pop_size - 1, dim)
    new_pop = np.ones((1, dim))
    indivs = np.vstack([indivs, new_pop])
    
    best_indiv = np.random.rand(dim)
    best_obj = -1e9
    count_fes = 0
    objective = -1e9 * np.ones(pop_size)
    
    # Evaluate initial population
    for i in range(pop_size):
        objective[i] = obj_func(indivs[i])
        count_fes += 1
        if objective[i] > best_obj:
            best_obj = objective[i]
            best_indiv = indivs[i].copy()
    
    generation = 0
    convergence = np.zeros(round(max_fes / pop_size))
    convergence[generation] = best_obj
    
    while count_fes < max_fes:
        generation += 1
        
        # ============ BEGIN REPRODUCTION ==================
        offspring = np.zeros((pop_size, dim))
        offspring_fitness = np.zeros(pop_size)
        count = 0
        
        while count < pop_size - 1:
            # Select two different parents randomly
            p1 = np.random.randint(0, len(indivs))
            p2 = np.random.randint(0, len(indivs))
            while p1 == p2:
                p2 = np.random.randint(0, len(indivs))
            
            # Crossover with 90% probability
            if np.random.rand() <= 0.9:
                children = sb_crossover(indivs[p1], indivs[p2])
                if count < pop_size:
                    offspring[count] = children[0]
                    count += 1
                if count < pop_size:
                    offspring[count] = children[1]
                    count += 1
        
        # Mutation
        for i in range(len(offspring)):
            if np.random.rand() < 0.01:
                offspring[i] = polynomial_mutation(offspring[i])
            
            offspring_fitness[i] = obj_func(offspring[i])
            count_fes += 1
        
        # ============== END REPRODUCTION ===================
        
        # Survival selection
        indivs, objective = selection(pop_size, indivs, objective, 
                                      offspring, offspring_fitness)
        
        # Update best solution
        if best_obj < objective[0]:
            best_obj = objective[0]
            best_indiv = indivs[0].copy()
        
        # Print progress
        if verbose and generation % 100 == 0:
            print(f"Generation {generation}, best objective = {best_obj}")
        
        convergence[generation] = best_obj
    
    best_solution = best_indiv
    
    return best_solution, convergence[:generation+1], best_obj
