"""
Improved Differential Evolution (IDE) implementation.
"""
import numpy as np
from ..utils.cauchy import cauchy
from ..utils.operators.de_operators import (
    operate_pbest_1_bin, 
    operate_current_to_pbest_1_bin
)


def run_ide(dim, obj_func, max_fes, pop_size, pbest_rate=0.1, 
            mem_size=5, arc_rate=2.6, verbose=True):
    """
    Run Improved Differential Evolution for maximization.
    
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
    pbest_rate : float
        Rate for selecting from top individuals
    mem_size : int
        Memory size for parameter adaptation
    arc_rate : float
        Archive rate
    verbose : bool
        Whether to print progress
        
    Returns:
    --------
    tuple : (best_solution, convergence, best_objective)
    """
    if verbose:
        print(f"IDE running for maximization, generation = {max_fes//pop_size}, n={pop_size} ...")
    
    count_fes = 0
    convergence = np.zeros(round(max_fes / pop_size))
    
    # Initialize population
    pop = np.random.rand(pop_size - 1, dim)
    new_pop = np.ones((1, dim))
    pop = np.vstack([pop, new_pop])
    
    fitness = -1e9 * np.ones(pop_size)
    best_solution = np.random.rand(dim)
    best_objective = -1e9
    
    # Evaluate initial population
    for i in range(pop_size):
        fitness[i] = obj_func(pop[i])
        count_fes += 1
        if fitness[i] > best_objective:
            best_objective = fitness[i]
            best_solution = pop[i].copy()
    
    # Parameter sampling setup
    M = 2  # Number of mutation operators
    success_sf = np.zeros((M, pop_size))
    success_cr = np.zeros((M, pop_size))
    dif_fitness = np.zeros((M, pop_size))
    
    mem_sf = 0.2 * np.ones((M, mem_size))
    mem_cr = 1.0 * np.ones((M, mem_size))
    mem_pos = np.ones(M, dtype=int)
    
    # Best operator tracking
    best_op = 0
    lambda_val = 0.2
    consumed_fes = np.ones(M)
    sum_improv = np.zeros(M)
    
    # Archive
    arc_size = round(pop_size * arc_rate)
    archive = np.zeros((arc_size, dim))
    arc_count = 0
    
    generation = 0
    if verbose:
        print(f"Generation {generation}, best objective = {best_objective}")
    convergence[generation] = best_objective
    
    offspring = np.zeros((pop_size, dim))
    offspring_fit = -1e9 * np.ones(pop_size)
    
    p_num = max(round(pbest_rate * pop_size), 2)
    
    while count_fes < max_fes:
        generation += 1
        
        # Sort for current-to-pbest
        sorted_idx = np.argsort(fitness)[::-1]  # Descending order
        
        # Reproduction
        next_pop = np.zeros_like(pop)
        next_pop_fitness = np.zeros_like(fitness)
        success_count = np.zeros(M, dtype=int)
        
        for i in range(pop_size):
            p_best_idx = sorted_idx[np.random.randint(0, p_num)]
            
            # Select operator
            rand_op = np.random.rand()
            opcode = -1
            for j in range(M):
                if j * lambda_val <= rand_op < (j + 1) * lambda_val:
                    opcode = j
                    break
            if opcode == -1:
                opcode = best_op
            
            consumed_fes[opcode] += 1
            
            # Sample F and CR
            rand_pos = np.random.randint(0, mem_size)
            mu_cr = mem_cr[opcode, rand_pos]
            mu_f = mem_sf[opcode, rand_pos]
            
            if mu_cr == -1:
                CR = 0
            else:
                CR = np.random.normal(mu_cr, 0.1)
                CR = np.clip(CR, 0, 1)
            
            F = cauchy(mu_f, 0.1)
            while F <= 0:
                F = cauchy(mu_f, 0.1)
            F = min(F, 1.0)
            
            # Apply mutation operator
            if opcode == 0:
                offspring[i] = operate_pbest_1_bin(pop, i, p_best_idx, CR, F, 
                                                   archive, arc_count)
            else:  # opcode == 1
                offspring[i] = operate_current_to_pbest_1_bin(pop, i, p_best_idx, CR, F,
                                                              archive, arc_count)
            
            offspring_fit[i] = obj_func(offspring[i])
            
            # Selection
            if fitness[i] <= offspring_fit[i]:
                next_pop[i] = offspring[i]
                next_pop_fitness[i] = offspring_fit[i]
                
                if offspring_fit[i] > best_objective:
                    best_objective = offspring_fit[i]
                    best_solution = offspring[i].copy()
                
                if fitness[i] < offspring_fit[i]:
                    pos = success_count[opcode]
                    success_count[opcode] += 1
                    success_sf[opcode, pos] = F
                    success_cr[opcode, pos] = CR
                    dif_fitness[opcode, pos] = offspring_fit[i] - fitness[i]
                    sum_improv[opcode] += offspring_fit[i] - fitness[i]
                
                # Update archive
                if arc_count < arc_size:
                    archive[arc_count] = pop[i]
                    arc_count += 1
                else:
                    rm_pos = np.random.randint(0, arc_size)
                    archive[rm_pos] = pop[i]
            else:
                next_pop[i] = pop[i]
                next_pop_fitness[i] = fitness[i]
        
        count_fes += pop_size
        convergence[generation] = best_objective
        
        pop = next_pop
        fitness = next_pop_fitness
        
        # Update parameter memory
        for m in range(M):
            sc = success_count[m]
            if sc > 0:
                sum_improvement = np.sum(dif_fitness[m, :sc])
                weight = dif_fitness[m, :sc] / sum_improvement
                
                mem_sf[m, mem_pos[m]] = (np.sum(weight * (success_sf[m, :sc] ** 2)) / 
                                         np.sum(weight * success_sf[m, :sc]))
                
                mem_cr[m, mem_pos[m]] = np.sum(weight * (success_cr[m, :sc] ** 2))
                tmp = np.sum(weight * success_cr[m, :sc])
                if tmp == 0 or mem_cr[m, mem_pos[m]] == -1:
                    mem_cr[m, mem_pos[m]] = -1
                else:
                    mem_cr[m, mem_pos[m]] = mem_cr[m, mem_pos[m]] / tmp
                
                mem_pos[m] += 1
                if mem_pos[m] == mem_size:
                    mem_pos[m] = 0
        
        # Update best operator
        if generation % 10 == 0:
            new_best_op = -1
            best_improve_rate = 0
            for m in range(M):
                improve_rate = sum_improv[m] / consumed_fes[m]
                if improve_rate > best_improve_rate:
                    best_improve_rate = improve_rate
                    new_best_op = m
            
            if new_best_op == -1:
                best_op = np.random.randint(0, M)
            else:
                best_op = new_best_op
            
            sum_improv = np.zeros(M)
            consumed_fes = np.ones(M)
        
        if verbose and generation % 100 == 0:
            print(f"Generation {generation}, best objective = {best_objective}")
    
    return best_solution, convergence[:generation+1], best_objective
