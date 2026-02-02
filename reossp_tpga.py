"""
REOSSP Two-Phase Genetic Algorithm
Implements a decomposed approach to REOSSP:
  Phase 1 (GA): Optimize orbital slot trajectories for each satellite across stages
  Phase 2 (Greedy): Schedule observations/downlinks/charging given fixed orbits

Chromosome Encoding:
  - Individual = 2D array [K satellites × S stages]
  - Each gene is an orbital slot index (1 to J_sk)
  - Example: [[s1_k1, s2_k1, ...], [s1_k2, s2_k2, ...], ...]

Key Advantages:
  - Smaller search space (K×S genes vs millions of MILP variables)
  - Fast fitness evaluation via greedy scheduling
  - Feasibility-preserving genetic operators
  - Naturally handles propellant budget constraints
"""

import numpy as np
import time
from deap import base, creator, tools, algorithms
import random
from parameters import InstanceParameters
from typing import List, Tuple, Dict


class GreedyScheduler:
    """
    Phase 2: Given fixed orbital trajectories, greedily schedule operations
    to maximize observations and downlinks while respecting battery/data constraints
    """
    
    def __init__(self, params: InstanceParameters):
        self.params = params
        self.T_per_stage = params.T // params.S
    
    def schedule(self, orbital_trajectory: np.ndarray) -> Dict:
        """
        Greedy scheduling given orbital slot assignments
        
        Args:
            orbital_trajectory: [K × S] array of slot assignments
        
        Returns:
            dict with schedule quality metrics and feasibility status
        """
        p = self.params
        K, S = orbital_trajectory.shape
        T_local = self.T_per_stage
        
        # Initialize state variables
        battery = np.zeros((S, K, T_local + 1))  # Battery state
        data = np.zeros((S, K, T_local + 1))      # Data storage state
        
        # Initialize at stage 1, time 1
        for k in range(K):
            slot_s1 = orbital_trajectory[k, 0] - 1  # Convert to 0-indexed
            # Battery starts at max minus cost of initial maneuver
            maneuver_cost = p.maneuver_costs[0, k, 0, slot_s1]
            battery[0, k, 0] = p.B_max - p.B_recon * (1 if maneuver_cost > 0 else 0)
            data[0, k, 0] = 0  # Start with empty storage
        
        # Track schedule decisions
        total_observations = 0
        total_downlinks = 0
        observations = []  # List of (s, k, t, p) tuples
        downlinks = []     # List of (s, k, t, g) tuples
        charges = []       # List of (s, k, t) tuples
        
        # Greedy scheduling loop: stage by stage, satellite by satellite, time by time
        for s in range(S):
            for k in range(K):
                slot = orbital_trajectory[k, s] - 1  # Current slot (0-indexed)
                
                for t in range(T_local):
                    # Get current state
                    b_curr = battery[s, k, t]
                    d_curr = data[s, k, t]
                    
                    # Check visibility for this slot
                    targets_visible = []
                    grounds_visible = []
                    sun_visible = False
                    
                    # Check target visibility
                    for target_idx in range(p.V_target.shape[4]):
                        if p.V_target[s, k, t, slot, target_idx]:
                            targets_visible.append(target_idx)
                    
                    # Check ground station visibility
                    for ground_idx in range(p.V_ground.shape[4]):
                        if p.V_ground[s, k, t, slot, ground_idx]:
                            grounds_visible.append(ground_idx)
                    
                    # Check sun visibility
                    sun_visible = p.V_sun[s, k, t, slot]
                    
                    # Decision priority: 
                    # 1. Downlink if data >= D_comm and ground visible (opportunistic)
                    # 2. Observe if target visible and enough storage
                    # 3. Charge if battery low and sun visible
                    
                    action_taken = False
                    
                    # Priority 1: Downlink if we have enough data (at least D_comm) and ground visible
                    # This is opportunistic - downlink whenever we can to free up storage
                    if len(grounds_visible) > 0 and d_curr >= p.D_comm:
                        if b_curr >= p.B_comm + p.B_time:  # Check battery sufficient
                            # Downlink to first visible ground station
                            ground_idx = grounds_visible[0]
                            downlinks.append((s, k, t, ground_idx))
                            total_downlinks += 1
                            
                            # Update state
                            b_next = b_curr - p.B_comm - p.B_time
                            d_next = max(0, d_curr - p.D_comm)
                            action_taken = True
                    
                    # Priority 2: Observe if target visible
                    if not action_taken and len(targets_visible) > 0:
                        if b_curr >= p.B_obs + p.B_time and d_curr + p.D_obs <= p.D_max:
                            # Observe first visible target (could prioritize by value)
                            target_idx = targets_visible[0]
                            observations.append((s, k, t, target_idx))
                            total_observations += 1
                            
                            # Update state
                            b_next = b_curr - p.B_obs - p.B_time
                            d_next = d_curr + p.D_obs
                            action_taken = True
                    
                    # Priority 3: Charge if battery low
                    if not action_taken and sun_visible and b_curr < 0.5 * p.B_max:
                        if b_curr + p.B_charge <= p.B_max:
                            charges.append((s, k, t))
                            
                            # Update state
                            b_next = min(p.B_max, b_curr + p.B_charge - p.B_time)
                            d_next = d_curr
                            action_taken = True
                    
                    # No action: idle (still consume B_time)
                    if not action_taken:
                        b_next = b_curr - p.B_time
                        d_next = d_curr
                    
                    # Store next state
                    if t < T_local - 1:
                        battery[s, k, t + 1] = b_next
                        data[s, k, t + 1] = d_next
                    else:
                        # Last timestep of stage - transition to next stage
                        if s < S - 1:
                            # Calculate maneuver cost for next stage
                            slot_next = orbital_trajectory[k, s + 1] - 1
                            slot_curr = orbital_trajectory[k, s] - 1
                            maneuver_cost = p.maneuver_costs[s + 1, k, slot_curr, slot_next]
                            recon_cost = p.B_recon if maneuver_cost > 0 else 0
                            
                            battery[s + 1, k, 0] = b_next - recon_cost
                            data[s + 1, k, 0] = d_next
                    
                    # Check feasibility
                    if b_next < 0 or d_next < 0 or d_next > p.D_max:
                        # Infeasible schedule
                        return {
                            'feasible': False,
                            'total_observations': total_observations,
                            'total_downlinks': total_downlinks,
                            'objective': 0,
                            'reason': f'Infeasible at stage {s+1}, satellite {k+1}, time {t+1}'
                        }
        
        # Calculate objective
        objective = p.C * total_downlinks + total_observations
        
        return {
            'feasible': True,
            'total_observations': total_observations,
            'total_downlinks': total_downlinks,
            'objective': objective,
            'observations': observations,
            'downlinks': downlinks,
            'charges': charges,
            'battery_final': battery[:, :, -1].copy(),
            'data_final': data[:, :, -1].copy()
        }


class REOSSPTwoPhaseGA:
    """
    Two-Phase Genetic Algorithm for REOSSP
    Phase 1: GA optimizes orbital trajectories
    Phase 2: Greedy scheduler evaluates fitness
    """
    
    def __init__(self, params: InstanceParameters):
        self.params = params
        self.scheduler = GreedyScheduler(params)
        
        # GA parameters
        self.pop_size = 100
        self.n_generations = 50
        self.cx_prob = 0.7  # Crossover probability
        self.mut_prob = 0.2  # Mutation probability
        self.tournament_size = 3
        
        # Statistics
        self.best_individual = None
        self.best_fitness = 0
        self.fitness_history = []
        
        # Initialize DEAP
        self._setup_deap()
    
    def _setup_deap(self):
        """Initialize DEAP framework"""
        # Clear any existing definitions
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
        
        # Define fitness (maximize) and individual
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        
        # Attribute generator: random slot index [1, J_sk]
        self.toolbox.register("attr_slot", random.randint, 1, self.params.J_sk)
        
        # Individual generator: K × S genes (one trajectory per satellite)
        def create_individual():
            # Create [K × S] trajectory matrix, flatten to list
            trajectory = [[self.toolbox.attr_slot() for _ in range(self.params.S)] 
                         for _ in range(self.params.K)]
            return creator.Individual(trajectory)
        
        self.toolbox.register("individual", create_individual)
        
        # Population generator
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Genetic operators
        self.toolbox.register("evaluate", self.evaluate_fitness)
        self.toolbox.register("mate", self.crossover)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
    
    def evaluate_fitness(self, individual: List[List[int]]) -> Tuple[float,]:
        """
        Evaluate fitness of an individual
        
        Args:
            individual: List of K trajectories, each with S slot assignments
        
        Returns:
            Tuple with single fitness value (for DEAP compatibility)
        """
        # Convert to numpy array [K × S]
        trajectory = np.array(individual, dtype=int)
        
        # Check propellant budget feasibility first (fast check)
        if not self.is_propellant_feasible(trajectory):
            return (0.0,)  # Infeasible - zero fitness
        
        # Check if any slots are unavailable
        if not self.check_slot_availability(trajectory):
            return (0.0,)  # Infeasible - using unavailable slots
        
        # Phase 2: Greedy scheduling
        schedule_result = self.scheduler.schedule(trajectory)
        
        if not schedule_result['feasible']:
            return (0.0,)  # Infeasible schedule
        
        return (schedule_result['objective'],)
    
    def is_propellant_feasible(self, trajectory: np.ndarray) -> bool:
        """
        Check if trajectory satisfies propellant budget for each satellite
        
        Args:
            trajectory: [K × S] array of slot assignments
        
        Returns:
            True if all satellites within budget
        """
        K, S = trajectory.shape
        
        for k in range(K):
            total_cost = 0
            
            # Stage 1: maneuver from initial slot (slot 1 = index 0)
            slot_s1 = trajectory[k, 0] - 1  # Convert to 0-indexed
            total_cost += self.params.maneuver_costs[0, k, 0, slot_s1]
            
            # Subsequent stages: maneuver from previous slot
            for s in range(1, S):
                slot_prev = trajectory[k, s-1] - 1
                slot_curr = trajectory[k, s] - 1
                total_cost += self.params.maneuver_costs[s, k, slot_prev, slot_curr]
            
            if total_cost > self.params.c_max:
                return False
        
        return True
    
    def check_slot_availability(self, trajectory: np.ndarray) -> bool:
        """
        Check if all assigned slots are available (not in unavailable_slots)
        
        Args:
            trajectory: [K × S] array of slot assignments
        
        Returns:
            True if all slots are available
        """
        K, S = trajectory.shape
        
        for k in range(K):
            for s in range(S):
                slot = trajectory[k, s] - 1  # Convert to 0-indexed
                if self.params.unavailable_slots[s, slot]:
                    return False  # This slot is unavailable
        
        return True
    
    def crossover(self, ind1: List, ind2: List) -> Tuple[List, List]:
        """
        Custom crossover: swap entire satellite trajectories
        Preserves per-satellite propellant budget constraints
        
        Args:
            ind1, ind2: Parent individuals
        
        Returns:
            Two offspring individuals
        """
        K = len(ind1)
        
        # Randomly decide which satellites to swap
        for k in range(K):
            if random.random() < 0.5:
                # Swap entire trajectory for satellite k
                ind1[k], ind2[k] = ind2[k][:], ind1[k][:]
        
        return ind1, ind2
    
    def mutate(self, individual: List) -> Tuple[List,]:
        """
        Custom mutation: change one satellite's slot in one stage
        Check propellant budget after mutation
        
        Args:
            individual: Individual to mutate
        
        Returns:
            Mutated individual (tuple for DEAP compatibility)
        """
        K = len(individual)
        S = len(individual[0])
        
        # Select random satellite and stage
        k = random.randint(0, K - 1)
        s = random.randint(0, S - 1)
        
        # Try to find a valid slot (max 10 attempts)
        max_attempts = 10
        for _ in range(max_attempts):
            # Generate new slot
            new_slot = random.randint(1, self.params.J_sk)
            
            # Check if this slot is available
            if self.params.unavailable_slots[s, new_slot - 1]:
                continue  # Slot unavailable, try again
            
            # Save old slot
            old_slot = individual[k][s]
            
            # Apply mutation
            individual[k][s] = new_slot
            
            # Check if still feasible (propellant budget)
            trajectory = np.array(individual, dtype=int)
            if self.is_propellant_feasible(trajectory):
                # Valid mutation
                return (individual,)
            else:
                # Revert mutation
                individual[k][s] = old_slot
        
        # Could not find valid mutation, return unchanged
        return (individual,)
    
    def initialize_population(self) -> List:
        """
        Initialize population with mix of strategies
        
        Returns:
            Initial population
        """
        population = []
        
        # Strategy 1: Static constellation (baseline - all satellites stay in initial positions)
        static_ind = [[1] * self.params.S for _ in range(self.params.K)]
        population.append(creator.Individual(static_ind))
        
        # Strategy 2: Each satellite takes a different fixed slot
        if self.params.K <= self.params.J_sk:
            diverse_ind = [[k + 1] * self.params.S for k in range(self.params.K)]
            population.append(creator.Individual(diverse_ind))
        
        # Strategy 3: Random feasible individuals
        max_attempts = self.pop_size * 10
        attempts = 0
        while len(population) < self.pop_size and attempts < max_attempts:
            attempts += 1
            
            # Create random individual
            ind = self.toolbox.individual()
            trajectory = np.array(ind, dtype=int)
            
            # Check feasibility
            if self.is_propellant_feasible(trajectory) and self.check_slot_availability(trajectory):
                population.append(ind)
        
        # Fill remaining with completely random (may be infeasible)
        while len(population) < self.pop_size:
            population.append(self.toolbox.individual())
        
        return population
    
    def solve(self, pop_size=100, n_generations=50, verbose=True):
        """
        Run the genetic algorithm
        
        Args:
            pop_size: Population size
            n_generations: Number of generations
            verbose: Print progress
        
        Returns:
            dict with results
        """
        self.pop_size = pop_size
        self.n_generations = n_generations
        
        start_time = time.time()
        
        # Initialize population
        if verbose:
            print(f"Initializing population of size {pop_size}...")
        population = self.initialize_population()
        
        # Evaluate initial population
        if verbose:
            print(f"Evaluating initial population...")
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Track statistics
        self.fitness_history = []
        
        # Evolution loop
        if verbose:
            print(f"\nStarting evolution for {n_generations} generations...")
        
        for gen in range(n_generations):
            # Select next generation
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cx_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Apply mutation
            for mutant in offspring:
                if random.random() < self.mut_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate individuals with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace population
            population[:] = offspring
            
            # Track best fitness
            fits = [ind.fitness.values[0] for ind in population]
            best_fit = max(fits)
            avg_fit = np.mean(fits)
            self.fitness_history.append({'gen': gen, 'best': best_fit, 'avg': avg_fit})
            
            if verbose and (gen % 10 == 0 or gen == n_generations - 1):
                print(f"Gen {gen:3d}: Best={best_fit:8.2f}, Avg={avg_fit:8.2f}")
        
        runtime = time.time() - start_time
        
        # Extract best solution
        best_ind = tools.selBest(population, 1)[0]
        best_trajectory = np.array(best_ind, dtype=int)
        best_schedule = self.scheduler.schedule(best_trajectory)
        
        self.best_individual = best_ind
        self.best_fitness = best_schedule['objective']
        
        # Calculate propellant used
        propellant_used = self._calculate_propellant(best_trajectory)
        
        results = {
            'status': 'completed',
            'objective': best_schedule['objective'],
            'runtime_minutes': runtime / 60,
            'total_observations': best_schedule['total_observations'],
            'total_downlinks': best_schedule['total_downlinks'],
            'data_downlinked_gb': best_schedule['total_downlinks'] * self.params.D_comm / 1024,
            'propellant_used': propellant_used,
            'best_trajectory': best_trajectory,
            'fitness_history': self.fitness_history,
            'final_population_size': len(population),
            'generations_completed': n_generations
        }
        
        return results
    
    def _calculate_propellant(self, trajectory: np.ndarray) -> float:
        """Calculate total propellant used by all satellites"""
        K, S = trajectory.shape
        total_cost = 0
        
        for k in range(K):
            # Stage 1: maneuver from initial slot
            slot_s1 = trajectory[k, 0] - 1
            total_cost += self.params.maneuver_costs[0, k, 0, slot_s1]
            
            # Subsequent stages
            for s in range(1, S):
                slot_prev = trajectory[k, s-1] - 1
                slot_curr = trajectory[k, s] - 1
                total_cost += self.params.maneuver_costs[s, k, slot_prev, slot_curr]
        
        return total_cost


if __name__ == "__main__":
    """Test the two-phase GA solver"""
    print("="*80)
    print("TESTING REOSSP TWO-PHASE GENETIC ALGORITHM")
    print("="*80)
    
    # Create test instance
    test_params = InstanceParameters(
        instance_id=999,
        S=8,
        K=5,
        J_sk=20,
        T=36*24*2
    )
    
    print(f"\nProblem size:")
    print(f"  Satellites (K): {test_params.K}")
    print(f"  Stages (S): {test_params.S}")
    print(f"  Slots per stage (J_sk): {test_params.J_sk}")
    print(f"  Total timesteps (T): {test_params.T}")
    print(f"  Timesteps per stage: {test_params.T // test_params.S}")
    print(f"  Chromosome size: {test_params.K} × {test_params.S} = {test_params.K * test_params.S} genes")
    
    # Run GA
    solver = REOSSPTwoPhaseGA(test_params)
    
    print(f"\nGA Parameters:")
    print(f"  Population size: {solver.pop_size}")
    print(f"  Generations: {solver.n_generations}")
    print(f"  Crossover prob: {solver.cx_prob}")
    print(f"  Mutation prob: {solver.mut_prob}")
    print(f"  Tournament size: {solver.tournament_size}")
    
    results = solver.solve(pop_size=50, n_generations=30, verbose=True)
    
    print(f"\n" + "="*80)
    print(f"RESULTS:")
    print(f"="*80)
    print(f"Status: {results['status']}")
    print(f"Objective: {results['objective']:.2f}")
    print(f"Runtime: {results['runtime_minutes']:.2f} minutes")
    print(f"Total observations: {results['total_observations']}")
    print(f"Total downlinks: {results['total_downlinks']}")
    print(f"Data downlinked: {results['data_downlinked_gb']:.4f} GB")
    print(f"Propellant used: {results['propellant_used']:.2f} m/s")
    print(f"Generations: {results['generations_completed']}")
    
    print(f"\n" + "="*80)
    print(f"BEST TRAJECTORY:")
    print(f"="*80)
    best_traj = results['best_trajectory']
    for k in range(test_params.K):
        traj_str = " → ".join([f"S{s+1}:slot{best_traj[k,s]}" for s in range(test_params.S)])
        print(f"Satellite {k+1}: {traj_str}")
    
    print(f"\n" + "="*80)
    print(f"FITNESS EVOLUTION:")
    print(f"="*80)
    for i, record in enumerate(results['fitness_history']):
        if i % 5 == 0 or i == len(results['fitness_history']) - 1:
            print(f"Gen {record['gen']:3d}: Best={record['best']:8.2f}, Avg={record['avg']:8.2f}")
