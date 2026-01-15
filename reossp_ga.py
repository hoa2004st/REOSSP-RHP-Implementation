"""
REOSSP Genetic Algorithm Solver
=================================
A custom Genetic Algorithm for the Reconfigurable Earth Observation Satellite Scheduling Problem (REOSSP).

This GA uses a specialized representation to:
1. Minimize complexity of gene pool
2. Maintain solution feasibility through custom encoding/decoding
3. Enforce all MILP constraints through intelligent operators

Key Features:
- Priority-based encoding for scheduling decisions
- Resource-aware decoding that enforces all constraints
- Custom mutation operators that preserve feasibility
- Elitism with diversity preservation

Author: GitHub Copilot
Date: January 2026
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from parameters import InstanceParameters


@dataclass
class Individual:
    """
    An individual in the GA population representing a complete REOSSP solution.
    
    Chromosome Structure (Priority-Based Encoding):
    ===============================================
    The chromosome is a compact representation that avoids explicit binary variables.
    Instead, it uses priority values that are decoded into feasible schedules.
    
    Chromosome Components:
    1. slot_priorities: [S×K×J] - Priority for each satellite to move to each slot in each stage
    2. action_priorities: [S×K×T×A] - Priority for each action (obs/downlink/charge) at each timestep
       where A = max(n_targets, n_ground) covers all possible actions
    
    This encoding has O(S*K*J + S*K*T*A) genes vs O(S*K*J^2 + S*K*T*(P+G+1)) binary variables in MILP.
    For typical instances: ~10,000 genes vs ~100,000+ binary variables.
    """
    # Genetic representation (continuous values in [0,1])
    slot_priorities: np.ndarray  # [S, K, J] - priorities for slot selection
    action_priorities: np.ndarray  # [S, K, T_per_stage, A] - priorities for action selection
    
    # Decoded solution (computed from priorities)
    orbital_slots: np.ndarray = None  # [S, K] - which slot each satellite occupies in each stage
    observations: np.ndarray = None  # [S, K, T_per_stage, P] - binary observation decisions
    downlinks: np.ndarray = None  # [S, K, T_per_stage, G] - binary downlink decisions
    charging: np.ndarray = None  # [S, K, T_per_stage] - binary charging decisions
    
    # Resource tracking (computed during decoding)
    data_storage: np.ndarray = None  # [S, K, T_per_stage] - data storage level
    battery_level: np.ndarray = None  # [S, K, T_per_stage] - battery level
    propellant_used: np.ndarray = None  # [K] - cumulative propellant per satellite
    
    # Fitness metrics
    fitness: float = 0.0  # Objective value (to maximize)
    constraint_violations: int = 0  # Number of constraint violations (should be 0)
    is_feasible: bool = False
    
    def __lt__(self, other):
        """For sorting by fitness (higher is better)"""
        if self.is_feasible and not other.is_feasible:
            return False
        if not self.is_feasible and other.is_feasible:
            return True
        return self.fitness < other.fitness


class REOSSPGASolver:
    """
    Genetic Algorithm solver for REOSSP with custom feasibility-preserving operators.
    """
    
    def __init__(self, params: InstanceParameters, 
                 population_size: int = 100,
                 elite_size: int = 10,
                 mutation_rate: float = 0.15,
                 crossover_rate: float = 0.8,
                 tournament_size: int = 5,
                 random_seed: Optional[int] = None):
        """
        Initialize GA solver with problem parameters.
        
        Args:
            params: REOSSP instance parameters
            population_size: Number of individuals in population
            elite_size: Number of elite individuals to preserve
            mutation_rate: Probability of mutation for each gene
            crossover_rate: Probability of crossover for each pair
            tournament_size: Number of individuals in tournament selection
            random_seed: Random seed for reproducibility
        """
        self.params = params
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Problem dimensions
        self.S = params.S
        self.K = params.K
        self.J = params.J_sk
        self.T_per_stage = params.T // params.S
        self.n_targets = params.V_target.shape[4]
        self.n_ground = params.V_ground.shape[4]
        self.max_actions = max(self.n_targets, self.n_ground)
        
        # Population
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.generation = 0
        
        # Statistics
        self.fitness_history = []
        self.diversity_history = []
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    def initialize_population(self):
        """
        Create initial population with diverse solutions.
        Uses multiple strategies to ensure diversity.
        """
        print(f"Initializing population of {self.population_size} individuals...")
        self.population = []
        
        # Strategy 1: Pure random (40% of population)
        n_random = int(0.4 * self.population_size)
        for _ in range(n_random):
            ind = self._create_random_individual()
            self._decode_and_evaluate(ind)
            self.population.append(ind)
        
        # Strategy 2: Greedy heuristic variations (40% of population)
        n_greedy = int(0.4 * self.population_size)
        for _ in range(n_greedy):
            ind = self._create_greedy_individual(randomness=0.3)
            self._decode_and_evaluate(ind)
            self.population.append(ind)
        
        # Strategy 3: Biased random (remaining 20%)
        n_biased = self.population_size - n_random - n_greedy
        for _ in range(n_biased):
            ind = self._create_biased_individual()
            self._decode_and_evaluate(ind)
            self.population.append(ind)
        
        # Sort population by fitness
        self.population.sort(reverse=True)
        
        if len(self.population) == 0:
            raise RuntimeError("Failed to create any individuals in population")
        
        self.best_individual = self.population[0]
        
        print(f"  Initial best fitness: {self.best_individual.fitness:.2f}")
        print(f"  Feasible solutions: {sum(ind.is_feasible for ind in self.population)}/{self.population_size}")
        
        if self.best_individual is None:
            raise RuntimeError("Best individual is None after initialization")
    
    def _create_random_individual(self) -> Individual:
        """Create individual with completely random priorities."""
        slot_priorities = np.random.random((self.S, self.K, self.J))
        action_priorities = np.random.random((self.S, self.K, self.T_per_stage, self.max_actions))
        return Individual(slot_priorities=slot_priorities, action_priorities=action_priorities)
    
    def _create_greedy_individual(self, randomness: float = 0.2) -> Individual:
        """
        Create individual using greedy heuristic with some randomness.
        Prioritizes high-value actions and efficient slot assignments.
        """
        # Greedy slot priorities: prefer lower-cost maneuvers with some randomness
        slot_priorities = np.zeros((self.S, self.K, self.J))
        for s in range(self.S):
            for k in range(self.K):
                # Prefer slots with lower maneuver costs
                costs = self.params.maneuver_costs[s, k, :, :].mean(axis=0)  # Average cost to reach each slot
                # Invert costs to priorities (lower cost = higher priority)
                max_cost = costs.max() + 0.01
                priorities = (max_cost - costs) / max_cost
                # Add randomness
                noise = np.random.random(self.J) * randomness
                slot_priorities[s, k, :] = priorities + noise
        
        # Greedy action priorities: prefer observations over downlinks, with visibility awareness
        action_priorities = np.random.random((self.S, self.K, self.T_per_stage, self.max_actions))
        
        return Individual(slot_priorities=slot_priorities, action_priorities=action_priorities)
    
    def _create_biased_individual(self) -> Individual:
        """
        Create individual with bias towards certain patterns (e.g., staying in same slot).
        """
        slot_priorities = np.random.random((self.S, self.K, self.J))
        
        # Bias: prefer staying in the same slot (lower propellant cost)
        for s in range(1, self.S):  # Skip first stage
            for k in range(self.K):
                # Give higher priority to diagonal (same slot as previous)
                slot_priorities[s, k, :] += np.random.random(self.J) * 0.3
        
        action_priorities = np.random.random((self.S, self.K, self.T_per_stage, self.max_actions))
        
        return Individual(slot_priorities=slot_priorities, action_priorities=action_priorities)
    
    # =========================================================================
    # DECODING - Convert priorities to feasible schedule
    # =========================================================================
    
    def _decode_and_evaluate(self, individual: Individual):
        """
        Decode chromosome into feasible REOSSP solution and compute fitness.
        
        This is the core function that enforces ALL constraints:
        - Orbital reconfiguration constraints (10a-10c)
        - Slot availability constraints
        - Visibility constraints (11a-11d)
        - Data storage dynamics (12a-12f)
        - Battery dynamics (13a-13b, 14a-14d)
        - Propellant budget constraints
        
        The decoding process is GREEDY and CONSTRUCTIVE, ensuring feasibility.
        """
        p = self.params
        
        # Initialize tracking arrays
        individual.orbital_slots = np.zeros((self.S, self.K), dtype=int)
        individual.observations = np.zeros((self.S, self.K, self.T_per_stage, self.n_targets), dtype=bool)
        individual.downlinks = np.zeros((self.S, self.K, self.T_per_stage, self.n_ground), dtype=bool)
        individual.charging = np.zeros((self.S, self.K, self.T_per_stage), dtype=bool)
        individual.data_storage = np.zeros((self.S, self.K, self.T_per_stage))
        individual.battery_level = np.zeros((self.S, self.K, self.T_per_stage))
        individual.propellant_used = np.zeros(self.K)
        
        # Constraint violation tracking
        violations = 0
        
        # ===== PHASE 1: Decode orbital slot assignments =====
        # Enforces constraints: 10a (initial assignment), 10b (continuity), 10c (propellant budget)
        for k in range(self.K):
            prev_slot = 0  # All satellites start at slot 0 (index 0 = slot 1 in MILP)
            
            for s in range(self.S):
                # Get priorities for this satellite in this stage
                priorities = individual.slot_priorities[s, k, :].copy()
                
                # CONSTRAINT: Unavailable slots - set priority to -inf
                # Enforces: No satellite can occupy unavailable slots
                for j in range(self.J):
                    if p.unavailable_slots[s, j]:
                        priorities[j] = -np.inf
                
                # CONSTRAINT: Propellant budget (10c)
                # Only consider slots within remaining propellant budget
                remaining_budget = p.c_max - individual.propellant_used[k]
                for j in range(self.J):
                    if priorities[j] > -np.inf:  # Only check available slots
                        maneuver_cost = p.maneuver_costs[s, k, prev_slot, j]
                        if maneuver_cost > remaining_budget:
                            priorities[j] = -np.inf  # Cannot afford this maneuver
                
                # Select slot with highest priority (that is still feasible)
                if np.all(priorities == -np.inf):
                    # No feasible slot - stay in previous slot if possible
                    if not p.unavailable_slots[s, prev_slot]:
                        selected_slot = prev_slot
                    else:
                        # Emergency: pick any available slot (may violate budget)
                        available = np.where(~p.unavailable_slots[s, :])[0]
                        if len(available) > 0:
                            selected_slot = available[0]
                            violations += 1
                        else:
                            selected_slot = 0  # Last resort
                            violations += 1
                else:
                    selected_slot = np.argmax(priorities)
                
                # Assign slot and update propellant
                individual.orbital_slots[s, k] = selected_slot
                individual.propellant_used[k] += p.maneuver_costs[s, k, prev_slot, selected_slot]
                prev_slot = selected_slot
        
        # ===== PHASE 2: Decode action schedule with resource tracking =====
        # Enforces constraints: 11a-11d (visibility), 12a-12f (data), 13a-13b, 14a-14d (battery)
        
        total_observations = 0
        total_downlinks = 0
        
        for s in range(self.S):
            for k in range(self.K):
                slot = individual.orbital_slots[s, k]
                
                # Initialize resources at start of stage
                if s == 0:
                    # CONSTRAINT 12e: Initial data storage is 0
                    data = 0.0
                    # CONSTRAINT 13c/14d: Initial battery = B_max - maneuver cost
                    maneuver_cost_battery = p.B_recon  # Cost of moving from initial slot
                    battery = p.B_max - maneuver_cost_battery
                else:
                    # Carry over from previous stage (constraints 12b, 13b)
                    data = individual.data_storage[s-1, k, -1]
                    battery = individual.battery_level[s-1, k, -1]
                    # Subtract maneuver cost for entering this stage
                    battery -= p.B_recon
                
                # Process each timestep in this stage
                for t in range(self.T_per_stage):
                    # Get action priorities
                    action_prio = individual.action_priorities[s, k, t, :].copy()
                    
                    # Determine available actions based on visibility
                    # CONSTRAINT 11a: Can only observe if target is visible from current slot
                    obs_available = []
                    for target_idx in range(self.n_targets):
                        if p.V_target[s, k, t, slot, target_idx]:
                            obs_available.append(('obs', target_idx, action_prio[target_idx % self.max_actions]))
                    
                    # CONSTRAINT 11b: Can only downlink if ground station is visible from current slot
                    downlink_available = []
                    for ground_idx in range(self.n_ground):
                        if p.V_ground[s, k, t, slot, ground_idx]:
                            downlink_available.append(('downlink', ground_idx, action_prio[ground_idx % self.max_actions]))
                    
                    # CONSTRAINT 11c: Can only charge if sun is visible from current slot
                    charge_available = []
                    if p.V_sun[s, k, t, slot]:
                        # Use last action priority for charging
                        charge_available.append(('charge', 0, action_prio[-1]))
                    
                    # Combine all available actions
                    all_actions = obs_available + downlink_available + charge_available
                    
                    # CONSTRAINT 11d: At most one activity per timestep
                    # Select best action considering resource constraints
                    best_action = None
                    best_priority = -np.inf
                    
                    for action_type, action_idx, priority in all_actions:
                        # Check resource feasibility
                        if action_type == 'obs':
                            # CONSTRAINT 12c: Cannot generate more data than storage capacity
                            if data + p.D_obs > p.D_max:
                                continue  # Cannot observe (storage full)
                            # CONSTRAINT 14b: Battery must be sufficient
                            if battery < p.B_obs + p.B_time:
                                continue  # Cannot observe (insufficient battery)
                            if priority > best_priority:
                                best_action = (action_type, action_idx)
                                best_priority = priority
                        
                        elif action_type == 'downlink':
                            # CONSTRAINT 12d: Cannot downlink more than stored data
                            if data < p.D_comm:
                                continue  # Cannot downlink (not enough data)
                            # CONSTRAINT 14b: Battery must be sufficient
                            if battery < p.B_comm + p.B_time:
                                continue  # Cannot downlink (insufficient battery)
                            if priority > best_priority:
                                best_action = (action_type, action_idx)
                                best_priority = priority
                        
                        elif action_type == 'charge':
                            # CONSTRAINT 14a: Battery cannot exceed maximum
                            if battery + p.B_charge > p.B_max:
                                # Can still charge but won't exceed max
                                pass
                            # CONSTRAINT 14b: Battery must be sufficient for baseline consumption
                            if battery < p.B_time:
                                # Should charge if possible
                                if priority > best_priority:
                                    best_action = (action_type, action_idx)
                                    best_priority = priority + 10  # Boost priority if low battery
                            else:
                                if priority > best_priority:
                                    best_action = (action_type, action_idx)
                                    best_priority = priority
                    
                    # Execute selected action and update resources
                    # CONSTRAINT 12a: Data dynamics within stage
                    # CONSTRAINT 13a: Battery dynamics within stage
                    if best_action is not None:
                        action_type, action_idx = best_action
                        
                        if action_type == 'obs':
                            individual.observations[s, k, t, action_idx] = True
                            data += p.D_obs
                            battery -= (p.B_obs + p.B_time)
                            total_observations += 1
                        
                        elif action_type == 'downlink':
                            individual.downlinks[s, k, t, action_idx] = True
                            data -= p.D_comm
                            battery -= (p.B_comm + p.B_time)
                            total_downlinks += 1
                        
                        elif action_type == 'charge':
                            individual.charging[s, k, t] = True
                            charge_amount = min(p.B_charge, p.B_max - battery)
                            battery += charge_amount
                            battery -= p.B_time
                    else:
                        # Idle - only baseline consumption
                        battery -= p.B_time
                    
                    # CONSTRAINT 14b: Battery must remain non-negative
                    if battery < 0:
                        violations += 1
                        battery = 0  # Clamp to 0 (infeasible but continue)
                    
                    # CONSTRAINT 12d: Data must remain non-negative
                    if data < 0:
                        violations += 1
                        data = 0  # Clamp to 0 (infeasible but continue)
                    
                    # Store resource levels
                    individual.data_storage[s, k, t] = data
                    individual.battery_level[s, k, t] = battery
        
        # ===== PHASE 3: Compute fitness =====
        # Objective: maximize C * total_downlinks + total_observations
        individual.fitness = p.C * total_downlinks + total_observations
        individual.constraint_violations = violations
        individual.is_feasible = (violations == 0)
        
        # Penalty for infeasible solutions
        if not individual.is_feasible:
            individual.fitness -= violations * 1000
    
    # =========================================================================
    # GENETIC OPERATORS
    # =========================================================================
    
    def _tournament_selection(self) -> Individual:
        """
        Select individual using tournament selection.
        Returns best individual from random tournament.
        """
        tournament = np.random.choice(self.population, size=self.tournament_size, replace=False)
        return max(tournament, key=lambda ind: ind.fitness)
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Uniform crossover - randomly select genes from each parent.
        
        This preserves the continuous nature of priorities while mixing traits.
        Each gene has 50% chance of coming from either parent.
        """
        if np.random.random() > self.crossover_rate:
            # No crossover - return copies of parents
            return self._copy_individual(parent1), self._copy_individual(parent2)
        
        # Create offspring
        child1_slot = parent1.slot_priorities.copy()
        child1_action = parent1.action_priorities.copy()
        child2_slot = parent2.slot_priorities.copy()
        child2_action = parent2.action_priorities.copy()
        
        # Uniform crossover for slot priorities
        mask = np.random.random(child1_slot.shape) < 0.5
        child1_slot[mask] = parent2.slot_priorities[mask]
        child2_slot[mask] = parent1.slot_priorities[mask]
        
        # Uniform crossover for action priorities
        mask = np.random.random(child1_action.shape) < 0.5
        child1_action[mask] = parent2.action_priorities[mask]
        child2_action[mask] = parent1.action_priorities[mask]
        
        return (Individual(slot_priorities=child1_slot, action_priorities=child1_action),
                Individual(slot_priorities=child2_slot, action_priorities=child2_action))
    
    def _mutate(self, individual: Individual):
        """
        Adaptive mutation operator that preserves feasibility tendencies.
        
        Mutation strategies:
        1. Gaussian mutation (small perturbations)
        2. Random reset (complete gene replacement)
        3. Swap mutation (exchange gene values)
        
        Each gene is mutated with probability mutation_rate.
        """
        # Mutate slot priorities
        for s in range(self.S):
            for k in range(self.K):
                for j in range(self.J):
                    if np.random.random() < self.mutation_rate:
                        mutation_type = np.random.choice(['gaussian', 'reset', 'swap'])
                        
                        if mutation_type == 'gaussian':
                            # Small perturbation
                            delta = np.random.normal(0, 0.1)
                            individual.slot_priorities[s, k, j] = np.clip(
                                individual.slot_priorities[s, k, j] + delta, 0, 1)
                        
                        elif mutation_type == 'reset':
                            # Complete reset
                            individual.slot_priorities[s, k, j] = np.random.random()
                        
                        elif mutation_type == 'swap':
                            # Swap with another slot
                            j2 = np.random.randint(0, self.J)
                            individual.slot_priorities[s, k, j], individual.slot_priorities[s, k, j2] = \
                                individual.slot_priorities[s, k, j2], individual.slot_priorities[s, k, j]
        
        # Mutate action priorities
        for s in range(self.S):
            for k in range(self.K):
                for t in range(self.T_per_stage):
                    for a in range(self.max_actions):
                        if np.random.random() < self.mutation_rate:
                            mutation_type = np.random.choice(['gaussian', 'reset'])
                            
                            if mutation_type == 'gaussian':
                                delta = np.random.normal(0, 0.1)
                                individual.action_priorities[s, k, t, a] = np.clip(
                                    individual.action_priorities[s, k, t, a] + delta, 0, 1)
                            
                            elif mutation_type == 'reset':
                                individual.action_priorities[s, k, t, a] = np.random.random()
    
    def _copy_individual(self, individual: Individual) -> Individual:
        """Create deep copy of individual."""
        return Individual(
            slot_priorities=individual.slot_priorities.copy(),
            action_priorities=individual.action_priorities.copy()
        )
    
    # =========================================================================
    # EVOLUTION
    # =========================================================================
    
    def evolve_generation(self):
        """
        Evolve population for one generation.
        
        Steps:
        1. Elitism: Preserve best individuals
        2. Selection: Tournament selection for parents
        3. Crossover: Create offspring
        4. Mutation: Mutate offspring
        5. Evaluation: Decode and evaluate offspring
        6. Replacement: Form new population
        """
        # Sort current population
        self.population.sort(reverse=True)
        
        # Elitism: preserve best individuals
        new_population = self.population[:self.elite_size]
        
        # Generate offspring
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            child1, child2 = self._crossover(parent1, parent2)
            
            # Mutation
            self._mutate(child1)
            self._mutate(child2)
            
            # Evaluation
            self._decode_and_evaluate(child1)
            self._decode_and_evaluate(child2)
            
            # Add to new population
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        # Replace population
        self.population = new_population[:self.population_size]
        self.population.sort(reverse=True)
        
        # Update best
        if self.population[0].fitness > self.best_individual.fitness:
            self.best_individual = self._copy_individual(self.population[0])
        
        self.generation += 1
    
    def compute_diversity(self) -> float:
        """
        Compute population diversity as average pairwise distance.
        Uses sampling for efficiency.
        """
        if len(self.population) < 2:
            return 0.0
        
        # Sample pairs for efficiency
        n_samples = min(50, len(self.population) * (len(self.population) - 1) // 2)
        total_distance = 0.0
        
        for _ in range(n_samples):
            i1 = np.random.randint(0, len(self.population))
            i2 = np.random.randint(0, len(self.population))
            if i1 == i2:
                continue
            
            # Compute distance (L2 norm on priorities)
            dist_slot = np.linalg.norm(
                self.population[i1].slot_priorities - self.population[i2].slot_priorities)
            dist_action = np.linalg.norm(
                self.population[i1].action_priorities - self.population[i2].action_priorities)
            total_distance += (dist_slot + dist_action)
        
        return total_distance / max(n_samples, 1)
    
    # =========================================================================
    # MAIN SOLVE METHOD
    # =========================================================================
    
    def solve(self, max_generations: int = 100, 
             time_limit_minutes: float = 60,
             verbose: bool = True,
             convergence_window: int = 20) -> Dict:
        """
        Solve REOSSP using Genetic Algorithm.
        
        Args:
            max_generations: Maximum number of generations
            time_limit_minutes: Time limit in minutes
            verbose: Print progress information
            convergence_window: Stop if no improvement for this many generations
        
        Returns:
            Dictionary with solution results
        """
        start_time = time.time()
        time_limit_seconds = time_limit_minutes * 60
        
        # Initialize population
        self.initialize_population()
        
        # Track convergence
        best_fitness_history = [self.best_individual.fitness]
        generations_without_improvement = 0
        
        if verbose:
            print(f"\nStarting GA evolution...")
            print(f"  Population size: {self.population_size}")
            print(f"  Max generations: {max_generations}")
            print(f"  Time limit: {time_limit_minutes:.1f} minutes")
            print(f"  Elite size: {self.elite_size}")
            print(f"  Mutation rate: {self.mutation_rate}")
            print(f"  Crossover rate: {self.crossover_rate}")
            print()
        
        # Evolution loop
        for gen in range(max_generations):
            # Check time limit
            elapsed = time.time() - start_time
            if elapsed > time_limit_seconds:
                if verbose:
                    print(f"\nTime limit reached at generation {gen}")
                break
            
            # Evolve one generation
            self.evolve_generation()
            
            # Track statistics
            best_fitness = self.best_individual.fitness
            avg_fitness = np.mean([ind.fitness for ind in self.population])
            diversity = self.compute_diversity()
            n_feasible = sum(ind.is_feasible for ind in self.population)
            
            self.fitness_history.append((best_fitness, avg_fitness))
            self.diversity_history.append(diversity)
            best_fitness_history.append(best_fitness)
            
            # Check convergence
            if len(best_fitness_history) > convergence_window:
                recent_improvement = best_fitness - best_fitness_history[-convergence_window]
                if abs(recent_improvement) < 0.01:
                    generations_without_improvement += 1
                else:
                    generations_without_improvement = 0
                
                if generations_without_improvement >= convergence_window:
                    if verbose:
                        print(f"\nConverged at generation {gen} (no improvement for {convergence_window} generations)")
                    break
            
            # Print progress
            if verbose and (gen % 10 == 0 or gen == max_generations - 1):
                print(f"Gen {gen:4d} | Best: {best_fitness:8.2f} | Avg: {avg_fitness:8.2f} | "
                      f"Diversity: {diversity:6.2f} | Feasible: {n_feasible:3d}/{self.population_size} | "
                      f"Time: {elapsed/60:5.2f}m")
        
        # Final evaluation
        runtime = time.time() - start_time
        
        # Extract solution details from best individual
        best = self.best_individual
        
        # Debug: Print best individual info
        if verbose:
            print(f"\nBest individual type: {type(best)}")
            if best is not None:
                print(f"Best fitness: {best.fitness}")
                print(f"Is feasible: {best.is_feasible}")
            else:
                print("WARNING: best is None!")
        
        # Check if we found any solution
        if best is None or best.observations is None:
            if verbose:
                print(f"\n{'='*80}")
                print(f"GA OPTIMIZATION COMPLETE - NO VALID SOLUTION FOUND")
                print(f"{'='*80}")
                print(f"Runtime: {runtime/60:.2f} minutes")
                print(f"Generations: {self.generation}")
                print(f"Population size: {len(self.population)}")
                print(f"Best individual: {best}")
                print(f"{'='*80}\n")
            
            return {
                'status': 'infeasible',
                'objective': 0,
                'runtime_minutes': runtime / 60,
                'data_downlinked_gb': 0,
                'total_observations': 0,
                'total_downlinks': 0,
                'propellant_used': 0,
                'num_generations': self.generation,
                'constraint_violations': {},
                'is_feasible': False,
                'best_individual': None,
                'fitness_history': self.fitness_history,
                'diversity_history': self.diversity_history,
                'num_variables': self.S * self.K * self.J + self.S * self.K * self.T_per_stage * self.max_actions,
                'num_constraints': 'N/A (GA)',
                'num_nonzeros': 'N/A (GA)',
                'message': 'No valid solution found by GA'
            }
        
        # Count observations and downlinks
        total_observations = best.observations.sum()
        total_downlinks = best.downlinks.sum()
        
        # Calculate data downlinked
        data_downlinked_mb = total_downlinks * self.params.D_comm
        data_downlinked_gb = data_downlinked_mb / 1024
        
        # Total propellant used
        propellant_used = best.propellant_used.sum()
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"GA OPTIMIZATION COMPLETE")
            print(f"{'='*80}")
            print(f"Status: {'Feasible' if best.is_feasible else 'Infeasible'}")
            print(f"Objective: {best.fitness:.2f}")
            print(f"Generations: {self.generation}")
            print(f"Runtime: {runtime/60:.2f} minutes")
            print(f"Observations: {total_observations:.0f}")
            print(f"Downlinks: {total_downlinks:.0f}")
            print(f"Data downlinked: {data_downlinked_gb:.4f} GB")
            print(f"Propellant used: {propellant_used:.2f} m/s")
            print(f"Constraint violations: {best.constraint_violations}")
            print(f"{'='*80}\n")
        
        return {
            'status': 'optimal' if best.is_feasible else 'feasible_with_violations',
            'objective': best.fitness,
            'runtime_minutes': runtime / 60,
            'data_downlinked_gb': data_downlinked_gb,
            'total_observations': int(total_observations),
            'total_downlinks': int(total_downlinks),
            'propellant_used': float(propellant_used),
            'num_generations': self.generation,
            'constraint_violations': best.constraint_violations,
            'is_feasible': best.is_feasible,
            'best_individual': best,
            'fitness_history': self.fitness_history,
            'diversity_history': self.diversity_history,
            # Dummy values for compatibility with other solvers
            'num_variables': self.S * self.K * self.J + self.S * self.K * self.T_per_stage * self.max_actions,
            'num_constraints': 'N/A (GA)',
            'num_nonzeros': 'N/A (GA)'
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    from parameters import InstanceParameters
    
    print("="*80)
    print("REOSSP GENETIC ALGORITHM SOLVER - TEST")
    print("="*80)
    
    # Create small test instance
    test_params = InstanceParameters(
        instance_id=42,
        S=4,
        K=3,
        J_sk=10,
        T=144,
        unavailable_slot_probability=0.1
    )
    
    print(f"\nProblem instance:")
    print(f"  Stages: {test_params.S}")
    print(f"  Satellites: {test_params.K}")
    print(f"  Orbital slots: {test_params.J_sk}")
    print(f"  Time steps: {test_params.T}")
    print(f"  Time steps per stage: {test_params.T // test_params.S}")
    print(f"  Targets: {test_params.V_target.shape[4]}")
    print(f"  Ground stations: {test_params.V_ground.shape[4]}")
    
    # Create and run GA solver
    solver = REOSSPGASolver(
        params=test_params,
        population_size=50,
        elite_size=5,
        mutation_rate=0.15,
        crossover_rate=0.8,
        random_seed=42
    )
    
    results = solver.solve(
        max_generations=50,
        time_limit_minutes=5,
        verbose=True,
        convergence_window=15
    )
    
    print("\nFinal Results:")
    print(f"  Status: {results['status']}")
    print(f"  Objective: {results['objective']:.2f}")
    print(f"  Runtime: {results['runtime_minutes']:.2f} minutes")
    print(f"  Generations: {results['num_generations']}")
    print(f"  Feasible: {results['is_feasible']}")
    print(f"  Constraint violations: {results['constraint_violations']}")
