"""
REOSSP Metaheuristic Solvers
Adapts the converted algorithms (BGA, IDE, PSO, RGA) to solve REOSSP
Uses the same greedy scheduling approach as Two-Phase GA for fitness evaluation
"""

import numpy as np
import time
from typing import Dict
from parameters import InstanceParameters
from reossp_two_phase_ga import GreedyScheduler


class REOSSPMetaheuristicSolver:
    """
    Wrapper to solve REOSSP using metaheuristic algorithms (BGA, IDE, PSO, RGA)
    
    Approach:
    - Chromosome: Flattened K×S vector of continuous values in [0,1]
    - Decoding: Map continuous values to discrete slot assignments
    - Fitness: Use greedy scheduler to evaluate orbital trajectory
    """
    
    def __init__(self, params: InstanceParameters):
        self.params = params
        self.scheduler = GreedyScheduler(params)
        
        # Chromosome dimension: K satellites × S stages
        self.dim = params.K * params.S
        
        # Track best solution found
        self.best_solution = None
        self.best_objective = -1e9
        self.best_trajectory = None
        
        # Statistics
        self.evaluations = 0
        self.feasible_count = 0
        self.infeasible_count = 0
    
    def decode_solution(self, x: np.ndarray) -> np.ndarray:
        """
        Decode continuous solution [0,1]^(K×S) to discrete trajectory [K×S]
        
        Args:
            x: Continuous vector in [0,1], length = K×S
            
        Returns:
            trajectory: [K×S] array of slot assignments (1 to J_sk)
        """
        # Reshape to K×S
        x_matrix = x.reshape(self.params.K, self.params.S)
        
        # Map [0,1] to slot indices [1, J_sk]
        # Use floor to discretize: x ∈ [0,1) → slot ∈ [1, J_sk]
        trajectory = np.floor(x_matrix * self.params.J_sk) + 1
        trajectory = np.clip(trajectory, 1, self.params.J_sk).astype(int)
        
        return trajectory
    
    def is_propellant_feasible(self, trajectory: np.ndarray) -> bool:
        """Check if trajectory satisfies propellant budget"""
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
        """Check if all assigned slots are available"""
        K, S = trajectory.shape
        
        for k in range(K):
            for s in range(S):
                slot = trajectory[k, s] - 1  # Convert to 0-indexed
                if self.params.unavailable_slots[s, slot]:
                    return False
        
        return True
    
    def objective_function(self, x: np.ndarray) -> float:
        """
        Evaluate objective for metaheuristic algorithms
        IMPORTANT: Uses EXACT same objective calculation as REOSSP Exact
        
        Args:
            x: Continuous solution vector [0,1]^(K×S)
            
        Returns:
            Objective value (for maximization)
        """
        self.evaluations += 1
        
        # Decode to trajectory
        trajectory = self.decode_solution(x)
        
        # Check propellant feasibility
        if not self.is_propellant_feasible(trajectory):
            self.infeasible_count += 1
            return -1e9  # Large penalty for infeasible solutions
        
        # Check slot availability
        if not self.check_slot_availability(trajectory):
            self.infeasible_count += 1
            return -1e9
        
        # Evaluate using greedy scheduler
        schedule_result = self.scheduler.schedule(trajectory)
        
        if not schedule_result['feasible']:
            self.infeasible_count += 1
            return -1e9
        
        self.feasible_count += 1
        
        # Calculate objective EXACTLY as REOSSP Exact does:
        # Objective = Total Observations + C * Total Downlinks
        total_observations = schedule_result['total_observations']
        total_downlinks = schedule_result['total_downlinks']
        objective = total_observations + self.params.C * total_downlinks
        
        # Track best solution
        if objective > self.best_objective:
            self.best_objective = objective
            self.best_solution = x.copy()
            self.best_trajectory = trajectory.copy()
        
        return objective
    
    def solve_bga(self, pop_size=50, max_fes=10000, verbose=True):
        """Solve using Binary Genetic Algorithm"""
        from integerGA_python.algos import run_bga
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Running BGA (Binary Genetic Algorithm)")
            print(f"{'='*80}")
            print(f"Problem dimension: {self.dim}")
            print(f"Population size: {pop_size}")
            print(f"Max evaluations: {max_fes}")
        
        start_time = time.time()
        self.evaluations = 0
        self.feasible_count = 0
        self.infeasible_count = 0
        
        best_sol, convergence, best_obj = run_bga(
            dim=self.dim,
            obj_func=self.objective_function,
            max_fes=max_fes,
            pop_size=pop_size,
            verbose=verbose,
            p_single_point=0.3,
            p_double_point=0.3
        )
        
        runtime = time.time() - start_time
        
        # Decode best solution
        best_trajectory = self.decode_solution(best_sol)
        schedule_result = self.scheduler.schedule(best_trajectory)
        
        return self._format_results('BGA', schedule_result, best_trajectory, 
                                    runtime, convergence)
    
    def solve_ide(self, pop_size=50, max_fes=10000, verbose=True):
        """Solve using Improved Differential Evolution"""
        from integerGA_python.algos import run_ide
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Running IDE (Improved Differential Evolution)")
            print(f"{'='*80}")
            print(f"Problem dimension: {self.dim}")
            print(f"Population size: {pop_size}")
            print(f"Max evaluations: {max_fes}")
        
        start_time = time.time()
        self.evaluations = 0
        self.feasible_count = 0
        self.infeasible_count = 0
        
        best_sol, convergence, best_obj = run_ide(
            dim=self.dim,
            obj_func=self.objective_function,
            max_fes=max_fes,
            pop_size=pop_size,
            pbest_rate=0.1,
            mem_size=5,
            arc_rate=2.6,
            verbose=verbose
        )
        
        runtime = time.time() - start_time
        
        # Decode best solution
        best_trajectory = self.decode_solution(best_sol)
        schedule_result = self.scheduler.schedule(best_trajectory)
        
        return self._format_results('IDE', schedule_result, best_trajectory, 
                                    runtime, convergence)
    
    def solve_pso(self, pop_size=50, max_fes=10000, verbose=True):
        """Solve using Particle Swarm Optimization"""
        from integerGA_python.algos import run_pso
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Running PSO (Particle Swarm Optimization)")
            print(f"{'='*80}")
            print(f"Problem dimension: {self.dim}")
            print(f"Swarm size: {pop_size}")
            print(f"Max evaluations: {max_fes}")
        
        start_time = time.time()
        self.evaluations = 0
        self.feasible_count = 0
        self.infeasible_count = 0
        
        best_sol, convergence, best_obj = run_pso(
            dim=self.dim,
            obj_func=self.objective_function,
            max_fes=max_fes,
            pop_size=pop_size,
            w=0.7,
            c1=1.5,
            c2=1.5,
            verbose=verbose
        )
        
        runtime = time.time() - start_time
        
        # Decode best solution
        best_trajectory = self.decode_solution(best_sol)
        schedule_result = self.scheduler.schedule(best_trajectory)
        
        return self._format_results('PSO', schedule_result, best_trajectory, 
                                    runtime, convergence)
    
    def solve_rga(self, pop_size=50, max_fes=10000, verbose=True):
        """Solve using Real-Coded Genetic Algorithm"""
        from integerGA_python.algos import run_rga
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Running RGA (Real-Coded Genetic Algorithm)")
            print(f"{'='*80}")
            print(f"Problem dimension: {self.dim}")
            print(f"Population size: {pop_size}")
            print(f"Max evaluations: {max_fes}")
        
        start_time = time.time()
        self.evaluations = 0
        self.feasible_count = 0
        self.infeasible_count = 0
        
        best_sol, convergence, best_obj = run_rga(
            dim=self.dim,
            obj_func=self.objective_function,
            max_fes=max_fes,
            pop_size=pop_size,
            verbose=verbose
        )
        
        runtime = time.time() - start_time
        
        # Decode best solution
        best_trajectory = self.decode_solution(best_sol)
        schedule_result = self.scheduler.schedule(best_trajectory)
        
        return self._format_results('RGA', schedule_result, best_trajectory, 
                                    runtime, convergence)
    
    def _format_results(self, algorithm_name: str, schedule_result: Dict, 
                       trajectory: np.ndarray, runtime: float, 
                       convergence: np.ndarray) -> Dict:
        """Format results in consistent structure"""
        
        # Calculate propellant used (same way as REOSSP Exact and Two-Phase GA)
        propellant_used = 0
        for k in range(self.params.K):
            # Stage 1: maneuver from initial slot
            slot_s1 = trajectory[k, 0] - 1
            propellant_used += self.params.maneuver_costs[0, k, 0, slot_s1]
            
            # Subsequent stages
            for s in range(1, self.params.S):
                slot_prev = trajectory[k, s-1] - 1
                slot_curr = trajectory[k, s] - 1
                propellant_used += self.params.maneuver_costs[s, k, slot_prev, slot_curr]
        
        # Calculate objective EXACTLY as REOSSP Exact does:
        # Objective = Total Observations + C * Total Downlinks
        total_observations = schedule_result['total_observations']
        total_downlinks = schedule_result['total_downlinks']
        
        # Use the EXACT same formula as REOSSP Exact
        objective = total_observations + self.params.C * total_downlinks
        
        return {
            'algorithm': algorithm_name,
            'status': 'Optimal' if schedule_result['feasible'] else 'Infeasible',
            'objective': objective,  # Recalculated to ensure consistency with REOSSP Exact
            'total_observations': total_observations,
            'total_downlinks': total_downlinks,
            'data_downlinked_gb': total_downlinks * self.params.D_comm / 1024,
            'runtime_seconds': runtime,
            'runtime_minutes': runtime / 60,
            'propellant_used': propellant_used,
            'best_trajectory': trajectory,
            'convergence': convergence,
            'total_evaluations': self.evaluations,
            'feasible_solutions': self.feasible_count,
            'infeasible_solutions': self.infeasible_count,
            'feasibility_rate': self.feasible_count / self.evaluations if self.evaluations > 0 else 0
        }


if __name__ == "__main__":
    """Test all metaheuristic solvers"""
    print("="*80)
    print("TESTING REOSSP METAHEURISTIC SOLVERS")
    print("="*80)
    
    # Create test instance
    test_params = InstanceParameters(
        instance_id=999,
        S=8,
        K=5,
        J_sk=20,
        T=36*24*2,
        unavailable_slot_probability=0.2
    )
    
    print(f"\nProblem size:")
    print(f"  Satellites (K): {test_params.K}")
    print(f"  Stages (S): {test_params.S}")
    print(f"  Slots per stage (J_sk): {test_params.J_sk}")
    print(f"  Total timesteps (T): {test_params.T}")
    print(f"  Chromosome dimension: {test_params.K * test_params.S}")
    print(f"  Unavailable slots: {test_params.unavailable_slots.sum()}/{test_params.S * test_params.J_sk}")
    
    # Algorithm parameters
    pop_size = 50
    max_fes = 5000
    
    print(f"\nAlgorithm parameters:")
    print(f"  Population size: {pop_size}")
    print(f"  Max evaluations: {max_fes}")
    
    solver = REOSSPMetaheuristicSolver(test_params)
    
    # Test all algorithms
    algorithms = {
        'BGA': solver.solve_bga,
        'IDE': solver.solve_ide,
        'PSO': solver.solve_pso,
        'RGA': solver.solve_rga
    }
    
    results = {}
    
    for alg_name, solve_func in algorithms.items():
        try:
            results[alg_name] = solve_func(pop_size=pop_size, max_fes=max_fes, verbose=True)
            
            print(f"\n{alg_name} Results:")
            print(f"  Status: {results[alg_name]['status']}")
            print(f"  Objective: {results[alg_name]['objective']:.2f}")
            print(f"  Runtime: {results[alg_name]['runtime_minutes']:.2f} minutes")
            print(f"  Observations: {results[alg_name]['total_observations']}")
            print(f"  Downlinks: {results[alg_name]['total_downlinks']}")
            print(f"  Propellant: {results[alg_name]['propellant_used']:.2f} m/s")
            print(f"  Evaluations: {results[alg_name]['total_evaluations']}")
            print(f"  Feasibility rate: {results[alg_name]['feasibility_rate']*100:.1f}%")
            
        except Exception as e:
            print(f"\n✗ {alg_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    # Comparison summary
    print(f"\n{'='*80}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Algorithm':<10} {'Objective':>12} {'Runtime (min)':>15} {'Feasibility %':>15}")
    print(f"{'-'*80}")
    
    for alg_name in ['BGA', 'IDE', 'PSO', 'RGA']:
        if alg_name in results:
            r = results[alg_name]
            print(f"{alg_name:<10} {r['objective']:>12.2f} {r['runtime_minutes']:>15.2f} "
                  f"{r['feasibility_rate']*100:>14.1f}%")
