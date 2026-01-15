"""
REOSSP Differential Evolution Solver
Implements REOSSP using DEAP (Differential Evolution) instead of MILP
Encodes binary/continuous variables as real-valued chromosome with repair mechanisms
"""

import numpy as np
import time
from deap import base, creator, tools, algorithms
from parameters import InstanceParameters


class REOSSPDESolver:
    """REOSSP solver using Differential Evolution (DEAP)"""
    
    def __init__(self, params: InstanceParameters):
        self.params = params
        self.T_per_stage = params.T // params.S
        self.n_targets = params.V_target.shape[4]
        self.n_ground = params.V_ground.shape[4]
        
        # Calculate chromosome dimensions
        self.n_y = params.S * params.K * self.T_per_stage * self.n_targets
        self.n_q = params.S * params.K * self.T_per_stage * self.n_ground
        self.n_h = params.S * params.K * self.T_per_stage
        self.n_x = params.S * params.K * params.J_sk * params.J_sk
        self.chrom_len = self.n_y + self.n_q + self.n_h + self.n_x
        
    def decode_chromosome(self, chrom):
        """Decode chromosome to decision variables"""
        p = self.params
        idx = 0
        
        # Decode y (observations) - threshold at 0.5
        y = np.zeros((p.S, p.K, self.T_per_stage, self.n_targets), dtype=int)
        for s in range(p.S):
            for k in range(p.K):
                for t in range(self.T_per_stage):
                    for pt in range(self.n_targets):
                        y[s,k,t,pt] = 1 if chrom[idx] >= 0.5 else 0
                        idx += 1
        
        # Decode q (downlinks)
        q = np.zeros((p.S, p.K, self.T_per_stage, self.n_ground), dtype=int)
        for s in range(p.S):
            for k in range(p.K):
                for t in range(self.T_per_stage):
                    for g in range(self.n_ground):
                        q[s,k,t,g] = 1 if chrom[idx] >= 0.5 else 0
                        idx += 1
        
        # Decode h (charging)
        h = np.zeros((p.S, p.K, self.T_per_stage), dtype=int)
        for s in range(p.S):
            for k in range(p.K):
                for t in range(self.T_per_stage):
                    h[s,k,t] = 1 if chrom[idx] >= 0.5 else 0
                    idx += 1
        
        # Decode x (reconfiguration)
        x = np.zeros((p.S, p.K, p.J_sk, p.J_sk), dtype=int)
        for s in range(p.S):
            for k in range(p.K):
                for i in range(p.J_sk):
                    for j in range(p.J_sk):
                        x[s,k,i,j] = 1 if chrom[idx] >= 0.5 else 0
                        idx += 1
        
        return y, q, h, x
    
    def encode_chromosome(self, y, q, h, x):
        """Encode decision variables back to chromosome"""
        chrom = []
        p = self.params
        
        for s in range(p.S):
            for k in range(p.K):
                for t in range(self.T_per_stage):
                    for pt in range(self.n_targets):
                        chrom.append(float(y[s,k,t,pt]))
        
        for s in range(p.S):
            for k in range(p.K):
                for t in range(self.T_per_stage):
                    for g in range(self.n_ground):
                        chrom.append(float(q[s,k,t,g]))
        
        for s in range(p.S):
            for k in range(p.K):
                for t in range(self.T_per_stage):
                    chrom.append(float(h[s,k,t]))
        
        for s in range(p.S):
            for k in range(p.K):
                for i in range(p.J_sk):
                    for j in range(p.J_sk):
                        chrom.append(float(x[s,k,i,j]))
        
        return chrom
    
    def repair_chromosome(self, chrom):
        """Repair chromosome to satisfy hard constraints"""
        p = self.params
        y, q, h, x = self.decode_chromosome(chrom)
        
        # Repair x: Initial slot assignment and continuity
        for k in range(p.K):
            # Stage 1: exactly one transition from slot 0 (initial)
            if x[0,k,0,:].sum() != 1:
                x[0,k,0,:] = 0
                j = np.random.randint(0, p.J_sk)
                x[0,k,0,j] = 1
            
            # Slot continuity
            for s in range(p.S - 1):
                for j in range(p.J_sk):
                    if s == 0:
                        incoming = x[0,k,0,j]
                    else:
                        incoming = x[s,k,:,j].sum()
                    
                    outgoing = x[s+1,k,j,:].sum()
                    
                    if incoming > 0 and outgoing == 0:
                        # Force transition
                        j_next = np.random.randint(0, p.J_sk)
                        x[s+1,k,j,j_next] = 1
                    elif incoming == 0:
                        x[s+1,k,j,:] = 0
        
        # Repair x: Unavailable slots
        for s in range(p.S):
            for j in range(p.J_sk):
                if p.unavailable_slots[s, j]:
                    if s == 0:
                        x[s,:,0,j] = 0
                    else:
                        x[s,:,:,j] = 0
        
        # Determine current slot for each satellite at each stage
        current_slots = np.zeros((p.S, p.K), dtype=int)
        for k in range(p.K):
            for s in range(p.S):
                if s == 0:
                    current_slots[s,k] = np.argmax(x[0,k,0,:]) if x[0,k,0,:].sum() > 0 else 0
                else:
                    prev_slot = current_slots[s-1,k]
                    current_slots[s,k] = np.argmax(x[s,k,prev_slot,:]) if x[s,k,prev_slot,:].sum() > 0 else prev_slot
        
        # Repair y, q, h: Visibility constraints
        for s in range(p.S):
            for k in range(p.K):
                j_curr = current_slots[s,k]
                for t in range(self.T_per_stage):
                    # Target visibility
                    for pt in range(self.n_targets):
                        if not p.V_target[s,k,t,j_curr,pt]:
                            y[s,k,t,pt] = 0
                    
                    # Ground visibility
                    for g in range(self.n_ground):
                        if not p.V_ground[s,k,t,j_curr,g]:
                            q[s,k,t,g] = 0
                    
                    # Sun visibility
                    if not p.V_sun[s,k,t,j_curr]:
                        h[s,k,t] = 0
                    
                    # One activity constraint
                    total = y[s,k,t,:].sum() + q[s,k,t,:].sum() + h[s,k,t]
                    if total > 1:
                        # Priority: h > q > y
                        if h[s,k,t] == 1:
                            q[s,k,t,:] = 0
                            y[s,k,t,:] = 0
                        elif q[s,k,t,:].sum() > 0:
                            y[s,k,t,:] = 0
                            first_g = np.argmax(q[s,k,t,:])
                            q[s,k,t,:] = 0
                            q[s,k,t,first_g] = 1
                        else:
                            first_p = np.argmax(y[s,k,t,:])
                            y[s,k,t,:] = 0
                            y[s,k,t,first_p] = 1
        
        return self.encode_chromosome(y, q, h, x)
    
    def compute_states(self, y, q, h, x):
        """Compute data and battery states from actions"""
        p = self.params
        d = np.zeros((p.S, p.K, self.T_per_stage))
        b = np.zeros((p.S, p.K, self.T_per_stage))
        
        for k in range(p.K):
            # Initial battery (reduced by first maneuver)
            b[0,k,0] = p.B_max - p.B_recon * x[0,k,0,:].sum()
            d[0,k,0] = 0.0
            
            for s in range(p.S):
                for t in range(self.T_per_stage):
                    if s == 0 and t == 0:
                        continue
                    
                    # Determine previous state
                    if t == 0:  # Stage transition
                        s_prev, t_prev = s-1, self.T_per_stage-1
                        d_prev = d[s_prev,k,t_prev]
                        b_prev = b[s_prev,k,t_prev]
                        
                        # Data dynamics
                        data_gen = y[s_prev,k,t_prev,:].sum() * p.D_obs
                        data_down = q[s_prev,k,t_prev,:].sum() * p.D_comm
                        d[s,k,t] = d_prev + data_gen - data_down
                        
                        # Battery dynamics (with maneuver cost)
                        charge = h[s_prev,k,t_prev] * p.B_charge
                        obs_cost = y[s_prev,k,t_prev,:].sum() * p.B_obs
                        comm_cost = q[s_prev,k,t_prev,:].sum() * p.B_comm
                        recon_cost = x[s,k,:,:].sum() * p.B_recon
                        b[s,k,t] = b_prev + charge - obs_cost - comm_cost - recon_cost - p.B_time
                    else:  # Within stage
                        d_prev = d[s,k,t-1]
                        b_prev = b[s,k,t-1]
                        
                        # Data dynamics
                        data_gen = y[s,k,t-1,:].sum() * p.D_obs
                        data_down = q[s,k,t-1,:].sum() * p.D_comm
                        d[s,k,t] = d_prev + data_gen - data_down
                        
                        # Battery dynamics
                        charge = h[s,k,t-1] * p.B_charge
                        obs_cost = y[s,k,t-1,:].sum() * p.B_obs
                        comm_cost = q[s,k,t-1,:].sum() * p.B_comm
                        b[s,k,t] = b_prev + charge - obs_cost - comm_cost - p.B_time
        
        return d, b
    
    def evaluate_fitness(self, individual):
        """Evaluate fitness = objective - penalties"""
        p = self.params
        
        # Repair chromosome
        individual[:] = self.repair_chromosome(individual)
        
        # Decode
        y, q, h, x = self.decode_chromosome(individual)
        
        # Compute states
        d, b = self.compute_states(y, q, h, x)
        
        # Objective
        total_downlinks = q.sum()
        total_observations = y.sum()
        objective = p.C * total_downlinks + total_observations
        
        # Penalties
        penalty = 0.0
        W = 10000.0  # Penalty weight
        
        # Data storage violations
        for s in range(p.S):
            for k in range(p.K):
                for t in range(self.T_per_stage):
                    data_gen = y[s,k,t,:].sum() * p.D_obs
                    if d[s,k,t] + data_gen > p.D_max:
                        penalty += W * (d[s,k,t] + data_gen - p.D_max)
                    
                    data_down = q[s,k,t,:].sum() * p.D_comm
                    if d[s,k,t] - data_down < 0:
                        penalty += W * abs(d[s,k,t] - data_down)
        
        # Battery violations
        for s in range(p.S):
            for k in range(p.K):
                for t in range(self.T_per_stage):
                    charge = h[s,k,t] * p.B_charge
                    if b[s,k,t] + charge > p.B_max:
                        penalty += W * (b[s,k,t] + charge - p.B_max)
                    
                    obs_cost = y[s,k,t,:].sum() * p.B_obs
                    comm_cost = q[s,k,t,:].sum() * p.B_comm
                    consumption = obs_cost + comm_cost + p.B_time
                    
                    if t == self.T_per_stage - 1 and s < p.S - 1:
                        consumption += x[s+1,k,:,:].sum() * p.B_recon
                    
                    if b[s,k,t] - consumption < 0:
                        penalty += W * abs(b[s,k,t] - consumption)
        
        # Propellant budget violation
        for k in range(p.K):
            prop_used = 0.0
            for s in range(p.S):
                if s == 0:
                    for j in range(p.J_sk):
                        if x[s,k,0,j] > 0:
                            prop_used += p.maneuver_costs[s,k,0,j]
                else:
                    for i in range(p.J_sk):
                        for j in range(p.J_sk):
                            if x[s,k,i,j] > 0:
                                prop_used += p.maneuver_costs[s,k,i,j]
            
            if prop_used > p.c_max:
                penalty += W * (prop_used - p.c_max)
        
        fitness = objective - penalty
        return (fitness,)
    
    def solve(self, pop_size=50, n_generations=100, verbose=True):
        """Run Differential Evolution optimization"""
        
        # Setup DEAP
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
        
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        toolbox.register("attr_float", np.random.random)
        toolbox.register("individual", tools.initRepeat, creator.Individual, 
                        toolbox.attr_float, n=self.chrom_len)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evaluate_fitness)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", np.max)
        stats.register("avg", np.mean)
        
        # Initialize population
        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)
        
        # Run evolution
        start_time = time.time()
        
        pop, logbook = algorithms.eaSimple(
            pop, toolbox,
            cxpb=0.7,
            mutpb=0.2,
            ngen=n_generations,
            stats=stats,
            halloffame=hof,
            verbose=verbose
        )
        
        runtime = time.time() - start_time
        
        # Extract best solution
        best = hof[0]
        best_fitness = best.fitness.values[0]
        
        # Decode best solution
        y, q, h, x = self.decode_chromosome(best)
        d, b = self.compute_states(y, q, h, x)
        
        # Calculate metrics
        total_downlinks = q.sum()
        total_observations = y.sum()
        data_downlinked_gb = (total_downlinks * self.params.D_comm) / 1024
        
        propellant_used = 0.0
        for k in range(self.params.K):
            for s in range(self.params.S):
                if s == 0:
                    for j in range(self.params.J_sk):
                        if x[s,k,0,j] > 0:
                            propellant_used += self.params.maneuver_costs[s,k,0,j]
                else:
                    for i in range(self.params.J_sk):
                        for j in range(self.params.J_sk):
                            if x[s,k,i,j] > 0:
                                propellant_used += self.params.maneuver_costs[s,k,i,j]
        
        results = {
            'status': 'completed',
            'objective': best_fitness,
            'runtime_minutes': runtime / 60,
            'data_downlinked_gb': data_downlinked_gb,
            'total_observations': int(total_observations),
            'total_downlinks': int(total_downlinks),
            'propellant_used': propellant_used,
            'num_variables': self.chrom_len,
            'num_constraints': 0,  # Constraints handled via repair
            'generations': n_generations,
            'population_size': pop_size
        }
        
        return results


if __name__ == "__main__":
    print("Testing REOSSP-DE Solver...")
    
    test_params = InstanceParameters(
        instance_id=999,
        S=8,
        K=5,
        J_sk=20,
        T=36*24*2,
        unavailable_slot_probability=0.0
    )
    
    solver = REOSSPDESolver(test_params)
    print(f"Chromosome length: {solver.chrom_len}")
    print(f"Running DE with pop_size=30, generations=50...")
    
    results = solver.solve(pop_size=30, n_generations=50, verbose=True)
    
    print("\nResults:")
    print(f"  Status: {results['status']}")
    print(f"  Objective: {results['objective']:.2f}")
    print(f"  Runtime: {results['runtime_minutes']:.2f} minutes")
    print(f"  Data downlinked: {results['data_downlinked_gb']:.2f} GB")
    print(f"  Observations: {results['total_observations']}")
    print(f"  Downlinks: {results['total_downlinks']}")
    print(f"  Propellant used: {results['propellant_used']:.2f} m/s")
