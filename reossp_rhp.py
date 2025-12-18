"""
REOSSP-RHP: Rolling Horizon Procedure
Fast heuristic approach for large instances
"""

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import numpy as np
import time
import os
from parameters import InstanceParameters


class REOSSPRHPSolver:
    """
    Rolling Horizon Procedure for REOSSP
    Solves the problem stage-by-stage with lookahead L=1
    Much faster than exact method but potentially suboptimal
    """
    
    def __init__(self, params: InstanceParameters, lookahead=1, weighted_objective=True):
        self.params = params
        self.lookahead = lookahead  # L=1 means optimize current + next stage
        self.weighted_objective = weighted_objective  # True: current 100%, lookahead 50%; False: equal weights
        self.results = None
        
        # Track solution across stages
        self.orbital_assignments = {}  # {stage: {sat: slot}}
        self.observation_plan = []
        self.downlink_plan = []
        self.satellite_propellant = {k: 0 for k in range(params.K)}  # Per-satellite cumulative propellant
        
        # Track state between stages
        self.battery_state = {k: params.B_max for k in range(params.K)}  # Initialize at full
        self.data_state = {k: 0 for k in range(params.K)}  # Initialize at empty
        
    def build_stage_model(self, current_stage, prev_assignments=None):
        """
        Build optimization model for current stage + lookahead stages
        
        Args:
            current_stage: Current stage index
            prev_assignments: Dict {sat: slot} from previous stage
        """
        p = self.params
        
        # Determine stages to optimize
        stages_to_optimize = min(self.lookahead + 1, p.S - current_stage)
        stage_range = range(current_stage, current_stage + stages_to_optimize)
        
        model = pyo.ConcreteModel(name=f"REOSSP_RHP_Stage_{current_stage}")
        
        # Time steps for these stages
        stage_boundaries = p.get_stage_boundaries()
        t_start = stage_boundaries[current_stage]
        t_end = stage_boundaries[current_stage + stages_to_optimize]
        time_range = range(t_start, t_end)
        
        # Sets
        model.S_local = pyo.Set(initialize=list(stage_range))
        model.K = pyo.RangeSet(0, p.K - 1)
        model.T_local = pyo.Set(initialize=list(time_range))
        model.J = pyo.RangeSet(0, p.J_sk - 1)
        model.N_target = pyo.RangeSet(0, p.V_target.shape[2] - 1)
        model.N_ground = pyo.RangeSet(0, p.V_ground.shape[2] - 1)
        
        # Decision Variables
        model.z = pyo.Var(model.S_local, model.K, model.J, domain=pyo.Binary)
        model.x = pyo.Var(model.K, model.T_local, model.N_target, domain=pyo.Binary)
        model.y = pyo.Var(model.K, model.T_local, model.N_ground, domain=pyo.Binary)
        model.c = pyo.Var(model.K, model.T_local, domain=pyo.Binary)  # Charging decision
        model.b = pyo.Var(model.K, model.T_local, domain=pyo.NonNegativeReals, bounds=(0, p.B_max))
        model.d = pyo.Var(model.K, model.T_local, domain=pyo.NonNegativeReals, bounds=(0, p.D_max))
        
        # Maneuver variables (only between stages in this window)
        maneuver_stages = [s for s in stage_range if s < p.S - 1 and s < current_stage + stages_to_optimize - 1]
        if maneuver_stages:
            model.S_maneuver = pyo.Set(initialize=maneuver_stages)
            model.u = pyo.Var(model.S_maneuver, model.K, model.J, model.J, domain=pyo.Binary)
        
        # Objective: maximize activities
        def objective_rule(m):
            obj = 0
            stage_boundaries = p.get_stage_boundaries()
            
            for s in m.S_local:
                # Optionally weight current stage higher than lookahead
                # Paper doesn't specify, but helps prioritize immediate decisions
                if self.weighted_objective:
                    weight = 1.0 if s == current_stage else 0.5  # Current stage priority
                else:
                    weight = 1.0  # Equal weights (standard RHP as in paper)
                
                # Time range for this stage
                t_start_s = stage_boundaries[s]
                t_end_s = stage_boundaries[s + 1]
                time_range_s = [t for t in m.T_local if t_start_s <= t < t_end_s]
                
                # Activities in this stage
                stage_downlinks = sum(m.y[k, t, g] 
                                    for k in m.K 
                                    for t in time_range_s
                                    for g in m.N_ground)
                stage_observations = sum(m.x[k, t, n] 
                                       for k in m.K 
                                       for t in time_range_s
                                       for n in m.N_target)
                
                obj += weight * (p.C * stage_downlinks + stage_observations)
            
            return obj
        
        model.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)
        
        # Constraints
        
        # 1. Orbital slot assignment
        def one_slot_per_stage_rule(m, s, k):
            return sum(m.z[s, k, j] for j in m.J) == 1
        model.one_slot = pyo.Constraint(model.S_local, model.K, rule=one_slot_per_stage_rule)
        
        def one_sat_per_slot_rule(m, s, j):
            return sum(m.z[s, k, j] for k in m.K) <= 1
        model.one_sat = pyo.Constraint(model.S_local, model.J, rule=one_sat_per_slot_rule)
        
        # 2. Fix initial orbital positions if this is not the first stage
        if prev_assignments is not None:
            def fix_initial_orbit_rule(m, k):
                initial_slot = prev_assignments[k]
                return m.z[current_stage, k, initial_slot] == 1
            model.fix_initial = pyo.Constraint(model.K, rule=fix_initial_orbit_rule)
        
        # 3. Maneuver constraints
        if maneuver_stages:
            def maneuver_consistency_rule(m, s, k):
                if s not in m.S_maneuver:
                    return pyo.Constraint.Skip
                return sum(m.u[s, k, j1, j2] for j1 in m.J for j2 in m.J) == 1
            model.maneuver_cons = pyo.Constraint(model.S_local, model.K, 
                                                rule=maneuver_consistency_rule)
            
            def maneuver_from_rule(m, s, k, j1):
                if s not in m.S_maneuver:
                    return pyo.Constraint.Skip
                return m.z[s, k, j1] == sum(m.u[s, k, j1, j2] for j2 in m.J)
            model.maneuver_from = pyo.Constraint(model.S_local, model.K, model.J,
                                                rule=maneuver_from_rule)
            
            def maneuver_to_rule(m, s, k, j2):
                if s not in m.S_maneuver or s + 1 not in m.S_local:
                    return pyo.Constraint.Skip
                return m.z[s + 1, k, j2] == sum(m.u[s, k, j1, j2] for j1 in m.J)
            model.maneuver_to = pyo.Constraint(model.S_local, model.K, model.J,
                                              rule=maneuver_to_rule)
            
            # Cumulative propellant constraint (per satellite)
            def propellant_rule(m, k):
                stage_propellant = sum(m.u[s, k, j1, j2] * p.maneuver_costs[s, k, j1, j2]
                                     for s in m.S_maneuver
                                     for j1 in m.J for j2 in m.J)
                # Subtract propellant already used by THIS satellite in previous stages
                prev_propellant_k = self.satellite_propellant.get(k, 0)
                return stage_propellant + prev_propellant_k <= p.c_max
            model.propellant = pyo.Constraint(model.K, rule=propellant_rule)
        
        # 4. Visibility and activity constraints
        def visibility_target_rule(m, k, t, n):
            if not p.V_target[k, t, n]:
                return m.x[k, t, n] == 0
            return pyo.Constraint.Skip
        model.vis_target = pyo.Constraint(model.K, model.T_local, model.N_target,
                                         rule=visibility_target_rule)
        
        def visibility_ground_rule(m, k, t, g):
            if not p.V_ground[k, t, g]:
                return m.y[k, t, g] == 0
            return pyo.Constraint.Skip
        model.vis_ground = pyo.Constraint(model.K, model.T_local, model.N_ground,
                                         rule=visibility_ground_rule)
        
        def visibility_sun_rule(m, k, t):
            if not p.V_sun[k, t]:
                return m.c[k, t] == 0
            return pyo.Constraint.Skip
        model.vis_sun = pyo.Constraint(model.K, model.T_local,
                                      rule=visibility_sun_rule)
        
        def one_activity_rule(m, k, t):
            total_obs = sum(m.x[k, t, n] for n in m.N_target)
            total_comm = sum(m.y[k, t, g] for g in m.N_ground)
            return total_obs + total_comm + m.c[k, t] <= 1
        model.one_activity = pyo.Constraint(model.K, model.T_local, rule=one_activity_rule)
        
        # 5. Battery and data dynamics
        def battery_dynamics_rule(m, k, t):
            if t == t_start:
                # Initial battery level (full) or inherited from previous stage
                if current_stage == 0:
                    # First stage starts with full battery
                    return m.b[k, t] == p.B_max
                else:
                    # Inherit battery level from end of previous stage
                    inherited_battery = self.battery_state.get(k, p.B_max)
                    return m.b[k, t] == inherited_battery
            
            b_prev = m.b[k, t-1]
            # Charging from sun (only when charging decision is active)
            charge = m.c[k, t-1] * p.B_charge
            obs_consumption = sum(m.x[k, t-1, n] for n in m.N_target) * p.B_obs
            comm_consumption = sum(m.y[k, t-1, g] for g in m.N_ground) * p.B_comm
            
            return m.b[k, t] == b_prev + charge - obs_consumption - comm_consumption
        model.battery_dynamics = pyo.Constraint(model.K, model.T_local,
                                               rule=battery_dynamics_rule)
        
        def data_dynamics_rule(m, k, t):
            if t == t_start:
                # Inherit data from previous stage (data accumulates)
                if current_stage == 0:
                    return m.d[k, t] == 0
                else:
                    # Inherit data level from end of previous stage
                    inherited_data = self.data_state.get(k, 0)
                    return m.d[k, t] == inherited_data
            
            d_prev = m.d[k, t-1]
            data_gen = sum(m.x[k, t-1, n] for n in m.N_target) * p.D_obs
            data_down = sum(m.y[k, t-1, g] for g in m.N_ground) * p.D_comm
            
            return m.d[k, t] == d_prev + data_gen - data_down
        model.data_dynamics = pyo.Constraint(model.K, model.T_local,
                                            rule=data_dynamics_rule)
        
        return model
    
    def solve(self, time_limit_per_stage_minutes=5, solver_name='gurobi'):
        """
        Solve using rolling horizon procedure with Gurobi solver
        
        Args:
            time_limit_per_stage_minutes: Time limit for each stage optimization
            solver_name: Solver to use (default: 'gurobi')
        """
        p = self.params
        total_runtime = 0
        total_vars = 0
        total_constrs = 0
        
        # Gurobi will use gurobi.lic file for licensing
        prev_assignments = None
        
        for stage in range(p.S):
                        
            # Build and solve stage model
            model = self.build_stage_model(stage, prev_assignments)
            
            # Initialize Gurobi solver
            try:
                solver = pyo.SolverFactory('gurobi')
            except:
                solver = pyo.SolverFactory('gurobi')
            
            if not solver.available():
                raise RuntimeError(f"Gurobi solver is not available. Please install it: pip install gurobipy")
            
            # Set Gurobi solver options
            solver.options['TimeLimit'] = time_limit_per_stage_minutes * 60
            solver.options['MIPGap'] = 0.02
            solver.options['Threads'] = 0  # Use all available threads
            solver.options['LogToConsole'] = 0  # Suppress console output
            
            start_time = time.time()
            results = solver.solve(model, tee=False, load_solutions=False)
            stage_runtime = time.time() - start_time
            total_runtime += stage_runtime
            
            # Track model size
            total_vars += model.nvariables()
            total_constrs += model.nconstraints()
            
            # Load solution if available
            if (results.solver.termination_condition == TerminationCondition.optimal or 
                results.solver.termination_condition == TerminationCondition.maxTimeLimit):
                model.solutions.load_from(results)
            
            # Extract solution for current stage
            if results.solver.status == SolverStatus.ok:
                # Save orbital assignments
                stage_assignments = {}
                for k in model.K:
                    for j in model.J:
                        if pyo.value(model.z[stage, k, j]) > 0.5:
                            stage_assignments[k] = j
                            break
                self.orbital_assignments[stage] = stage_assignments
                
                # Save observations and downlinks for current stage only
                stage_boundaries = p.get_stage_boundaries()
                t_start = stage_boundaries[stage]
                t_end = stage_boundaries[stage + 1]
                
                for k in model.K:
                    for t in range(t_start, t_end):
                        if t not in model.T_local:
                            continue
                        for n in model.N_target:
                            if pyo.value(model.x[k, t, n]) > 0.5:
                                self.observation_plan.append((k, t, n))
                        for g in model.N_ground:
                            if pyo.value(model.y[k, t, g]) > 0.5:
                                self.downlink_plan.append((k, t, g))
                
                # Calculate propellant used in this stage transition (per satellite)
                if stage < p.S - 1 and hasattr(model, 'u'):
                    for k in model.K:
                        for j1 in model.J:
                            for j2 in model.J:
                                if pyo.value(model.u[stage, k, j1, j2]) > 0.5:
                                    prop_cost = p.maneuver_costs[stage, k, j1, j2]
                                    self.satellite_propellant[k] = self.satellite_propellant.get(k, 0) + prop_cost
                
                # Save battery and data state at end of stage for next stage
                t_final = t_end - 1
                if t_final in model.T_local:
                    for k in model.K:
                        self.battery_state[k] = max(0, min(p.B_max, pyo.value(model.b[k, t_final])))
                        self.data_state[k] = max(0, min(p.D_max, pyo.value(model.d[k, t_final])))
                        
                
                # Stage summary
                stage_obs = sum(1 for (k, t, n) in self.observation_plan if t >= t_start and t < t_end)
                stage_down = sum(1 for (k, t, g) in self.downlink_plan if t >= t_start and t < t_end)
                print(f"  Stage {stage+1} solved: {stage_obs} observations, {stage_down} downlinks")
                
                # Update for next stage
                prev_assignments = stage_assignments
            else:
                print(f"    ❌ WARNING: Stage {stage+1} failed to solve! Status: {results.solver.status}")
                print(f"    Termination: {results.solver.termination_condition}")
        
        # Calculate final metrics
        total_observations = len(self.observation_plan)
        total_downlinks = len(self.downlink_plan)
        data_downlinked_gb = (total_downlinks * p.D_comm) / 1024
        objective_value = p.C * total_downlinks + total_observations
        
        self.results = {
            'status': 'completed',
            'objective': objective_value,
            'runtime_minutes': total_runtime / 60,
            'data_downlinked_gb': data_downlinked_gb,
            'total_observations': total_observations,
            'total_downlinks': total_downlinks,
            'satellite_propellant': self.satellite_propellant,
            'num_variables': total_vars,
            'num_constraints': total_constrs,
            'num_stages_solved': p.S
        }
        
        return self.results


if __name__ == "__main__":
    from parameters import InstanceParameters
    
    print("Testing REOSSP-RHP Solver with FREE Solver (CBC)...")
    
    test_params = InstanceParameters(
        instance_id=999,
        S=8,
        K=5,
        J_sk=20
    )
    
    solver = REOSSPRHPSolver(test_params, lookahead=1)
    print(f"Solving with rolling horizon for S={test_params.S}, K={test_params.K}, J_sk={test_params.J_sk}...")
    
    results = solver.solve(time_limit_per_stage_minutes=0.5, solver_name='gurobi')
    
    print("\nResults:")
    print(f"  Status: {results['status']}")
    print(f"  Objective: {results['objective']:.2f}")
    print(f"  Runtime: {results['runtime_minutes']:.2f} minutes")
    print(f"  Data downlinked: {results['data_downlinked_gb']:.2f} GB")
    print(f"  Observations: {results['total_observations']:.0f}")
    total_prop = sum(results['satellite_propellant'].values())
    print(f"  Propellant used: {total_prop:.2f} m/s (total across all satellites)")
