"""
REOSSP-RHP: Rolling Horizon Procedure
Fast heuristic approach for large instances
"""

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import numpy as np
import time
from parameters import InstanceParameters


class REOSSPRHPSolver:
    """
    Rolling Horizon Procedure for REOSSP
    Solves the problem stage-by-stage with lookahead L=1
    Much faster than exact method but potentially suboptimal
    """
    
    def __init__(self, params: InstanceParameters, lookahead=1):
        self.params = params
        self.lookahead = lookahead  # L=1 means optimize current + next stage
        self.results = None
        
        # Track solution across stages
        self.orbital_assignments = {}  # {stage: {sat: slot}}
        self.observation_plan = []
        self.downlink_plan = []
        self.total_propellant = 0
        
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
        model.b = pyo.Var(model.K, model.T_local, domain=pyo.NonNegativeReals, bounds=(0, p.B_max))
        model.d = pyo.Var(model.K, model.T_local, domain=pyo.NonNegativeReals, bounds=(0, p.D_max))
        
        # Maneuver variables (only between stages in this window)
        maneuver_stages = [s for s in stage_range if s < p.S - 1 and s < current_stage + stages_to_optimize - 1]
        if maneuver_stages:
            model.S_maneuver = pyo.Set(initialize=maneuver_stages)
            model.u = pyo.Var(model.S_maneuver, model.K, model.J, model.J, domain=pyo.Binary)
        
        # Objective: maximize activities in current stage (prioritize near-term)
        def objective_rule(m):
            # Current stage time steps
            current_stage_times = range(stage_boundaries[current_stage], 
                                       stage_boundaries[current_stage + 1])
            
            total_downlinks = sum(m.y[k, t, g] 
                                for k in m.K 
                                for t in current_stage_times if t in m.T_local
                                for g in m.N_ground)
            total_observations = sum(m.x[k, t, n] 
                                   for k in m.K 
                                   for t in current_stage_times if t in m.T_local
                                   for n in m.N_target)
            
            # Add lookahead with reduced weight
            if stages_to_optimize > 1:
                lookahead_times = range(stage_boundaries[current_stage + 1], t_end)
                lookahead_weight = 0.5
                
                lookahead_downlinks = sum(m.y[k, t, g] 
                                        for k in m.K 
                                        for t in lookahead_times if t in m.T_local
                                        for g in m.N_ground)
                lookahead_observations = sum(m.x[k, t, n] 
                                           for k in m.K 
                                           for t in lookahead_times if t in m.T_local
                                           for n in m.N_target)
                
                total_downlinks += lookahead_weight * lookahead_downlinks
                total_observations += lookahead_weight * lookahead_observations
            
            return p.C * total_downlinks + total_observations
        
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
            
            # Cumulative propellant constraint
            def propellant_rule(m, k):
                stage_propellant = sum(m.u[s, k, j1, j2] * p.maneuver_costs[s, k, j1, j2]
                                     for s in m.S_maneuver
                                     for j1 in m.J for j2 in m.J)
                # Add propellant already used in previous stages
                prev_propellant = sum(self.total_propellant for _ in [0]) if hasattr(self, 'total_propellant') else 0
                return stage_propellant <= p.c_max - prev_propellant
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
        
        def one_activity_rule(m, k, t):
            total_obs = sum(m.x[k, t, n] for n in m.N_target)
            total_comm = sum(m.y[k, t, g] for g in m.N_ground)
            return total_obs + total_comm <= 1
        model.one_activity = pyo.Constraint(model.K, model.T_local, rule=one_activity_rule)
        
        # 5. Battery and data dynamics
        def battery_dynamics_rule(m, k, t):
            if t == t_start:
                # Initial or inherited battery level
                return m.b[k, t] == p.B_max * 0.9
            
            b_prev = m.b[k, t-1]
            charge = p.B_charge if p.V_sun[k, t-1] else 0
            obs_consumption = sum(m.x[k, t-1, n] for n in m.N_target) * p.B_obs
            comm_consumption = sum(m.y[k, t-1, g] for g in m.N_ground) * p.B_comm
            
            return m.b[k, t] == b_prev + charge - obs_consumption - comm_consumption
        model.battery_dynamics = pyo.Constraint(model.K, model.T_local,
                                               rule=battery_dynamics_rule)
        
        def data_dynamics_rule(m, k, t):
            if t == t_start:
                return m.d[k, t] == 0
            
            d_prev = m.d[k, t-1]
            data_gen = sum(m.x[k, t-1, n] for n in m.N_target) * p.D_obs
            data_down = sum(m.y[k, t-1, g] for g in m.N_ground) * p.D_comm
            
            return m.d[k, t] == d_prev + data_gen - data_down
        model.data_dynamics = pyo.Constraint(model.K, model.T_local,
                                            rule=data_dynamics_rule)
        
        return model
    
    def solve(self, time_limit_per_stage_minutes=5, solver_name='highs'):
        """
        Solve using rolling horizon procedure with HiGHS solver
        
        Args:
            time_limit_per_stage_minutes: Time limit for each stage optimization
            solver_name: Solver to use (default: 'highs')
        """
        p = self.params
        total_runtime = 0
        
        prev_assignments = None
        
        for stage in range(p.S):
            print(f"  Solving stage {stage + 1}/{p.S}...")
            
            # Build and solve stage model
            model = self.build_stage_model(stage, prev_assignments)
            
            # Use HiGHS solver
            try:
                solver = pyo.SolverFactory('highs')
            except:
                solver = pyo.SolverFactory('highs')
            
            if not solver.available():
                raise RuntimeError(f"HiGHS solver is not available. Please install it: pip install highspy")
            
            # Set HiGHS solver options
            solver.options['time_limit'] = time_limit_per_stage_minutes * 60
            solver.options['mip_rel_gap'] = 0.02
            solver.options['parallel'] = 'on'
            
            start_time = time.time()
            results = solver.solve(model, tee=False, load_solutions=False)
            stage_runtime = time.time() - start_time
            total_runtime += stage_runtime
            
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
                
                # Calculate propellant used in this stage transition
                if stage < p.S - 1 and hasattr(model, 'u'):
                    for k in model.K:
                        for j1 in model.J:
                            for j2 in model.J:
                                if pyo.value(model.u[stage, k, j1, j2]) > 0.5:
                                    self.total_propellant += p.maneuver_costs[stage, k, j1, j2]
                
                # Update for next stage
                prev_assignments = stage_assignments
            else:
                print(f"    Warning: Stage {stage} failed to solve")
        
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
            'propellant_used': self.total_propellant
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
    
    results = solver.solve(time_limit_per_stage_minutes=0.5, solver_name='highs')
    
    print("\nResults:")
    print(f"  Status: {results['status']}")
    print(f"  Objective: {results['objective']:.2f}")
    print(f"  Runtime: {results['runtime_minutes']:.2f} minutes")
    print(f"  Data downlinked: {results['data_downlinked_gb']:.2f} GB")
    print(f"  Observations: {results['total_observations']:.0f}")
    print(f"  Propellant used: {results['propellant_used']:.2f} m/s")
