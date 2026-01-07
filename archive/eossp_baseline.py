"""
EOSSP Baseline Formulation (without reconfiguration)
Implements baseline MILP from paper Section 3.2 (constraints 4a-6c)
Satellites remain in fixed initial orbital slots throughout mission

Key simplifications vs REOSSP:
- No orbital reconfiguration (x variables not needed)
- Satellites stay at j=0 (initial slot) for all stages
- Simpler visibility constraints (fixed slot position)
"""

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import numpy as np
import time
import os
from parameters import InstanceParameters


class EOSSPSolver:
    """
    Earth Observation Satellite Scheduling Problem (baseline)
    No orbital reconfiguration - satellites stay in initial orbits
    """
    
    def __init__(self, params: InstanceParameters):
        self.params = params
        self.model = None
        self.results = None
        
    def build_model(self):
        """
        Build the Pyomo MILP model for EOSSP baseline
        Implements constraints (4a-6c) from paper Section 3.2
        """
        p = self.params
        model = pyo.ConcreteModel(name="EOSSP_Baseline")
        
        # Convert 4D visibility [S, K, T_per_stage, J, targets] to global time indexing
        # For EOSSP: satellites stay at slot j=0 throughout mission
        T_per_stage = p.T // p.S
        fixed_slot = 0  # All satellites stay at initial slot 0
        
        # Flatten visibility to [K, T_global, targets/ground/sun]
        n_targets = p.V_target.shape[4]
        n_ground = p.V_ground.shape[4]
        V_target_flat = np.zeros((p.K, p.T, n_targets), dtype=bool)
        V_ground_flat = np.zeros((p.K, p.T, n_ground), dtype=bool)
        V_sun_flat = np.zeros((p.K, p.T), dtype=bool)
        
        for s in range(p.S):
            t_start = s * T_per_stage
            for t_local in range(T_per_stage):
                t_global = t_start + t_local
                if t_global < p.T:
                    V_target_flat[:, t_global, :] = p.V_target[s, :, t_local, fixed_slot, :]
                    V_ground_flat[:, t_global, :] = p.V_ground[s, :, t_local, fixed_slot, :]
                    V_sun_flat[:, t_global] = p.V_sun[s, :, t_local, fixed_slot]
        
        # Sets (1-based indexing to match paper notation)
        model.K = pyo.RangeSet(1, p.K)  # Satellites k ∈ {1,...,K}
        model.T = pyo.RangeSet(1, p.T)  # Time steps t ∈ {1,...,T}
        model.P = pyo.RangeSet(1, n_targets)  # Targets p ∈ {1,...,P}
        model.G = pyo.RangeSet(1, n_ground)  # Ground stations g ∈ {1,...,G}
        
        # Decision Variables (matching paper notation)
        # Paper Eq (1a): y_ktp ∈ {0,1} - observation of target p by satellite k at time t
        model.y = pyo.Var(model.K, model.T, model.P, domain=pyo.Binary)
        
        # Paper Eq (1b): q_ktg ∈ {0,1} - downlink to ground station g by satellite k at time t
        model.q = pyo.Var(model.K, model.T, model.G, domain=pyo.Binary)
        
        # Paper Eq (1c): h_kt ∈ {0,1} - charging from sun by satellite k at time t
        model.h = pyo.Var(model.K, model.T, domain=pyo.Binary)
        
        # State Variables (continuous)
        # Paper Eq (1d): d_kt ∈ [D_min, D_max] - data stored on satellite k at time t
        model.d = pyo.Var(model.K, model.T, domain=pyo.NonNegativeReals, bounds=(0, p.D_max))
        
        # Paper Eq (1e): b_kt ∈ [B_min, B_max] - battery level of satellite k at time t
        model.b = pyo.Var(model.K, model.T, domain=pyo.NonNegativeReals, bounds=(0, p.B_max))
        
        # =====================================================================
        # OBJECTIVE FUNCTION - Paper Eq (2)
        # Maximize: sum over k,t,g of (C * q_ktg) + sum over k,t,p of y_ktp
        # =====================================================================
        def objective_rule(m):
            total_downlinks = sum(m.q[k, t, g] 
                                for k in m.K for t in m.T for g in m.G)
            total_observations = sum(m.y[k, t, p] 
                                   for k in m.K for t in m.T for p in m.P)
            return p.C * total_downlinks + total_observations
        
        model.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)
        
        # =====================================================================
        # CONSTRAINTS
        # =====================================================================
        
        # ---------------------------------------------------------------------
        # Visibility Constraints - Paper Eq (4a, 4b, 4c)
        # y_ktp, q_ktg, h_kt can only be 1 when visibility window is available
        # ---------------------------------------------------------------------
        def visibility_target_rule(m, k, t, p):
            # Paper Eq (4a): y_ktp <= V_ktp (target visibility)
            if not V_target_flat[k-1, t-1, p-1]:  # Convert to 0-based indexing
                return m.y[k, t, p] == 0
            return pyo.Constraint.Skip
        model.vis_target = pyo.Constraint(model.K, model.T, model.P, 
                                         rule=visibility_target_rule)
        
        def visibility_ground_rule(m, k, t, g):
            # Paper Eq (4b): q_ktg <= W_ktg (ground station visibility)
            if not V_ground_flat[k-1, t-1, g-1]:
                return m.q[k, t, g] == 0
            return pyo.Constraint.Skip
        model.vis_ground = pyo.Constraint(model.K, model.T, model.G, 
                                         rule=visibility_ground_rule)
        
        def visibility_sun_rule(m, k, t):
            # Paper Eq (4c): h_kt <= H_kt (sun visibility for charging)
            if not V_sun_flat[k-1, t-1]:
                return m.h[k, t] == 0
            return pyo.Constraint.Skip
        model.vis_sun = pyo.Constraint(model.K, model.T, 
                                      rule=visibility_sun_rule)
        
        # ---------------------------------------------------------------------
        # One Activity Per Time Step - Paper Eq (4d)
        # sum_p y_ktp + sum_g q_ktg + h_kt <= 1
        # (observe OR downlink OR charge, at most one activity)
        # ---------------------------------------------------------------------
        def one_activity_rule(m, k, t):
            total_obs = sum(m.y[k, t, p] for p in m.P)
            total_comm = sum(m.q[k, t, g] for g in m.G)
            return total_obs + total_comm + m.h[k, t] <= 1
        model.one_activity = pyo.Constraint(model.K, model.T, rule=one_activity_rule)
        
        # ---------------------------------------------------------------------
        # Data Storage Dynamics - Paper Eq (5a, 5b, 5c)
        # Tracks data accumulated from observations and removed by downlinks
        # ---------------------------------------------------------------------
        def data_dynamics_rule(m, k, t):
            if t == 1:
                # Initial condition: d_k1 = D_min (start with empty storage)
                return m.d[k, t] == 0
            
            # Paper Eq (5a): d_kt+1 = d_kt + D_obs*sum_p(y_ktp) - D_comm*sum_g(q_ktg)
            d_prev = m.d[k, t-1]
            
            # Data generated from observations
            data_gen = sum(m.y[k, t-1, p] for p in m.P) * p.D_obs
            
            # Data downlinked
            data_down = sum(m.q[k, t-1, g] for g in m.G) * p.D_comm
            
            return m.d[k, t] == d_prev + data_gen - data_down
        
        model.data_dynamics = pyo.Constraint(model.K, model.T, 
                                            rule=data_dynamics_rule)
        
        def data_upper_bound_rule(m, k, t):
            # Paper Eq (5b): d_kt + D_obs*sum_p(y_ktp) <= D_max
            # (cannot generate more data than storage capacity)
            data_gen = sum(m.y[k, t, p] for p in m.P) * p.D_obs
            return m.d[k, t] + data_gen <= p.D_max
        
        model.data_upper = pyo.Constraint(model.K, model.T, 
                                         rule=data_upper_bound_rule)
        
        def data_lower_bound_rule(m, k, t):
            # Paper Eq (5c): d_kt - D_comm*sum_g(q_ktg) >= D_min
            # (cannot downlink more data than stored)
            data_down = sum(m.q[k, t, g] for g in m.G) * p.D_comm
            return m.d[k, t] - data_down >= 0
        
        model.data_lower = pyo.Constraint(model.K, model.T, 
                                         rule=data_lower_bound_rule)
        
        # ---------------------------------------------------------------------
        # Battery Dynamics - Paper Eq (6a, 6b, 6c)
        # Tracks battery charge/discharge from sun, observations, downlinks
        # ---------------------------------------------------------------------
        def battery_dynamics_rule(m, k, t):
            if t == 1:
                # Initial condition: b_k1 = B_max (start with full battery)
                # Paper: No initial maneuver cost for EOSSP (satellites don't move)
                return m.b[k, t] == p.B_max
            
            # Paper Eq (6a): b_kt+1 = b_kt + B_charge*h_kt - B_obs*sum_p(y_ktp) - B_comm*sum_g(q_ktg)
            b_prev = m.b[k, t-1]
            
            # Charging from sun
            charge = m.h[k, t-1] * p.B_charge
            
            # Consumption from observation
            obs_consumption = sum(m.y[k, t-1, p] for p in m.P) * p.B_obs
            
            # Consumption from communication
            comm_consumption = sum(m.q[k, t-1, g] for g in m.G) * p.B_comm

            # Idle power consumption (always occurs)
            idle_consumption = p.B_time
            
            return m.b[k, t] == b_prev + charge - obs_consumption - comm_consumption - idle_consumption
        
        model.battery_dynamics = pyo.Constraint(model.K, model.T, 
                                               rule=battery_dynamics_rule)
        
        def battery_upper_bound_rule(m, k, t):
            # Paper Eq (6b): b_kt + B_charge*h_kt <= B_max
            # (battery cannot exceed maximum capacity)
            charge = m.h[k, t] * p.B_charge
            return m.b[k, t] + charge <= p.B_max
        
        model.battery_upper = pyo.Constraint(model.K, model.T, 
                                            rule=battery_upper_bound_rule)
        
        def battery_lower_bound_rule(m, k, t):
            # Paper Eq (6c): b_kt - B_obs*sum_p(y_ktp) - B_comm*sum_g(q_ktg) >= B_min
            # (battery cannot drop below minimum level)
            obs_consumption = sum(m.y[k, t, p] for p in m.P) * p.B_obs
            comm_consumption = sum(m.q[k, t, g] for g in m.G) * p.B_comm
            idle_consumption = p.B_time
            return m.b[k, t] - obs_consumption - comm_consumption - idle_consumption >= 0
        
        model.battery_lower = pyo.Constraint(model.K, model.T, 
                                            rule=battery_lower_bound_rule)
        
        self.model = model
        return model
    
    def solve(self, time_limit_minutes=60, solver_name='gurobi'):
        """
        Solve the model using Gurobi solver
        
        Args:
            time_limit_minutes: Time limit in minutes
            solver_name: Solver to use (default: 'gurobi')
        
        Returns:
            dict: Results including objective value, runtime, and solution status
        """
        if self.model is None:
            self.build_model()
        
        # Gurobi will use gurobi.lic file for licensing
        # Map solver names to factory names
        solver_map = {
            'gurobi': 'gurobi',
            'gurobi_direct': 'gurobi_direct',
            'gurobi_persistent': 'gurobi_persistent',
        }
        
        factory_name = solver_map.get(solver_name, solver_name)
        
        # Initialize Gurobi solver
        try:
            solver = pyo.SolverFactory(factory_name)
        except:
            # Fallback to standard interface
            solver = pyo.SolverFactory('gurobi')
        
        if not solver.available():
            raise RuntimeError(f"Solver '{factory_name}' is not available. Please install it first.")
        
        # Set Gurobi solver options
        solver.options['TimeLimit'] = time_limit_minutes * 60
        solver.options['MIPGap'] = 0.01
        solver.options['Threads'] = 0  # Use all available threads
        solver.options['LogToConsole'] = 0  # Suppress console output
        
        start_time = time.time()
        
        # Solve with minimal output
        results = solver.solve(self.model, tee=False, load_solutions=False)
        runtime = time.time() - start_time
        
        # Check if solution exists before loading
        if (results.solver.termination_condition == TerminationCondition.optimal or
            results.solver.termination_condition == TerminationCondition.maxTimeLimit):
            # Load the solution
            self.model.solutions.load_from(results)
        
        # Extract results
        status = results.solver.status
        termination = results.solver.termination_condition
        
        if termination == TerminationCondition.infeasible:
            self.results = {
                'status': 'infeasible',
                'objective': 0,
                'runtime_minutes': runtime / 60,
                'data_downlinked_gb': 0,
                'total_observations': 0,
                'total_downlinks': 0,
                'propellant_used': 0.0
            }
        elif status == SolverStatus.ok and termination in [
            TerminationCondition.optimal, 
            TerminationCondition.maxTimeLimit
        ]:
            objective_value = pyo.value(self.model.obj)
            
            # Calculate figure of merit: total data downlinked
            total_downlinks = sum(pyo.value(self.model.q[k, t, g])
                                for k in self.model.K 
                                for t in self.model.T 
                                for g in self.model.G)
            data_downlinked_mb = total_downlinks * self.params.D_comm
            data_downlinked_gb = data_downlinked_mb / 1024
            
            # Count observations
            total_observations = sum(pyo.value(self.model.y[k, t, p])
                                   for k in self.model.K 
                                   for t in self.model.T 
                                   for p in self.model.P)
            
            self.results = {
                'status': 'optimal' if termination == TerminationCondition.optimal else 'time_limit',
                'objective': objective_value,
                'runtime_minutes': runtime / 60,
                'data_downlinked_gb': data_downlinked_gb,
                'total_observations': total_observations,
                'total_downlinks': total_downlinks,
                'propellant_used': 0.0,  # No maneuvers in baseline
                'num_variables': self.model.nvariables(),
                'num_constraints': self.model.nconstraints(),
                'num_nonzeros': sum(c.body.polynomial_degree() if hasattr(c.body, 'polynomial_degree') else 0 
                                   for c in self.model.component_data_objects(pyo.Constraint, active=True))
            }
        else:
            self.results = {
                'status': 'failed',
                'objective': 0,
                'runtime_minutes': runtime / 60,
                'data_downlinked_gb': 0,
                'total_observations': 0,
                'total_downlinks': 0,
                'propellant_used': 0.0,
                'num_variables': self.model.nvariables(),
                'num_constraints': self.model.nconstraints(),
                'num_nonzeros': 0
            }
        
        return self.results


if __name__ == "__main__":
    # Test with a small instance
    from parameters import InstanceParameters
    
    print("Testing EOSSP Baseline Solver with FREE Solver (CBC)...")
    
    # Create small test instance
    test_params = InstanceParameters(
        instance_id=999,
        S=8,
        K=5,
        J_sk=20
    )
    
    solver = EOSSPSolver(test_params)
    print(f"Building model for S={test_params.S}, K={test_params.K}, J_sk={test_params.J_sk}...")
    
    results = solver.solve(time_limit_minutes=1, solver_name='gurobi')
    
    print("\nResults:")
    print(f"  Status: {results['status']}")
    print(f"  Objective: {results['objective']:.2f}")
    print(f"  Runtime: {results['runtime_minutes']:.2f} minutes")
    print(f"  Data downlinked: {results['data_downlinked_gb']:.2f} GB")
    print(f"  Observations: {results['total_observations']:.0f}")
    print(f"  Downlinks: {results['total_downlinks']:.0f}")
