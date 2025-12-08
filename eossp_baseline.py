"""
EOSSP Baseline Formulation (without reconfiguration)
Simplified MILP implementation using Pyomo
"""

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import numpy as np
import time
import os
from parameters import InstanceParameters
import api_key


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
        """Build the Pyomo MILP model"""
        p = self.params
        model = pyo.ConcreteModel(name="EOSSP_Baseline")
        
        # Sets
        model.K = pyo.RangeSet(0, p.K - 1)  # Satellites
        model.T = pyo.RangeSet(0, p.T - 1)  # Time steps
        model.N_target = pyo.RangeSet(0, p.V_target.shape[2] - 1)  # Targets
        model.N_ground = pyo.RangeSet(0, p.V_ground.shape[2] - 1)  # Ground stations
        
        # Decision Variables
        # x[k, t, n] = 1 if satellite k observes target n at time t
        model.x = pyo.Var(model.K, model.T, model.N_target, domain=pyo.Binary)
        
        # y[k, t, g] = 1 if satellite k downlinks to ground station g at time t
        model.y = pyo.Var(model.K, model.T, model.N_ground, domain=pyo.Binary)
        
        # c[k, t] = 1 if satellite k is charging at time t
        model.c = pyo.Var(model.K, model.T, domain=pyo.Binary)
        
        # State Variables (continuous for efficiency)
        # b[k, t] = battery level of satellite k at time t
        model.b = pyo.Var(model.K, model.T, domain=pyo.NonNegativeReals, bounds=(0, p.B_max))
        
        # d[k, t] = data stored on satellite k at time t
        model.d = pyo.Var(model.K, model.T, domain=pyo.NonNegativeReals, bounds=(0, p.D_max))
        
        # Objective: maximize (C * downlinks + observations)
        def objective_rule(m):
            total_downlinks = sum(m.y[k, t, g] 
                                for k in m.K for t in m.T for g in m.N_ground)
            total_observations = sum(m.x[k, t, n] 
                                   for k in m.K for t in m.T for n in m.N_target)
            return p.C * total_downlinks + total_observations
        
        model.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)
        
        # Constraints
        
        # 1. Visibility constraints: can only observe/downlink when visible
        def visibility_target_rule(m, k, t, n):
            if not p.V_target[k, t, n]:
                return m.x[k, t, n] == 0
            return pyo.Constraint.Skip
        model.vis_target = pyo.Constraint(model.K, model.T, model.N_target, 
                                         rule=visibility_target_rule)
        
        def visibility_ground_rule(m, k, t, g):
            if not p.V_ground[k, t, g]:
                return m.y[k, t, g] == 0
            return pyo.Constraint.Skip
        model.vis_ground = pyo.Constraint(model.K, model.T, model.N_ground, 
                                         rule=visibility_ground_rule)
        
        def visibility_sun_rule(m, k, t):
            if not p.V_sun[k, t]:
                return m.c[k, t] == 0
            return pyo.Constraint.Skip
        model.vis_sun = pyo.Constraint(model.K, model.T, 
                                      rule=visibility_sun_rule)
        
        # 2. One activity per satellite per time step (observe, downlink, or charge)
        def one_activity_rule(m, k, t):
            total_obs = sum(m.x[k, t, n] for n in m.N_target)
            total_comm = sum(m.y[k, t, g] for g in m.N_ground)
            return total_obs + total_comm + m.c[k, t] <= 1
        model.one_activity = pyo.Constraint(model.K, model.T, rule=one_activity_rule)
        
        # 3. Battery dynamics
        def battery_dynamics_rule(m, k, t):
            if t == 0:
                # Initial battery level
                return m.b[k, t] == p.B_max  # Start at 100%
            
            b_prev = m.b[k, t-1]
            
            # Charging from sun (only when charging decision is active)
            charge = m.c[k, t-1] * p.B_charge
            
            # Consumption from observation
            obs_consumption = sum(m.x[k, t-1, n] for n in m.N_target) * p.B_obs
            
            # Consumption from communication
            comm_consumption = sum(m.y[k, t-1, g] for g in m.N_ground) * p.B_comm
            
            return m.b[k, t] == b_prev + charge - obs_consumption - comm_consumption
        
        model.battery_dynamics = pyo.Constraint(model.K, model.T, 
                                               rule=battery_dynamics_rule)
        
        # 4. Data storage dynamics
        def data_dynamics_rule(m, k, t):
            if t == 0:
                # Initial data storage
                return m.d[k, t] == 0
            
            d_prev = m.d[k, t-1]
            
            # Data generated from observations
            data_gen = sum(m.x[k, t-1, n] for n in m.N_target) * p.D_obs
            
            # Data downlinked
            data_down = sum(m.y[k, t-1, g] for g in m.N_ground) * p.D_comm
            
            return m.d[k, t] == d_prev + data_gen - data_down
        
        model.data_dynamics = pyo.Constraint(model.K, model.T, 
                                            rule=data_dynamics_rule)
        
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
        
        # Get Gurobi license credentials and write to gurobi.env file
        key_instance = api_key.key()
        gurobi_options = key_instance.get_options()
        
        # Write credentials to gurobi.env file in current directory
        with open('gurobi.env', 'w') as f:
            f.write(f"WLSACCESSID={gurobi_options['WLSACCESSID']}\n")
            f.write(f"WLSSECRET={gurobi_options['WLSSECRET']}\n")
            f.write(f"LICENSEID={gurobi_options['LICENSEID']}\n")
        
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
        
        # Solve
        print(f"Starting solver {factory_name}...")
        print(f"Model statistics:")
        print(f"  Variables: {self.model.nvariables()}")
        print(f"  Constraints: {self.model.nconstraints()}")
        print(f"  Time limit: {time_limit_minutes} minutes")
        print("Solving (this may take a while)...")
        
        start_time = time.time()
        
        # Set load_solutions=False to handle infeasibility
        results = solver.solve(self.model, tee=True, load_solutions=False)
        runtime = time.time() - start_time
        print(f"Solver finished in {runtime:.2f} seconds")
        
        # Check if solution exists before loading
        if (results.solver.termination_condition == TerminationCondition.optimal or
            results.solver.termination_condition == TerminationCondition.maxTimeLimit):
            # Load the solution
            self.model.solutions.load_from(results)
        else:
            print(f"Warning: Solver terminated with condition: {results.solver.termination_condition}")
        
        # Extract results
        status = results.solver.status
        termination = results.solver.termination_condition
        
        if termination == TerminationCondition.infeasible:
            print("Model is INFEASIBLE - no solution exists with current constraints")
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
            total_downlinks = sum(pyo.value(self.model.y[k, t, g])
                                for k in self.model.K 
                                for t in self.model.T 
                                for g in self.model.N_ground)
            data_downlinked_mb = total_downlinks * self.params.D_comm
            data_downlinked_gb = data_downlinked_mb / 1024
            
            # Count observations
            total_observations = sum(pyo.value(self.model.x[k, t, n])
                                   for k in self.model.K 
                                   for t in self.model.T 
                                   for n in self.model.N_target)
            
            self.results = {
                'status': 'optimal' if termination == TerminationCondition.optimal else 'time_limit',
                'objective': objective_value,
                'runtime_minutes': runtime / 60,
                'data_downlinked_gb': data_downlinked_gb,
                'total_observations': total_observations,
                'total_downlinks': total_downlinks,
                'propellant_used': 0.0  # No maneuvers in baseline
            }
        else:
            self.results = {
                'status': 'failed',
                'objective': 0,
                'runtime_minutes': runtime / 60,
                'data_downlinked_gb': 0,
                'total_observations': 0,
                'total_downlinks': 0,
                'propellant_used': 0.0
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
