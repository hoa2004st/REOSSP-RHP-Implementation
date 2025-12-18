"""
REOSSP-Exact Formulation (with constellation reconfiguration)
Optimal solution with orbital maneuvers between stages
"""

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import numpy as np
import time
import os
from parameters import InstanceParameters


class REOSSPExactSolver:
    """
    Reconfigurable Earth Observation Satellite Scheduling Problem
    Allows orbital maneuvers between stages with delta-v costs
    """
    
    def __init__(self, params: InstanceParameters):
        self.params = params
        self.model = None
        self.results = None
        
    def build_model(self):
        """Build the Pyomo MILP model with reconfiguration"""
        p = self.params
        model = pyo.ConcreteModel(name="REOSSP_Exact")
        
        # Sets
        model.S = pyo.RangeSet(0, p.S - 1)  # Stages
        model.K = pyo.RangeSet(0, p.K - 1)  # Satellites
        model.T = pyo.RangeSet(0, p.T - 1)  # Time steps
        model.J = pyo.RangeSet(0, p.J_sk - 1)  # Orbital slots
        model.N_target = pyo.RangeSet(0, p.V_target.shape[2] - 1)  # Targets
        model.N_ground = pyo.RangeSet(0, p.V_ground.shape[2] - 1)  # Ground stations
        
        # Stage boundaries
        stage_boundaries = p.get_stage_boundaries()
        
        # Decision Variables
        
        # z[s, k, j] = 1 if satellite k is in orbital slot j during stage s
        model.z = pyo.Var(model.S, model.K, model.J, domain=pyo.Binary)
        
        # x[k, t, n] = 1 if satellite k observes target n at time t
        model.x = pyo.Var(model.K, model.T, model.N_target, domain=pyo.Binary)
        
        # y[k, t, g] = 1 if satellite k downlinks to ground station g at time t
        model.y = pyo.Var(model.K, model.T, model.N_ground, domain=pyo.Binary)
        
        # c[k, t] = 1 if satellite k is charging at time t
        model.c = pyo.Var(model.K, model.T, domain=pyo.Binary)
        
        # State Variables
        model.b = pyo.Var(model.K, model.T, domain=pyo.NonNegativeReals, bounds=(0, p.B_max))
        model.d = pyo.Var(model.K, model.T, domain=pyo.NonNegativeReals, bounds=(0, p.D_max))
        
        # Maneuver variables
        # u[s, k, j_from, j_to] = 1 if satellite k maneuvers from slot j_from to j_to between stage s and s+1
        model.u = pyo.Var(pyo.RangeSet(0, p.S - 2), model.K, model.J, model.J, domain=pyo.Binary)
        
        # Objective: maximize (C * downlinks + observations)
        # Note: Maneuvers are constrained by propellant budget, not penalized in objective
        def objective_rule(m):
            total_downlinks = sum(m.y[k, t, g] 
                                for k in m.K for t in m.T for g in m.N_ground)
            total_observations = sum(m.x[k, t, n] 
                                   for k in m.K for t in m.T for n in m.N_target)
            
            return p.C * total_downlinks + total_observations
        
        model.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)
        
        # Constraints
        
        # 1. Each satellite assigned to exactly one orbital slot per stage
        def one_slot_per_stage_rule(m, s, k):
            return sum(m.z[s, k, j] for j in m.J) == 1
        model.one_slot = pyo.Constraint(model.S, model.K, rule=one_slot_per_stage_rule)
        
        # 2. Each orbital slot occupied by at most one satellite per stage
        def one_sat_per_slot_rule(m, s, j):
            return sum(m.z[s, k, j] for k in m.K) <= 1
        model.one_sat = pyo.Constraint(model.S, model.J, rule=one_sat_per_slot_rule)
        
        # 3. Maneuver consistency: link orbital slots between stages
        def maneuver_from_rule(m, s, k):
            if s >= p.S - 1:
                return pyo.Constraint.Skip
            return sum(m.u[s, k, j1, j2] for j1 in m.J for j2 in m.J) == 1
        model.maneuver_from = pyo.Constraint(model.S, model.K, rule=maneuver_from_rule)
        
        def maneuver_link_from_rule(m, s, k, j1):
            if s >= p.S - 1:
                return pyo.Constraint.Skip
            return m.z[s, k, j1] == sum(m.u[s, k, j1, j2] for j2 in m.J)
        model.maneuver_link_from = pyo.Constraint(model.S, model.K, model.J, 
                                                  rule=maneuver_link_from_rule)
        
        def maneuver_link_to_rule(m, s, k, j2):
            if s >= p.S - 1:
                return pyo.Constraint.Skip
            return m.z[s + 1, k, j2] == sum(m.u[s, k, j1, j2] for j1 in m.J)
        model.maneuver_link_to = pyo.Constraint(model.S, model.K, model.J, 
                                               rule=maneuver_link_to_rule)
        
        # 4. Total propellant constraint
        def propellant_constraint_rule(m, k):
            total_deltav = sum(m.u[s, k, j1, j2] * p.maneuver_costs[s, k, j1, j2]
                             for s in range(p.S - 1)
                             for j1 in m.J
                             for j2 in m.J)
            return total_deltav <= p.c_max
        model.propellant = pyo.Constraint(model.K, rule=propellant_constraint_rule)
        
        # 5. Visibility constraints (simplified: assume visibility doesn't depend on orbital slot)
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
        
        # 6. One activity per satellite per time step (observe, downlink, or charge)
        def one_activity_rule(m, k, t):
            total_obs = sum(m.x[k, t, n] for n in m.N_target)
            total_comm = sum(m.y[k, t, g] for g in m.N_ground)
            return total_obs + total_comm + m.c[k, t] <= 1
        model.one_activity = pyo.Constraint(model.K, model.T, rule=one_activity_rule)
        
        # 7. Battery dynamics
        def battery_dynamics_rule(m, k, t):
            if t == 0:
                return m.b[k, t] == p.B_max
            
            b_prev = m.b[k, t-1]
            # Charging from sun (only when charging decision is active)
            charge = m.c[k, t-1] * p.B_charge
            obs_consumption = sum(m.x[k, t-1, n] for n in m.N_target) * p.B_obs
            comm_consumption = sum(m.y[k, t-1, g] for g in m.N_ground) * p.B_comm
            
            return m.b[k, t] == b_prev + charge - obs_consumption - comm_consumption
        model.battery_dynamics = pyo.Constraint(model.K, model.T, 
                                               rule=battery_dynamics_rule)
        
        # 8. Data storage dynamics
        def data_dynamics_rule(m, k, t):
            if t == 0:
                return m.d[k, t] == 0
            
            d_prev = m.d[k, t-1]
            data_gen = sum(m.x[k, t-1, n] for n in m.N_target) * p.D_obs
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
        
        factory_name = solver_map.get(solver_name, 'gurobi')
        
        # Initialize Gurobi solver
        try:
            solver = pyo.SolverFactory(factory_name)
        except:
            solver = pyo.SolverFactory('gurobi')
        
        if not solver.available():
            raise RuntimeError(f"Gurobi solver is not available. Please install it: pip install gurobipy")
        
        # Set Gurobi solver options
        solver.options['TimeLimit'] = time_limit_minutes * 60
        solver.options['MIPGap'] = 0.01
        solver.options['Threads'] = 0  # Use all available threads
        solver.options['LogToConsole'] = 0  # Suppress console output
        
        start_time = time.time()
        results = solver.solve(self.model, tee=False, load_solutions=False)
        runtime = time.time() - start_time
        
        status = results.solver.status
        termination = results.solver.termination_condition
        
        # Load solution if available
        if termination == TerminationCondition.optimal or termination == TerminationCondition.maxTimeLimit:
            self.model.solutions.load_from(results)
        
        if status == SolverStatus.ok and termination in [
            TerminationCondition.optimal, 
            TerminationCondition.maxTimeLimit
        ]:
            objective_value = pyo.value(self.model.obj)
            
            # Calculate metrics
            total_downlinks = sum(pyo.value(self.model.y[k, t, g])
                                for k in self.model.K 
                                for t in self.model.T 
                                for g in self.model.N_ground)
            data_downlinked_gb = (total_downlinks * self.params.D_comm) / 1024
            
            total_observations = sum(pyo.value(self.model.x[k, t, n])
                                   for k in self.model.K 
                                   for t in self.model.T 
                                   for n in self.model.N_target)
            
            # Calculate total propellant used
            p = self.params
            propellant_used = sum(pyo.value(self.model.u[s, k, j1, j2]) * p.maneuver_costs[s, k, j1, j2]
                                for s in range(p.S - 1)
                                for k in self.model.K
                                for j1 in self.model.J
                                for j2 in self.model.J)
            
            self.results = {
                'status': 'optimal' if termination == TerminationCondition.optimal else 'time_limit',
                'objective': objective_value,
                'runtime_minutes': runtime / 60,
                'data_downlinked_gb': data_downlinked_gb,
                'total_observations': total_observations,
                'total_downlinks': total_downlinks,
                'propellant_used': propellant_used,
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
    from parameters import InstanceParameters
    
    print("Testing REOSSP-Exact Solver with Gurobi...")
    
    test_params = InstanceParameters(
        instance_id=999,
        S=8,
        K=5,
        J_sk=20
    )
    
    solver = REOSSPExactSolver(test_params)
    print(f"Building model for S={test_params.S}, K={test_params.K}, J_sk={test_params.J_sk}...")
    
    results = solver.solve(time_limit_minutes=1, solver_name='gurobi')
    
    print("\nResults:")
    print(f"  Status: {results['status']}")
    print(f"  Objective: {results['objective']:.2f}")
    print(f"  Runtime: {results['runtime_minutes']:.2f} minutes")
    print(f"  Data downlinked: {results['data_downlinked_gb']:.2f} GB")
    print(f"  Observations: {results['total_observations']:.0f}")
    print(f"  Propellant used: {results['propellant_used']:.2f} m/s")
