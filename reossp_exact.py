"""
REOSSP-Exact Formulation (with constellation reconfiguration)
Implements complete REOSSP model from paper Section 3.3 (constraints 7a-14d)
Allows orbital reconfiguration between stages with propellant budget constraints

Key additions vs EOSSP:
- Orbital reconfiguration variables x^s_{kij} (satellite k moves from slot i to j in stage s)
- Slot-dependent visibility (V^s_{ktjp} depends on orbital slot j)
- Propellant budget constraints (10c)
- Battery cost for maneuvers (14d)
- Stage-to-stage transitions with maneuver costs (13b, 14c)
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
        """
        Build the Pyomo MILP model for REOSSP-Exact
        Implements constraints (7a-14d) from paper Section 3.3
        """
        p = self.params
        model = pyo.ConcreteModel(name="REOSSP_Exact")
        
        # Calculate time steps per stage
        T_per_stage = p.T // p.S
        
        # Extract dimensions from 4D visibility [S, K, T_per_stage, J, targets]
        n_targets = p.V_target.shape[4]
        n_ground = p.V_ground.shape[4]
        
        # =====================================================================
        # SETS - Paper notation: s ∈ S, k ∈ K, t ∈ T, j ∈ J, p ∈ P, g ∈ G
        # Using 1-based indexing to match paper
        # =====================================================================
        model.S = pyo.RangeSet(1, p.S)  # Stages s ∈ {1,...,S}
        model.K = pyo.RangeSet(1, p.K)  # Satellites k ∈ {1,...,K}
        model.T_local = pyo.RangeSet(1, T_per_stage)  # Local time within stage t ∈ {1,...,T_s}
        model.J = pyo.RangeSet(1, p.J_sk)  # Orbital slots j ∈ {1,...,J}
        model.P = pyo.RangeSet(1, n_targets)  # Targets p ∈ {1,...,P}
        model.G = pyo.RangeSet(1, n_ground)  # Ground stations g ∈ {1,...,G}
        
        # Initial slots (J_0) - satellites start at slot 1
        model.J_0 = pyo.RangeSet(1, 1)  # Initial slot set (only slot 1)
        
        # =====================================================================
        # DECISION VARIABLES - Paper Section 3.3.1, Eq (7a-7f)
        # =====================================================================
        
        # Paper Eq (7a): x^s_{kij} ∈ {0,1} - orbital reconfiguration decision
        # = 1 if satellite k moves from slot i to slot j entering stage s
        model.x = pyo.Var(model.S, model.K, model.J, model.J, domain=pyo.Binary)
        
        # Paper Eq (7b): y^s_{ktp} ∈ {0,1} - observation decision
        # = 1 if satellite k observes target p at local time t in stage s
        model.y = pyo.Var(model.S, model.K, model.T_local, model.P, domain=pyo.Binary)
        
        # Paper Eq (7c): q^s_{ktg} ∈ {0,1} - downlink decision
        # = 1 if satellite k downlinks to ground station g at local time t in stage s
        model.q = pyo.Var(model.S, model.K, model.T_local, model.G, domain=pyo.Binary)
        
        # Paper Eq (7d): h^s_{kt} ∈ {0,1} - charging decision
        # = 1 if satellite k charges from sun at local time t in stage s
        model.h = pyo.Var(model.S, model.K, model.T_local, domain=pyo.Binary)
        
        # Paper Eq (7e): d^s_{kt} ∈ [D_min, D_max] - data storage state
        # Amount of data stored on satellite k at local time t in stage s
        model.d = pyo.Var(model.S, model.K, model.T_local, domain=pyo.NonNegativeReals, 
                         bounds=(0, p.D_max))
        
        # Paper Eq (7f): b^s_{kt} ∈ [B_min, B_max] - battery state
        # Battery charge level of satellite k at local time t in stage s
        model.b = pyo.Var(model.S, model.K, model.T_local, domain=pyo.NonNegativeReals, 
                         bounds=(0, p.B_max))
        
        # =====================================================================
        # OBJECTIVE FUNCTION - Paper Eq (8)
        # Maximize: sum_{s,k,t,g} C*q^s_{ktg} + sum_{s,k,t,p} y^s_{ktp}
        # =====================================================================
        def objective_rule(m):
            total_downlinks = sum(m.q[s, k, t, g] 
                                for s in m.S for k in m.K for t in m.T_local for g in m.G)
            total_observations = sum(m.y[s, k, t, p] 
                                   for s in m.S for k in m.K for t in m.T_local for p in m.P)
            return p.C * total_downlinks + total_observations
        
        model.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)
        
        # =====================================================================
        # CONSTRAINTS
        # =====================================================================
        
        # ---------------------------------------------------------------------
        # Orbital Reconfiguration Constraints - Paper Eq (10a-10c)
        # Control satellite movement between orbital slots
        # ---------------------------------------------------------------------
        
        def initial_slot_assignment_rule(m, k):
            # Paper Eq (10a): sum_{j in J^1} x^1_{k,i_init,j} = 1
            # Each satellite must move from initial slot (slot 1) to exactly one slot in stage 1
            i_init = 1  # All satellites start at slot 1
            return sum(m.x[1, k, i_init, j] for j in m.J) == 1
        model.initial_slot = pyo.Constraint(model.K, rule=initial_slot_assignment_rule)
        
        def slot_continuity_rule(m, s, k, i):
            # Paper Eq (10b): sum_{j in J^{s+1}} x^{s+1}_{k,i,j} = sum_{j' in J^{s-1}} x^s_{k,j',i}
            # If satellite is in slot j at stage s, it must move from j to some slot in stage s+1
            if s >= p.S:  # No transition after last stage
                return pyo.Constraint.Skip
            
            # Satellites entering slot j in stage s
            if s == 1:
                # First stage: coming from initial slot 1
                incoming = m.x[s, k, 1, i]
            else:
                # Later stages: coming from any slot in previous stage
                incoming = sum(m.x[s, k, j_prime, i] for j_prime in m.J)
            
            # Satellites leaving slot j to go to next stage
            outgoing = sum(m.x[s+1, k, i, j] for j in m.J)
            
            return outgoing == incoming
        
        model.slot_continuity = pyo.Constraint(model.S, model.K, model.J, 
                                              rule=slot_continuity_rule)
        
        def propellant_budget_rule(m, k):
            # Paper Eq (10c): sum_{s,i,j} c^s_{kij} * x^s_{kij} <= c^max_k
            # Total propellant used by satellite k must not exceed budget
            total_cost = 0
            for s in m.S:
                if s == 1:
                    # Stage 1: maneuver from initial slot (slot 1)
                    for j in m.J:
                        total_cost += m.x[s, k, 1, j] * p.maneuver_costs[s-1, k-1, 0, j-1]
                else:
                    # Later stages: maneuver from previous stage slot
                    for i in m.J:
                        for j in m.J:
                            total_cost += m.x[s, k, i, j] * p.maneuver_costs[s-1, k-1, i-1, j-1]
            
            return total_cost <= p.c_max
        
        model.propellant_budget = pyo.Constraint(model.K, rule=propellant_budget_rule)
        
        # ---------------------------------------------------------------------
        # Visibility Constraints - Paper Eq (11a-11d)
        # Actions depend on slot-dependent visibility windows
        # ---------------------------------------------------------------------
        
        def visibility_target_rule(m, s, k, t, target_idx):
            # Paper Eq (11a): y^s_{ktp} <= sum_{i,j} V^s_{ktjp} * x^s_{kij}
            # Can only observe target p if visible from current orbital slot
            if s == 1:
                visibility_sum = sum(int(p.V_target[s-1, k-1, t-1, j-1, target_idx-1]) * m.x[s, k, 1, j] 
                                    for j in m.J)
            else:
                visibility_sum = sum(int(p.V_target[s-1, k-1, t-1, j-1, target_idx-1]) * m.x[s, k, i, j] 
                                    for i in m.J for j in m.J)
            return m.y[s, k, t, target_idx] <= visibility_sum
        
        model.vis_target = pyo.Constraint(model.S, model.K, model.T_local, model.P, 
                                         rule=visibility_target_rule)
        
        def visibility_ground_rule(m, s, k, t, ground_idx):
            # Paper Eq (11b): q^s_{ktg} <= sum_{i,j} W^s_{ktjg} * x^s_{kij}
            # Can only downlink to ground station g if visible from current orbital slot
            if s == 1:
                visibility_sum = sum(int(p.V_ground[s-1, k-1, t-1, j-1, ground_idx-1]) * m.x[s, k, 1, j] 
                                    for j in m.J)
            else:
                visibility_sum = sum(int(p.V_ground[s-1, k-1, t-1, j-1, ground_idx-1]) * m.x[s, k, i, j] 
                                    for i in m.J for j in m.J)
            return m.q[s, k, t, ground_idx] <= visibility_sum
        
        model.vis_ground = pyo.Constraint(model.S, model.K, model.T_local, model.G, 
                                         rule=visibility_ground_rule)
        
        def visibility_sun_rule(m, s, k, t):
            # Paper Eq (11c): h^s_{kt} <= sum_{i,j} H^s_{ktj} * x^s_{kij}
            # Can only charge if in sunlight at current orbital slot
            if s == 1:
                visibility_sum = sum(int(p.V_sun[s-1, k-1, t-1, j-1]) * m.x[s, k, 1, j] 
                                    for j in m.J)
            else:
                visibility_sum = sum(int(p.V_sun[s-1, k-1, t-1, j-1]) * m.x[s, k, i, j] 
                                    for i in m.J for j in m.J)
            return m.h[s, k, t] <= visibility_sum
        
        model.vis_sun = pyo.Constraint(model.S, model.K, model.T_local, 
                                      rule=visibility_sun_rule)
        
        def one_activity_rule(m, s, k, t):
            # Paper Eq (11d): sum_p y^s_{ktp} + sum_g q^s_{ktg} + h^s_{kt} <= 1
            # At most one activity per time step (observe OR downlink OR charge)
            total_obs = sum(m.y[s, k, t, p] for p in m.P)
            total_comm = sum(m.q[s, k, t, g] for g in m.G)
            return total_obs + total_comm + m.h[s, k, t] <= 1
        
        model.one_activity = pyo.Constraint(model.S, model.K, model.T_local, 
                                           rule=one_activity_rule)
        
        # ---------------------------------------------------------------------
        # Data Storage Dynamics - Paper Eq (12a-12f)
        # Tracks data accumulation and downlink
        # ---------------------------------------------------------------------
        
        def data_tracking_within_stage_rule(m, s, k, t):
            # Paper Eq (12a): d^s_{kt+1} = d^s_{kt} + D_obs*sum_p(y^s_{ktp}) - D_comm*sum_g(q^s_{ktg})
            # Data dynamics within a stage
            if t >= T_per_stage:  # Last time step of stage
                return pyo.Constraint.Skip
            
            d_curr = m.d[s, k, t]
            data_gen = sum(m.y[s, k, t, p] for p in m.P) * p.D_obs
            data_down = sum(m.q[s, k, t, g] for g in m.G) * p.D_comm
            
            return m.d[s, k, t+1] == d_curr + data_gen - data_down
        
        model.data_tracking_within = pyo.Constraint(model.S, model.K, model.T_local, 
                                                   rule=data_tracking_within_stage_rule)
        
        def data_tracking_between_stages_rule(m, s, k):
            # Paper Eq (12b): d^{s+1}_{k1} = d^s_{kT_s} + D_obs*sum_p(y^s_{kT_s,p}) - D_comm*sum_g(q^s_{kT_s,g})
            # Data carries over from one stage to next (no data loss during maneuver)
            if s >= p.S:  # No transition after last stage
                return pyo.Constraint.Skip
            
            t_last = T_per_stage
            d_end = m.d[s, k, t_last]
            data_gen = sum(m.y[s, k, t_last, p] for p in m.P) * p.D_obs
            data_down = sum(m.q[s, k, t_last, g] for g in m.G) * p.D_comm
            
            return m.d[s+1, k, 1] == d_end + data_gen - data_down
        
        model.data_tracking_between = pyo.Constraint(model.S, model.K, 
                                                    rule=data_tracking_between_stages_rule)
        
        def data_upper_bound_rule(m, s, k, t):
            # Paper Eq (12c): d^s_{kt} + D_obs*sum_p(y^s_{ktp}) <= D_max
            # Cannot generate more data than storage capacity
            data_gen = sum(m.y[s, k, t, p] for p in m.P) * p.D_obs
            return m.d[s, k, t] + data_gen <= p.D_max
        
        model.data_upper = pyo.Constraint(model.S, model.K, model.T_local, 
                                         rule=data_upper_bound_rule)
        
        def data_lower_bound_rule(m, s, k, t):
            # Paper Eq (12d): d^s_{kt} - D_comm*sum_g(q^s_{ktg}) >= D_min = 0
            # Cannot downlink more data than stored
            data_down = sum(m.q[s, k, t, g] for g in m.G) * p.D_comm
            return m.d[s, k, t] - data_down >= 0
        
        model.data_lower = pyo.Constraint(model.S, model.K, model.T_local, 
                                         rule=data_lower_bound_rule)
        
        def data_initial_rule(m, k):
            # Paper: d^1_{k1} = D_min = 0 (start with empty storage)
            return m.d[1, k, 1] == 0
        
        model.data_initial = pyo.Constraint(model.K, rule=data_initial_rule)
        
        # ---------------------------------------------------------------------
        # Battery Dynamics - Paper Eq (13a-13b, 14a-14d)
        # Tracks battery charge/discharge including maneuver costs
        # ---------------------------------------------------------------------
        
        def battery_tracking_within_stage_rule(m, s, k, t):
            # Paper Eq (13a): b^s_{kt+1} = b^s_{kt} + B_charge*h^s_{kt} - B_obs*sum_p(y^s_{ktp}) - B_comm*sum_g(q^s_{ktg}) - B_time
            # Battery dynamics within a stage (no maneuver cost)
            if t >= T_per_stage:  # Last time step of stage
                return pyo.Constraint.Skip
            
            b_curr = m.b[s, k, t]
            charge = m.h[s, k, t] * p.B_charge
            obs_consumption = sum(m.y[s, k, t, p] for p in m.P) * p.B_obs
            comm_consumption = sum(m.q[s, k, t, g] for g in m.G) * p.B_comm
            idle_consumption = p.B_time
            return m.b[s, k, t+1] == b_curr + charge - obs_consumption - comm_consumption - idle_consumption
        
        model.battery_tracking_within = pyo.Constraint(model.S, model.K, model.T_local, 
                                                      rule=battery_tracking_within_stage_rule)
        
        def battery_tracking_between_stages_rule(m, s, k):
            # Paper Eq (13b): b^{s+1}_{k1} = b^s_{kT_s} + B_charge*h^s_{kT_s} - B_obs*sum_p(y^s_{kT_s,p}) 
            #                                - B_comm*sum_g(q^s_{kT_s,g}) - B_recon*sum_{i,j} x^{s+1}_{kij} - B_time
            # Battery transition between stages includes maneuver cost!
            if s >= p.S:  # No transition after last stage
                return pyo.Constraint.Skip
            
            t_last = T_per_stage
            b_end = m.b[s, k, t_last]
            charge = m.h[s, k, t_last] * p.B_charge
            obs_consumption = sum(m.y[s, k, t_last, p] for p in m.P) * p.B_obs
            comm_consumption = sum(m.q[s, k, t_last, g] for g in m.G) * p.B_comm
            idle_consumption = p.B_time
            
            # Battery consumption of maneuver for transitioning to next stage
            recon_consumption = sum(m.x[s+1, k, i, j] for i in m.J for j in m.J) * p.B_recon
            
            return m.b[s+1, k, 1] == b_end + charge - obs_consumption - comm_consumption - recon_consumption - idle_consumption
        
        model.battery_tracking_between = pyo.Constraint(model.S, model.K, 
                                                       rule=battery_tracking_between_stages_rule)
        
        def battery_initial_rule(m, k):
            # Paper Eq (13c): b^{1}_{k1} = B_max - B_recon*sum_j x^{1}_{k,1,j}
            recon_consumption = sum(m.x[1, k, 1, j] for j in m.J) * p.B_recon
            return m.b[1, k, 1] == p.B_max - recon_consumption

        model.battery_init = pyo.Constraint(model.K, rule=battery_initial_rule)     
        
        def battery_upper_bound_rule(m, s, k, t):
            # Paper Eq (14a): b^s_{kt} + B_charge*h^s_{kt} <= B_max
            # Battery cannot exceed maximum capacity
            charge = m.h[s, k, t] * p.B_charge
            return m.b[s, k, t] + charge <= p.B_max
        
        model.battery_upper = pyo.Constraint(model.S, model.K, model.T_local, 
                                            rule=battery_upper_bound_rule)
        
        def battery_lower_within_rule(m, s, k, t):
            # Paper Eq (14b): b^s_{kt} - B_obs*sum_p(y^s_{ktp}) - B_comm*sum_g(q^s_{ktg}) - B_time>= B_min (B_min=0)
            # Battery cannot drop below minimum level during normal operations
            if t >= T_per_stage:  # Last time step handled separately
                return pyo.Constraint.Skip
            
            obs_consumption = sum(m.y[s, k, t, p] for p in m.P) * p.B_obs
            comm_consumption = sum(m.q[s, k, t, g] for g in m.G) * p.B_comm
            idle_consumption = p.B_time
            
            return m.b[s, k, t] - obs_consumption - comm_consumption - idle_consumption >= 0
        
        model.battery_lower_within = pyo.Constraint(model.S, model.K, model.T_local, 
                                                   rule=battery_lower_within_rule)
        
        def battery_lower_between_rule(m, s, k):
            # Paper Eq (14c): b^s_{kT_s} - B_obs*sum_p(y^s_{kT_s,p}) - B_comm*sum_g(q^s_{kT_s,g}) 
            #                              - B_recon*sum_{i,j} x^{s+1}_{kij} - B_time>= B_min (B_min=0)
            # Battery must remain above minimum even after maneuver cost
            if s >= p.S:  # No transition after last stage, use 14b instead
                t_last = T_per_stage
                obs_consumption = sum(m.y[s, k, t_last, p] for p in m.P) * p.B_obs
                comm_consumption = sum(m.q[s, k, t_last, g] for g in m.G) * p.B_comm
                idle_consumption = p.B_time
                return m.b[s, k, t_last] - obs_consumption - comm_consumption - idle_consumption >= 0
            
            t_last = T_per_stage
            b_end = m.b[s, k, t_last]
            obs_consumption = sum(m.y[s, k, t_last, p] for p in m.P) * p.B_obs
            comm_consumption = sum(m.q[s, k, t_last, g] for g in m.G) * p.B_comm
            recon_consumption = sum(m.x[s+1, k, i, j] for i in m.J for j in m.J) * p.B_recon
            idle_consumption = p.B_time
            
            return b_end - obs_consumption - comm_consumption - recon_consumption - idle_consumption >= 0
        
        model.battery_lower_between = pyo.Constraint(model.S, model.K, 
                                                    rule=battery_lower_between_rule)
        
        def battery_lower_init_rule(m, k):
            # Paper Eq (14d): B_max - B_recon*sum_j x^1_{k,i,j} >= B_min (B_min=0)
            # Initial battery reduced by cost of first maneuver from initial slot
            recon_consumption = sum(m.x[1, k, i, j] for i in m.J for j in m.J) * p.B_recon
            return p.B_max - recon_consumption >= 0
        
        model.battery_lower_init = pyo.Constraint(model.K, rule=battery_lower_init_rule)
        
        self.model = model
        return model
    
    def solve(self, time_limit_minutes=60, solver_name='gurobi'):
        """
        Solve the model using specified solver
        
        Args:
            time_limit_minutes: Time limit in minutes
            solver_name: Solver to use (default: 'gurobi')
        
        Returns:
            dict: Results including objective value, runtime, and solution status
        """
        if self.model is None:
            self.build_model()
        
        # Initialize solver
        solver_map = {
            'gurobi': 'gurobi',
            'gurobi_direct': 'gurobi_direct',
            'gurobi_persistent': 'gurobi_persistent',
        }
        
        factory_name = solver_map.get(solver_name, solver_name)
        
        try:
            solver = pyo.SolverFactory(factory_name)
        except:
            solver = pyo.SolverFactory('gurobi')
        
        if not solver.available():
            raise RuntimeError(f"Solver '{factory_name}' is not available. Please install it first.")
        
        # Set solver options
        solver.options['TimeLimit'] = time_limit_minutes * 60
        solver.options['MIPGap'] = 0.01
        solver.options['Threads'] = 0
        solver.options['LogToConsole'] = 0
        
        start_time = time.time()
        
        # Solve
        results = solver.solve(self.model, tee=False, load_solutions=False)
        runtime = time.time() - start_time
        
        # Check if solution exists before loading
        if (results.solver.termination_condition == TerminationCondition.optimal or
            results.solver.termination_condition == TerminationCondition.maxTimeLimit):
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
            
            # Calculate total data downlinked
            total_downlinks = sum(pyo.value(self.model.q[s, k, t, g])
                                for s in self.model.S 
                                for k in self.model.K 
                                for t in self.model.T_local 
                                for g in self.model.G)
            data_downlinked_mb = total_downlinks * self.params.D_comm
            data_downlinked_gb = data_downlinked_mb / 1024
            
            # Count observations
            total_observations = sum(pyo.value(self.model.y[s, k, t, p])
                                   for s in self.model.S 
                                   for k in self.model.K 
                                   for t in self.model.T_local 
                                   for p in self.model.P)
            
            # Calculate propellant used
            propellant_used = 0
            for k in self.model.K:
                for s in self.model.S:
                    if s == 1:
                        for j in self.model.J:
                            x_val = pyo.value(self.model.x[s, k, 1, j])
                            if x_val > 0.5:
                                propellant_used += self.params.maneuver_costs[s-1, k-1, 0, j-1]
                    else:
                        for i in self.model.J:
                            for j in self.model.J:
                                x_val = pyo.value(self.model.x[s, k, i, j])
                                if x_val > 0.5:
                                    propellant_used += self.params.maneuver_costs[s-1, k-1, i-1, j-1]
            
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
    # Test with a small instance
    from parameters import InstanceParameters
    
    print("Testing REOSSP-Exact Solver...")
    
    # Create small test instance
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
    print(f"  Downlinks: {results['total_downlinks']:.0f}")
    print(f"  Propellant used: {results['propellant_used']:.2f} m/s")
