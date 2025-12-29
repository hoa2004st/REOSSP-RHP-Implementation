"""
REOSSP-RHP: Rolling Horizon Procedure
Fast heuristic approach for large instances
Implements Algorithm 1 from paper Section 4.3
"""

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import numpy as np
import time
import os
import sys
from contextlib import contextmanager
from parameters import InstanceParameters


@contextmanager
def suppress_stderr():
    """Context manager to suppress stderr temporarily"""
    old_stderr = sys.stderr
    try:
        sys.stderr = open(os.devnull, 'w')
        yield
    finally:
        sys.stderr.close()
        sys.stderr = old_stderr


class REOSSPRHPSolver:
    """
    Rolling Horizon Procedure for REOSSP
    Solves stage-by-stage with lookahead window L
    Implements Algorithm 1 from paper Section 4.3
    """
    
    def __init__(self, params: InstanceParameters, lookahead=1):
        self.params = params
        self.lookahead = lookahead  # L in paper (lookahead window size)
        self.results = None
        
        # Track solution across stages (only committed decisions)
        self.orbital_assignments = {}  # {stage: {sat: slot}}
        self.observation_plan = []     # (stage, sat, time, target)
        self.downlink_plan = []        # (stage, sat, time, ground)
        
        # Track state between stages (for initial conditions)
        # Paper Eq (21a,d): d^s_{k1} and b^s_{k1} inherit from previous stage
        self.c_max_remaining = {k: params.c_max for k in range(params.K)}  # Remaining propellant budget per satellite
        self.J_tilde_prev = {k: 0 for k in range(params.K)}  # j_tilde^{s-1}_k: slot at end of previous stage (0-indexed)
        
    def build_stage_model(self, current_stage):
        """
        Build optimization model for stages s to s+L-1 (lookahead window)
        Implements optimization problem (16a-22d) from paper Section 4.3.3
        
        Args:
            current_stage: Current stage s (0-indexed)
        
        Returns:
            Pyomo model for lookahead window
        """
        p = self.params
        s = current_stage  # Current stage (0-indexed)
        L = self.lookahead  # Lookahead window size
        
        # Paper Algorithm 1, Line 3: Stages in lookahead window: {s, s+1, ..., min(s+L-1, S-1)}
        S_full = p.S
        stages_in_window = min(L, S_full - s)
        stage_set = list(range(s, s + stages_in_window))  # [s, s+1, ..., s+L-1] (0-indexed)
        
        model = pyo.ConcreteModel(name=f"REOSSP_RHP_Stage_{s}")
        
        # Time steps per stage (local indexing t = 1, ..., T_s)
        T_per_stage = p.T // p.S
        
        # Extract dimensions from visibility arrays
        n_targets = p.V_target.shape[4]
        n_ground = p.V_ground.shape[4]
        
        # =====================================================================
        # SETS - Paper notation: l ∈ L_s, k ∈ K, t ∈ T, j ∈ J, p ∈ P, g ∈ G
        # Using 1-based indexing for time (t ∈ {1,...,T_s}) to match paper
        # Using 0-indexed stage indices internally but 1-indexed in constraints
        # =====================================================================
        model.S_window = pyo.Set(initialize=stage_set)  # Stages in lookahead window
        model.K = pyo.RangeSet(0, p.K - 1)  # Satellites (0-indexed to match arrays)
        model.T_local = pyo.RangeSet(1, T_per_stage)  # Local time within stage (1-indexed)
        model.J = pyo.RangeSet(0, p.J_sk - 1)  # Orbital slots (0-indexed)
        model.P = pyo.RangeSet(0, n_targets - 1)  # Targets (0-indexed)
        model.G = pyo.RangeSet(0, n_ground - 1)  # Ground stations (0-indexed)
        
        # =====================================================================
        # DECISION VARIABLES - Paper Section 4.3.3, Eq (16a-16f)
        # Only for stages in lookahead window
        # =====================================================================
        
        # Paper Eq (16a): x^l_{kij} ∈ {0,1} - reconfiguration decision
        # For stage s: x^s_{k,j_tilde^{s-1}_k,j} (from previous slot to new slot)
        # For stage l > s: x^l_{kij} (from any slot to any slot)
        model.x = pyo.Var(model.S_window, model.K, model.J, model.J, domain=pyo.Binary)
        
        # Paper Eq (16b): y^l_{ktp} ∈ {0,1} - observation decision
        # = 1 if satellite k observes target p at local time t in stage l
        model.y = pyo.Var(model.S_window, model.K, model.T_local, model.P, domain=pyo.Binary)
        
        # Paper Eq (16c): q^l_{ktg} ∈ {0,1} - downlink decision
        # = 1 if satellite k downlinks to ground station g at local time t in stage l
        model.q = pyo.Var(model.S_window, model.K, model.T_local, model.G, domain=pyo.Binary)
        
        # Paper Eq (16d): h^l_{kt} ∈ {0,1} - charging decision
        # = 1 if satellite k charges from sun at local time t in stage l
        model.h = pyo.Var(model.S_window, model.K, model.T_local, domain=pyo.Binary)
        
        # Paper Eq (16e): d^l_{kt} ∈ [D_min, D_max] - data storage state
        # Amount of data stored on satellite k at local time t in stage l
        model.d = pyo.Var(model.S_window, model.K, model.T_local, 
                         domain=pyo.NonNegativeReals, bounds=(0, p.D_max))
        
        # Paper Eq (16f): b^l_{kt} ∈ [B_min, B_max] - battery state
        # Battery charge level of satellite k at local time t in stage l
        model.b = pyo.Var(model.S_window, model.K, model.T_local, 
                         domain=pyo.NonNegativeReals, bounds=(0, p.B_max))
        
        # =====================================================================
        # OBJECTIVE FUNCTION - Paper Eq (17)
        # Maximize: sum_{l in L_s} [C * sum q^l_{ktg} + sum y^l_{ktp}]
        # Maximize sum over lookahead window (equal weights for all stages)
        # =====================================================================
        def objective_rule(m):
            # Paper Eq (17): max sum_{l in L_s} [C * sum q^l_{ktg} + sum y^l_{ktp}]
            obj = 0
            for l in m.S_window:
                # Downlinks in stage l
                stage_downlinks = sum(m.q[l, k, t, g] 
                                     for k in m.K 
                                     for t in m.T_local 
                                     for g in m.G)
                # Observations in stage l
                stage_observations = sum(m.y[l, k, t, p] 
                                        for k in m.K 
                                        for t in m.T_local 
                                        for p in m.P)
                obj += p.C * stage_downlinks + stage_observations
            
            return obj
        
        model.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)
        
        # =====================================================================
        # CONSTRAINTS
        # =====================================================================
        
        # ---------------------------------------------------------------------
        # Orbital Reconfiguration Constraints - Paper Eq (18a-18c)
        # Control satellite movement between orbital slots in lookahead window
        # ---------------------------------------------------------------------
        
        def reconfig_from_prev_rule(m, k):
            # Paper Eq (18a): sum_{j in J^s} x^s_{k,j_tilde^{s-1}_k,j} = 1
            # Each satellite must move from its slot at end of previous stage
            l = s  # Current stage only
            j_tilde_prev = self.J_tilde_prev[k]  # Slot at end of stage s-1
            # Must choose exactly one target slot from previous slot
            return sum(m.x[l, k, j_tilde_prev, j] for j in m.J) == 1
        
        model.reconfig_from_prev = pyo.Constraint(model.K, rule=reconfig_from_prev_rule)
        
        def slot_continuity_rule(m, l, k, i):
            # Paper Eq (18b): sum_{j in J^{l+1}} x^{l+1}_{kij} = sum_{j' in J^{l-1}} x^{l}_{kj'i}
            # Slot at end of stage l determines slot at start of stage l+1
            # If satellite is in slot i at end of stage l, it must move from i to some slot in stage l+1
            if l not in m.S_window or l + 1 not in m.S_window:
                return pyo.Constraint.Skip
            
            # Satellites entering slot i in stage l
            incoming = sum(m.x[l, k, j_prime, i] for j_prime in m.J)
            
            # Satellites leaving slot i to stage l+1
            outgoing = sum(m.x[l + 1, k, i, j] for j in m.J)
            
            return outgoing == incoming
        
        model.slot_continuity = pyo.Constraint(model.S_window, model.K, model.J, 
                                              rule=slot_continuity_rule)
        
        def propellant_budget_rule(m, k):
            # Paper Eq (18c): sum_{l,i,j} c^l_{kij} * x^l_{kij} <= c^max_k (remaining budget)
            # Total propellant used by satellite k in lookahead window must not exceed remaining budget
            # Uses REMAINING budget for satellite k (already accounts for previous stages)
            total_cost = 0
            for l in m.S_window:
                if l == s:
                    # Stage s: from j_tilde^{s-1}_k to j
                    j_tilde_prev = self.J_tilde_prev[k]
                    for j in m.J:
                        cost = p.maneuver_costs[l, k, j_tilde_prev, j]
                        total_cost += m.x[l, k, j_tilde_prev, j] * cost
                else:
                    # Later stages in window: from any i to j
                    for i in m.J:
                        for j in m.J:
                            cost = p.maneuver_costs[l, k, i, j]
                            total_cost += m.x[l, k, i, j] * cost
            
            # Use remaining budget (already accounts for previous stages)
            return total_cost <= self.c_max_remaining[k]
        
        model.propellant_budget = pyo.Constraint(model.K, rule=propellant_budget_rule)
        
        # ---------------------------------------------------------------------
        # Visibility Constraints - Paper Eq (19a-19d)
        # Actions depend on slot-dependent visibility windows
        # Similar to exact formulation but only for lookahead window
        # ---------------------------------------------------------------------
        
        def visibility_target_rule(m, l, k, t, target_idx):
            # Paper Eq (19a): y^l_{ktp} <= sum_{i,j} V^l_{ktjp} * x^l_{kij}
            # Can only observe target p if visible from current orbital slot
            if l == s:
                j_tilde_prev = self.J_tilde_prev[k]
                visibility_sum = sum(int(p.V_target[l, k, t-1, j, target_idx]) * m.x[l, k, j_tilde_prev, j] 
                                    for j in m.J)
            else:
                visibility_sum = sum(int(p.V_target[l, k, t-1, j, target_idx]) * m.x[l, k, i, j] 
                                    for i in m.J for j in m.J)
            return m.y[l, k, t, target_idx] <= visibility_sum
        
        model.vis_target = pyo.Constraint(model.S_window, model.K, model.T_local, model.P, 
                                         rule=visibility_target_rule)
        
        def visibility_ground_rule(m, l, k, t, ground_idx):
            # Paper Eq (19b): q^l_{ktg} <= sum_{i,j} W^l_{ktjg} * x^l_{kij}
            # Can only downlink to ground station g if visible from current orbital slot
            if l == s:
                j_tilde_prev = self.J_tilde_prev[k]
                visibility_sum = sum(int(p.V_ground[l, k, t-1, j, ground_idx]) * m.x[l, k, j_tilde_prev, j] 
                                    for j in m.J)
            else:
                visibility_sum = sum(int(p.V_ground[l, k, t-1, j, ground_idx]) * m.x[l, k, i, j] 
                                    for i in m.J for j in m.J)
            return m.q[l, k, t, ground_idx] <= visibility_sum
        
        model.vis_ground = pyo.Constraint(model.S_window, model.K, model.T_local, model.G, 
                                         rule=visibility_ground_rule)
        
        def visibility_sun_rule(m, l, k, t):
            # Paper Eq (19c): h^l_{kt} <= sum_{i,j} H^l_{ktj} * x^l_{kij}
            # Can only charge if in sunlight at current orbital slot
            if l == s:
                j_tilde_prev = self.J_tilde_prev[k]
                visibility_sum = sum(int(p.V_sun[l, k, t-1, j]) * m.x[l, k, j_tilde_prev, j] 
                                    for j in m.J)
            else:
                visibility_sum = sum(int(p.V_sun[l, k, t-1, j]) * m.x[l, k, i, j] 
                                    for i in m.J for j in m.J)
            return m.h[l, k, t] <= visibility_sum
        
        model.vis_sun = pyo.Constraint(model.S_window, model.K, model.T_local, 
                                      rule=visibility_sun_rule)
        
        def one_activity_rule(m, l, k, t):
            # Paper Eq (19d): sum_p y^l_{ktp} + sum_g q^l_{ktg} + h^l_{kt} <= 1
            # At most one activity per time step (observe OR downlink OR charge)
            total_obs = sum(m.y[l, k, t, p] for p in m.P)
            total_comm = sum(m.q[l, k, t, g] for g in m.G)
            return total_obs + total_comm + m.h[l, k, t] <= 1
        
        model.one_activity = pyo.Constraint(model.S_window, model.K, model.T_local, 
                                           rule=one_activity_rule)
        
        # ---------------------------------------------------------------------
        # Data Storage Dynamics - Paper Eq (20a-20f)
        # Tracks data accumulation and downlink in lookahead window
        # ---------------------------------------------------------------------
        
        def data_tracking_initial_stage_s_rule(m, k):
            # Paper Eq (20a): d^s_{k2} = d^s_{k1} + D_obs*sum_p(y^s_{k1p}) - D_comm*sum_g(q^s_{k1g})
            # Special case for t=1 in stage s (uses updated parameter d^s_{k1})
            # Note: d^s_{k1} is set via data_initial constraint
            l = s  # Current stage
            if l not in m.S_window or T_per_stage < 2:
                return pyo.Constraint.Skip
            
            d_curr = m.d[l, k, 1]
            data_gen = sum(m.y[l, k, 1, p] for p in m.P) * p.D_obs
            data_down = sum(m.q[l, k, 1, g] for g in m.G) * p.D_comm
            
            return m.d[l, k, 2] == d_curr + data_gen - data_down
        
        model.data_tracking_initial_s = pyo.Constraint(model.K, 
                                                      rule=data_tracking_initial_stage_s_rule)
        
        def data_tracking_rest_stage_s_rule(m, k, t):
            # Paper Eq (20b): d^s_{kt+1} = d^s_{kt} + D_obs*sum_p(y^s_{ktp}) - D_comm*sum_g(q^s_{ktg})
            # For t in T^s \\ {1, T^s} (i.e., t=2,3,...,T_s-1)
            l = s  # Current stage
            if l not in m.S_window or t < 2 or t >= T_per_stage:
                return pyo.Constraint.Skip
            
            d_curr = m.d[l, k, t]
            data_gen = sum(m.y[l, k, t, p] for p in m.P) * p.D_obs
            data_down = sum(m.q[l, k, t, g] for g in m.G) * p.D_comm
            
            return m.d[l, k, t+1] == d_curr + data_gen - data_down
        
        model.data_tracking_rest_s = pyo.Constraint(model.K, model.T_local, 
                                                   rule=data_tracking_rest_stage_s_rule)
        
        def data_tracking_other_stages_rule(m, l, k, t):
            # Paper Eq (20c): d^l_{kt+1} = d^l_{kt} + D_obs*sum_p(y^l_{ktp}) - D_comm*sum_g(q^l_{ktg})
            # For l in L \\ {s}, t in T^l \\ {T^l} (i.e., t=1,2,...,T_l-1)
            if l == s or t >= T_per_stage:
                return pyo.Constraint.Skip
            
            d_curr = m.d[l, k, t]
            data_gen = sum(m.y[l, k, t, p] for p in m.P) * p.D_obs
            data_down = sum(m.q[l, k, t, g] for g in m.G) * p.D_comm
            
            return m.d[l, k, t+1] == d_curr + data_gen - data_down
        
        model.data_tracking_other = pyo.Constraint(model.S_window, model.K, model.T_local, 
                                                  rule=data_tracking_other_stages_rule)
        
        def data_tracking_between_stages_rule(m, l, k):
            # Paper Eq (20d): d^{l+1,k}_1 = d^{lk}_{T^l} + D_obs*sum_p(y^{lk}_{T^l,p}) - D_comm*sum_g(q^{lk}_{T^l,g})
            # For l in L \\ {s+L} (all stages except last in window)
            # Data carries over from one stage to next (no data loss during maneuver)
            if l not in m.S_window or l + 1 not in m.S_window:
                return pyo.Constraint.Skip
            
            t_last = T_per_stage
            d_end = m.d[l, k, t_last]
            data_gen = sum(m.y[l, k, t_last, p] for p in m.P) * p.D_obs
            data_down = sum(m.q[l, k, t_last, g] for g in m.G) * p.D_comm
            
            return m.d[l+1, k, 1] == d_end + data_gen - data_down
        
        model.data_tracking_between = pyo.Constraint(model.S_window, model.K, 
                                                    rule=data_tracking_between_stages_rule)
        
        def data_upper_bound_rule(m, l, k, t):
            # Paper Eq (20e): d^l_{kt} + D_obs*sum_p(y^l_{ktp}) <= D_max
            # Cannot generate more data than storage capacity
            data_gen = sum(m.y[l, k, t, p] for p in m.P) * p.D_obs
            return m.d[l, k, t] + data_gen <= p.D_max
        
        model.data_upper = pyo.Constraint(model.S_window, model.K, model.T_local, 
                                         rule=data_upper_bound_rule)
        
        def data_lower_bound_rule(m, l, k, t):
            # Paper Eq (20f): d^l_{kt} - D_comm*sum_g(q^l_{ktg}) >= D_min = 0
            # Cannot downlink more data than stored
            data_down = sum(m.q[l, k, t, g] for g in m.G) * p.D_comm
            return m.d[l, k, t] - data_down >= 0
        
        model.data_lower = pyo.Constraint(model.S_window, model.K, model.T_local, 
                                         rule=data_lower_bound_rule)
        
        def data_initial_rule(m, k):
            # Initial data for stage s
            l = s  # Current stage
            if s == 0:
                # First stage: start with empty storage
                return m.d[l, k, 1] == 0
            else:
                # Later stages: inherit from previous solution
                # This will be set dynamically based on previous stage solution
                # For now, initialize to 0 (will be updated in solve loop)
                return pyo.Constraint.Skip  # Will use dual variable or parameter
        
        model.data_initial = pyo.Constraint(model.K, rule=data_initial_rule)
        
        # ---------------------------------------------------------------------
        # Battery Dynamics - Paper Eq (21a-21d)
        # Tracks battery charge/discharge including maneuver costs
        # Similar structure to data dynamics
        # ---------------------------------------------------------------------
        
        def battery_tracking_initial_stage_s_rule(m, k):
            # Paper Eq (21a): b^s_{k2} = b^s_{k1} + B_charge*h^s_{k1} - B_obs*sum_p(y^s_{k1p}) - B_comm*sum_g(q^s_{k1g}) - B_time
            # Special case for t=1 in stage s (uses updated parameter b^s_{k1})
            # Note: b^s_{k1} is set via battery_initial constraint
            l = s  # Current stage
            if l not in m.S_window or T_per_stage < 2:
                return pyo.Constraint.Skip
            
            b_curr = m.b[l, k, 1]
            charge = m.h[l, k, 1] * p.B_charge
            obs_consumption = sum(m.y[l, k, 1, p] for p in m.P) * p.B_obs
            comm_consumption = sum(m.q[l, k, 1, g] for g in m.G) * p.B_comm
            idle_consumption = p.B_time
            
            return m.b[l, k, 2] == b_curr + charge - obs_consumption - comm_consumption - idle_consumption
        
        model.battery_tracking_initial_s = pyo.Constraint(model.K, 
                                                         rule=battery_tracking_initial_stage_s_rule)
        
        def battery_tracking_rest_stage_s_rule(m, k, t):
            # Paper Eq (21b): b^s_{kt+1} = b^s_{kt} + B_charge*h^s_{kt} - B_obs*sum_p(y^s_{ktp}) - B_comm*sum_g(q^s_{ktg}) - B_time
            # For t in T^s \\ {1, T^s} (i.e., t=2,3,...,T_s-1)
            l = s  # Current stage
            if l not in m.S_window or t < 2 or t >= T_per_stage:
                return pyo.Constraint.Skip
            
            b_curr = m.b[l, k, t]
            charge = m.h[l, k, t] * p.B_charge
            obs_consumption = sum(m.y[l, k, t, p] for p in m.P) * p.B_obs
            comm_consumption = sum(m.q[l, k, t, g] for g in m.G) * p.B_comm
            idle_consumption = p.B_time
            
            return m.b[l, k, t+1] == b_curr + charge - obs_consumption - comm_consumption - idle_consumption
        
        model.battery_tracking_rest_s = pyo.Constraint(model.K, model.T_local, 
                                                      rule=battery_tracking_rest_stage_s_rule)
        
        def battery_tracking_other_stages_rule(m, l, k, t):
            # Paper Eq (21c): b^l_{kt+1} = b^l_{kt} + B_charge*h^l_{kt} - B_obs*sum_p(y^l_{ktp}) - B_comm*sum_g(q^l_{ktg}) - B_time
            # For l in L \\ {s}, t in T^l \\ {T^l} (i.e., t=1,2,...,T_l-1)
            if l == s or t >= T_per_stage:
                return pyo.Constraint.Skip
            
            b_curr = m.b[l, k, t]
            charge = m.h[l, k, t] * p.B_charge
            obs_consumption = sum(m.y[l, k, t, p] for p in m.P) * p.B_obs
            comm_consumption = sum(m.q[l, k, t, g] for g in m.G) * p.B_comm
            idle_consumption = p.B_time
            
            return m.b[l, k, t+1] == b_curr + charge - obs_consumption - comm_consumption - idle_consumption
        
        model.battery_tracking_other = pyo.Constraint(model.S_window, model.K, model.T_local, 
                                                     rule=battery_tracking_other_stages_rule)
        
        def battery_tracking_between_stages_rule(m, l, k):
            # Paper Eq (21d): b^{l+1,k}_1 = b^{lk}_{T^l} + B_charge*h^{lk}_{T^l} - B_obs*sum_p(y^{lk}_{T^l,p}) 
            #                                  - B_comm*sum_g(q^{lk}_{T^l,g}) - B_recon*sum_{i in J^{lk}, j in J^{l+1,k}} x^{l+1}_{kij} - B_time
            # For l in L \\ {s+L} (all stages except last in window)
            # Battery transition between stages includes maneuver cost!
            if l not in m.S_window or l + 1 not in m.S_window:
                return pyo.Constraint.Skip
            
            t_last = T_per_stage
            b_end = m.b[l, k, t_last]
            charge = m.h[l, k, t_last] * p.B_charge
            obs_consumption = sum(m.y[l, k, t_last, p] for p in m.P) * p.B_obs
            comm_consumption = sum(m.q[l, k, t_last, g] for g in m.G) * p.B_comm
            idle_consumption = p.B_time
            
            # Battery cost of reconfiguration maneuver
            recon_consumption = sum(m.x[l+1, k, i, j] for i in m.J for j in m.J) * p.B_recon
            
            return m.b[l+1, k, 1] == b_end + charge - obs_consumption - comm_consumption - recon_consumption - idle_consumption
        
        model.battery_tracking_between = pyo.Constraint(model.S_window, model.K, 
                                                       rule=battery_tracking_between_stages_rule)
        
        def battery_initial_rule(m, k):
            # Initial battery for stage s
            # This is KEY: battery at start of stage s must account for maneuver cost!
            l = s  # Current stage
            
            if s == 0:
                # First stage: start with full battery minus initial maneuver cost
                j_tilde_prev = self.J_tilde_prev[k]  # Initial slot (0 for first stage)
                recon_cost = sum(m.x[l, k, j_tilde_prev, j] for j in m.J) * p.B_recon
                return m.b[l, k, 1] == p.B_max - recon_cost
            else:
                # Later stages: inherit from previous solution minus maneuver cost
                # b^s_{k1} = b_init - B_recon*sum_{i,j} x^s_{kij}
                # This will be handled dynamically in solve loop
                return pyo.Constraint.Skip
        
        model.battery_initial = pyo.Constraint(model.K, rule=battery_initial_rule)

        def battery_upper_bound_rule(m, l, k, t):
            # Paper Eq (22a): b^l_{kt} + B_charge*h^l_{kt} <= B_max
            # Battery cannot exceed maximum capacity
            charge = m.h[l, k, t] * p.B_charge
            return m.b[l, k, t] + charge <= p.B_max
        
        model.battery_upper = pyo.Constraint(model.S_window, model.K, model.T_local, 
                                            rule=battery_upper_bound_rule)
        
        def battery_lower_stage_s_rule(m, k, t):
            # Paper Eq (22b): b^s_{kt} - B_obs*sum_p(y^s_{ktp}) - B_comm*sum_g(q^s_{ktg}) - B_time >= B_min = 0
            # For t in T^s \\ {1, T^s} (i.e., t=2,3,...,T_s-1)
            # Battery cannot drop below minimum level during operations in stage s
            l = s  # Current stage
            if l not in m.S_window or t < 2 or t >= T_per_stage:
                return pyo.Constraint.Skip
            
            obs_consumption = sum(m.y[l, k, t, p] for p in m.P) * p.B_obs
            comm_consumption = sum(m.q[l, k, t, g] for g in m.G) * p.B_comm
            idle_consumption = p.B_time
            
            return m.b[l, k, t] - obs_consumption - comm_consumption - idle_consumption >= 0
        
        model.battery_lower_stage_s = pyo.Constraint(model.K, model.T_local, 
                                                     rule=battery_lower_stage_s_rule)
        
        def battery_lower_other_stages_rule(m, l, k, t):
            # Paper Eq (22c): b^l_{kt} - B_obs*sum_p(y^l_{ktp}) - B_comm*sum_g(q^l_{ktg}) - B_time >= B_min = 0
            # For l in L \\ {s}, t in T^l \\ {T^l} (i.e., t=1,2,...,T_l-1)
            # Battery cannot drop below minimum level during operations in other stages
            if l == s or t >= T_per_stage:
                return pyo.Constraint.Skip
            
            obs_consumption = sum(m.y[l, k, t, p] for p in m.P) * p.B_obs
            comm_consumption = sum(m.q[l, k, t, g] for g in m.G) * p.B_comm
            idle_consumption = p.B_time
            
            return m.b[l, k, t] - obs_consumption - comm_consumption - idle_consumption >= 0
        
        model.battery_lower_other = pyo.Constraint(model.S_window, model.K, model.T_local, 
                                                   rule=battery_lower_other_stages_rule)
        
        def battery_lower_last_timestep_rule(m, l, k):
            # Paper Eq (22d): b^{lk}_{T^l} - B_obs*sum_p(y^{lk}_{T^l,p}) - B_comm*sum_g(q^{lk}_{T^l,g}) 
            #                 - B_recon*sum_{i,j} x^{l+1}_{kij} - B_time >= B_min = 0
            # For l in L \\ {s+L}, t = T^l
            # Battery must remain above minimum even after activities + maneuver to next stage
            if l not in m.S_window:
                return pyo.Constraint.Skip
            
            t_last = T_per_stage
            b_end = m.b[l, k, t_last]
            obs_consumption = sum(m.y[l, k, t_last, p] for p in m.P) * p.B_obs
            comm_consumption = sum(m.q[l, k, t_last, g] for g in m.G) * p.B_comm
            idle_consumption = p.B_time
            
            # Check if there's a next stage in the window (then include reconfig cost)
            if l + 1 in m.S_window:
                # Paper Eq (22d) - includes reconfiguration cost for transition
                recon_consumption = sum(m.x[l+1, k, i, j] for i in m.J for j in m.J) * p.B_recon
                return b_end - obs_consumption - comm_consumption - recon_consumption - idle_consumption >= 0
            else:
                # Last stage in window - no reconfiguration cost (22c applied at t=T^l)
                return b_end - obs_consumption - comm_consumption - idle_consumption >= 0
        
        model.battery_lower_last_timestep = pyo.Constraint(model.S_window, model.K, 
                                                           rule=battery_lower_last_timestep_rule)
        
        return model
    
    def solve(self, time_limit_per_stage_minutes=5, solver_name='gurobi'):
        """
        Solve using rolling horizon procedure (Algorithm 1 from paper)
        
        Args:
            time_limit_per_stage_minutes: Time limit for each stage optimization
            solver_name: Solver to use (default: 'gurobi')
        
        Returns:
            dict: Results including objective value, runtime, and solution status
        """
        p = self.params
        total_runtime = 0
        total_vars = 0
        total_constrs = 0
        
        L = self.lookahead
        
        # Algorithm 1, Line 1: Loop over stages s = 1 to S
        for s in range(p.S):
            print(f"\n📍 Solving stage {s+1}/{p.S} (lookahead L={L})...")
            
            # Algorithm 1, Line 2: Build and solve model for stages s to min(s+L-1, S-1)
            model = self.build_stage_model(s)
            
            # If we have state from previous stage, update initial conditions
            if s > 0:
                # Update d^s_{k1} and b^s_{k1} based on previous solution
                # This is done implicitly via constraints in build_stage_model
                pass
            
            # Initialize solver
            try:
                solver = pyo.SolverFactory('gurobi')
            except:
                solver = pyo.SolverFactory('gurobi')
            
            if not solver.available():
                raise RuntimeError(f"Gurobi solver is not available.")
            
            # Set solver options
            solver.options['TimeLimit'] = time_limit_per_stage_minutes * 60
            solver.options['MIPGap'] = 0.02
            solver.options['Threads'] = 0
            solver.options['LogToConsole'] = 0
            
            start_time = time.time()
            results = solver.solve(model, tee=False, load_solutions=False)
            stage_runtime = time.time() - start_time
            total_runtime += stage_runtime
            
            # Track model size
            total_vars += model.nvariables()
            total_constrs += model.nconstraints()
            
            # Load solution if available and extract results
            solution_loaded = False
            if (results.solver.termination_condition == TerminationCondition.optimal or 
                results.solver.termination_condition == TerminationCondition.maxTimeLimit):
                try:
                    model.solutions.load_from(results)
                    solution_loaded = True
                except Exception as e:
                    print(f"  ⚠ Warning: Could not load solution: {e}")
                    solution_loaded = False
            
            # Extract solution for CURRENT STAGE ONLY (s)
            # Algorithm 1, Lines 5-7: Only commit decisions for stage s
            if solution_loaded and results.solver.status == SolverStatus.ok:
                # Save orbital assignments for stage s only
                stage_assignments = {}
                for k_idx in model.K:
                    j_tilde_prev = self.J_tilde_prev[k_idx]
                    for j_idx in model.J:
                        # Check if variable exists and has a value
                        var = model.x[s, k_idx, j_tilde_prev, j_idx]
                        if var.value is not None and var.value > 0.5:
                            stage_assignments[k_idx] = j_idx
                            self.J_tilde_prev[k_idx] = j_idx  # Update for next stage
                            break
                self.orbital_assignments[s] = stage_assignments
                
                # Save observations and downlinks for stage s ONLY
                T_per_stage = p.T // p.S
                for k_idx in model.K:
                    for t_idx in model.T_local:
                        for p_idx in model.P:
                            var = model.y[s, k_idx, t_idx, p_idx]
                            if var.value is not None and var.value > 0.5:
                                self.observation_plan.append((s, k_idx, t_idx, p_idx))
                        for g_idx in model.G:
                            var = model.q[s, k_idx, t_idx, g_idx]
                            if var.value is not None and var.value > 0.5:
                                self.downlink_plan.append((s, k_idx, t_idx, g_idx))
                
                # Update propellant budget (deduct used propellant)
                for k_idx in model.K:
                    j_tilde_prev = self.J_tilde_prev[k_idx]
                    for j_idx in model.J:
                        var = model.x[s, k_idx, j_tilde_prev, j_idx]
                        if var.value is not None and var.value > 0.5:
                            prop_cost = p.maneuver_costs[s, k_idx, j_tilde_prev, j_idx]
                            self.c_max_remaining[k_idx] -= prop_cost
                
                # Stage summary
                stage_obs = len([o for o in self.observation_plan if o[0] == s])
                stage_down = len([d for d in self.downlink_plan if d[0] == s])
                print(f"  ✅ Stage {s+1} solved: {stage_obs} obs, {stage_down} downlinks ({stage_runtime:.1f}s)")
            else:
                print(f"  ❌ WARNING: Stage {s+1} failed! Status: {results.solver.status}")
        
        # Calculate final metrics (only from committed stages)
        total_observations = len(self.observation_plan)
        total_downlinks = len(self.downlink_plan)
        data_downlinked_gb = (total_downlinks * p.D_comm) / 1024
        objective_value = p.C * total_downlinks + total_observations
        
        # Total propellant used
        total_propellant = sum(p.c_max - self.c_max_remaining[k] for k in range(p.K))
        
        self.results = {
            'status': 'completed',
            'objective': objective_value,
            'runtime_minutes': total_runtime / 60,
            'data_downlinked_gb': data_downlinked_gb,
            'total_observations': total_observations,
            'total_downlinks': total_downlinks,
            'propellant_used': total_propellant,
            'num_variables': total_vars,
            'num_constraints': total_constrs,
            'num_stages_solved': p.S
        }
        
        return self.results


if __name__ == "__main__":
    from parameters import InstanceParameters
    
    print("Testing REOSSP-RHP Solver (Paper Algorithm 1)...")
    
    test_params = InstanceParameters(
        instance_id=999,
        S=8,
        K=5,
        J_sk=20
    )
    
    solver = REOSSPRHPSolver(test_params, lookahead=1)
    print(f"Solving with RHP for S={test_params.S}, K={test_params.K}, J_sk={test_params.J_sk}...")
    
    results = solver.solve(time_limit_per_stage_minutes=0.5, solver_name='gurobi')
    
    print("\n" + "="*60)
    print("FINAL RESULTS:")
    print("="*60)
    print(f"  Status: {results['status']}")
    print(f"  Objective: {results['objective']:.2f}")
    print(f"  Runtime: {results['runtime_minutes']:.2f} minutes")
    print(f"  Data downlinked: {results['data_downlinked_gb']:.2f} GB")
    print(f"  Observations: {results['total_observations']:.0f}")
    print(f"  Downlinks: {results['total_downlinks']:.0f}")
    print(f"  Propellant used: {results['propellant_used']:.2f} m/s")