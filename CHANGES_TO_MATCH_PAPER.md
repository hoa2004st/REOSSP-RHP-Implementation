# Changes Made to Match Paper Implementation

This document explains all changes made to match the paper "The Reconfigurable Earth Observation Satellite Scheduling Problem" more closely.

## 1. Parameters (parameters.py)

### Changes Made:
1. **4D Visibility Matrices** (CRITICAL)
   - **Old**: 3D `V_target[K, T, targets]` - visibility per satellite/time
   - **New**: 4D `V_target[S, K, T_per_stage, J, targets]` - visibility per satellite/time/**orbital slot**
   - **Paper Reference**: Constraint (11a-c) shows `V^s_{ktjp}` depends on stage s, time t, slot j
   - **Why**: Different orbital slots see different targets - this is the KEY innovation of REOSSP

2. **Propellant Budget**
   - **Old**: `c_max = 750 m/s`
   - **New**: `c_max = 1800 m/s`
   - **Paper Reference**: Paper mentions ~1.8 km/s for constellation reconfiguration
   - **Why**: More realistic LEO orbital transfer budget

3. **Battery Reconfiguration Cost**
   - **Old**: Not included
   - **New**: `B_recon = 0.50 kJ` per maneuver
   - **Paper Reference**: Section 3.3.4, Constraint (14d)
   - **Why**: Maneuvers consume battery energy for thruster firing

4. **Maneuver Cost Structure**
   - **Old**: Complex formula with randomness, indexed [S-1, K, J, J]
   - **New**: Simple distance-based `abs(i - j) * 0.02`, indexed [S, K, J, J]
   - **Paper Reference**: Section 3.3.2 notation `c^s_{kij}`
   - **Why**: Matches all.py and paper indexing (costs exist for all stages including first)

### Code Comments Added:
```python
# Paper constraint (11a-c): y^s_{ktp} <= sum_{i,j} V^s_{ktjp} * x^s_{kij}
# Visibility depends on which orbital slot j the satellite is in
V_target: np.ndarray = None  # [S, K, T_per_stage, J_sk, targets]
```

---

## 2. EOSSP Baseline (eossp_baseline.py)

### Changes Made:

1. **1-Based Indexing**
   - **Old**: 0-based Python indexing
   - **New**: 1-based to match paper notation exactly
   - **Paper Reference**: All equations use k ∈ {1,...,K}, t ∈ {1,...,T}
   - **Why**: Makes comparing code to paper equations easier

2. **Variable Naming**
   - **Old**: `x` for observations, `y` for downlinks, `c` for charging
   - **New**: `y` for observations, `q` for downlinks, `h` for charging
   - **Paper Reference**: Equations (2a-2e)
   - **Why**: Direct match to paper notation

3. **Fixed Slot Assumption**
   - **Old**: Used 3D visibility directly
   - **New**: Extracts fixed slot (j=0) from 4D visibility
   - **Paper Reference**: EOSSP baseline assumes no reconfiguration
   - **Why**: Satellites stay at initial slot throughout mission

4. **Detailed Constraint Comments**
   - Added references to specific paper equations for EVERY constraint
   - Example:
   ```python
   def visibility_target_rule(m, k, t, p):
       # Paper Eq (4a): y_ktp <= V_ktp (target visibility)
       if not V_target_flat[k-1, t-1, p-1]:  # Convert to 0-based indexing
           return m.y[k, t, p] == 0
       return pyo.Constraint.Skip
   ```

5. **Constraint Organization**
   - Grouped by paper sections:
     - Visibility Constraints (4a-4d)
     - Data Storage Dynamics (5a-5c)
     - Battery Dynamics (6a-6c)

### Paper Equation Mapping:
- **Eq (3)**: Objective function
- **Eq (4a)**: `vis_target` constraint
- **Eq (4b)**: `vis_ground` constraint
- **Eq (4c)**: `vis_sun` constraint
- **Eq (4d)**: `one_activity` constraint
- **Eq (5a)**: `data_dynamics` constraint
- **Eq (5b)**: `data_upper` constraint
- **Eq (5c)**: `data_lower` constraint
- **Eq (6a)**: `battery_dynamics` constraint
- **Eq (6b)**: `battery_upper` constraint
- **Eq (6c)**: `battery_lower` constraint

---

## 3. REOSSP Exact (reossp_exact.py)

### Changes Made:

1. **Slot-Dependent Visibility** (CRITICAL)
   - **Old**: `if not p.V_target[k, t, n]: return m.x[k, t, n] == 0`
   - **New**:
   ```python
   def visibility_target_rule(m, s, k, t, p):
       # Paper Eq (11a): y^s_{ktp} <= sum_{i,j} V^s_{ktjp} * x^s_{kij}
       if s == 1:
           visibility_sum = sum(p.V_target[s-1, k-1, t-1, j-1, p-1] * m.x[s, k, 1, j] 
                               for j in m.J)
       else:
           visibility_sum = sum(p.V_target[s-1, k-1, t-1, j-1, p-1] * m.x[s, k, i, j] 
                               for i in m.J for j in m.J)
       return m.y[s, k, t, p] <= visibility_sum
   ```
   - **Paper Reference**: Constraint (11a)
   - **Why**: Visibility now depends on which slot satellite occupies

2. **Initial Battery with Maneuver Cost** (CRITICAL)
   - **Old**: `return m.b[k, 0] == p.B_max`
   - **New**:
   ```python
   def battery_initial_rule(m, k):
       # Paper Eq (14d): b^1_{k1} = B_max - B_recon*sum_j x^1_{k,i_init,j}
       i_init = 1
       maneuver_cost_initial = sum(m.x[1, k, i_init, j] for j in m.J) * p.B_recon
       return m.b[1, k, 1] == p.B_max - maneuver_cost_initial
   ```
   - **Paper Reference**: Constraint (14d)
   - **Why**: First maneuver from initial position costs battery energy

3. **Stage Transition with Maneuver Cost** (CRITICAL)
   - **Old**: Battery/data just carry over between stages
   - **New**:
   ```python
   def battery_tracking_between_stages_rule(m, s, k):
       # Paper Eq (13b): includes - B_recon*sum_{i,j} x^{s+1}_{kij}
       maneuver_cost = sum(m.x[s+1, k, i, j] for i in m.J for j in m.J) * p.B_recon
       return m.b[s+1, k, 1] == b_end + charge - obs - comm - maneuver_cost
   ```
   - **Paper Reference**: Constraints (13b), (14c)
   - **Why**: Maneuvers between stages consume battery

4. **Complete Constraint Set**
   - Added ALL paper constraints with exact equation references:
   
   **Orbital Reconfiguration (10a-10c)**:
   - (10a) `initial_slot`: First maneuver from initial position
   - (10b) `slot_continuity`: Slot assignments linked across stages
   - (10c) `propellant_budget`: Total delta-v constraint
   
   **Visibility (11a-11d)**:
   - (11a) `vis_target`: Observation depends on slot
   - (11b) `vis_ground`: Downlink depends on slot
   - (11c) `vis_sun`: Charging depends on slot
   - (11d) `one_activity`: At most one action per timestep
   
   **Data Storage (12a-12f)**:
   - (12a) `data_tracking_within`: Within-stage dynamics
   - (12b) `data_tracking_between`: Between-stage transitions
   - (12c) `data_upper`: Storage capacity
   - (12d) `data_lower`: Non-negative storage
   - (12e) Implicit in initial condition
   - (12f) Implicit in initial condition
   
   **Battery (13a-13b, 14a-14d)**:
   - (13a) `battery_tracking_within`: Within-stage dynamics
   - (13b) `battery_tracking_between`: Between-stage with maneuver cost
   - (14a) `battery_upper`: Max capacity
   - (14b) `battery_lower_within`: Min level during operations
   - (14c) `battery_lower_between`: Min level after maneuvers
   - (14d) `battery_initial`: Initial battery minus first maneuver

---

## 4. Expected Behavior Changes

### Why REOSSP Should Now Outperform EOSSP:

1. **Slot-Dependent Visibility**
   - Different orbital slots have different visibility windows
   - REOSSP can move to slots with better target/ground access
   - EOSSP stuck at initial slot with limited visibility

2. **Realistic Costs**
   - Higher propellant budget (1800 vs 750 m/s) allows more maneuvers
   - Battery cost makes maneuvers non-trivial but feasible
   - More opportunities for optimization

3. **Proper Constraint Enforcement**
   - Initial maneuver cost prevents "free" reconfiguration
   - Stage transitions properly account for energy costs
   - Matches all.py behavior (which shows clear REOSSP advantage)

### Test Results Expected:

```
EOSSP Baseline:  Fixed at slot 1, limited visibility
REOSSP Exact:    Can reconfigure to better slots → Higher objective
REOSSP RHP:      Faster than Exact, slightly lower objective
```

---

## 5. How to Verify Code Matches Paper

### For Each Constraint, Check:

1. **Find the equation number** in the paper
2. **Locate the corresponding constraint** in code
3. **Compare term-by-term**:
   - Variable names match?
   - Summation indices correct?
   - <= vs >= direction?
   - Constants (D_obs, B_charge, etc.) correct?

### Example Verification:

**Paper Eq (11a)**: `y^s_{ktp} <= sum_{i in J^{s-1}, j in J^s} V^s_{ktjp} * x^s_{kij}`

**Code**:
```python
def visibility_target_rule(m, s, k, t, p):
    # Paper Eq (11a): y^s_{ktp} <= sum_{i,j} V^s_{ktjp} * x^s_{kij}
    visibility_sum = sum(p.V_target[s-1, k-1, t-1, j-1, p-1] * m.x[s, k, i, j] 
                        for i in m.J for j in m.J)
    return m.y[s, k, t, p] <= visibility_sum
```

✓ **Matches**: Variable names, inequality direction, summation structure all correct!

---

## 6. Quick Reference: Variable Mapping

| Paper | Code (Old) | Code (New) | Description |
|-------|-----------|-----------|-------------|
| `y^s_{ktp}` | `x[k,t,n]` | `y[s,k,t,p]` | Observation decision |
| `q^s_{ktg}` | `y[k,t,g]` | `q[s,k,t,g]` | Downlink decision |
| `h^s_{kt}` | `c[k,t]` | `h[s,k,t]` | Charging decision |
| `x^s_{kij}` | `z[s,k,j]` + `u[s,k,i,j]` | `x[s,k,i,j]` | Reconfiguration decision |
| `d^s_{kt}` | `d[k,t]` | `d[s,k,t]` | Data storage state |
| `b^s_{kt}` | `b[k,t]` | `b[s,k,t]` | Battery state |
| `V^s_{ktjp}` | `V_target[k,t,n]` | `V_target[s,k,t,j,p]` | Target visibility |
| `W^s_{ktjg}` | `V_ground[k,t,g]` | `V_ground[s,k,t,j,g]` | Ground visibility |
| `H^s_{ktj}` | `V_sun[k,t]` | `V_sun[s,k,t,j]` | Sun visibility |

---

## 7. Files Changed Summary

1. ✅ **parameters.py**
   - 4D visibility structure
   - Increased propellant budget
   - Added B_recon parameter
   - Simplified maneuver costs

2. ✅ **eossp_baseline.py**
   - 1-based indexing
   - Renamed variables to match paper
   - Added constraint equation references
   - Fixed slot extraction from 4D visibility

3. ✅ **reossp_exact.py**
   - Complete rewrite with paper constraints
   - Slot-dependent visibility (11a-c)
   - Initial battery with maneuver cost (14d)
   - Stage transitions with maneuver costs (13b, 14c)
   - All constraints documented with equation numbers

4. ⚠️ **reossp_rhp.py** (needs update)
   - Should follow same pattern as reossp_exact.py
   - Apply to subproblems in rolling horizon
   - Same constraints but for subset of stages

---

## 8. Testing Recommendations

Run the updated code and check for:

1. **REOSSP > EOSSP**: With slot-dependent visibility, REOSSP should show clear advantage
2. **Propellant usage**: Should see ~0-1.8 km/s used (not 0 and not exceeding budget)
3. **Battery depletion**: Initial battery should be < B_max due to first maneuver
4. **Feasibility**: All constraints should be satisfied
5. **Objective improvement**: 10-30% improvement typical in paper results

---

## 9. Paper Section References

- **Section 3.2**: EOSSP formulation (constraints 2-6)
- **Section 3.3**: REOSSP formulation (constraints 7-14)
- **Section 3.3.1**: Decision variables
- **Section 3.3.2**: Orbital reconfiguration constraints (10)
- **Section 3.3.3**: Visibility constraints (11)
- **Section 3.3.4**: Resource dynamics (12-14)
- **Section 4.3**: RHP algorithm
- **Section 5**: Computational experiments

---

## Contact for Questions

If you find discrepancies between code and paper:
1. Check the equation number in code comments
2. Look up that equation in the paper
3. Compare term-by-term
4. Verify indexing (1-based in paper, 0-based in arrays)
5. Check for off-by-one errors in conversions
