# REOSSP Metaheuristic Algorithms Integration

This integration brings the converted metaheuristic algorithms (BGA, IDE, PSO, RGA) from MATLAB to solve your REOSSP problem alongside your custom Two-Phase GA.

## New Files Created

### 1. `reossp_metaheuristics.py`
Main integration file that wraps the converted algorithms to solve REOSSP.

**Key Components:**
- `REOSSPMetaheuristicSolver`: Main solver class
- `decode_solution()`: Converts continuous [0,1] solutions to discrete orbital slot assignments
- `objective_function()`: Evaluates fitness using your greedy scheduler
- `solve_bga()`, `solve_ide()`, `solve_pso()`, `solve_rga()`: Individual algorithm solvers

**Algorithm Mapping:**
- **Chromosome Encoding**: Flattened K×S vector of continuous values in [0,1]
- **Decoding**: Maps each continuous value to a slot index [1, J_sk]
- **Fitness Evaluation**: Uses your `GreedyScheduler` from Two-Phase GA

### 2. `run_all_metaheuristics.py`
Comprehensive experiment runner that tests all algorithms on all problem instances.

**Runs:**
- BGA (Binary Genetic Algorithm)
- IDE (Improved Differential Evolution)
- PSO (Particle Swarm Optimization)
- RGA (Real-Coded Genetic Algorithm)
- Two-Phase GA (your custom algorithm)

**Output:** `results/results_all_metaheuristics.csv`

### 3. `visualize_all_metaheuristics.py`
Creates comprehensive visualizations comparing all algorithms.

**Generates:**
- Line charts for: Objective, Runtime, Figure of Merit, Feasibility Rate
- Bar charts for overall comparison
- Statistical summaries

**Output:** 
- `visualizations/all_metaheuristics_comparison.png`
- `visualizations/overall_comparison_bars.png`

### 4. `test_metaheuristics.py`
Quick test script to verify all algorithms work correctly on a small instance.

## How to Use

### Quick Test (Recommended First Step)
```bash
python test_metaheuristics.py
```
This runs a quick test on a small instance (K=3, S=6) to verify everything works.

### Run Full Experiment
```bash
python run_all_metaheuristics.py
```
This runs all 5 algorithms on all 30 problem instances (5 unavailable probabilities × 6 S-K pairs).

**Warning:** This will take considerable time as it runs:
- 30 instances × 5 algorithms = 150 algorithm runs
- Each algorithm runs for max_fes = 5000 evaluations

**Estimated Time:** 2-4 hours (depending on your hardware)

### Visualize Results
```bash
python visualize_all_metaheuristics.py
```
After running the full experiment, this creates comprehensive comparison charts.

## Algorithm Comparison

| Algorithm | Type | Search Space | Strengths |
|-----------|------|--------------|-----------|
| **BGA** | Binary GA | Binary {0,1} | Simple, good for discrete problems |
| **IDE** | Differential Evolution | Continuous [0,1] | Adaptive parameters, archive |
| **PSO** | Swarm Intelligence | Continuous [0,1] | Fast convergence, simple |
| **RGA** | Real-Coded GA | Continuous [0,1] | SBX crossover, polynomial mutation |
| **Two-Phase GA** | Custom GA | Discrete slots | Problem-specific, custom operators |

## Configuration Parameters

### Metaheuristic Algorithms (BGA, IDE, PSO, RGA)
```python
pop_size = 50      # Population size
max_fes = 5000     # Maximum function evaluations
```

### Two-Phase GA
```python
pop_size = 50         # Population size
n_generations = 30    # Number of generations
```

**Note:** Two-Phase GA uses generations instead of function evaluations because it has a different evaluation model (custom chromosome, problem-specific operators).

## Understanding the Integration

### Chromosome Encoding

**Converted Algorithms (BGA, IDE, PSO, RGA):**
```
Chromosome = [x1, x2, ..., x(K×S)]
where each xi ∈ [0, 1]

Decoded to:
Trajectory[k,s] = floor(x[k*S + s] * J_sk) + 1
```

**Two-Phase GA:**
```
Chromosome = [[s1_k1, s2_k1, ..., sS_k1],
              [s1_k2, s2_k2, ..., sS_k2],
              ...
              [s1_kK, s2_kK, ..., sS_kK]]
where each s_ks ∈ {1, 2, ..., J_sk}
```

### Fitness Evaluation
All algorithms use the same `GreedyScheduler` from your Two-Phase GA:
1. Decode chromosome to orbital trajectory
2. Check propellant feasibility
3. Check slot availability
4. Run greedy scheduler
5. Return objective value

### Feasibility Handling
- **Propellant Budget**: Delta-v must be within budget for each satellite
- **Slot Availability**: Cannot use unavailable slots
- **Penalty**: Infeasible solutions receive -1e9 fitness

## Output Files

### CSV Format
The results CSV includes for each algorithm:
- `{alg}_objective`: Final objective value
- `{alg}_total_observations`: Number of observations
- `{alg}_total_downlinks`: Number of downlinks
- `{alg}_runtime_minutes`: Execution time
- `{alg}_propellant_used`: Total propellant consumption
- `{alg}_feasibility_rate`: Percentage of feasible solutions found
- `{alg}_figure_of_merit`: Objective / Runtime

### Visualization Outputs
1. **Line Charts**: Show how each metric varies with unavailable probability
2. **Bar Charts**: Overall performance comparison across all instances
3. **Statistics**: Mean and standard deviation for each algorithm

## Tips for Usage

### For Quick Exploration
```python
from reossp_metaheuristics import REOSSPMetaheuristicSolver
from parameters import InstanceParameters

# Create instance
params = InstanceParameters(instance_id=1, S=8, K=5, J_sk=20, T=36*24*2)

# Create solver
solver = REOSSPMetaheuristicSolver(params)

# Run single algorithm
results = solver.solve_rga(pop_size=50, max_fes=5000, verbose=True)

print(f"Objective: {results['objective']}")
print(f"Runtime: {results['runtime_minutes']:.2f} minutes")
```

### For Custom Experiments
Modify `run_all_metaheuristics.py`:
- Change `pop_size` and `max_fes`
- Add/remove problem instances
- Adjust algorithm parameters in `reossp_metaheuristics.py`

### For Algorithm Tuning
Each algorithm has parameters you can tune in `reossp_metaheuristics.py`:

**BGA:**
- `p_single_point`: Single point crossover probability
- `p_double_point`: Double point crossover probability

**IDE:**
- `pbest_rate`: Rate for selecting from top individuals
- `mem_size`: Memory size for parameter adaptation
- `arc_rate`: Archive rate

**PSO:**
- `w`: Inertia weight
- `c1`: Cognitive coefficient
- `c2`: Social coefficient

## Expected Results

Based on the algorithm characteristics:

1. **IDE** - Likely best objective (adaptive, sophisticated)
2. **RGA** - Good balance of quality and speed
3. **PSO** - Fast convergence but may get stuck
4. **BGA** - Simple but effective for discrete problems
5. **Two-Phase GA** - Problem-specific, should be competitive

Run the experiments to see which algorithm works best for your REOSSP instances!

## Troubleshooting

### Import Errors
Make sure `integerGA_python` folder is in the same directory as your REOSSP files.

### Memory Issues
Reduce `pop_size` or `max_fes` if you run out of memory.

### Slow Execution
- Reduce number of instances in `run_all_metaheuristics.py`
- Decrease `max_fes` for faster (but less optimized) results
- Run individual algorithms instead of all at once

### Low Feasibility Rate
If feasibility rate is very low (<10%), the problem might be over-constrained:
- Check propellant budget settings
- Verify unavailable slot configuration
- Inspect `unavailable_slot_probability`

## Next Steps

1. ✅ Run `test_metaheuristics.py` to verify setup
2. ✅ Run `run_all_metaheuristics.py` for full comparison
3. ✅ Run `visualize_all_metaheuristics.py` to analyze results
4. 📊 Compare with your existing results from exact/RHP methods
5. 📝 Write up findings and select best algorithm for your problem

Good luck with your experiments! 🚀
