"""
Parameter Generation Module for Satellite Scheduling Experiments
Generates random instances with synthetic visibility matrices
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class InstanceParameters:
    """Parameters for a single satellite scheduling instance"""
    # Instance ID
    instance_id: int
    
    # Variable parameters
    S: int  # Number of stages
    K: int  # Number of satellites
    J_sk: int  # Number of orbital slots per satellite per stage
    
    # Fixed parameters from Table 1
    T_r: int = 1  # Schedule duration in days
    dt: int = 100  # Time step in seconds
    T: int = 144  # Total time steps - 4 hours (144*100s = 14400s = 4h) for faster solving
    
    # Data parameters (in MB for easier computation)
    D_max: float = 128 * 1024  # 128 GB in MB
    D_obs: float = 102.5  # MB per observation
    D_comm: float = 100  # MB per downlink
    
    # Battery parameters (in kJ)
    B_max: float = 1647  # kJ
    B_charge: float = 41.48  # kJ per time step when in sun
    B_obs: float = 16.26  # kJ per observation
    B_comm: float = 1.20  # kJ per downlink
    
    # Propellant constraint
    c_max: float = 750  # m/s per satellite
    
    # Objective weight
    C: float = 2.0  # Weight for downlinks vs observations
    
    # Visibility matrices (synthetic)
    V_target: np.ndarray = None  # [K, T, targets]
    V_ground: np.ndarray = None  # [K, T, ground_stations]
    V_sun: np.ndarray = None  # [K, T]
    
    # Maneuver costs (for REOSSP)
    maneuver_costs: np.ndarray = None  # [S-1, K, J_sk, J_sk] delta-v costs
    
    def __post_init__(self):
        """Generate synthetic visibility matrices and maneuver costs"""
        # Calculate T if not provided (14 days by default)
        if self.T is None:
            self.T = (self.T_r * 86400) // self.dt  # 14 days * 86400 s/day / 100 s = 12096
        
        if self.V_target is None:
            self._generate_visibility_matrices()
        if self.maneuver_costs is None:
            self._generate_maneuver_costs()
    
    def _generate_visibility_matrices(self):
        """
        Generate synthetic visibility matrices
        Simplification: Use random binary visibility instead of SGP4 propagation
        """
        np.random.seed(self.instance_id)
        
        # Number of targets and ground stations (scaled by instance complexity)
        n_targets = 50 + self.S * 5
        n_ground = 5
        
        # Target visibility: ~5% probability per time step
        V_target_raw = np.random.random((self.K, self.T, n_targets)) < 0.10
        
        # Apply constraint: only one target visible per satellite per time step
        self.V_target = np.zeros((self.K, self.T, n_targets), dtype=bool)
        for k in range(self.K):
            for t in range(self.T):
                visible_targets = np.where(V_target_raw[k, t, :])[0]
                if len(visible_targets) > 0:
                    # Randomly select one visible target
                    selected = np.random.choice(visible_targets)
                    self.V_target[k, t, selected] = True
        
        # Ground station visibility: ~10% probability per time step
        self.V_ground = np.random.random((self.K, self.T, n_ground)) < 0.20
        
        # Sun visibility: roughly 60% of orbit (simplified eclipse model)
        # Assume ~90 minute orbit → ~54 minutes in sun
        orbit_period = 90 * 60 // self.dt  # ~54 time steps
        self.V_sun = np.zeros((self.K, self.T), dtype=bool)
        for k in range(self.K):
            # Phase offset for each satellite
            phase = np.random.randint(0, orbit_period)
            for t in range(self.T):
                # In sun for 60% of orbit
                orbit_position = (t + phase) % orbit_period
                self.V_sun[k, t] = orbit_position < int(0.6 * orbit_period)
    
    def _generate_maneuver_costs(self):
        """
        Generate maneuver costs (delta-v) between orbital slots
        Simplification: Use simple distance-based heuristic instead of orbital mechanics
        """
        np.random.seed(self.instance_id + 1000)
        
        # Maneuver costs between stages: [S-1, K, J_from, J_to]
        self.maneuver_costs = np.zeros((self.S - 1, self.K, self.J_sk, self.J_sk))
        
        for s in range(self.S - 1):
            for k in range(self.K):
                for j_from in range(self.J_sk):
                    for j_to in range(self.J_sk):
                        if j_from == j_to:
                            # No maneuver needed
                            self.maneuver_costs[s, k, j_from, j_to] = 0
                        else:
                            # Cost proportional to orbital slot distance
                            # Scale by slot spacing (more slots = closer spacing = lower cost)
                            slot_distance = abs(j_to - j_from)
                            base_cost = 50 + 150 * (slot_distance / self.J_sk)
                            # Add randomness ±20%
                            variation = np.random.uniform(0.8, 1.2)
                            self.maneuver_costs[s, k, j_from, j_to] = base_cost * variation
    
    def get_stage_boundaries(self) -> List[int]:
        """Return time step boundaries for each stage"""
        stage_length = self.T // self.S
        return [s * stage_length for s in range(self.S + 1)]


def generate_instance_set() -> List[InstanceParameters]:
    """
    Generate 24 random instances with different parameter combinations
    S ∈ {8, 9, 12}, K ∈ {5, 6}, J_sk ∈ {20, 40, 60, 80}
    """
    S_values = [8, 9, 12]
    K_values = [5, 6]
    J_sk_values = [20, 40, 60, 80]
    
    instances = []
    instance_id = 1
    
    # Generate all combinations
    for S in S_values:
        for K in K_values:
            for J_sk in J_sk_values:
                params = InstanceParameters(
                    instance_id=instance_id,
                    S=S,
                    K=K,
                    J_sk=J_sk
                )
                instances.append(params)
                instance_id += 1
    
    return instances


if __name__ == "__main__":
    # Test parameter generation
    instances = generate_instance_set()
    print(f"Generated {len(instances)} instances")
    
    # Show first instance details
    inst = instances[0]
    print(f"\nInstance 1: S={inst.S}, K={inst.K}, J_sk={inst.J_sk}")
    print(f"  Target visibility shape: {inst.V_target.shape}")
    print(f"  Ground visibility shape: {inst.V_ground.shape}")
    print(f"  Sun visibility shape: {inst.V_sun.shape}")
    print(f"  Maneuver costs shape: {inst.maneuver_costs.shape}")
    print(f"  Total possible observations: {inst.V_target.sum()}")
    print(f"  Total possible downlinks: {inst.V_ground.sum()}")
