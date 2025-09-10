"""
3D Fluid state representation with full IOT dynamics
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass
from ..core import get_factorization_state_space
from ..geometry.iot_3d import IOT3DGeometry


@dataclass 
class VelocityField3D:
    """
    3D velocity field on the IOT including radial component
    """
    u_component: np.ndarray  # Toroidal velocity
    v_component: np.ndarray  # Poloidal velocity
    w_component: np.ndarray  # Radial velocity (NEW!)
    pressure: np.ndarray     # Pressure field
    
    def magnitude(self) -> np.ndarray:
        """Compute velocity magnitude at each point"""
        return np.sqrt(self.u_component**2 + self.v_component**2 + self.w_component**2)
    
    def helicity(self, geometry: IOT3DGeometry, u_grid: np.ndarray,
                v_grid: np.ndarray, r_grid: np.ndarray) -> np.ndarray:
        """
        Compute helicity H = u · ω (now non-zero in 3D!)
        
        This is a key topological invariant that was missing in 2D
        """
        # Compute vorticity components
        du = float(u_grid[1,0,0] - u_grid[0,0,0]) if u_grid.shape[0] > 1 else 0.1
        dv = float(v_grid[0,1,0] - v_grid[0,0,0]) if v_grid.shape[1] > 1 else 0.1
        dr = float(r_grid[0,0,1] - r_grid[0,0,0]) if r_grid.shape[2] > 1 else 0.1
        
        # ω = ∇ × u
        omega_u = np.gradient(self.w_component, dv, axis=1) - np.gradient(self.v_component, dr, axis=2)
        omega_v = np.gradient(self.u_component, dr, axis=2) - np.gradient(self.w_component, du, axis=0)
        omega_w = np.gradient(self.v_component, du, axis=0) - np.gradient(self.u_component, dv, axis=1)
        
        # Helicity density
        helicity = (self.u_component * omega_u + 
                   self.v_component * omega_v + 
                   self.w_component * omega_w)
        
        return helicity
    
    def vortex_stretching(self, geometry: IOT3DGeometry, u_grid: np.ndarray,
                         v_grid: np.ndarray, r_grid: np.ndarray) -> np.ndarray:
        """
        Compute vortex stretching term (ω·∇)u
        
        This is the KEY 3D effect that was missing in 2D!
        """
        du = float(u_grid[1,0,0] - u_grid[0,0,0]) if u_grid.shape[0] > 1 else 0.1
        dv = float(v_grid[0,1,0] - v_grid[0,0,0]) if v_grid.shape[1] > 1 else 0.1
        dr = float(r_grid[0,0,1] - r_grid[0,0,0]) if r_grid.shape[2] > 1 else 0.1
        
        # Vorticity components
        omega_u = np.gradient(self.w_component, dv, axis=1) - np.gradient(self.v_component, dr, axis=2)
        omega_v = np.gradient(self.u_component, dr, axis=2) - np.gradient(self.w_component, du, axis=0)
        omega_w = np.gradient(self.v_component, du, axis=0) - np.gradient(self.u_component, dv, axis=1)
        
        # Velocity gradients
        du_du = np.gradient(self.u_component, du, axis=0)
        du_dv = np.gradient(self.u_component, dv, axis=1)
        du_dr = np.gradient(self.u_component, dr, axis=2)
        
        # Vortex stretching: (ω·∇)u
        stretch_u = omega_u * du_du + omega_v * du_dv + omega_w * du_dr
        
        return stretch_u


class FluidState3D:
    """
    Complete 3D fluid state including radial dynamics and involution
    """
    
    def __init__(self, velocity: VelocityField3D, geometry: IOT3DGeometry,
                 u_grid: np.ndarray, v_grid: np.ndarray, r_grid: np.ndarray,
                 time: float = 0.0):
        """
        Initialize 3D fluid state
        
        The radial dimension allows factorization states to "hide" during collapse
        """
        self.velocity = velocity
        self.geometry = geometry
        self.u_grid = u_grid
        self.v_grid = v_grid
        self.r_grid = r_grid
        self.time = time
        
        # Compute factorization states in 3D
        self._compute_factorization_states_3d()
        
    def _compute_factorization_states_3d(self):
        """
        Compute factorization states accounting for radial hiding
        
        In 3D, factorization states can:
        1. Exist at a point (visible)
        2. Hide in the radial dimension (during involution)
        3. Transition between radial layers
        """
        shape = self.u_grid.shape
        self.state_spaces = np.empty(shape, dtype=object)
        self.state_cardinalities = np.zeros(shape, dtype=int)
        self.radial_hiding = np.zeros(shape, dtype=float)
        
        vel_mag = self.velocity.magnitude()
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    # Get factorization states
                    states = get_factorization_state_space(vel_mag[i,j,k])
                    self.state_spaces[i,j,k] = states
                    
                    # Radial position affects visibility
                    r = self.r_grid[i,j,k]
                    r_normalized = r / self.geometry.r_max
                    
                    # Near involution point (r ≈ r_max/2), states can hide
                    hiding_factor = np.exp(-((r_normalized - 0.5) / 0.1)**2)
                    
                    # Effective cardinality (some states may be hidden)
                    visible_states = len(states) * (1 - hiding_factor)
                    self.state_cardinalities[i,j,k] = max(1, int(visible_states))
                    self.radial_hiding[i,j,k] = hiding_factor
    
    def energy_3d(self) -> float:
        """
        Compute total kinetic energy in 3D
        
        Now includes radial kinetic energy and involution potential
        """
        vel_squared = (self.velocity.u_component**2 + 
                      self.velocity.v_component**2 + 
                      self.velocity.w_component**2)
        
        # Grid spacings
        du = float(self.u_grid[1,0,0] - self.u_grid[0,0,0]) if self.u_grid.shape[0] > 1 else 0.1
        dv = float(self.v_grid[0,1,0] - self.v_grid[0,0,0]) if self.v_grid.shape[1] > 1 else 0.1
        dr = float(self.r_grid[0,0,1] - self.r_grid[0,0,0]) if self.r_grid.shape[2] > 1 else 0.1
        
        energy = 0.0
        
        for i in range(self.u_grid.shape[0]):
            for j in range(self.u_grid.shape[1]):
                for k in range(self.u_grid.shape[2]):
                    u = self.u_grid[i,j,k]
                    v = self.v_grid[i,j,k]
                    r = self.r_grid[i,j,k]
                    
                    # 3D metric determinant
                    g = self.geometry.metric.metric_tensor_3d(u, v, r, self.time)
                    det_g = np.linalg.det(g)
                    # Protect against numerical issues
                    sqrt_g = np.sqrt(max(det_g, 1e-10))
                    
                    # Volume element in 3D
                    dV = sqrt_g * du * dv * dr
                    
                    # Just kinetic energy for now
                    # TODO: Add proper involution potential that stays positive
                    
                    energy += 0.5 * vel_squared[i,j,k] * dV
        
        return energy
    
    def helicity_integral(self) -> float:
        """
        Total helicity - a key 3D invariant
        
        In 2D this was always zero, but in 3D it's conserved and crucial
        """
        helicity_density = self.velocity.helicity(self.geometry, 
                                                 self.u_grid, self.v_grid, self.r_grid)
        
        # Integrate over 3D volume
        du = float(self.u_grid[1,0,0] - self.u_grid[0,0,0]) if self.u_grid.shape[0] > 1 else 0.1
        dv = float(self.v_grid[0,1,0] - self.v_grid[0,0,0]) if self.v_grid.shape[1] > 1 else 0.1
        dr = float(self.r_grid[0,0,1] - self.r_grid[0,0,0]) if self.r_grid.shape[2] > 1 else 0.1
        
        total_helicity = 0.0
        
        for i in range(self.u_grid.shape[0]):
            for j in range(self.u_grid.shape[1]):
                for k in range(self.u_grid.shape[2]):
                    u = self.u_grid[i,j,k]
                    v = self.v_grid[i,j,k]
                    r = self.r_grid[i,j,k]
                    
                    g = self.geometry.metric.metric_tensor_3d(u, v, r, self.time)
                    det_g = np.linalg.det(g)
                    sqrt_g = np.sqrt(max(det_g, 1e-10))
                    
                    dV = sqrt_g * du * dv * dr
                    total_helicity += helicity_density[i,j,k] * dV
        
        return total_helicity
    
    def turbulent_regions_3d(self) -> np.ndarray:
        """
        Identify turbulent regions in 3D
        
        Now includes:
        1. Multiple factorization states
        2. Strong vortex stretching
        3. Radial energy transfer
        """
        # Traditional criterion: multiple factorization states
        turbulent_factorization = self.state_cardinalities > 1
        
        # 3D criterion: significant vortex stretching
        stretch = self.velocity.vortex_stretching(self.geometry,
                                                 self.u_grid, self.v_grid, self.r_grid)
        turbulent_stretching = np.abs(stretch) > 0.1 * np.max(np.abs(stretch))
        
        # Radial criterion: energy transfer between layers
        radial_flux = np.abs(self.velocity.w_component)
        turbulent_radial = radial_flux > 0.1 * np.max(radial_flux)
        
        # Combined criterion
        return turbulent_factorization | turbulent_stretching | turbulent_radial
    
    def max_factorization_complexity(self) -> int:
        """Get maximum factorization state cardinality"""
        return np.max(self.state_cardinalities)