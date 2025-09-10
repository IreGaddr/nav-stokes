"""
Observational Density functional for fluid dynamics

Implements the observer effect in the fluid evolution
"""

import numpy as np
from typing import Tuple
from ..geometry import IOTGeometry
from .fluid_state import FluidState


class ObservationalDensity:
    """
    Implements the Observational Density functional O[u]
    
    This represents the effect of observation/measurement on the fluid state
    """
    
    def __init__(self, geometry: IOTGeometry):
        """
        Initialize Observational Density operator
        
        Args:
            geometry: IOT geometry
        """
        self.geometry = geometry
        
    def complexity_function(self, u: float, v: float, t: float,
                          factorization_cardinality: int) -> float:
        """
        Compute complexity function C(x,t)
        
        The complexity increases with:
        - Factorization state space cardinality
        - Warping function magnitude
        - Distance from fixed points
        """
        # Base complexity from factorization
        C_base = np.log1p(factorization_cardinality)
        
        # Warping contribution
        W = self.geometry.warping_function(u, v, t)
        C_warp = np.abs(W)
        
        # Distance from fixed points (involution fixed points at v=0,π)
        dist_to_fixed = min(abs(v), abs(v - np.pi))
        C_fixed = 1.0 / (1.0 + dist_to_fixed)
        
        # Total complexity
        C_total = C_base + 0.5 * C_warp + 0.3 * C_fixed
        
        return C_total
    
    def smoothing_kernel(self, x1: Tuple[float, float], 
                        x2: Tuple[float, float]) -> float:
        """
        Compute smoothing kernel K(x,y)
        
        Uses a Gaussian-like kernel on the torus
        """
        u1, v1 = x1
        u2, v2 = x2
        
        # Toroidal distance
        du = min(abs(u1 - u2), 2*np.pi - abs(u1 - u2))
        dv = min(abs(v1 - v2), 2*np.pi - abs(v1 - v2))
        
        dist_squared = du**2 + dv**2
        
        # Gaussian kernel
        sigma = 0.2  # Kernel width
        kernel = np.exp(-dist_squared / (2 * sigma**2))
        
        return kernel
    
    def apply_to_velocity(self, state: FluidState) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Observational Density to velocity field
        
        O[u_i](x,t) = ∫ C(y,t)|u(y,t)|² u_i(y,t) K(x,y) dμ_f(y)
        
        Args:
            state: Current fluid state
            
        Returns:
            O[u], O[v] as arrays
        """
        u = state.velocity.u_component
        v = state.velocity.v_component
        u_grid = state.u_grid
        v_grid = state.v_grid
        
        # Initialize results
        O_u = np.zeros_like(u)
        O_v = np.zeros_like(v)
        
        # Grid spacing
        du = u_grid[1, 0] - u_grid[0, 0]
        dv = v_grid[0, 1] - v_grid[0, 0]
        
        # Apply observational operator
        for i in range(u_grid.shape[0]):
            for j in range(u_grid.shape[1]):
                x1 = (u_grid[i, j], v_grid[i, j])
                
                # Integrate over domain
                integral_u = 0.0
                integral_v = 0.0
                
                for i2 in range(u_grid.shape[0]):
                    for j2 in range(u_grid.shape[1]):
                        x2 = (u_grid[i2, j2], v_grid[i2, j2])
                        
                        # Complexity at point y
                        C = self.complexity_function(
                            x2[0], x2[1], state.time,
                            state.state_cardinalities[i2, j2]
                        )
                        
                        # Velocity magnitude squared
                        u_mag_sq = u[i2, j2]**2 + v[i2, j2]**2
                        
                        # Kernel
                        K = self.smoothing_kernel(x1, x2)
                        
                        # Fractal measure element (simplified)
                        W = self.geometry.warping_function(x2[0], x2[1], state.time)
                        measure = (1 + W)**(self.geometry.d_f - 2) * du * dv
                        
                        # Contribution to integral
                        weight = C * u_mag_sq * K * measure
                        integral_u += weight * u[i2, j2]
                        integral_v += weight * v[i2, j2]
                
                O_u[i, j] = integral_u
                O_v[i, j] = integral_v
        
        # Normalize (prevent runaway growth)
        max_O = max(np.max(np.abs(O_u)), np.max(np.abs(O_v)))
        if max_O > 1e-10:
            O_u = O_u / max_O
            O_v = O_v / max_O
            
        return O_u, O_v
    
    def decoherence_rate(self, state: FluidState) -> np.ndarray:
        """
        Compute decoherence rate at each point
        
        This determines how quickly quantum superposition collapses
        """
        rate = np.zeros_like(state.u_grid)
        
        for i in range(state.u_grid.shape[0]):
            for j in range(state.u_grid.shape[1]):
                u, v = state.u_grid[i, j], state.v_grid[i, j]
                
                # Base rate depends on complexity
                C = self.complexity_function(
                    u, v, state.time,
                    state.state_cardinalities[i, j]
                )
                
                # Anisotropic factor (from rotating IOT)
                # Decoherence is stronger along rotation axis
                anisotropy = 1.0 + 0.5 * np.cos(v)**2
                
                rate[i, j] = C * anisotropy
                
        return rate