"""
Initial conditions for PPF Navier-Stokes simulations
"""

import numpy as np
from typing import Tuple
from ..physics import VelocityField, FluidState
from ..geometry import IOTGeometry


def taylor_green_vortex(geometry: IOTGeometry, 
                       u_grid: np.ndarray,
                       v_grid: np.ndarray,
                       amplitude: float = 1.0) -> FluidState:
    """
    Taylor-Green vortex initial condition adapted to toroidal geometry
    
    Args:
        geometry: IOT geometry
        u_grid: u coordinate grid
        v_grid: v coordinate grid
        amplitude: Velocity amplitude
        
    Returns:
        Initial fluid state
    """
    # Adapt Taylor-Green to toroidal coordinates
    u_component = amplitude * np.sin(2 * u_grid) * np.cos(2 * v_grid)
    v_component = -amplitude * np.cos(2 * u_grid) * np.sin(2 * v_grid)
    
    # Initial pressure (derived from velocity)
    pressure = -amplitude**2 / 4 * (np.cos(4 * u_grid) + np.cos(4 * v_grid))
    
    velocity = VelocityField(
        u_component=u_component,
        v_component=v_component,
        pressure=pressure
    )
    
    return FluidState(velocity, geometry, u_grid, v_grid, time=0.0)


def random_turbulence(geometry: IOTGeometry,
                     u_grid: np.ndarray, 
                     v_grid: np.ndarray,
                     energy_level: float = 1.0,
                     n_modes: int = 10) -> FluidState:
    """
    Random turbulent initial condition
    
    Args:
        geometry: IOT geometry
        u_grid: u coordinate grid
        v_grid: v coordinate grid
        energy_level: Target energy level
        n_modes: Number of Fourier modes
        
    Returns:
        Initial fluid state
    """
    n_u, n_v = u_grid.shape
    
    # Initialize in Fourier space
    u_hat = np.zeros((n_u, n_v), dtype=complex)
    v_hat = np.zeros((n_u, n_v), dtype=complex)
    
    # Random phases
    for k_u in range(-n_modes, n_modes+1):
        for k_v in range(-n_modes, n_modes+1):
            if k_u == 0 and k_v == 0:
                continue
                
            k = np.sqrt(k_u**2 + k_v**2)
            if k > n_modes:
                continue
                
            # Random amplitude with energy spectrum E(k) ~ k^{-2}
            amplitude = np.random.randn() / (k**2 + 1)
            phase = np.random.rand() * 2 * np.pi
            
            # Ensure divergence-free
            theta = np.arctan2(k_v, k_u)
            
            # Indices with periodic wrapping
            i_u = k_u % n_u
            i_v = k_v % n_v
            
            u_hat[i_u, i_v] = amplitude * np.exp(1j * phase) * (-np.sin(theta))
            v_hat[i_u, i_v] = amplitude * np.exp(1j * phase) * np.cos(theta)
    
    # Transform to real space
    from scipy.fft import ifft2
    u_component = np.real(ifft2(u_hat))
    v_component = np.real(ifft2(v_hat))
    
    # Normalize to desired energy
    current_energy = np.mean(u_component**2 + v_component**2)
    scale = np.sqrt(energy_level / (current_energy + 1e-10))
    u_component *= scale
    v_component *= scale
    
    # Initial pressure (zero)
    pressure = np.zeros_like(u_component)
    
    velocity = VelocityField(
        u_component=u_component,
        v_component=v_component,
        pressure=pressure
    )
    
    return FluidState(velocity, geometry, u_grid, v_grid, time=0.0)


def shear_flow(geometry: IOTGeometry,
              u_grid: np.ndarray,
              v_grid: np.ndarray,
              shear_rate: float = 1.0,
              perturbation: float = 0.1) -> FluidState:
    """
    Shear flow with small perturbation
    
    Args:
        geometry: IOT geometry
        u_grid: u coordinate grid
        v_grid: v coordinate grid
        shear_rate: Shear rate
        perturbation: Perturbation amplitude
        
    Returns:
        Initial fluid state
    """
    # Base shear flow
    u_component = shear_rate * np.sin(v_grid)
    v_component = np.zeros_like(u_grid)
    
    # Add small perturbation
    u_component += perturbation * np.random.randn(*u_grid.shape)
    v_component += perturbation * np.random.randn(*v_grid.shape)
    
    # Initial pressure
    pressure = np.zeros_like(u_component)
    
    velocity = VelocityField(
        u_component=u_component,
        v_component=v_component,
        pressure=pressure
    )
    
    return FluidState(velocity, geometry, u_grid, v_grid, time=0.0)


def vortex_pair(geometry: IOTGeometry,
               u_grid: np.ndarray,
               v_grid: np.ndarray,
               separation: float = np.pi,
               strength: float = 1.0) -> FluidState:
    """
    Counter-rotating vortex pair
    
    Args:
        geometry: IOT geometry
        u_grid: u coordinate grid
        v_grid: v coordinate grid
        separation: Separation between vortices
        strength: Vortex strength
        
    Returns:
        Initial fluid state
    """
    # Centers of vortices
    u1, v1 = np.pi - separation/2, np.pi
    u2, v2 = np.pi + separation/2, np.pi
    
    # Initialize velocity
    u_component = np.zeros_like(u_grid)
    v_component = np.zeros_like(v_grid)
    
    # Add vortices
    for sign, (u0, v0) in [(1, (u1, v1)), (-1, (u2, v2))]:
        # Distance from vortex center (with periodic wrapping)
        du = u_grid - u0
        dv = v_grid - v0
        
        # Wrap distances
        du = np.where(du > np.pi, du - 2*np.pi, du)
        du = np.where(du < -np.pi, du + 2*np.pi, du)
        dv = np.where(dv > np.pi, dv - 2*np.pi, dv)
        dv = np.where(dv < -np.pi, dv + 2*np.pi, dv)
        
        r_squared = du**2 + dv**2 + 0.1  # Regularization
        
        # Velocity field of point vortex
        u_component += sign * strength * dv / r_squared
        v_component -= sign * strength * du / r_squared
    
    # Initial pressure
    pressure = np.zeros_like(u_component)
    
    velocity = VelocityField(
        u_component=u_component,
        v_component=v_component,
        pressure=pressure
    )
    
    return FluidState(velocity, geometry, u_grid, v_grid, time=0.0)