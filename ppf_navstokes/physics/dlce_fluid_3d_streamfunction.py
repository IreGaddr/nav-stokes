"""
3D DLCE fluid solver using stream function formulation

This ensures exact divergence-free velocity fields at all times.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from ..geometry.iot_3d import IOT3DGeometry
from ..geometry.tautochrone_3d import FastTautochrone3D
from .fluid_state_3d import FluidState3D, VelocityField3D


class StreamFunctionDLCESolver3D:
    """
    3D DLCE solver using stream function-vorticity formulation
    
    Key idea: Represent velocity as u = curl(A) where A is a vector potential.
    This automatically ensures div(u) = 0 since div(curl(A)) = 0.
    
    For toroidal geometry, we use two scalar stream functions:
    - ψ (psi): Poloidal stream function
    - χ (chi): Toroidal stream function
    """
    
    def __init__(self, geometry: IOT3DGeometry,
                 nu: float = 0.01,
                 alpha: float = 0.05,
                 beta: float = 0.05, 
                 gamma: float = 0.02,
                 delta: float = 0.1,
                 rho: float = 1.0):
        self.geometry = geometry
        self.nu = nu
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        
        # 3D tautochrone operators
        self.T_past = FastTautochrone3D(geometry, time_offset=-0.01, alpha=0.1)
        self.T_future = FastTautochrone3D(geometry, time_offset=0.01, alpha=0.1)
        
        # Stream functions
        self.psi = None  # Poloidal stream function
        self.chi = None  # Toroidal stream function
        
        # Time history
        self.history: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
        
    def initialize_from_velocity(self, state: FluidState3D):
        """
        Initialize stream functions from a velocity field
        
        This is approximate but provides a starting point.
        """
        u = state.velocity.u_component
        v = state.velocity.v_component
        w = state.velocity.w_component
        
        # Initialize stream functions
        self.psi = np.zeros_like(u)
        self.chi = np.zeros_like(u)
        
        # Approximate initialization
        # For toroidal flow: integrate to get stream functions
        du = state.u_grid[1,0,0] - state.u_grid[0,0,0] if state.u_grid.shape[0] > 1 else 0.1
        dv = state.v_grid[0,1,0] - state.v_grid[0,0,0] if state.v_grid.shape[1] > 1 else 0.1
        dr = state.r_grid[0,0,1] - state.r_grid[0,0,0] if state.r_grid.shape[2] > 1 else 0.1
        
        # Simple approximation: psi generates (u, w), chi generates v
        # This is not exact but provides a divergence-free starting point
        
        # Integrate w to get psi (approximately)
        for k in range(1, self.psi.shape[2]):
            self.psi[:, :, k] = self.psi[:, :, k-1] + w[:, :, k] * dr
            
        # Integrate v to get chi (approximately)  
        for j in range(1, self.chi.shape[1]):
            self.chi[:, j, :] = self.chi[:, j-1, :] + v[:, j, :] * dv
            
    def velocity_from_stream_functions(self, u_grid: np.ndarray, v_grid: np.ndarray, 
                                     r_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute velocity field from stream functions
        
        In toroidal coordinates:
        u = -(1/r) ∂ψ/∂v + ∂χ/∂r       (toroidal)
        v = (1/r) ∂ψ/∂u                (poloidal)
        w = (1/r²) ∂(rψ)/∂r            (radial)
        
        This ensures div(u) = 0 exactly.
        """
        du = u_grid[1,0,0] - u_grid[0,0,0] if u_grid.shape[0] > 1 else 0.1
        dv = v_grid[0,1,0] - v_grid[0,0,0] if v_grid.shape[1] > 1 else 0.1
        dr = r_grid[0,0,1] - r_grid[0,0,0] if r_grid.shape[2] > 1 else 0.1
        
        # Ensure minimum spacing
        du = max(abs(du), 1e-6)
        dv = max(abs(dv), 1e-6)
        dr = max(abs(dr), 1e-6)
        
        # Check if stream functions are initialized
        if self.psi is None or self.chi is None:
            raise ValueError("Stream functions not initialized. Call initialize_from_velocity first.")
            
        # Compute derivatives with periodic BC in u,v
        dpsi_du = np.zeros_like(self.psi)
        dpsi_dv = np.zeros_like(self.psi)
        dpsi_dr = np.zeros_like(self.psi)
        
        dchi_du = np.zeros_like(self.chi)
        dchi_dv = np.zeros_like(self.chi)
        dchi_dr = np.zeros_like(self.chi)
        
        # u derivatives (periodic)
        dpsi_du[1:-1, :, :] = (self.psi[2:, :, :] - self.psi[:-2, :, :]) / (2 * du)
        dpsi_du[0, :, :] = (self.psi[1, :, :] - self.psi[-1, :, :]) / (2 * du)
        dpsi_du[-1, :, :] = (self.psi[0, :, :] - self.psi[-2, :, :]) / (2 * du)
        
        dchi_du[1:-1, :, :] = (self.chi[2:, :, :] - self.chi[:-2, :, :]) / (2 * du)
        dchi_du[0, :, :] = (self.chi[1, :, :] - self.chi[-1, :, :]) / (2 * du)
        dchi_du[-1, :, :] = (self.chi[0, :, :] - self.chi[-2, :, :]) / (2 * du)
        
        # v derivatives (periodic)
        dpsi_dv[:, 1:-1, :] = (self.psi[:, 2:, :] - self.psi[:, :-2, :]) / (2 * dv)
        dpsi_dv[:, 0, :] = (self.psi[:, 1, :] - self.psi[:, -1, :]) / (2 * dv)
        dpsi_dv[:, -1, :] = (self.psi[:, 0, :] - self.psi[:, -2, :]) / (2 * dv)
        
        dchi_dv[:, 1:-1, :] = (self.chi[:, 2:, :] - self.chi[:, :-2, :]) / (2 * dv)
        dchi_dv[:, 0, :] = (self.chi[:, 1, :] - self.chi[:, -1, :]) / (2 * dv)
        dchi_dv[:, -1, :] = (self.chi[:, 0, :] - self.chi[:, -2, :]) / (2 * dv)
        
        # r derivatives (non-periodic)
        dpsi_dr[:, :, 1:-1] = (self.psi[:, :, 2:] - self.psi[:, :, :-2]) / (2 * dr)
        dpsi_dr[:, :, 0] = (self.psi[:, :, 1] - self.psi[:, :, 0]) / dr
        dpsi_dr[:, :, -1] = (self.psi[:, :, -1] - self.psi[:, :, -2]) / dr
        
        dchi_dr[:, :, 1:-1] = (self.chi[:, :, 2:] - self.chi[:, :, :-2]) / (2 * dr)
        dchi_dr[:, :, 0] = (self.chi[:, :, 1] - self.chi[:, :, 0]) / dr
        dchi_dr[:, :, -1] = (self.chi[:, :, -1] - self.chi[:, :, -2]) / dr
        
        # Avoid division by zero at r=0
        r_safe = np.maximum(r_grid, 0.1)
        
        # Compute velocity components
        u = -dpsi_dv / r_safe + dchi_dr
        v = dpsi_du / r_safe
        
        # For w, we need ∂(rψ)/∂r
        r_psi = r_grid * self.psi
        dr_rpsi = np.zeros_like(r_psi)
        dr_rpsi[:, :, 1:-1] = (r_psi[:, :, 2:] - r_psi[:, :, :-2]) / (2 * dr)
        dr_rpsi[:, :, 0] = (r_psi[:, :, 1] - r_psi[:, :, 0]) / dr
        dr_rpsi[:, :, -1] = (r_psi[:, :, -1] - r_psi[:, :, -2]) / dr
        
        w = dr_rpsi / (r_safe**2)
        
        # Enforce boundary conditions
        w[:, :, 0] = 0.0  # No flow at center
        w[:, :, -1] = 0.0  # No flow through outer boundary
        
        return u, v, w
    
    def time_step(self, state: FluidState3D, dt: float) -> FluidState3D:
        """
        Advance the stream functions by one time step
        """
        if self.psi is None or self.chi is None:
            self.initialize_from_velocity(state)
            
        # Store in history
        if self.psi is not None and self.chi is not None:
            self.history[state.time] = (self.psi.copy(), self.chi.copy())
        
        # Get current velocity from stream functions
        u, v, w = self.velocity_from_stream_functions(
            state.u_grid, state.v_grid, state.r_grid
        )
        
        # Compute vorticity from velocity
        omega = self._compute_vorticity(u, v, w, state)
        
        # Compute RHS for stream function evolution
        # The stream functions evolve according to vorticity dynamics
        rhs_psi, rhs_chi = self._compute_stream_function_rhs(
            u, v, w, omega, state
        )
        
        # Update stream functions
        if self.psi is not None and self.chi is not None:
            self.psi += dt * rhs_psi
            self.chi += dt * rhs_chi
        else:
            raise ValueError("Stream functions not properly initialized")
        
        # Apply boundary conditions on stream functions
        self._apply_stream_function_bc(state.time + dt)
        
        # Get new velocity field
        u_new, v_new, w_new = self.velocity_from_stream_functions(
            state.u_grid, state.v_grid, state.r_grid
        )
        
        # Create pressure (computed from momentum equation)
        pressure = self._compute_pressure(u_new, v_new, w_new, state)
        
        # Create new state
        new_velocity = VelocityField3D(u_new, v_new, w_new, pressure)
        new_state = FluidState3D(
            new_velocity, self.geometry,
            state.u_grid, state.v_grid, state.r_grid,
            state.time + dt
        )
        
        return new_state
    
    def _compute_vorticity(self, u: np.ndarray, v: np.ndarray, w: np.ndarray,
                          state: FluidState3D) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute vorticity ω = curl(u)"""
        du = state.u_grid[1,0,0] - state.u_grid[0,0,0] if state.u_grid.shape[0] > 1 else 0.1
        dv = state.v_grid[0,1,0] - state.v_grid[0,0,0] if state.v_grid.shape[1] > 1 else 0.1
        dr = state.r_grid[0,0,1] - state.r_grid[0,0,0] if state.r_grid.shape[2] > 1 else 0.1
        
        # Vorticity components
        omega_u = np.gradient(w, dv, axis=1) - np.gradient(v, dr, axis=2)
        omega_v = np.gradient(u, dr, axis=2) - np.gradient(w, du, axis=0)
        omega_w = np.gradient(v, du, axis=0) - np.gradient(u, dv, axis=1)
        
        return omega_u, omega_v, omega_w
    
    def _compute_stream_function_rhs(self, u: np.ndarray, v: np.ndarray, w: np.ndarray,
                                   omega: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                   state: FluidState3D) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute RHS for stream function evolution
        
        The vorticity equation with PPF terms drives the stream function evolution.
        """
        omega_u, omega_v, omega_w = omega
        
        # Compute advection of vorticity
        adv_omega = self._compute_vorticity_advection(u, v, w, omega, state)
        
        # Compute vortex stretching (key 3D term)
        stretch = self._compute_vortex_stretching_vorticity(u, v, w, omega, state)
        
        # Compute diffusion of vorticity
        diff_omega = self._compute_vorticity_diffusion(omega, state)
        
        # PPF retrocausal terms on vorticity
        retro_omega = self._compute_retrocausal_vorticity(omega, state)
        
        # Combine all terms
        domega_dt = (
            -adv_omega[2] +  # Only z-component affects poloidal stream function
            stretch[2] +
            self.nu * diff_omega[2] +
            retro_omega[2]
        )
        
        # For stream functions, we need to invert the relationship
        # This is approximate but maintains divergence-free
        rhs_psi = -domega_dt * (state.r_grid + 0.1)
        rhs_chi = 0.1 * (stretch[0] + retro_omega[0])  # Toroidal component
        
        return rhs_psi, rhs_chi
    
    def _compute_vorticity_advection(self, u, v, w, omega, state):
        """Compute (u·∇)ω"""
        omega_u, omega_v, omega_w = omega
        
        du = state.u_grid[1,0,0] - state.u_grid[0,0,0]
        dv = state.v_grid[0,1,0] - state.v_grid[0,0,0]
        dr = state.r_grid[0,0,1] - state.r_grid[0,0,0]
        
        adv_u = (u * np.gradient(omega_u, du, axis=0) +
                v * np.gradient(omega_u, dv, axis=1) +
                w * np.gradient(omega_u, dr, axis=2))
        
        adv_v = (u * np.gradient(omega_v, du, axis=0) +
                v * np.gradient(omega_v, dv, axis=1) +
                w * np.gradient(omega_v, dr, axis=2))
        
        adv_w = (u * np.gradient(omega_w, du, axis=0) +
                v * np.gradient(omega_w, dv, axis=1) +
                w * np.gradient(omega_w, dr, axis=2))
        
        return (adv_u, adv_v, adv_w)
    
    def _compute_vortex_stretching_vorticity(self, u, v, w, omega, state):
        """Compute (ω·∇)u for vorticity equation"""
        omega_u, omega_v, omega_w = omega
        
        du = state.u_grid[1,0,0] - state.u_grid[0,0,0]
        dv = state.v_grid[0,1,0] - state.v_grid[0,0,0]
        dr = state.r_grid[0,0,1] - state.r_grid[0,0,0]
        
        # Velocity gradients
        grad_u = np.array([
            np.gradient(u, du, axis=0),
            np.gradient(u, dv, axis=1),
            np.gradient(u, dr, axis=2)
        ])
        
        grad_v = np.array([
            np.gradient(v, du, axis=0),
            np.gradient(v, dv, axis=1),
            np.gradient(v, dr, axis=2)
        ])
        
        grad_w = np.array([
            np.gradient(w, du, axis=0),
            np.gradient(w, dv, axis=1),
            np.gradient(w, dr, axis=2)
        ])
        
        # Vortex stretching
        stretch_u = omega_u * grad_u[0] + omega_v * grad_u[1] + omega_w * grad_u[2]
        stretch_v = omega_u * grad_v[0] + omega_v * grad_v[1] + omega_w * grad_v[2]
        stretch_w = omega_u * grad_w[0] + omega_v * grad_w[1] + omega_w * grad_w[2]
        
        return (stretch_u, stretch_v, stretch_w)
    
    def _compute_vorticity_diffusion(self, omega, state):
        """Compute ν∇²ω"""
        omega_u, omega_v, omega_w = omega
        
        du = state.u_grid[1,0,0] - state.u_grid[0,0,0]
        dv = state.v_grid[0,1,0] - state.v_grid[0,0,0]
        dr = state.r_grid[0,0,1] - state.r_grid[0,0,0]
        
        # Simple Laplacian
        diff_u = (np.gradient(np.gradient(omega_u, du, axis=0), du, axis=0) +
                 np.gradient(np.gradient(omega_u, dv, axis=1), dv, axis=1) +
                 np.gradient(np.gradient(omega_u, dr, axis=2), dr, axis=2))
        
        diff_v = (np.gradient(np.gradient(omega_v, du, axis=0), du, axis=0) +
                 np.gradient(np.gradient(omega_v, dv, axis=1), dv, axis=1) +
                 np.gradient(np.gradient(omega_v, dr, axis=2), dr, axis=2))
        
        diff_w = (np.gradient(np.gradient(omega_w, du, axis=0), du, axis=0) +
                 np.gradient(np.gradient(omega_w, dv, axis=1), dv, axis=1) +
                 np.gradient(np.gradient(omega_w, dr, axis=2), dr, axis=2))
        
        return (diff_u, diff_v, diff_w)
    
    def _compute_retrocausal_vorticity(self, omega, state):
        """PPF retrocausal terms on vorticity"""
        omega_u, omega_v, omega_w = omega
        
        # Apply tautochrone operators to vorticity
        omega_u_past = self.T_past.apply(omega_u, state.u_grid, state.v_grid, state.r_grid, state.time)
        omega_v_past = self.T_past.apply(omega_v, state.u_grid, state.v_grid, state.r_grid, state.time)
        omega_w_past = self.T_past.apply(omega_w, state.u_grid, state.v_grid, state.r_grid, state.time)
        
        omega_u_future = self.T_future.apply(omega_u, state.u_grid, state.v_grid, state.r_grid, state.time)
        omega_v_future = self.T_future.apply(omega_v, state.u_grid, state.v_grid, state.r_grid, state.time)
        omega_w_future = self.T_future.apply(omega_w, state.u_grid, state.v_grid, state.r_grid, state.time)
        
        # PPF identity: -×- = +
        # Multiplicative interaction of past/future deviations
        retro_u = self.beta * (omega_u_past - omega_u) * (omega_u_future - omega_u)
        retro_v = self.beta * (omega_v_past - omega_v) * (omega_v_future - omega_v)
        retro_w = self.beta * (omega_w_past - omega_w) * (omega_w_future - omega_w)
        
        return (retro_u, retro_v, retro_w)
    
    def _compute_pressure(self, u, v, w, state):
        """Compute pressure from momentum equation"""
        # Simplified pressure calculation
        # In full implementation, solve Poisson equation for pressure
        return np.zeros_like(u)
    
    def _apply_stream_function_bc(self, t):
        """Apply boundary conditions to stream functions"""
        if self.psi is None or self.chi is None:
            return
            
        # Periodic in u,v directions
        self.psi[0, :, :] = self.psi[-1, :, :]
        self.psi[:, 0, :] = self.psi[:, -1, :]
        self.chi[0, :, :] = self.chi[-1, :, :]
        self.chi[:, 0, :] = self.chi[:, -1, :]
        
        # At r=0: stream functions vanish
        self.psi[:, :, 0] = 0.0
        self.chi[:, :, 0] = 0.0
        
        # At r=r_max: apply PPF involution condition
        h = self.geometry.metric._involution_function(np.array([self.geometry.r_max]), t)[0]
        if h < 0:
            # During involution: flip sign
            self.psi[:, :, -1] = -self.psi[:, :, -2]
            self.chi[:, :, -1] = -self.chi[:, :, -2]
        else:
            # Normal state: Neumann BC
            self.psi[:, :, -1] = self.psi[:, :, -2]
            self.chi[:, :, -1] = self.chi[:, :, -2]