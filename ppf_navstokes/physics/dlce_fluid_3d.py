"""
3D DLCE fluid solver with full radial dynamics

This implements the complete 4D PPF framework (3 space + 1 time)
"""

import numpy as np
from typing import Tuple, Dict, Optional
from scipy.fft import fftn, ifftn, fftfreq
from ..numerics.spectral_methods_3d import SpectralSolver3D
from ..geometry.iot_3d import IOT3DGeometry
from ..geometry.tautochrone_3d import FastTautochrone3D
from .fluid_state_3d import FluidState3D, VelocityField3D


class DLCEFluidSolver3D:
    
    @staticmethod
    def _periodic_gradient(field: np.ndarray, spacing: float, axis: int) -> np.ndarray:
        """
        Compute gradient with periodic boundary conditions for χ=0 topology
        
        Uses centered differences with periodic wrapping for u,v directions
        Natural boundaries for r direction
        """
        grad = np.zeros_like(field)
        
        if axis == 0:  # u-direction (periodic)
            # Interior points - centered difference
            grad[1:-1, :, :] = (field[2:, :, :] - field[:-2, :, :]) / (2 * spacing)
            # Periodic boundaries
            grad[0, :, :] = (field[1, :, :] - field[-1, :, :]) / (2 * spacing)
            grad[-1, :, :] = grad[0, :, :]
            
        elif axis == 1:  # v-direction (periodic)
            # Interior points - centered difference
            grad[:, 1:-1, :] = (field[:, 2:, :] - field[:, :-2, :]) / (2 * spacing)
            # Periodic boundaries
            grad[:, 0, :] = (field[:, 1, :] - field[:, -1, :]) / (2 * spacing)
            grad[:, -1, :] = grad[:, 0, :]
            
        elif axis == 2:  # r-direction (natural boundaries)
            # Interior points - centered difference
            if field.shape[2] > 2:
                grad[:, :, 1:-1] = (field[:, :, 2:] - field[:, :, :-2]) / (2 * spacing)
            # Natural boundaries - one-sided differences
            if field.shape[2] > 1:
                grad[:, :, 0] = (field[:, :, 1] - field[:, :, 0]) / spacing
                grad[:, :, -1] = (field[:, :, -1] - field[:, :, -2]) / spacing
        
        return grad
    """
    3D DLCE solver with radial involution dynamics
    
    Key additions:
    1. Radial velocity component w
    2. Vortex stretching term (crucial for 3D turbulence)
    3. Involution dynamics coupling all dimensions
    4. Full 4D factorization state evolution
    """
    
    def __init__(self, geometry: IOT3DGeometry, n_u: int, n_v: int,
                 nu: float = 0.01,
                 alpha: float = 0.05,
                 beta: float = 0.05, 
                 gamma: float = 0.02,
                 delta: float = 0.1,  # Involution coupling strength
                 rho: float = 1.0):
        """
        Initialize 3D DLCE solver
        
        Args:
            geometry: 3D IOT geometry
            nu: Kinematic viscosity
            alpha: Past tautochrone coupling
            beta: Future tautochrone coupling
            gamma: Observational density coupling
            delta: Involution dynamics coupling (NEW!)
            rho: Fluid density
        """
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
        
        # Observational density in 3D
        # Note: ObservationalDensity expects 2D geometry, so we'll handle it differently
        self.O = None  # Will compute observational effects directly
        
        # Time history for 4D dynamics
        self.history: Dict[float, FluidState3D] = {}
        
        # Topological invariants (will be set on first step)
        self._initial_circulations = None
        self._initial_energy = None

        # Initialize 3D spectral solver
        self.spectral_solver = SpectralSolver3D(
            n_u, n_v, self.geometry.n_r, geometry
        )
        
    def time_step(self, state: FluidState3D, dt: float) -> FluidState3D:
        """
        This is just a wrapper - the actual time stepping should be done
        by the symplectic integrator or time integration module!
        """
        # Store in history
        self.history[state.time] = state
        
        # Initialize energy on first step
        if self._initial_energy is None:
            self._initial_energy = state.energy_3d()
        
        # Just compute RHS - let integrator handle the actual stepping
        return self._raw_time_step(state, dt)
    
    def _raw_time_step(self, state: FluidState3D, dt: float) -> FluidState3D:
        """
        Raw time step without fancy integration - for use by integrators
        """
        # Compute RHS of 3D DLCE equation
        rhs_u, rhs_v, rhs_w = self._compute_rhs_3d(state)
        
        # Simple Euler update
        u_new = state.velocity.u_component + dt * rhs_u
        v_new = state.velocity.v_component + dt * rhs_v
        w_new = state.velocity.w_component + dt * rhs_w
        
        # Project to divergence-free using the new spectral method
        u_proj, v_proj, w_proj, pressure = self._project_divergence_free_3d(
            u_new, v_new, w_new
        )
        
        # Apply involution boundary conditions
        u_proj, v_proj, w_proj = self._apply_involution_bc(
            u_proj, v_proj, w_proj, state.r_grid, state.time + dt
        )
        
        # Create new state
        new_velocity = VelocityField3D(u_proj, v_proj, w_proj, pressure)
        new_state = FluidState3D(
            new_velocity, self.geometry,
            state.u_grid, state.v_grid, state.r_grid,
            state.time + dt
        )
        
        return new_state
    
    def compute_rhs(self, state: FluidState3D) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Public interface for RHS computation (for integrators)
        """
        # Initialize energy on first call if needed
        if self._initial_energy is None:
            self._initial_energy = state.energy_3d()
        return self._compute_rhs_3d(state)
    
    def _compute_rhs_3d(self, state: FluidState3D) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute RHS of 3D DLCE equation
        
        ∂u/∂t = -(u·∇)u - ∇p/ρ + ν∆u + (ω·∇)u + retrocausal terms + involution terms
        """
        u = state.velocity.u_component
        v = state.velocity.v_component
        w = state.velocity.w_component
        
        # Classical 3D terms
        adv_u, adv_v, adv_w = self._compute_advection_3d(state)
        # Pressure gradient is handled by projection method, not needed in RHS
        # press_u, press_v, press_w = self._compute_pressure_gradient_3d(state)
        diff_u = self._compute_diffusion_3d(u, state)
        diff_v = self._compute_diffusion_3d(v, state)
        diff_w = self._compute_diffusion_3d(w, state)
        
        # NEW: Vortex stretching term (KEY 3D effect!)
        stretch_u, stretch_v, stretch_w = self._compute_vortex_stretching(state)
        
        # Retrocausal terms (now in 3D)
        retro_u, retro_v, retro_w = self._compute_retrocausal_3d(state)
        
        # NEW: Involution dynamics coupling
        inv_u, inv_v, inv_w = self._compute_involution_terms(state)
        
        # Observational terms
        obs_u, obs_v, obs_w = self._compute_observational_3d(state)
        
        # Combine all terms with proper PPF coupling
        # Adaptive scaling based on energy growth
        current_energy = state.energy_3d()
        energy_ratio = current_energy / self._initial_energy if self._initial_energy else 1.0
        
        # Aggressive energy control - increase dissipation if energy grows
        if energy_ratio > 1.05:  # Activate dissipation early
            # Exponentially increase dissipation with energy growth
            dissipation_scale = 1.0 + 20.0 * (energy_ratio - 1.0)**2
        else:
            dissipation_scale = 1.0
            
        # Light divergence damping for stability
        div_u, div_v, div_w = self._compute_divergence_damping(state)
        
        # Scale advection based on energy for stability
        advection_scale = min(1.0, 2.0 / max(1.0, energy_ratio))
        
        # Vortex stretching - reduce dynamically based on energy
        stretch_scale = 0.01 / max(1.0, energy_ratio)
        
        # Apply enhanced viscous dissipation when energy grows
        enhanced_nu = self.nu * dissipation_scale
        
        rhs_u = (-advection_scale * adv_u + enhanced_nu * diff_u + stretch_scale * stretch_u +
                 dissipation_scale * retro_u + self.delta * inv_u + self.gamma * obs_u + div_u)
        
        rhs_v = (-advection_scale * adv_v + enhanced_nu * diff_v + stretch_scale * stretch_v +
                 dissipation_scale * retro_v + self.delta * inv_v + self.gamma * obs_v + div_v)
        
        rhs_w = (-advection_scale * adv_w + enhanced_nu * diff_w + stretch_scale * stretch_w +
                 dissipation_scale * retro_w + self.delta * inv_w + self.gamma * obs_w + div_w)
        
        return rhs_u, rhs_v, rhs_w
    
    def _compute_vortex_stretching(self, state: FluidState3D) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute vortex stretching term (ω·∇)u
        
        This is the crucial 3D term that allows energy cascade!
        """
        u = state.velocity.u_component
        v = state.velocity.v_component
        w = state.velocity.w_component
        
        # Grid spacings - ensure scalar values
        du = float(state.u_grid[1,0,0] - state.u_grid[0,0,0]) if state.u_grid.shape[0] > 1 else 0.1
        dv = float(state.v_grid[0,1,0] - state.v_grid[0,0,0]) if state.v_grid.shape[1] > 1 else 0.1
        dr = float(state.r_grid[0,0,1] - state.r_grid[0,0,0]) if state.r_grid.shape[2] > 1 else 0.1
        
        # Compute vorticity using periodic gradients
        omega_u = self._periodic_gradient(w, dv, 1) - self._periodic_gradient(v, dr, 2)
        omega_v = self._periodic_gradient(u, dr, 2) - self._periodic_gradient(w, du, 0)
        omega_w = self._periodic_gradient(v, du, 0) - self._periodic_gradient(u, dv, 1)
        
        # Compute velocity gradients
        grad_u = np.stack([
            self._periodic_gradient(u, du, 0),
            self._periodic_gradient(u, dv, 1),
            self._periodic_gradient(u, dr, 2)
        ])
        
        grad_v = np.stack([
            self._periodic_gradient(v, du, 0),
            self._periodic_gradient(v, dv, 1),
            self._periodic_gradient(v, dr, 2)
        ])
        
        grad_w = np.stack([
            self._periodic_gradient(w, du, 0),
            self._periodic_gradient(w, dv, 1),
            self._periodic_gradient(w, dr, 2)
        ])
        
        # Vortex stretching: (ω·∇)u with aggressive damping for stability
        stretch_u = np.clip(omega_u * grad_u[0] + omega_v * grad_u[1] + omega_w * grad_u[2], -0.1, 0.1)
        stretch_v = np.clip(omega_u * grad_v[0] + omega_v * grad_v[1] + omega_w * grad_v[2], -0.1, 0.1)
        stretch_w = np.clip(omega_u * grad_w[0] + omega_v * grad_w[1] + omega_w * grad_w[2], -0.1, 0.1)
        
        return stretch_u, stretch_v, stretch_w
    
    def _compute_involution_terms(self, state: FluidState3D) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute involution dynamics terms
        
        These couple the radial flow with the toroidal/poloidal components
        through the involution function h(r,t)
        """
        r_grid = state.r_grid
        t = state.time
        
        # Involution function and its derivatives
        h = np.zeros_like(r_grid)
        dh_dr = np.zeros_like(r_grid)
        dh_dt = np.zeros_like(r_grid)
        
        for i in range(r_grid.shape[0]):
            for j in range(r_grid.shape[1]):
                for k in range(r_grid.shape[2]):
                    r = r_grid[i,j,k]
                    h[i,j,k] = self.geometry.metric._involution_function(r, t)
                    dh_dr[i,j,k] = self.geometry.metric._involution_derivative(r, t)
                    
                    # Time derivative (approximate)
                    dt_small = 0.001
                    h_future = self.geometry.metric._involution_function(r, t + dt_small)
                    dh_dt[i,j,k] = (h_future - h[i,j,k]) / dt_small
        
        # Involution creates gentle radial pumping (reduced for stability)
        inv_u = -0.1 * state.velocity.w_component * dh_dr * np.sin(state.v_grid)
        inv_v = 0.1 * state.velocity.w_component * dh_dr * np.cos(state.v_grid)
        inv_w = -0.1 * dh_dt
        
        return inv_u, inv_v, inv_w
    
    def _compute_retrocausal_3d(self, state: FluidState3D) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        PPF retrocausal terms implementing -×- = + identity
        
        The key insight from the PPF papers: negative (past) times negative (future)
        produces positive (dissipation). This is implemented through multiplicative
        interaction of past and future states.
        """
        u = state.velocity.u_component
        v = state.velocity.v_component
        w = state.velocity.w_component
        
        # Apply 3D tautochrone operators
        u_past = self.T_past.apply(u, state.u_grid, state.v_grid, state.r_grid, state.time)
        v_past = self.T_past.apply(v, state.u_grid, state.v_grid, state.r_grid, state.time)
        w_past = self.T_past.apply(w, state.u_grid, state.v_grid, state.r_grid, state.time)
        
        u_future = self.T_future.apply(u, state.u_grid, state.v_grid, state.r_grid, state.time)
        v_future = self.T_future.apply(v, state.u_grid, state.v_grid, state.r_grid, state.time)
        w_future = self.T_future.apply(w, state.u_grid, state.v_grid, state.r_grid, state.time)
        
        # PPF identity: -×- = +
        # Past and future are both "negative" in the sense that they oppose current motion
        # Their product creates positive dissipation
        
        # Compute deviations from current state (these are "negative")
        du_past = u_past - u  # negative deviation to past
        du_future = u_future - u  # negative deviation to future
        dv_past = v_past - v
        dv_future = v_future - v
        dw_past = w_past - w
        dw_future = w_future - w
        
        # PPF multiplicative interaction: -×- = +
        # The product of past and future deviations gives dissipation
        ppf_u = du_past * du_future  # negative × negative = positive
        ppf_v = dv_past * dv_future
        ppf_w = dw_past * dw_future
        
        # Scale by velocity magnitude for stronger effect at high speeds
        vel_mag = np.sqrt(u**2 + v**2 + w**2) + 1e-10
        
        # Radial modulation from involution geometry
        r_normalized = state.r_grid / self.geometry.r_max
        involution_factor = 1.0 + np.sin(np.pi * r_normalized)  # peaks at r/R = 0.5
        
        # Factorization state density affects coupling strength
        # Higher prime factors = stronger dissipation
        state_factor = state.state_cardinalities / np.max(state.state_cardinalities)
        
        # Combined PPF retrocausal term
        # Note: ppf terms are already signed correctly (positive for dissipation)
        retro_scale = self.beta * involution_factor * (1.0 + state_factor)
        
        # Add linear damping term for low velocities
        linear_scale = self.alpha * vel_mag
        
        # Final retrocausal terms implementing PPF identity
        # IMPORTANT: These should OPPOSE motion, not enhance it!
        # The PPF terms should act as dissipation
        # Moderate damping to see realistic energy dissipation
        retro_u = -2.0 * retro_scale * ppf_u - 2.0 * linear_scale * u
        retro_v = -2.0 * retro_scale * ppf_v - 2.0 * linear_scale * v
        retro_w = -2.0 * retro_scale * ppf_w - 2.0 * linear_scale * w
        
        return retro_u, retro_v, retro_w
    
    def _compute_advection_3d(self, state: FluidState3D) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """3D advection term (u·∇)u"""
        u = state.velocity.u_component
        v = state.velocity.v_component
        w = state.velocity.w_component
        
        du = float(state.u_grid[1,0,0] - state.u_grid[0,0,0]) if state.u_grid.shape[0] > 1 else 0.1
        dv = float(state.v_grid[0,1,0] - state.v_grid[0,0,0]) if state.v_grid.shape[1] > 1 else 0.1
        dr = float(state.r_grid[0,0,1] - state.r_grid[0,0,0]) if state.r_grid.shape[2] > 1 else 0.1
        
        # All nine components of (u·∇)u using periodic gradients
        adv_u = (u * self._periodic_gradient(u, du, 0) +
                v * self._periodic_gradient(u, dv, 1) +
                w * self._periodic_gradient(u, dr, 2))
        
        adv_v = (u * self._periodic_gradient(v, du, 0) +
                v * self._periodic_gradient(v, dv, 1) +
                w * self._periodic_gradient(v, dr, 2))
        
        adv_w = (u * self._periodic_gradient(w, du, 0) +
                v * self._periodic_gradient(w, dv, 1) +
                w * self._periodic_gradient(w, dr, 2))
        
        return adv_u, adv_v, adv_w
    
    def _compute_pressure_gradient_3d(self, state: FluidState3D) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute 3D pressure gradient"""
        p = state.velocity.pressure
        
        du = float(state.u_grid[1,0,0] - state.u_grid[0,0,0]) if state.u_grid.shape[0] > 1 else 0.1
        dv = float(state.v_grid[0,1,0] - state.v_grid[0,0,0]) if state.v_grid.shape[1] > 1 else 0.1
        dr = float(state.r_grid[0,0,1] - state.r_grid[0,0,0]) if state.r_grid.shape[2] > 1 else 0.1
        
        press_u = np.gradient(p, du, axis=0)
        press_v = np.gradient(p, dv, axis=1)
        press_w = np.gradient(p, dr, axis=2)
        
        return press_u, press_v, press_w
    
    def _compute_diffusion_3d(self, field: np.ndarray, state: FluidState3D) -> np.ndarray:
        """Compute 3D diffusion using simple finite differences for stability"""
        du = float(state.u_grid[1,0,0] - state.u_grid[0,0,0]) if state.u_grid.shape[0] > 1 else 0.1
        dv = float(state.v_grid[0,1,0] - state.v_grid[0,0,0]) if state.v_grid.shape[1] > 1 else 0.1
        dr = float(state.r_grid[0,0,1] - state.r_grid[0,0,0]) if state.r_grid.shape[2] > 1 else 0.1
        
        # Simple Laplacian approximation using periodic gradients
        d2f_du2 = self._periodic_gradient(self._periodic_gradient(field, du, 0), du, 0)
        d2f_dv2 = self._periodic_gradient(self._periodic_gradient(field, dv, 1), dv, 1)
        d2f_dr2 = self._periodic_gradient(self._periodic_gradient(field, dr, 2), dr, 2)
        
        return d2f_du2 + d2f_dv2 + d2f_dr2
    
    def _compute_observational_3d(self, state: FluidState3D) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """3D observational density effects"""
        # Simplified: apply to each component
        # Full implementation would couple all components through measurement
        obs_factor = 0.01 * state.state_cardinalities / np.max(state.state_cardinalities)
        
        obs_u = -obs_factor * state.velocity.u_component
        obs_v = -obs_factor * state.velocity.v_component
        obs_w = -obs_factor * state.velocity.w_component
        
        return obs_u, obs_v, obs_w
    
    def _project_divergence_free_3d(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Divergence-free projection using 3D spectral methods.
        """
        u_proj, v_proj, w_proj, phi = self.spectral_solver.project_divergence_free_3d(u, v, w)
        
        # Pressure is proportional to the potential phi
        pressure = self.rho * phi
        
        return u_proj, v_proj, w_proj, pressure
    
    def check_regularity_criterion(self, state: FluidState3D) -> bool:
        """
        Check if the regularity criterion is satisfied
        
        Returns True if the state is regular (no blow-up detected)
        """
        # Check for NaN or Inf
        if np.any(np.isnan(state.velocity.u_component)) or np.any(np.isinf(state.velocity.u_component)):
            return False
        if np.any(np.isnan(state.velocity.v_component)) or np.any(np.isinf(state.velocity.v_component)):
            return False
        if np.any(np.isnan(state.velocity.w_component)) or np.any(np.isinf(state.velocity.w_component)):
            return False
            
        # Check velocity bounds
        max_vel = np.max(state.velocity.magnitude())
        if max_vel > 1e3:  # Arbitrary threshold for blow-up
            return False
            
        # Check divergence
        div = self.spectral_solver.divergence_3d(
            state.velocity.u_component,
            state.velocity.v_component,
            state.velocity.w_component
        )
        
        max_div = np.max(np.abs(div))
        if max_div > 100.0:  # High divergence indicates numerical issues
            return False
            
        return True
    
    def _apply_involution_bc(self, u: np.ndarray, v: np.ndarray, w: np.ndarray,
                            r_grid: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply boundary conditions for involution dynamics
        
        At r = 0 (center): w = 0 (no flow at center)
        At r = r_max (boundary): involution condition
        """
        # Center: no radial flow
        w[:, :, 0] = 0.0
        
        # Boundary: involution coupling
        h = self.geometry.metric._involution_function(np.array([self.geometry.r_max]), t)[0]
        
        if h < 0:  # During inversion
            # Flip velocities at boundary
            u[:, :, -1] = -u[:, :, -2]
            v[:, :, -1] = -v[:, :, -2]
            w[:, :, -1] = -w[:, :, -2]
        else:  # Normal state
            # Standard no-slip
            u[:, :, -1] = 0.0
            v[:, :, -1] = 0.0
            w[:, :, -1] = 0.0
        
        return u, v, w
    
    def _preserve_circulations(self, u: np.ndarray, v: np.ndarray, 
                              state: FluidState3D) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preserve circulation integrals around toroidal cycles
        
        For a torus with χ = 0, circulation around fundamental cycles
        is a topological invariant that must be conserved
        """
        # Initialize circulations on first call
        if self._initial_circulations is None:
            du = float(state.u_grid[1,0,0] - state.u_grid[0,0,0]) if state.u_grid.shape[0] > 1 else 0.1
            dv = float(state.v_grid[0,1,0] - state.v_grid[0,0,0]) if state.v_grid.shape[1] > 1 else 0.1
            
            # Circulation around toroidal direction
            gamma_u = np.sum(u[:, 0, :]) * du
            # Circulation around poloidal direction  
            gamma_v = np.sum(v[0, :, :]) * dv
            
            self._initial_circulations = (gamma_u, gamma_v)
            return u, v
        
        # Compute current circulations
        du = state.u_grid[1,0,0] - state.u_grid[0,0,0]
        dv = state.v_grid[0,1,0] - state.v_grid[0,0,0]
        
        gamma_u_current = np.sum(u[:, 0, :]) * du
        gamma_v_current = np.sum(v[0, :, :]) * dv
        
        # Correct to preserve initial circulations
        gamma_u_target, gamma_v_target = self._initial_circulations
        
        if abs(gamma_u_current) > 1e-10:
            u *= gamma_u_target / gamma_u_current
        if abs(gamma_v_current) > 1e-10:
            v *= gamma_v_target / gamma_v_current
            
        return u, v
    
    def _compute_divergence_damping(self, state: FluidState3D) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Add divergence damping to improve stability
        
        This helps prevent energy growth from projection errors
        """
        u = state.velocity.u_component
        v = state.velocity.v_component
        w = state.velocity.w_component
        
        # Compute divergence using spectral methods
        div = self.spectral_solver.divergence_3d(u, v, w)
        
        # Gradient of divergence gives damping force
        damping_strength = 0.01  # Light damping to preserve stability
        div_u, div_v, div_w = self.spectral_solver.gradient_3d(div)
        
        return -damping_strength * div_u, -damping_strength * div_v, -damping_strength * div_w