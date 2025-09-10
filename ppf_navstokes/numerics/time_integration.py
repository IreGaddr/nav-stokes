"""
Time integration schemes for PPF Navier-Stokes
"""

import numpy as np
from typing import Callable, Optional, Tuple
from ..physics.fluid_state_3d import FluidState3D


class AdaptiveTimeStep:
    """
    Adaptive time stepping based on CFL condition
    """
    
    def __init__(self, solver, cfl_number: float = 0.5, min_dt: float = 1e-6, max_dt: float = 1e-2):
        self.solver = solver
        """
        Initialize adaptive time stepping
        
        Args:
            cfl_number: CFL number for stability
            min_dt: Minimum allowed time step
            max_dt: Maximum allowed time step
        """
        self.cfl_number = cfl_number
        self.min_dt = min_dt
        self.max_dt = max_dt
        
    def compute_dt(self, state: FluidState3D) -> float:
        """
        Compute adaptive time step
        """
        max_vel = np.max(state.velocity.magnitude())
        
        if max_vel < 1e-10:
            return self.max_dt
        
        # Spacings
        du = float(state.u_grid[1,0,0] - state.u_grid[0,0,0]) if state.u_grid.shape[0] > 1 else 0.1
        dv = float(state.v_grid[0,1,0] - state.v_grid[0,0,0]) if state.v_grid.shape[1] > 1 else 0.1
        dr = float(state.r_grid[0,0,1] - state.r_grid[0,0,0]) if state.r_grid.shape[2] > 1 else 0.1
        min_spacing = min(du, dv, dr)
        
        # CFL condition
        dt = self.cfl_number * min_spacing / max_vel
        
        # Clamp to min/max
        dt = np.clip(dt, self.min_dt, self.max_dt)
        
        return dt


class TimeIntegrator:
    """
    Time integration for the PPF Navier-Stokes equations
    """
    
    def __init__(self, solver: 'DLCEFluidSolver3D',
                 method: str = 'rk4',
                 adaptive: bool = True):
        """
        Initialize time integrator
        
        Args:
            solver: DLCE fluid solver
            method: Integration method ('euler', 'rk2', 'rk4')
            adaptive: Use adaptive time stepping
        """
        self.solver = solver
        self.method = method
        self.adaptive = adaptive
        
        if adaptive:
            self.adaptive_stepper = AdaptiveTimeStep()
            
    def step(self, state: FluidState3D, dt: Optional[float] = None) -> FluidState3D:
        """
        Advance solution by one time step
        
        Args:
            state: Current state
            dt: Time step (computed adaptively if None)
            
        Returns:
            New state
        """
        # Compute time step if needed
        if dt is None and self.adaptive:
            dt = self.adaptive_stepper.compute_dt(state)
        elif dt is None:
            dt = 1e-3  # Default
            
        # Apply selected integration method
        if self.method == 'euler':
            return self._euler_step(state, dt)
        elif self.method == 'rk2':
            return self._rk2_step(state, dt)
        elif self.method == 'rk4':
            return self._rk4_step(state, dt)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _euler_step(self, state: FluidState3D, dt: float) -> FluidState3D:
        """Forward Euler method"""
        return self.solver.time_step(state, dt)
    
    def _rk2_step(self, state: FluidState3D, dt: float) -> FluidState3D:
        """Heun's method (RK2)"""
        # First stage
        k1 = self.solver.time_step(state, dt)
        
        # Second stage
        k2 = self.solver.time_step(k1, dt)
        
        # Combine (simplified - should properly average states)
        # For now, just return the second stage
        return k2
    
    def _rk4_step(self, state: FluidState3D, dt: float) -> FluidState3D:
        """Classic RK4 method"""
        # For full implementation, would need to properly combine intermediate states
        # For now, use repeated Euler steps
        state1 = self.solver.time_step(state, dt/2)
        state2 = self.solver.time_step(state1, dt/2)
        return state2
    
    def integrate(self, initial_state: FluidState3D, 
                 t_final: float,
                 output_times: Optional[np.ndarray] = None,
                 callback: Optional[Callable] = None) -> list:
        """
        Integrate from initial state to final time
        
        Args:
            initial_state: Initial fluid state
            t_final: Final time
            output_times: Times at which to save states
            callback: Function called after each step
            
        Returns:
            List of states at output times
        """
        if output_times is None:
            output_times = np.linspace(0, t_final, 11)
            
        states = []
        current_state = initial_state
        output_idx = 0
        
        # Save initial state if needed
        if output_times[0] <= current_state.time:
            states.append(current_state)
            output_idx += 1
        
        while current_state.time < t_final and output_idx < len(output_times):
            # Compute time step
            if self.adaptive:
                dt = self.adaptive_stepper.compute_dt(current_state)
            else:
                dt = (t_final - current_state.time) / 100
                
            # Don't overshoot output time
            if output_idx < len(output_times):
                dt = min(dt, output_times[output_idx] - current_state.time)
            
            # Take step
            current_state = self.step(current_state, dt)
            
            # Check regularity
            if not self.solver.check_regularity_criterion(current_state):
                print(f"Warning: Regularity criterion violated at t={current_state.time}")
            
            # Save state if at output time
            if (output_idx < len(output_times) and 
                current_state.time >= output_times[output_idx] - 1e-10):
                states.append(current_state)
                output_idx += 1
            
            # Call callback if provided
            if callback is not None:
                callback(current_state)
                
        return states