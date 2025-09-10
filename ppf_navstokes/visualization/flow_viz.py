"""
Flow visualization for PPF Navier-Stokes
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from typing import List, Optional, Tuple
from ..physics import FluidState
from ..geometry import IOTGeometry


class FlowVisualizer:
    """
    Visualize fluid flow on the IOT
    """
    
    def __init__(self, geometry: IOTGeometry, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize flow visualizer
        
        Args:
            geometry: IOT geometry
            figsize: Figure size
        """
        self.geometry = geometry
        self.figsize = figsize
        
    def plot_velocity_field(self, state: FluidState, 
                           show_vectors: bool = True,
                           show_magnitude: bool = True,
                           show_factorization: bool = False) -> plt.Figure:
        """
        Plot velocity field
        
        Args:
            state: Fluid state
            show_vectors: Show velocity vectors
            show_magnitude: Show velocity magnitude as color
            show_factorization: Show factorization complexity
            
        Returns:
            Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        axes = axes.flatten()
        
        # Get grid
        u_grid = state.u_grid
        v_grid = state.v_grid
        
        # 1. Velocity magnitude
        ax = axes[0]
        vel_mag = state.velocity.magnitude()
        im1 = ax.contourf(u_grid, v_grid, vel_mag, levels=20, cmap='viridis')
        
        if show_vectors:
            # Subsample for vectors
            skip = max(1, len(u_grid) // 20)
            ax.quiver(u_grid[::skip, ::skip], v_grid[::skip, ::skip],
                     state.velocity.u_component[::skip, ::skip],
                     state.velocity.v_component[::skip, ::skip],
                     color='white', alpha=0.7)
        
        ax.set_xlabel('u (toroidal angle)')
        ax.set_ylabel('v (poloidal angle)')
        ax.set_title(f'Velocity Magnitude (t={state.time:.3f})')
        plt.colorbar(im1, ax=ax)
        
        # 2. Vorticity
        ax = axes[1]
        vorticity = state.velocity.vorticity(self.geometry, u_grid, v_grid)
        v_max = np.max(np.abs(vorticity))
        if v_max > 1e-10:
            im2 = ax.contourf(u_grid, v_grid, vorticity, levels=20, 
                             cmap='RdBu_r', vmin=-v_max, vmax=v_max)
            plt.colorbar(im2, ax=ax)
        else:
            ax.text(0.5, 0.5, 'Zero vorticity', transform=ax.transAxes,
                   ha='center', va='center')
        
        ax.set_xlabel('u (toroidal angle)')
        ax.set_ylabel('v (poloidal angle)')
        ax.set_title('Vorticity')
        
        # 3. Pressure
        ax = axes[2]
        im3 = ax.contourf(u_grid, v_grid, state.velocity.pressure, 
                         levels=20, cmap='plasma')
        ax.set_xlabel('u (toroidal angle)')
        ax.set_ylabel('v (poloidal angle)')
        ax.set_title('Pressure')
        plt.colorbar(im3, ax=ax)
        
        # 4. Factorization complexity
        ax = axes[3]
        if show_factorization:
            im4 = ax.contourf(u_grid, v_grid, state.state_cardinalities,
                             levels=20, cmap='hot')
            ax.set_title('Factorization State Cardinality')
            plt.colorbar(im4, ax=ax)
        else:
            # Show turbulent regions
            turbulent = state.is_turbulent()
            im4 = ax.contourf(u_grid, v_grid, turbulent.astype(float),
                             levels=[0, 0.5, 1], colors=['blue', 'red'])
            ax.set_title('Flow Regime (Blue=Laminar, Red=Turbulent)')
        
        ax.set_xlabel('u (toroidal angle)')
        ax.set_ylabel('v (poloidal angle)')
        
        plt.tight_layout()
        return fig
    
    def plot_streamlines(self, state: FluidState, 
                        n_streamlines: int = 20) -> plt.Figure:
        """
        Plot streamlines of the flow
        
        Args:
            state: Fluid state
            n_streamlines: Number of streamlines
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        u_grid = state.u_grid
        v_grid = state.v_grid
        
        # Velocity magnitude background
        vel_mag = state.velocity.magnitude()
        im = ax.contourf(u_grid, v_grid, vel_mag, levels=20, 
                        cmap='viridis', alpha=0.6)
        
        # Create streamlines
        # Note: matplotlib's streamplot expects y,x ordering
        ax.streamplot(u_grid.T, v_grid.T, 
                     state.velocity.u_component.T,
                     state.velocity.v_component.T,
                     color='white', density=1.5, linewidth=1.5)
        
        ax.set_xlabel('u (toroidal angle)')
        ax.set_ylabel('v (poloidal angle)')
        ax.set_title(f'Streamlines (t={state.time:.3f})')
        plt.colorbar(im, ax=ax, label='Velocity magnitude')
        
        return fig
    
    def animate_flow(self, states: List[FluidState],
                    filename: Optional[str] = None,
                    interval: int = 100) -> animation.FuncAnimation:
        """
        Create animation of flow evolution
        
        Args:
            states: List of fluid states
            filename: Save animation to file
            interval: Milliseconds between frames
            
        Returns:
            Animation object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Initial plot
        u_grid = states[0].u_grid
        v_grid = states[0].v_grid
        vel_mag = states[0].velocity.magnitude()
        
        im = ax.contourf(u_grid, v_grid, vel_mag, levels=20, cmap='viridis')
        ax.set_xlabel('u (toroidal angle)')
        ax.set_ylabel('v (poloidal angle)')
        title = ax.set_title(f'Velocity Magnitude (t={states[0].time:.3f})')
        
        # Color limits
        vmin = min(s.velocity.magnitude().min() for s in states)
        vmax = max(s.velocity.magnitude().max() for s in states)
        
        def update(frame):
            ax.clear()
            state = states[frame]
            vel_mag = state.velocity.magnitude()
            
            im = ax.contourf(u_grid, v_grid, vel_mag, levels=20, 
                           cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_xlabel('u (toroidal angle)')
            ax.set_ylabel('v (poloidal angle)')
            ax.set_title(f'Velocity Magnitude (t={state.time:.3f})')
            
            return [im]
        
        anim = animation.FuncAnimation(fig, update, frames=len(states),
                                     interval=interval, blit=False)
        
        if filename is not None:
            anim.save(filename, writer='pillow')
            
        return anim
    
    def plot_energy_spectrum(self, state: FluidState) -> plt.Figure:
        """
        Plot energy spectrum E(k)
        
        Args:
            state: Fluid state
            
        Returns:
            Figure object
        """
        from ..numerics import SpectralSolver
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create spectral solver
        n_u, n_v = state.u_grid.shape
        spectral = SpectralSolver(n_u, n_v, self.geometry)
        
        # Compute spectrum
        k, E_k = spectral.energy_spectrum(state.velocity.u_component,
                                        state.velocity.v_component)
        
        # Plot
        ax.loglog(k[1:], E_k[1:], 'b-', linewidth=2, label='E(k)')
        
        # Reference slopes
        k_ref = k[len(k)//4:len(k)//2]
        
        # Kolmogorov -5/3
        E_kolm = E_k[len(k)//4] * (k_ref / k[len(k)//4])**(-5/3)
        ax.loglog(k_ref, E_kolm, 'r--', label='k^{-5/3}')
        
        # Enstrophy cascade -3
        E_enst = E_k[len(k)//4] * (k_ref / k[len(k)//4])**(-3)
        ax.loglog(k_ref, E_enst, 'g--', label='k^{-3}')
        
        ax.set_xlabel('Wavenumber k')
        ax.set_ylabel('Energy E(k)')
        ax.set_title(f'Energy Spectrum (t={state.time:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig