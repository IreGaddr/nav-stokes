"""
Diagnostic plotting for PPF Navier-Stokes simulations
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from ..physics import FluidState


class DiagnosticPlotter:
    """
    Plot diagnostic quantities for fluid simulations
    """
    
    def __init__(self):
        """Initialize diagnostic plotter"""
        self.history: Dict[str, List[float]] = {
            'time': [],
            'energy': [],
            'enstrophy': [],
            'max_velocity': [],
            'max_vorticity': [],
            'max_factorization': [],
            'turbulent_fraction': []
        }
        
    def update(self, state: FluidState):
        """
        Update diagnostic history with current state
        
        Args:
            state: Current fluid state
        """
        self.history['time'].append(state.time)
        self.history['energy'].append(state.energy())
        self.history['enstrophy'].append(state.enstrophy())
        
        vel_mag = state.velocity.magnitude()
        self.history['max_velocity'].append(np.max(vel_mag))
        
        vorticity = state.velocity.vorticity(
            state.geometry, state.u_grid, state.v_grid
        )
        self.history['max_vorticity'].append(np.max(np.abs(vorticity)))
        
        self.history['max_factorization'].append(
            state.max_factorization_complexity()
        )
        
        turbulent_fraction = np.mean(state.is_turbulent())
        self.history['turbulent_fraction'].append(turbulent_fraction)
        
    def plot_time_series(self) -> plt.Figure:
        """
        Plot time series of diagnostic quantities
        
        Returns:
            Figure object
        """
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        t = np.array(self.history['time'])
        
        # Energy
        ax = axes[0]
        ax.plot(t, self.history['energy'], 'b-', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Energy')
        ax.set_title('Total Energy')
        ax.grid(True, alpha=0.3)
        
        # Enstrophy
        ax = axes[1]
        ax.semilogy(t, self.history['enstrophy'], 'r-', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Enstrophy')
        ax.set_title('Total Enstrophy')
        ax.grid(True, alpha=0.3)
        
        # Max velocity
        ax = axes[2]
        ax.plot(t, self.history['max_velocity'], 'g-', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Max |u|')
        ax.set_title('Maximum Velocity')
        ax.grid(True, alpha=0.3)
        
        # Max vorticity
        ax = axes[3]
        ax.semilogy(t, self.history['max_vorticity'], 'm-', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Max |ω|')
        ax.set_title('Maximum Vorticity')
        ax.grid(True, alpha=0.3)
        
        # Max factorization complexity
        ax = axes[4]
        ax.plot(t, self.history['max_factorization'], 'c-', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Max |S(u)|')
        ax.set_title('Maximum Factorization Complexity')
        ax.grid(True, alpha=0.3)
        
        # Turbulent fraction
        ax = axes[5]
        ax.plot(t, self.history['turbulent_fraction'], 'orange', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Fraction')
        ax.set_title('Turbulent Area Fraction')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_conservation_check(self) -> plt.Figure:
        """
        Check conservation properties
        
        Returns:
            Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        t = np.array(self.history['time'])
        energy = np.array(self.history['energy'])
        
        if len(energy) > 0:
            # Energy conservation (relative change)
            energy_change = (energy - energy[0]) / energy[0]
            ax1.plot(t, energy_change * 100, 'b-', linewidth=2)
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Energy Change (%)')
            ax1.set_title('Energy Conservation Check')
            ax1.grid(True, alpha=0.3)
            
            # Energy dissipation rate
            if len(t) > 1:
                dt = np.diff(t)
                dE_dt = np.diff(energy) / dt
                ax2.plot(t[1:], -dE_dt, 'r-', linewidth=2)
                ax2.set_xlabel('Time')
                ax2.set_ylabel('-dE/dt')
                ax2.set_title('Energy Dissipation Rate')
                ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_regularity_indicators(self) -> plt.Figure:
        """
        Plot indicators for solution regularity
        
        Returns:
            Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        t = np.array(self.history['time'])
        
        # Palinstrophy (enstrophy growth rate)
        enstrophy = np.array(self.history['enstrophy'])
        if len(t) > 1 and len(enstrophy) > 1:
            dt = np.diff(t)
            palinstrophy = np.diff(np.log(enstrophy + 1e-10)) / dt
            
            ax1.plot(t[1:], palinstrophy, 'b-', linewidth=2)
            ax1.set_xlabel('Time')
            ax1.set_ylabel('d(log Ω)/dt')
            ax1.set_title('Palinstrophy (Enstrophy Growth Rate)')
            ax1.grid(True, alpha=0.3)
            
            # Add danger zone
            ax1.axhline(y=10, color='r', linestyle='--', alpha=0.5)
            ax1.text(t[-1]*0.1, 11, 'Potential singularity', color='r')
        
        # Factorization complexity growth
        max_fact = np.array(self.history['max_factorization'])
        ax2.semilogy(t, max_fact, 'g-', linewidth=2)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Max Factorization Complexity')
        ax2.set_title('Factorization State Space Growth')
        ax2.grid(True, alpha=0.3)
        
        # Add threshold
        ax2.axhline(y=1000, color='r', linestyle='--', alpha=0.5)
        ax2.text(t[-1]*0.1, 1200, 'Regularity threshold', color='r')
        
        plt.tight_layout()
        return fig
    
    def save_diagnostics(self, filename: str):
        """
        Save diagnostic data to file
        
        Args:
            filename: Output filename
        """
        import json
        
        # Convert to serializable format
        data = {}
        for key, values in self.history.items():
            data[key] = [float(v) for v in values]
            
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
    def load_diagnostics(self, filename: str):
        """
        Load diagnostic data from file
        
        Args:
            filename: Input filename
        """
        import json
        
        with open(filename, 'r') as f:
            data = json.load(f)
            
        self.history = data