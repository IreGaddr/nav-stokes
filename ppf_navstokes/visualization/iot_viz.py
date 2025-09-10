"""
Visualization of the IOT geometry
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from typing import Optional, Tuple
from ..geometry import IOTGeometry
from ..physics import FluidState


class IOTVisualizer:
    """
    Visualize the Involuted Oblate Toroid geometry
    """
    
    def __init__(self, geometry: IOTGeometry):
        """
        Initialize IOT visualizer
        
        Args:
            geometry: IOT geometry to visualize
        """
        self.geometry = geometry
        
    def plot_torus_3d(self, n_u: int = 50, n_v: int = 50,
                     show_warping: bool = True,
                     time: float = 0.0) -> plt.Figure:
        """
        Plot 3D visualization of the torus
        
        Args:
            n_u: Number of points in u direction
            n_v: Number of points in v direction
            show_warping: Color by warping function
            time: Time for warping function evaluation
            
        Returns:
            Figure object
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create grid
        u = np.linspace(0, 2*np.pi, n_u)
        v = np.linspace(0, 2*np.pi, n_v)
        u_grid, v_grid = np.meshgrid(u, v)
        
        # Convert to Cartesian coordinates
        x, y, z = self.geometry.to_cartesian(u_grid, v_grid)
        
        if show_warping:
            # Color by warping function
            colors = np.zeros_like(u_grid)
            for i in range(n_u):
                for j in range(n_v):
                    colors[i, j] = self.geometry.warping_function(
                        u_grid[i, j], v_grid[i, j], time
                    )
            
            surf = ax.plot_surface(x, y, z, facecolors=cm.viridis(colors),
                                 alpha=0.8, linewidth=0, antialiased=True)
        else:
            # Uniform color
            surf = ax.plot_surface(x, y, z, color='lightblue',
                                 alpha=0.8, linewidth=0, antialiased=True)
        
        # Add wireframe
        ax.plot_wireframe(x, y, z, color='gray', alpha=0.2, linewidth=0.5)
        
        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Involuted Oblate Toroid (R={self.geometry.metric.R}, r={self.geometry.metric.r})')
        
        # Equal aspect ratio
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        return fig
    
    def plot_metric_components(self, n_points: int = 100) -> plt.Figure:
        """
        Plot metric tensor components as functions of coordinates
        
        Args:
            n_points: Number of points for plotting
            
        Returns:
            Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        u = np.linspace(0, 2*np.pi, n_points)
        v = np.linspace(0, 2*np.pi, n_points)
        
        # g_uu component
        ax = axes[0, 0]
        g_uu = np.zeros((n_points, n_points))
        for i, ui in enumerate(u):
            for j, vj in enumerate(v):
                metric = self.geometry.metric.metric_tensor(ui, vj)
                g_uu[j, i] = metric[0, 0]
        
        im1 = ax.contourf(u, v, g_uu, levels=20, cmap='viridis')
        ax.set_xlabel('u')
        ax.set_ylabel('v')
        ax.set_title('Metric component g_uu')
        plt.colorbar(im1, ax=ax)
        
        # g_vv component
        ax = axes[0, 1]
        g_vv = np.full((n_points, n_points), self.geometry.metric.r**2)
        im2 = ax.contourf(u, v, g_vv, levels=20, cmap='viridis')
        ax.set_xlabel('u')
        ax.set_ylabel('v')
        ax.set_title('Metric component g_vv')
        plt.colorbar(im2, ax=ax)
        
        # Christoffel symbols (example: Γ^u_vv)
        ax = axes[1, 0]
        gamma_u_vv = np.zeros((n_points, n_points))
        for i, ui in enumerate(u):
            for j, vj in enumerate(v):
                gamma = self.geometry.metric.christoffel_symbols(ui, vj)
                gamma_u_vv[j, i] = gamma[0, 1, 1]
        
        im3 = ax.contourf(u, v, gamma_u_vv, levels=20, cmap='RdBu_r')
        ax.set_xlabel('u')
        ax.set_ylabel('v')
        ax.set_title('Christoffel symbol Γ^u_vv')
        plt.colorbar(im3, ax=ax)
        
        # Warping function
        ax = axes[1, 1]
        W = np.zeros((n_points, n_points))
        for i, ui in enumerate(u):
            for j, vj in enumerate(v):
                W[j, i] = self.geometry.warping_function(ui, vj, 0.0)
        
        im4 = ax.contourf(u, v, W, levels=20, cmap='plasma')
        ax.set_xlabel('u')
        ax.set_ylabel('v')
        ax.set_title('Warping function W(u,v,t=0)')
        plt.colorbar(im4, ax=ax)
        
        plt.tight_layout()
        return fig
    
    def plot_tautochrone_paths(self, start_point: Tuple[float, float],
                             n_paths: int = 8) -> plt.Figure:
        """
        Plot tautochrone paths from a starting point
        
        Args:
            start_point: Starting (u, v) coordinates
            n_paths: Number of paths to plot
            
        Returns:
            Figure object
        """
        from ..geometry import compute_tautochrone_paths
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Compute paths
        paths = compute_tautochrone_paths(self.geometry, start_point, n_paths)
        
        # Plot in (u,v) space
        for i, path in enumerate(paths):
            color = plt.cm.rainbow(i / n_paths)
            ax1.plot(path.u_path, path.v_path, color=color, linewidth=2)
        
        ax1.scatter(*start_point, color='red', s=100, marker='o', zorder=5)
        ax1.set_xlabel('u (toroidal angle)')
        ax1.set_ylabel('v (poloidal angle)')
        ax1.set_title('Tautochrone Paths in (u,v) Space')
        ax1.set_xlim(0, 2*np.pi)
        ax1.set_ylim(0, 2*np.pi)
        ax1.grid(True, alpha=0.3)
        
        # Plot in 3D
        ax2 = fig.add_subplot(122, projection='3d')
        
        for i, path in enumerate(paths):
            # Convert to Cartesian
            x, y, z = self.geometry.to_cartesian(path.u_path, path.v_path)
            color = plt.cm.rainbow(i / n_paths)
            ax2.plot(x, y, z, color=color, linewidth=2)
        
        # Mark start point
        x0, y0, z0 = self.geometry.to_cartesian(
            np.array([start_point[0]]), np.array([start_point[1]])
        )
        ax2.scatter(x0, y0, z0, color='red', s=100, marker='o')
        
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('Tautochrone Paths in 3D')
        
        return fig
    
    def plot_flow_on_torus(self, state: FluidState) -> plt.Figure:
        """
        Plot flow velocity on 3D torus surface
        
        Args:
            state: Fluid state
            
        Returns:
            Figure object
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get grid
        u_grid = state.u_grid
        v_grid = state.v_grid
        
        # Convert to Cartesian
        x, y, z = self.geometry.to_cartesian(u_grid, v_grid)
        
        # Color by velocity magnitude
        vel_mag = state.velocity.magnitude()
        colors = cm.viridis(vel_mag / vel_mag.max())
        
        # Plot surface
        surf = ax.plot_surface(x, y, z, facecolors=colors,
                             alpha=0.9, linewidth=0, antialiased=True)
        
        # Add velocity vectors (subsampled)
        skip = max(1, len(u_grid) // 15)
        u_sub = u_grid[::skip, ::skip]
        v_sub = v_grid[::skip, ::skip]
        x_sub, y_sub, z_sub = self.geometry.to_cartesian(u_sub, v_sub)
        
        # Transform velocity vectors to 3D
        # This is simplified - proper transformation would use metric
        vel_u = state.velocity.u_component[::skip, ::skip]
        vel_v = state.velocity.v_component[::skip, ::skip]
        
        # Approximate 3D velocity components
        scale = 2.0
        dx = -scale * vel_u * np.sin(u_sub)
        dy = scale * vel_u * np.cos(u_sub)
        dz = scale * vel_v * np.cos(v_sub)
        
        ax.quiver(x_sub, y_sub, z_sub, dx, dy, dz,
                 color='black', alpha=0.6, arrow_length_ratio=0.3)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Flow on IOT Surface (t={state.time:.3f})')
        
        return fig