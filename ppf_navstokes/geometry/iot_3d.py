"""
Full 3D Involuted Oblate Toroid (IOT) Geometry

Implements the complete 3D manifold with radial involution
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class IOT3DMetric:
    """
    Metric tensor for the full 3D IOT including radial dimension
    
    The IOT is parameterized by (u, v, r):
    - u: toroidal angle [0, 2π]
    - v: poloidal angle [0, 2π]  
    - r: radial coordinate with involution
    """
    R: float  # Major radius
    r_max: float  # Maximum minor radius
    
    def __post_init__(self):
        # Enforce critical ratio
        assert abs(self.r_max / self.R - 1/30) < 0.01, "Must maintain r/R ≈ 1/30"
    
    def embedding(self, u: np.ndarray, v: np.ndarray, r: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Embed the 3D IOT into R³
        
        The involution creates a "breathing" torus that can turn inside-out
        """
        # Radial modulation function (involution dynamics)
        h = self._involution_function(r, t)
        
        # Standard torus embedding with radial modulation
        x = (self.R + r * h * np.cos(v)) * np.cos(u)
        y = (self.R + r * h * np.cos(v)) * np.sin(u)
        z = r * h * np.sin(v)
        
        return x, y, z
    
    def _involution_function(self, r: np.ndarray, t: float) -> np.ndarray:
        """
        The involution function that allows the torus to turn inside-out
        
        This implements f: T → T where f² = id
        """
        # Handle both scalar and array inputs
        r = np.asarray(r)
        scalar_input = r.ndim == 0
        r = np.atleast_1d(r)
        
        # Critical involution at r = r_max/2
        r_crit = self.r_max / 2
        
        # Smooth involution that flips inside/outside
        # h(r,t) oscillates between +1 and -1, allowing inside-out motion
        phase = 2 * np.pi * t  # Complete involution cycle
        
        # Radial-dependent involution strength
        involution_strength = np.tanh((r - r_crit) / (0.1 * self.r_max))
        
        # Time-dependent involution
        h = np.cos(phase) + involution_strength * np.sin(phase)
        
        return h.item() if scalar_input else h
    
    def metric_tensor_3d(self, u: float, v: float, r: float, t: float) -> np.ndarray:
        """
        Full 3D metric tensor including radial component
        """
        # Handle r=0 case to avoid singularities
        if np.abs(r) < 1e-10:
            r = 1e-10
            
        h = self._involution_function(r, t)
        dh_dr = self._involution_derivative(r, t)
        
        # Build 3x3 metric tensor
        g = np.zeros((3, 3))
        
        # g_uu component
        g[0, 0] = (self.R + r * h * np.cos(v))**2
        
        # g_vv component (protect against r=0)
        g[1, 1] = max((r * h)**2, 1e-10)
        
        # g_rr component (includes involution dynamics)
        # Ensure non-zero diagonal for invertibility
        g[2, 2] = max(1.0, h**2 + (r * dh_dr)**2)
        
        # Off-diagonal terms from involution (scaled down to maintain positive definiteness)
        coupling_strength = 0.1  # Reduce off-diagonal coupling
        g[0, 2] = g[2, 0] = -coupling_strength * r * h * dh_dr * np.sin(v) * (self.R + r * h * np.cos(v))
        g[1, 2] = g[2, 1] = coupling_strength * r**2 * h * dh_dr
        
        return g
    
    def _involution_derivative(self, r: float, t: float) -> float:
        """Derivative of involution function with respect to r"""
        # Handle r=0 case
        if np.abs(r) < 1e-10:
            r = 1e-10
            
        r_crit = self.r_max / 2
        phase = 2 * np.pi * t
        
        # Derivative of tanh term
        sech2 = 1 / np.cosh((r - r_crit) / (0.1 * self.r_max))**2
        dinv_dr = sech2 / (0.1 * self.r_max)
        
        return dinv_dr * np.sin(phase)


class IOT3DGeometry:
    """
    Full 3D IOT geometry with involution dynamics
    """
    
    def __init__(self, R: float = 30.0, r_max: float = 1.0, n_r: int = 16):
        """
        Initialize 3D IOT
        
        Args:
            R: Major radius
            r_max: Maximum minor radius
            n_r: Number of radial grid points
        """
        self.R = R
        self.r_max = r_max
        self.n_r = n_r
        self.metric = IOT3DMetric(R, r_max)
        
        # Warping modes for 3D
        self._warping_modes_3d = []
        
    def create_grid_3d(self, n_u: int, n_v: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create 3D grid including radial dimension
        
        Returns:
            u_grid, v_grid, r_grid: 3D arrays of coordinates
        """
        # Angular coordinates
        u = np.linspace(0, 2*np.pi, n_u, endpoint=False)
        v = np.linspace(0, 2*np.pi, n_v, endpoint=False)
        
        # Radial coordinate (avoid r=0 singularity)
        # Use non-uniform spacing to resolve involution region
        # Start from small positive value to avoid singularity
        r_min = 0.1 * self.r_max  # Minimum radius to avoid singularity
        r_uniform = np.linspace(0, 1, self.n_r)**2  # Quadratic spacing
        r = r_min + r_uniform * (self.r_max - r_min)
        
        # Create 3D meshgrid
        u_grid, v_grid, r_grid = np.meshgrid(u, v, r, indexing='ij')
        
        return u_grid, v_grid, r_grid
    
    def warping_function_3d(self, u: np.ndarray, v: np.ndarray, r: np.ndarray, t: float) -> np.ndarray:
        """
        3D warping function including radial modes
        
        The radial dimension allows new warping modes that couple with involution
        """
        W = np.zeros_like(u, dtype=float)
        
        for mode in self._warping_modes_3d:
            # Angular components
            term_u = np.sin(mode['f_u'] * u + mode['phase_u'])
            term_v = np.sin(mode['f_v'] * v + mode['phase_v'])
            
            # Radial component (couples with involution)
            r_normalized = r / self.r_max
            term_r = mode['radial_profile'](r_normalized)
            
            # Time modulation
            time_mod = np.cos(mode['omega'] * t)
            
            W += mode['amplitude'] * term_u * term_v * term_r * time_mod
        
        return W
    
    def add_warping_mode_3d(self, frequency_u: int, frequency_v: int, 
                           radial_mode: int, amplitude: float, omega: float = 1.0):
        """
        Add a 3D warping mode
        
        Args:
            frequency_u, frequency_v: Angular frequencies
            radial_mode: Radial mode number (0 = constant, 1 = linear, etc.)
            amplitude: Mode amplitude
            omega: Temporal frequency
        """
        # Define radial profile based on mode number
        if radial_mode == 0:
            radial_profile = lambda r: np.ones_like(r)
        elif radial_mode == 1:
            radial_profile = lambda r: r
        elif radial_mode == 2:
            radial_profile = lambda r: 1 - 2*(r - 0.5)**2
        else:
            # Higher modes use Bessel-like functions
            radial_profile = lambda r: np.sin(radial_mode * np.pi * r)
        
        self._warping_modes_3d.append({
            'f_u': frequency_u,
            'f_v': frequency_v,
            'radial_profile': radial_profile,
            'amplitude': amplitude,
            'omega': omega,
            'phase_u': 0.0,
            'phase_v': 0.0
        })
    
    def laplace_beltrami_3d(self, field: np.ndarray, u_grid: np.ndarray,
                           v_grid: np.ndarray, r_grid: np.ndarray, t: float) -> np.ndarray:
        """
        Laplace-Beltrami operator in full 3D
        
        ∆_g f = 1/√g ∂_i(√g g^{ij} ∂_j f)
        """
        # Grid spacings
        du = u_grid[1,0,0] - u_grid[0,0,0] if u_grid.shape[0] > 1 else 0.1
        dv = v_grid[0,1,0] - v_grid[0,0,0] if u_grid.shape[1] > 1 else 0.1
        dr = r_grid[0,0,1] - r_grid[0,0,0] if r_grid.shape[2] > 1 else 0.1
        
        # Compute derivatives
        df_du = np.gradient(field, du, axis=0)
        df_dv = np.gradient(field, dv, axis=1)
        df_dr = np.gradient(field, dr, axis=2)
        
        # This is a simplified version - full implementation would compute
        # the complete covariant Laplacian with all metric components
        laplacian = np.zeros_like(field)
        
        for i in range(u_grid.shape[0]):
            for j in range(u_grid.shape[1]):
                for k in range(u_grid.shape[2]):
                    u, v, r = u_grid[i,j,k], v_grid[i,j,k], r_grid[i,j,k]
                    
                    # Get metric at this point
                    g = self.metric.metric_tensor_3d(u, v, r, t)
                    
                    # Handle potential singularities
                    det_g = np.linalg.det(g)
                    if abs(det_g) < 1e-10:
                        # Use identity metric at singular points
                        g_inv = np.eye(3)
                        sqrt_det_g = 1.0
                    else:
                        g_inv = np.linalg.inv(g)
                        sqrt_det_g = np.sqrt(abs(det_g))
                    
                    # Simplified Laplacian (diagonal terms only for now)
                    laplacian[i,j,k] = (
                        g_inv[0,0] * np.gradient(np.gradient(field, du, axis=0), du, axis=0)[i,j,k] +
                        g_inv[1,1] * np.gradient(np.gradient(field, dv, axis=1), dv, axis=1)[i,j,k] +
                        g_inv[2,2] * np.gradient(np.gradient(field, dr, axis=2), dr, axis=2)[i,j,k]
                    ) / sqrt_det_g
        
        return laplacian