"""
Spectral methods for efficient computation on the 3D IOT geometry
"""

import numpy as np
from scipy.fft import fftn, ifftn, fftfreq
from typing import Tuple
from ..geometry import IOT3DGeometry


class SpectralSolver3D:
    """
    3D Spectral methods for solving PDEs on the IOT geometry.
    
    Uses 3D FFT for efficient computation of derivatives and solving elliptic equations.
    """
    
    def __init__(self, n_u: int, n_v: int, n_z: int, geometry: IOT3DGeometry):
        """
        Initialize 3D spectral solver
        
        Args:
            n_u: Number of grid points in u direction
            n_v: Number of grid points in v direction
            n_z: Number of grid points in z direction
            geometry: IOT geometry object
        """
        self.n_u = n_u
        self.n_v = n_v
        self.n_z = n_z
        self.geometry = geometry
        
        # Precompute wavenumbers for a real-to-complex FFT
        du = 2 * np.pi / n_u
        dv = 2 * np.pi / n_v
        dz = geometry.r_max / n_z
        
        # Fixed wavenumber computation
        self.k_u = fftfreq(n_u, d=du/(2*np.pi))
        self.k_v = fftfreq(n_v, d=dv/(2*np.pi))
        # For z-direction (radial), use proper spacing
        self.k_z = fftfreq(n_z, d=dz) * 2 * np.pi

        # 3D wavenumber grids
        self.k_u_grid, self.k_v_grid, self.k_z_grid = np.meshgrid(
            self.k_u, self.k_v, self.k_z, indexing='ij'
        )
        
        self.k_squared = self.k_u_grid**2 + self.k_v_grid**2 + self.k_z_grid**2
        
        # Avoid division by zero for the k=0 mode
        self.k_squared_safe = self.k_squared.copy()
        self.k_squared_safe[0, 0, 0] = 1.0
        
    def gradient_3d(self, field: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gradient of a 3D scalar field.
        """
        field_hat = fftn(field)
        
        d_du_hat = 1j * self.k_u_grid * field_hat
        d_dv_hat = 1j * self.k_v_grid * field_hat
        d_dz_hat = 1j * self.k_z_grid * field_hat
        
        d_du = np.real(ifftn(d_du_hat))
        d_dv = np.real(ifftn(d_dv_hat))
        d_dz = np.real(ifftn(d_dz_hat))
        
        return d_du, d_dv, d_dz

    def divergence_3d(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Compute divergence of a 3D vector field.
        """
        u_hat = fftn(u)
        v_hat = fftn(v)
        w_hat = fftn(w)
        
        div_hat = (1j * self.k_u_grid * u_hat +
                   1j * self.k_v_grid * v_hat +
                   1j * self.k_z_grid * w_hat)
        
        return np.real(ifftn(div_hat))

    def solve_poisson_3d(self, rhs: np.ndarray) -> np.ndarray:
        """
        Solve 3D Poisson equation: ∆φ = rhs
        """
        rhs_hat = fftn(rhs)
        
        # Solve in spectral space with proper regularization
        phi_hat = np.zeros_like(rhs_hat)
        
        # Avoid the k=0 mode (set to 0)
        mask = self.k_squared > 1e-12
        phi_hat[mask] = -rhs_hat[mask] / self.k_squared[mask]
        
        # Explicitly set mean to zero
        phi_hat[0, 0, 0] = 0.0
            
        return np.real(ifftn(phi_hat))
    
    def project_divergence_free_3d(self, u: np.ndarray, v: np.ndarray, w: np.ndarray
                                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Project a 3D velocity field onto the divergence-free subspace.
        Returns the divergence-free velocity field and the pressure potential phi.
        """
        # 1. Compute the divergence of the velocity field
        div = self.divergence_3d(u, v, w)
        
        # 2. Solve the Poisson equation for the pressure-like potential
        #    ∇²φ = ∇ ⋅ u
        phi = self.solve_poisson_3d(div)
        
        # 3. Compute the gradient of the potential
        dphi_du, dphi_dv, dphi_dz = self.gradient_3d(phi)
        
        # 4. Subtract the gradient from the original field to get the divergence-free part
        u_df = u - dphi_du
        v_df = v - dphi_dv
        w_df = w - dphi_dz
        
        return u_df, v_df, w_df, phi
