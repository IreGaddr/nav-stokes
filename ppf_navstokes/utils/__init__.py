"""Utility functions for PPF Navier-Stokes"""

from .initial_conditions import (
    taylor_green_vortex,
    random_turbulence,
    shear_flow,
    vortex_pair
)

__all__ = [
    'taylor_green_vortex',
    'random_turbulence',
    'shear_flow',
    'vortex_pair'
]