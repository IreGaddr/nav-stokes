"""
Factorization utilities for PPF
"""

import numpy as np
from typing import List, Set, Tuple
import sympy
from .ppf import PPFInteger, FactorizationState


def factorize_ppf(n: int) -> Set[FactorizationState]:
    """
    Get all PPF factorizations for an integer
    
    Args:
        n: Integer to factorize
        
    Returns:
        Set of all valid factorization states
    """
    ppf_int = PPFInteger(n)
    return ppf_int.state_space


def get_factorization_state_space(velocity_magnitude: float, 
                                  tolerance: float = 1e-6) -> Set[FactorizationState]:
    """
    Get factorization state space for a continuous velocity magnitude
    
    Args:
        velocity_magnitude: Magnitude of velocity vector
        tolerance: Tolerance for discretization
        
    Returns:
        Set of factorization states
    """
    # Discretize to nearest integer for factorization
    # In full implementation, we'd use rational approximation
    discrete_val = int(round(velocity_magnitude / tolerance))
    if discrete_val == 0:
        discrete_val = 1
    
    return factorize_ppf(discrete_val)


def factorization_complexity(states: Set[FactorizationState]) -> float:
    """
    Compute average complexity of a factorization state space
    
    Args:
        states: Set of factorization states
        
    Returns:
        Average complexity
    """
    if not states:
        return 0.0
    
    total_complexity = sum(state.complexity() for state in states)
    return total_complexity / len(states)


def is_turbulent(states: Set[FactorizationState]) -> bool:
    """
    Determine if flow is turbulent based on factorization state space
    
    Args:
        states: Factorization state space
        
    Returns:
        True if turbulent (multiple states), False if laminar (single state)
    """
    return len(states) > 1