"""
Physics-Prime Factorization Core Implementation

This module implements the fundamental PPF concepts including:
- Sign Prime (-1) as a prime number
- Factorization state spaces
- Multiple factorization representations
"""

import numpy as np
from typing import List, Tuple, Set, Optional
from dataclasses import dataclass
import sympy


class SignPrime:
    """The Sign Prime: -1"""
    value = -1
    symbol = "σ"
    
    def __repr__(self):
        return f"SignPrime({self.value})"


@dataclass(frozen=True)
class FactorizationState:
    """
    Represents a single factorization state in PPF
    
    Attributes:
        magnitude_primes: List of (prime, power) tuples for magnitude primes
        sign_count: Number of sign primes in this factorization
    """
    magnitude_primes: Tuple[Tuple[int, int], ...]  # (prime, power) pairs
    sign_count: int
    
    def __repr__(self):
        parts = []
        if self.sign_count > 0:
            parts.append(f"(-1)^{self.sign_count}")
        for prime, power in self.magnitude_primes:
            if power == 1:
                parts.append(str(prime))
            else:
                parts.append(f"{prime}^{power}")
        return " × ".join(parts) if parts else "1"
    
    def evaluate(self) -> int:
        """Compute the integer value of this factorization"""
        result = (-1) ** self.sign_count
        for prime, power in self.magnitude_primes:
            result *= prime ** power
        return result
    
    def complexity(self) -> int:
        """Compute the complexity of this factorization (total prime count)"""
        return self.sign_count + sum(power for _, power in self.magnitude_primes)


class PPFInteger:
    """
    Represents an integer in the PPF framework with its factorization state space
    """
    
    def __init__(self, value: int):
        self.value = value
        self._state_space: Optional[Set[FactorizationState]] = None
        
    @property
    def state_space(self) -> Set[FactorizationState]:
        """Get the factorization state space for this integer"""
        if self._state_space is None:
            self._state_space = self._compute_state_space()
        return self._state_space
    
    def _compute_state_space(self) -> Set[FactorizationState]:
        """Compute all valid PPF factorizations for this integer"""
        states = set()
        
        if self.value == 0:
            return states
        
        # Get standard prime factorization
        abs_val = abs(self.value)
        if abs_val == 1:
            # Special case: 1 = (-1)^0 or -1 = (-1)^1
            if self.value > 0:
                states.add(FactorizationState((), 0))
            else:
                states.add(FactorizationState((), 1))
            return states
        
        # Standard factorization using sympy
        factors = sympy.factorint(abs_val)
        magnitude_primes = tuple(sorted(factors.items()))
        
        # For positive integers: even number of negative magnitude primes
        if self.value > 0:
            num_primes = len(magnitude_primes)
            # We can distribute negative signs to pairs of primes
            for i in range(0, num_primes + 1, 2):
                # i primes will be negative
                states.add(FactorizationState(magnitude_primes, 0))
                # Note: In full implementation, we'd generate all combinations
                # For now, we add the base state and one with sign primes
                if i > 0:
                    states.add(FactorizationState(magnitude_primes, 2))
        
        # For negative integers: odd number of negative magnitude primes or sign prime
        else:
            # Option 1: Use sign prime with all positive magnitude primes
            states.add(FactorizationState(magnitude_primes, 1))
            
            # Option 2: Odd number of negative magnitude primes (simplified)
            states.add(FactorizationState(magnitude_primes, 3))
            
        return states
    
    def is_collapsed(self) -> bool:
        """Check if this is a collapsed (classical) state"""
        return self.value > 0
    
    def is_superposed(self) -> bool:
        """Check if this is a superposed (quantum) state"""
        return self.value < 0
    
    def state_cardinality(self) -> int:
        """Return the number of states in the factorization state space"""
        return len(self.state_space)
    
    def __repr__(self):
        return f"PPFInteger({self.value}, |S|={self.state_cardinality()})"