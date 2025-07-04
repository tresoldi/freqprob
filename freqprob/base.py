"""
Base classes for frequency probability scoring methods.

This module provides the abstract base class and common functionality
for all smoothing methods in the freqprob library.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ScoringMethodConfig:
    """Configuration for scoring methods."""
    
    unobs_prob: Optional[float] = None
    gamma: Optional[float] = None
    bins: Optional[int] = None
    logprob: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.unobs_prob is not None:
            if not 0.0 <= self.unobs_prob <= 1.0:
                raise ValueError("The reserved mass probability must be between 0.0 and 1.0")
        
        if self.gamma is not None:
            if self.gamma < 0:
                raise ValueError("Gamma must be a non-negative real number.")
        
        if self.bins is not None:
            if self.bins < 1:
                raise ValueError("Number of bins must be a positive integer.")


class ScoringMethod(ABC):
    """
    Abstract base class for smoothing methods.

    This class provides a common interface for all smoothing methods.
    """
    
    __slots__ = ('_unobs', '_prob', 'logprob', 'name', 'config')
    
    def __init__(self, config: ScoringMethodConfig):
        """
        Initialize the scoring method.
        
        Parameters
        ----------
        config : ScoringMethodConfig
            Configuration object containing method parameters.
        """
        self.config = config
        self._unobs = 1e-10  # Default value to avoid domain errors
        self._prob: Dict[str, float] = {}
        self.logprob: Optional[bool] = config.logprob
        self.name: Optional[str] = None

    def __call__(self, element: str) -> float:
        """
        Score one element.

        Parameters
        ----------
        element : str
            Element to be scored.

        Returns
        -------
        float
            The probability or log-probability of the element.
        """
        return self._prob.get(element, self._unobs)

    def __str__(self) -> str:
        """
        Return a string representation of the smoothing method.

        Returns
        -------
        str
            String representation of the smoothing method.
        """
        if self.name is None:
            raise ValueError("The smoothing method has not been (properly) initialized.")

        buffer = []
        if self.logprob:
            buffer.append(f"{self.name} log-scorer")
        else:
            buffer.append(f"{self.name} scorer")

        buffer.append(f"{len(self._prob)} elements.")

        return ", ".join(buffer)

    @abstractmethod
    def _compute_probabilities(self, freqdist: Dict[str, int]) -> None:
        """
        Compute probabilities for the given frequency distribution.
        
        This method must be implemented by subclasses to compute the
        actual probability values.
        
        Parameters
        ----------
        freqdist : Dict[str, int]
            Frequency distribution of samples and their counts.
        """
        pass

    def fit(self, freqdist: Dict[str, int]) -> 'ScoringMethod':
        """
        Fit the scoring method to a frequency distribution.
        
        Parameters
        ----------
        freqdist : Dict[str, int]
            Frequency distribution of samples and their counts.
            
        Returns
        -------
        ScoringMethod
            Self for method chaining.
        """
        self._compute_probabilities(freqdist)
        return self