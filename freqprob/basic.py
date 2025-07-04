"""
Basic probability scoring methods.

This module implements simple probability scoring methods including
Uniform, Random, and Maximum Likelihood Estimation (MLE).
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, Optional

from .base import ScoringMethod, ScoringMethodConfig


@dataclass
class UniformConfig(ScoringMethodConfig):
    """Configuration for Uniform distribution."""
    
    unobs_prob: float = 0.0
    logprob: bool = True


@dataclass
class RandomConfig(ScoringMethodConfig):
    """Configuration for Random distribution."""
    
    unobs_prob: float = 0.0
    logprob: bool = True
    seed: Optional[int] = None


@dataclass
class MLEConfig(ScoringMethodConfig):
    """Configuration for MLE distribution."""
    
    unobs_prob: float = 0.0
    logprob: bool = True


class Uniform(ScoringMethod):
    """
    Uniform distribution.

    This is the simplest smoothing method, which assigns the same
    probability to all elements.

    Parameters
    ----------
    freqdist : dict
        Frequency distribution of samples (keys) and counts (values) from
        which the log-probability distribution will be calculated.
    unobs_prob : float
        An optional mass probability to be reserved for unobserved states,
        from 0.0 to 1.0. If not provided, the probability mass is
        distributed evenly among all states (i.e, the probability of
        unobserved states is 0.0).
    logprob : bool
        Whether to return the log-probabilities (default) or the
        probabilities themselves. When using the log-probabilities, the
        counts are automatically corrected to avoid domain errors.
    """
    
    __slots__ = ()
    
    def __init__(self, freqdist: Dict[str, int], unobs_prob: float = 0.0, logprob: bool = True):
        config = UniformConfig(unobs_prob=unobs_prob, logprob=logprob)
        super().__init__(config)
        self.name = "Uniform"
        self.fit(freqdist)
    
    def _compute_probabilities(self, freqdist: Dict[str, int]) -> None:
        """Compute uniform probabilities."""
        unobs_prob = self.config.unobs_prob
        
        if self.logprob:
            unobs_prob = max(unobs_prob, self._unobs)  # avoids domain errors
            prob = math.log((1.0 - unobs_prob) / len(freqdist))
            self._prob = {elem: prob for elem in freqdist}
            self._unobs = math.log(unobs_prob)
        else:
            prob = (1.0 - unobs_prob) / len(freqdist)
            self._prob = {elem: prob for elem in freqdist}
            self._unobs = unobs_prob


class Random(ScoringMethod):
    """
    Random distribution.

    In a random log-probability distribution all samples, no matter the
    observed counts, will have a random log-probability computed from a set of
    randomly drawn floating point values. A mass probability can optionally be
    reserved for unobserved samples.

    Parameters
    ----------
    freqdist : dict
        Frequency distribution of samples (keys) and counts (values) from
        which the probability distribution will be calculated.
    unobs_prob : float
        An optional mass probability to be reserved for unobserved states,
        from 0.0 to 1.0.
    logprob : bool
        Whether to return the log-probabilities (default) or the
        probabilities themselves. When using the log-probabilities, the
        counts are automatically corrected to avoid domain errors.
    seed :
        An optional seed for the random number generator, defaulting to None.
    """
    
    __slots__ = ()
    
    def __init__(self, freqdist: Dict[str, int], unobs_prob: float = 0.0, logprob: bool = True, seed=None):
        config = RandomConfig(unobs_prob=unobs_prob, logprob=logprob, seed=seed)
        super().__init__(config)
        self.name = "Random"
        self.fit(freqdist)
    
    def _compute_probabilities(self, freqdist: Dict[str, int]) -> None:
        """Compute random probabilities."""
        unobs_prob = self.config.unobs_prob
        
        # Store the probabilities, which are computed from a random sampling based (if
        # possible) on the observed counts.
        min_counts, max_counts = min(freqdist.values()), max(freqdist.values())
        random.seed(self.config.seed)
        random_freqdist = {elem: random.randint(max(min_counts, 1), max_counts) for elem in freqdist}
        value_sum = sum(random_freqdist.values())

        if self.logprob:
            unobs_prob = max(unobs_prob, self._unobs)
            self._prob = {
                elem: math.log((count / value_sum) * (1.0 - unobs_prob)) for elem, count in random_freqdist.items()
            }
            self._unobs = math.log(unobs_prob)
        else:
            self._prob = {elem: (count / value_sum) * (1.0 - unobs_prob) for elem, count in random_freqdist.items()}
            self._unobs = unobs_prob


class MLE(ScoringMethod):
    """
    Returns a Maximum-Likelihood Estimation log-probability distribution.

    In an MLE log-probability distribution the probability of each sample is
    approximated as the frequency of the same sample in the frequency
    distribution of observed samples. It is the distribution people intuitively
    adopt when thinking of probability distributions. A mass probability can
    optionally be reserved for unobserved samples.

    Parameters
    ----------
    freqdist : dict
        Frequency distribution of samples (keys) and counts (values) from
        which the probability distribution will be calculated.
    unobs_prob : float
        An optional mass probability to be reserved for unobserved states,
        from 0.0 to 1.0.
    logprob : bool
        Whether to return the log-probabilities (default) or the
        probabilities themselves. When using the log-probabilities, the
        counts are automatically corrected to avoid domain errors.
    """
    
    __slots__ = ()
    
    def __init__(self, freqdist: Dict[str, int], unobs_prob: float = 0.0, logprob: bool = True):
        config = MLEConfig(unobs_prob=unobs_prob, logprob=logprob)
        super().__init__(config)
        self.name = "MLE"
        self.fit(freqdist)
    
    def _compute_probabilities(self, freqdist: Dict[str, int]) -> None:
        """Compute MLE probabilities."""
        unobs_prob = self.config.unobs_prob
        
        # Store the probabilities
        value_sum = sum(freqdist.values())
        if self.logprob:
            unobs_prob = max(unobs_prob, self._unobs)
            self._prob = {elem: math.log((count / value_sum) * (1.0 - unobs_prob)) for elem, count in freqdist.items()}
            self._unobs = math.log(unobs_prob)
        else:
            self._prob = {elem: (count / value_sum) * (1.0 - unobs_prob) for elem, count in freqdist.items()}
            self._unobs = unobs_prob