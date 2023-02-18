"""
Module providing various methods for frequency smoothing.

The smoothing methods are implemented as scoring classes, which are
instantiated with the observed distribution and then called with the
elements to be scored. The scoring classes are implemented as
subclasses of the abstract class :class:`ScoringMethod`, which
provides a common interface for all smoothing methods.
"""

# Import standard modules
import math
import random
from typing import Dict

# TODO: add querying functions to the ScoringMethod to get 1. the method name 2. whether logprobs are used


class ScoringMethod:
    """
    Abstract class for smoothing methods.

    This class provides a common interface for all smoothing methods.
    """

    def __init__(self):
        # Set a default value for the unobserved probability, which is
        # also used to avoid domain errors when calculating the log
        # probabilities.
        self._unobs = 1e-10
        self._prob = {}

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

    def __init__(self, freqdist: Dict[str, int], unobs_prob: float = 0.0, logprob: bool = True):
        # Call the parent constructor
        super().__init__()

        # Confirm that the reserved mass probability is valid (between 0.0 and 1.0)
        if not 0.0 <= unobs_prob <= 1.0:
            raise ValueError("The reserved mass probability must be between 0.0 and 1.0")

        # Store the parameters
        self.logprob = logprob

        # Calculation couldn't be easier: we just subtract the reserved mass
        # probability from 1.0 and divide by the number of samples.
        if self.logprob:
            unobs_prob = max(unobs_prob, self._unobs)
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

    def __init__(self, freqdist: Dict[str, int], unobs_prob: float = 0.0, logprob: bool = True, seed=None):
        # Call the parent constructor
        super().__init__()

        # Confirm that the reserved mass probability is valid (between 0.0 and 1.0)
        if not 0.0 <= unobs_prob <= 1.0:
            raise ValueError("The reserved mass probability must be between 0.0 and 1.0")

        # Store the parameters
        self.logprob = logprob

        # Store the probabilities, which are computed from a random sampling based (if
        # possible) on the observed counts.
        min_counts, max_counts = min(freqdist.values()), max(freqdist.values())
        random.seed(seed)
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

    def __init__(self, freqdist: Dict[str, int], unobs_prob: float = 0.0, logprob: bool = True):
        # Call the parent constructor
        super().__init__()

        # Confirm that the reserved mass probability is valid (between 0.0 and 1.0)
        if not 0.0 <= unobs_prob <= 1.0:
            raise ValueError("The reserved mass probability must be between 0.0 and 1.0")

        # Store the parameters
        self.logprob = logprob

        # Store the probabilities
        value_sum = sum(freqdist.values())
        if self.logprob:
            unobs_prob = max(unobs_prob, self._unobs)
            self._prob = {elem: math.log((count / value_sum) * (1.0 - unobs_prob)) for elem, count in freqdist.items()}
            self._unobs = math.log(unobs_prob)
        else:
            self._prob = {elem: (count / value_sum) * (1.0 - unobs_prob) for elem, count in freqdist.items()}
            self._unobs = unobs_prob
