"""
Module providing various methods for frequence smoothing.

The smoothing methods are implemented as scoring classes, which are
instantiated with the observed distribution and then called with the
elements to be scored. The scoring classes are implemented as
subclasses of the abstract class :class:`ScoringMethod`, which
provides a common interface for all smoothing methods.
"""

import math
from typing import Dict


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

    def __call__(self, element: str) -> float:
        raise NotImplementedError


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
