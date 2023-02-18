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
from typing import Dict, Optional

# TODO: add querying functions to the ScoringMethod to get 1. the method name 2. whether logprobs are used


class ScoringMethod:
    """
    Abstract class for smoothing methods.

    This class provides a common interface for all smoothing methods.
    """

    def __init__(self, unobs_prob: Optional[float] = None, gamma: Optional[float] = None, bins: Optional[int] = None):
        # Set a default value for the observed and unobserved probabilities.
        # The default value for the unobserved probability is 1e-10, which
        # is also used to avoid domain errors when calculating the log
        # probabilities. All methods should take the maximum value between
        # this value and the reserved mass probability when computing
        # log-probabilities.
        self._unobs = 1e-10
        self._prob: Dict[str, float] = {}

        # Set .logprob to None, so that calling this superclass directly will raise an error.
        self.logprob: Optional[bool] = None
        self.name: Optional[str] = None

        # Check values for all distributions; this allows to have a single check,
        # so arguments with the same name in different distributions will behave
        # the same way.
        if unobs_prob is not None:
            # Confirm that the reserved mass probability is valid (between 0.0 and 1.0)
            if not 0.0 <= unobs_prob <= 1.0:
                raise ValueError("The reserved mass probability must be between 0.0 and 1.0")

        if gamma is not None:
            if gamma < 0:
                # TODO: check if it can/makes sense to have a gamma of zero (<= 0.0)
                raise ValueError("Gamma must be a real number.")

        if bins is not None:
            if bins < 1:
                # TODO: check if it can/makes sense to have a bins of zero (<= 0)
                raise ValueError("Number of bins must be a real number.")

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
        super().__init__(unobs_prob=unobs_prob)

        # Store the parameters
        self.logprob = logprob
        self.name = "Uniform"

        # Calculation couldn't be easier: we just subtract the reserved mass
        # probability from 1.0 and divide by the number of samples.
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

    def __init__(self, freqdist: Dict[str, int], unobs_prob: float = 0.0, logprob: bool = True, seed=None):
        # Call the parent constructor
        super().__init__(unobs_prob=unobs_prob)

        # Store the parameters
        self.logprob = logprob
        self.name = "Random"

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
        super().__init__(unobs_prob=unobs_prob)

        # Store the parameters
        self.logprob = logprob
        self.name = "MLE"

        # Store the probabilities
        value_sum = sum(freqdist.values())
        if self.logprob:
            unobs_prob = max(unobs_prob, self._unobs)
            self._prob = {elem: math.log((count / value_sum) * (1.0 - unobs_prob)) for elem, count in freqdist.items()}
            self._unobs = math.log(unobs_prob)
        else:
            self._prob = {elem: (count / value_sum) * (1.0 - unobs_prob) for elem, count in freqdist.items()}
            self._unobs = unobs_prob


class Lidstone(ScoringMethod):
    """
    Returns a Lidstone estimate log-probability distribution.

    In a Lidstone estimate log-probability the frequency distribution of
    observed samples is used to estimate the probability distribution of the
    experiment that generated such observation, following a parameter given by
    a real number *gamma* typically randing from 0.0 to 1.0. The Lidstone
    estimate approximates the probability of a sample with count *c* from an
    experiment with *N* outcomes and *B* bins as *(c+gamma)/(N+B*gamma)*. This
    is equivalent to adding *gamma* to the count of each bin and taking the
    Maximum-Likelihood estimate of the resulting frequency distribution, with
    the corrected space of observation; the probability for an unobserved
    sample is given by frequency of a sample with gamma observations.
    Also called "additive smoothing", this estimation method is frequently
    used with a *gamma* of 1.0 (the so-called "Laplace smoothing") or of 0.5
    (the so-called "Expected likelihood estimate", or ELE).

    Parameters
    ----------
    freqdist : dict
        Frequency distribution of samples (keys) and counts (values) from
        which the probability distribution will be calculated.
    gamma : float
        A real number used to parameterize the estimate.
    bins: int
        The optional number of sample bins that can be generated by the
        experiment that is described by the probability distribution. If not
        specified, it will default to the number of samples in the frequency
        distribution.
    logprob : bool
        Whether to return the log-probabilities (default) or the
        probabilities themselves. When using the log-probabilities, the
        counts are automatically corrected to avoid domain errors.
    """

    def __init__(self, freqdist: Dict[str, int], gamma: float, bins: Optional[int] = None, logprob: bool = True):
        # Collect `bins` if necessary before validating with the parent
        if bins is None:
            bins = len(freqdist)

        # Call the parent constructor
        super().__init__(gamma=gamma, bins=bins)

        # Store the parameters
        self.logprob = logprob
        self.name = "Lidstone"

        # Store the probabilities
        n = sum(freqdist.values())
        if self.logprob:
            self._prob = {sample: math.log((count + gamma) / (n + bins * gamma)) for sample, count in freqdist.items()}
            self._unobs = math.log(gamma / (n + bins * gamma))
        else:
            self._prob = {sample: (count + gamma) / (n + bins * gamma) for sample, count in freqdist.items()}
            self._unobs = gamma / (n + bins * gamma)


class Laplace(Lidstone):
    """
    Returns a Laplace estimate probability distribution.

    In a Laplace estimate log-probability the frequency distribution of
    observed samples is used to estimate the probability distribution of the
    experiment that generated such observation, following a parameter given by
    a real number *gamma* set by definition to 1. As such, it is a
    generalization of the Lidstone estimate.

    Parameters
    ----------
    freqdist : dict
        Frequency distribution of samples (keys) and counts (values) from
        which the probability distribution will be calculated.
    bins: int
        The optional number of sample bins that can be generated by the
        experiment that is described by the probability distribution. If
        not specified, it will default to the number of samples in
        the frequency distribution.
    logprob : bool
        Whether to return the log-probabilities (default) or the
        probabilities themselves. When using the log-probabilities, the
        counts are automatically corrected to avoid domain errors.
    """

    def __init__(self, freqdist: Dict[str, int], bins: Optional[int] = None, logprob: bool = True):
        # Call the parent constructor
        super().__init__(freqdist, gamma=1.0, bins=bins, logprob=logprob)
        self.name = "Laplace"
