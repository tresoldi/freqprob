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

# Import 3rd-party modules
import numpy as np
import scipy  # type: ignore

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


class ELE(Lidstone):
    """
    Returns a Expected-Likelihood estimate probability distribution.

    In an Expected-Likelihood estimate log-probability the frequency
    distribution of observed samples is used to estimate the probability
    distribution of the experiment that generated such observation, following a
    parameter given by a real number *gamma* set by definition to 0.5. As such,
    it is a generalization of the Lidstone estimate.

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
        super().__init__(freqdist, gamma=0.5, bins=bins, logprob=logprob)
        self.name = "ELE"


class WittenBell(ScoringMethod):
    """
    Returns a Witten-Bell estimate probability distribution.

    In a Witten-Bell estimate log-probability a uniform probability mass is
    allocated to yet unobserved samples by using the number of samples that
    have only been observed once. The probability mass reserved for unobserved
    samples is equal to *T / (N +T)*, where *T* is the number of observed
    samples and *N* the number of total observations. This equates to the
    Maximum-Likelihood Estimate of a new type of sample occurring. The
    remaining probability mass is discounted such that all probability
    estimates sum to one, yielding:
        - *p = T / Z (N + T)*, if count == 0
        - *p = c / (N + T)*, otherwise

    Parameters
    ----------
    freqdist : dict
        Frequency distribution of samples (keys) and counts (values) from
        which the probability distribution will be calculated.
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

    def __init__(self, freqdist: Dict[str, int], bins: Optional[int] = None, logprob: bool = True):
        # TODO: decide how to operate if no bins are given
        # TODO: have tests specifying the number of bins (10 and 99 for OBS1)

        # Call the parent constructor; note that we don't pass the `bins`
        # in this case
        super().__init__(bins=bins)

        # Store the parameters
        self.logprob = logprob
        self.name = "Witten-Bell"

        # Store the probabilities
        n = sum(freqdist.values())
        t = len(freqdist)
        z = 1.0 if bins is None or bins == t else bins - t

        if self.logprob:
            self._prob = {sample: math.log(count / (n + t)) for sample, count in freqdist.items()}
            if n == 0:
                self._unobs = math.log(1.0 / z)
            else:
                self._unobs = math.log(t / (z * (n + t)))
        else:
            self._prob = {sample: count / (n + t) for sample, count in freqdist.items()}
            if n == 0:
                self._unobs = 1.0 / z
            else:
                self._unobs = t / (z * (n + t))


class CertaintyDegree(ScoringMethod):
    """
    Returns a Certainty Degree probability distribution.

    In this distribution a mass probability is reserved for unobserved samples
    from a computation of the degree of certainty that the are no unobserved
    samples.

    Under development and test by Tiago Tresoldi, this is an experimental
    probability distribution that should not be used as the sole or main
    distribution for the time being.

    Parameters
    ----------
    freqdist : dict
        Frequency distribution of samples (keys) and counts (values) from
        which the probability distribution will be calculated.
    bins: int
        The optional number of sample bins that can be generated by the
        experiment that is described by the probability distribution. If not
        specified, it will default to the number of samples in the frequency
        distribution.
    unobs_prob : float
        An optional mass probability to be reserved for unobserved states,
        from 0.0 to 1.0. If not specified, it will default to 0.0.
    logprob : bool
        Whether to return the log-probabilities (default) or the
        probabilities themselves. When using the log-probabilities, the
        counts are automatically corrected to avoid domain errors.
    """

    def __init__(
        self, freqdist: Dict[str, int], bins: Optional[int] = None, unobs_prob: float = 0.0, logprob: bool = True
    ):
        # Call the parent constructor; note that we don't pass the `bins`
        # in this case
        super().__init__(unobs_prob=unobs_prob, bins=bins)

        # Store the parameters
        self.logprob = logprob
        self.name = "Certainty Degree"

        # Obtain the parameters for probability calculation.
        n = sum(freqdist.values())
        b = len(freqdist)
        z = bins or b

        # Calculate the mass of probability space to reserve and use this value to
        # correct the Maximum-Likelihood Estimate for each sample.
        # Note that, for very large values of N, this will underflow because we
        # effectively have a large confidence of having observed all the
        # samples that matter; this is a problem when taking the
        # log-probability, as we'll ultimately raise a math domain error by
        # asking for the logarithm of what is machine-represented as zero;
        # for this reason, we take as the probability space the minimum value
        # between 1.0 discounted the calculated mass and 1.0 discounted the
        # minimum mass probability reserved.
        if self.logprob:
            unobs_prob = max(unobs_prob, self._unobs)
            prob_space = min(1.0 - (b / (z + 1)) ** n, 1.0 - unobs_prob)
            self._prob = {sample: math.log((count / n) * prob_space) for sample, count in freqdist.items()}
            self._unobs = math.log(-(prob_space - 1.0))
        else:
            prob_space = min(1.0 - (b / (z + 1)) ** n, 1.0 - unobs_prob)
            self._prob = {sample: (count / n) * prob_space for sample, count in freqdist.items()}
            self._unobs = -(prob_space - 1.0)


class SimpleGoodTuring(ScoringMethod):
    """
    Returns a Simple Good-Turing estimate probability distribution.

    The returned probability distribution is based on the Good-Turing
    frequency estimation, as first developed by Alan Turing and I. J. Good and
    implemented in a more easily computable way by Gale and Sampson's
    (1995/2001 reprint) in the so-called "Simple Good-Turing".
    This implementation is based mostly in the one by "maxbane" (2011)
    (https://github.com/maxbane/simplegoodturing/blob/master/sgt.py), as well
    as in the original one in C by Geoffrey Sampson (1995; 2000; 2005; 2008)
    (https://www.grsampson.net/Resources.html), and in the one by
    Loper, Bird et al. (2001-2018, NLTK Project). Please note that
    due to minor differences in implementation intended to guarantee non-zero
    probabilities even in cases of expected underflow, as well as our
    reliance on scipy's libraries for speed and our way of handling
    probabilities that are not computable when the assumptions of SGT are
    not met, most results will not exactly match those of the 'gold standard'
    of Gale and Sampson, even though the differences are never expected to
    be significant and are equally distributed across the samples.

    freqdist : dict
        Frequency distribution of samples (keys) and counts (values) from
        which the probability distribution will be calculated.
    p_value : float
        The p-value for calculating the confidence interval of the empirical
        Turing estimate, which guides the decision of using either the Turing
        estimate "x" or the log-linear smoothed "y". Defaults to 0.05, as per
        the reference implementation by Sampson, but consider that the authors,
        both in their paper and in the code following suggestions credited to
        private communication with Fan Yang, suggest using a value of 0.1.
    default_p0 : float
        An optional value indicating the probability for unobserved samples
        ("p0") in cases where no samples with a single count are observed; if
        this value is not specified, "p0" will default to a Laplace estimation
        for the current frequency distribution. Please note that this is
        intended change from the reference implementation by Gale and Sampson.
    logprob : bool
        Whether to return the log-probabilities (default) or the
        probabilities themselves. When using the log-probabilities, the
        counts are automatically corrected to avoid domain errors.
    allow_fail : bool
        A logic value informing if the function is allowed to fail, throwing
        RuntimeWarning exceptions, if the essential assumptions on the
        frequency distribution are not met, i.e., if the slope of the log-linear
        regression is > -1.0 or if an unobserved count is reached before we are
        able to cross the smoothing threshold. If set to False, the estimation
        might result in an unreliable probability distribution; defaults to
        True.
    """

    def __init__(
        self,
        freqdist: Dict[str, int],
        p_value: float = 0.05,
        default_p0: Optional[float] = None,
        logprob: bool = True,
        allow_fail: bool = True,
    ):
        # Call the parent constructor
        # TODO: add checks
        super().__init__()

        # Store the parameters
        self.logprob = logprob
        self.name = "Simple Good-Turing"

        # Calculate the confidence level from the p_value.
        confidence_level = scipy.stats.norm.ppf(1.0 - (p_value / 2.0))

        # Remove all samples with `count` equal to zero.
        # TODO: consider counts of zero in all other tests
        freqdist = {sample: count for sample, count in freqdist.items() if count > 0}

        # Prepare vectors for frequencies (`r` in G&S) and frequencies of
        # frequencies (`Nr` in G&S). freqdist.values() is cast to a tuple because
        # we can't consume the iterable a single time. `freqs_keys` is sorted to
        # make vector computations faster later on (so we query lists and not
        # dictionaries).
        freqs = tuple(freqdist.values())
        freqs_keys = sorted(set(freqs))  # r -> n (G&S)
        freqs_of_freqs = {c: freqs.count(c) for c in freqs_keys}

        # The papers and the implementations are not clear on how to calculate the
        # probability of unobserved states in case of missing single-count samples
        # (unless we just fail, of course); Gale and Sampson's C implementation
        # defaults to 0.0, which is not acceptable for our purposes. The solution
        # here offered is to either use an user-provided probability (but in this
        # case we are not necessarily defaulting to _UNOBS, and, in fact, the
        # function argument name is `default_p0` and not `unobs_prob`) or default
        # to a Lidstone smoothing with a gamma of 1.0 (i.e., using Laplace
        # smoothing constant).
        # TODO: Investigate and discuss other possible solutions, including
        #       user-defined `gamma`, `bins`, and/or `N`.
        if 1 in freqs_keys:
            p0 = freqs_of_freqs[1] / sum(freqs)
        else:
            p0 = default_p0 or (1.0 / (sum(freqs) + 1))

        # Compute Sampson's Z: for each count `j`, we set Z[j] to the linear
        # interpolation of {i, j, k}, where `i` is the greatest observed count less
        # than `j`, and `k` the smallest observed count greater than `j`.
        i = [0] + freqs_keys[:-1]
        k = freqs_keys[1:] + [2 * freqs_keys[-1] - i[-1]]
        z = {j: 2 * freqs_of_freqs[j] / (k - i) for i, j, k in zip(i, freqs_keys, k)}

        # Compute a loglinear regression of Z[r] over r. We cast keys and values to
        # a list for the computation with `linalg.lstsq`.
        z_keys = list(z.keys())
        z_values = list(z.values())
        slope, intercept = scipy.linalg.lstsq(np.c_[np.log(z_keys), (1,) * len(z_keys)], np.log(z_values))[0]
        # print ('Regression: log(z) = %f*log(r) + %f' % (slope, intercept))
        if slope > -1.0 and allow_fail:
            raise RuntimeWarning("In SGT, linear regression slope is > -1.0.")

        # Apply Gale and Sampson's "simple" log-linear smoothing method.
        r_smoothed = {}
        use_y = False
        for r in freqs_keys:
            # `y` is the log-linear smoothing.
            y = float(r + 1) * np.exp(slope * np.log(r + 1) + intercept) / np.exp(slope * np.log(r) + intercept)

            # If we've already started using `y` as the estimate for `r`, then
            # continue doing so; also start doing so if no samples were observed
            # with count equal to `r+1` (following comments and variable names in
            # both Sampson's C implementation and in NLTK, we check at which
            # point we should `switch`)
            if r + 1 not in freqs_of_freqs:
                if not use_y:
                    # An unobserved count was reached before we were able to cross
                    # the smoothing threshold; this means that assumptions were
                    # not met and the results will likely be off.
                    if allow_fail:
                        raise RuntimeWarning("In SGT, unobserved count before smoothing threshold.")

                use_y = True

            # If we are using `y`, just copy its value to `r_smoothed`, otherwise
            # perform the actual calculation.
            if use_y:
                r_smoothed[r] = y
            else:
                # `estim` is the empirical Turing estimate for `r` (equivalent to
                # `x` in G&S)
                estim = (float(r + 1) * freqs_of_freqs[r + 1]) / freqs_of_freqs[r]

                nr = float(freqs_of_freqs[r])
                nr1 = float(freqs_of_freqs[r + 1])

                # `width` is the width of the confidence interval of the empirical
                # Turing estimate (for which Sampson uses 95% but suggests 90%),
                # when assuming independence.
                width = confidence_level * np.sqrt(float(r + 1) ** 2 * (nr1 / nr**2) * (1.0 + (nr1 / nr)))

                # If the difference between `x` and `y` is more than `t`, then the
                # empirical Turing estimate `x` tends to be more accurate.
                # Otherwise, use the loglinear smoothed value `y`.
                if abs(estim - y) > width:
                    r_smoothed[r] = estim
                else:
                    use_y = True
                    r_smoothed[r] = y

        # (Re)normalize and return the resulting smoothed probabilities, less the
        # estimated probability mass of unseen species; please note that we might
        # be unable to calculate some probabilities if the function was not allowed
        # to fail, mostly due to math domain errors. We default to `p0` in all such
        # cases.
        smooth_sum = sum([freqs_of_freqs[r] * r_smooth for r, r_smooth in r_smoothed.items()])

        # Build the probability distribution for the observed samples and for
        # unobserved ones.
        if self.logprob:
            self._unobs = math.log(p0)
            for sample, count in freqdist.items():
                prob = (1.0 - p0) * (r_smoothed[count] / smooth_sum)
                if prob == 0.0:
                    self._prob[sample] = math.log(p0)
                else:
                    self._prob[sample] = math.log(prob)
        else:
            self._unobs = p0
            for sample, count in freqdist.items():
                prob = (1.0 - p0) * (r_smoothed[count] / smooth_sum)
                if prob == 0.0:
                    self._prob[sample] = p0
                else:
                    self._prob[sample] = prob
