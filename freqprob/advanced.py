"""Advanced probability scoring methods.

This module implements sophisticated smoothing methods including
Witten-Bell discounting, Certainty Degree estimation, and
Simple Good-Turing smoothing. These methods use more complex
statistical techniques to estimate probability distributions.
"""

import math
from dataclasses import dataclass

# Import 3rd-party modules
import numpy as np
import scipy
import scipy.linalg
import scipy.stats

from .base import FrequencyDistribution, Probability, ScoringMethod, ScoringMethodConfig
from .cache import cached_computation, cached_sgt_computation


@dataclass
class WittenBellConfig(ScoringMethodConfig):
    """Configuration for Witten-Bell smoothing.

    Attributes:
    ----------
    bins : int | None
        Total number of possible bins/elements (default: vocabulary size)
    logprob : bool
        Whether to return log-probabilities (default: True)
    """

    bins: int | None = None

    logprob: bool = True


@dataclass
class CertaintyDegreeConfig(ScoringMethodConfig):
    """Configuration for Certainty Degree estimation.

    Attributes:
    ----------
    bins : int | None
        Total number of possible bins/elements (default: vocabulary size)
    unobs_prob : Probability
        Reserved probability mass for unobserved elements (default: 0.0)
    logprob : bool
        Whether to return log-probabilities (default: True)
    """

    bins: int | None = None

    unobs_prob: Probability = 0.0
    logprob: bool = True


@dataclass
class SimpleGoodTuringConfig(ScoringMethodConfig):
    """Configuration for Simple Good-Turing smoothing.

    Attributes:
    ----------
    p_value : float
        Confidence level for smoothing threshold (default: 0.05)
    default_p0 : float | None
        Fallback probability for unobserved elements (default: None)
    logprob : bool
        Whether to return log-probabilities (default: True)
    allow_fail : bool
        Whether to raise errors on invalid assumptions (default: True)
    """

    p_value: float = 0.05

    default_p0: float | None = None
    logprob: bool = True
    allow_fail: bool = True


class WittenBell(ScoringMethod):
    """Returns a Witten-Bell estimate probability distribution.

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

    __slots__ = ()

    def __init__(
        self,
        freqdist: FrequencyDistribution,
        bins: int | None = None,
        logprob: bool = True,
    ) -> None:
        """Initialize Witten-Bell smoothing."""
        config = WittenBellConfig(bins=bins, logprob=logprob)
        super().__init__(config)
        self.name = "Witten-Bell"
        self.fit(freqdist)

    @cached_computation()
    def _compute_probabilities(self, freqdist: FrequencyDistribution) -> None:
        """Compute Witten-Bell probabilities."""
        bins = self.config.bins

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
    """Returns a Certainty Degree probability distribution.

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

    __slots__ = ()

    def __init__(
        self,
        freqdist: FrequencyDistribution,
        bins: int | None = None,
        unobs_prob: Probability = 0.0,
        logprob: bool = True,
    ) -> None:
        """Initialize Certainty Degree estimation."""
        config = CertaintyDegreeConfig(bins=bins, unobs_prob=unobs_prob, logprob=logprob)
        super().__init__(config)
        self.name = "Certainty Degree"
        self.fit(freqdist)

    @cached_computation()
    def _compute_probabilities(self, freqdist: FrequencyDistribution) -> None:
        """Compute Certainty Degree probabilities."""
        bins = self.config.bins

        unobs_prob = self.config.unobs_prob

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
            # Ensure unobs_prob is not None and handle self._unobs initialization
            unobs_prob = unobs_prob or 0.0
            current_unobs = getattr(self, "_unobs", 0.0) or 0.0
            unobs_prob = max(unobs_prob, current_unobs)
            prob_space = min(1.0 - (b / (z + 1)) ** n, 1.0 - unobs_prob)
            self._prob = {
                sample: math.log((count / n) * prob_space) for sample, count in freqdist.items()
            }
            self._unobs = math.log(-(prob_space - 1.0))
        else:
            # Ensure unobs_prob is not None
            unobs_prob = unobs_prob or 0.0
            prob_space = min(1.0 - (b / (z + 1)) ** n, 1.0 - unobs_prob)
            self._prob = {sample: (count / n) * prob_space for sample, count in freqdist.items()}
            self._unobs = -(prob_space - 1.0)


class SimpleGoodTuring(ScoringMethod):
    """Returns a Simple Good-Turing estimate probability distribution.

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

    __slots__ = ()

    def __init__(
        self,
        freqdist: FrequencyDistribution,
        p_value: float = 0.05,
        default_p0: float | None = None,
        logprob: bool = True,
        allow_fail: bool = True,
    ) -> None:
        """Initialize Simple Good-Turing smoothing."""
        config = SimpleGoodTuringConfig(
            p_value=p_value,
            default_p0=default_p0,
            logprob=logprob,
            allow_fail=allow_fail,
        )
        super().__init__(config)
        self.name = "Simple Good-Turing"
        self.fit(freqdist)

    @cached_sgt_computation
    def _compute_probabilities(self, freqdist: FrequencyDistribution) -> None:
        """Compute Simple Good-Turing probabilities."""
        p_value = self.config.p_value  # type: ignore

        default_p0 = self.config.default_p0  # type: ignore
        allow_fail = self.config.allow_fail  # type: ignore

        # Calculate the confidence level from the p_value.
        confidence_level = scipy.stats.norm.ppf(1.0 - (p_value / 2.0))

        # Remove all samples with `count` equal to zero.
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
        if 1 in freqs_keys:
            p0 = freqs_of_freqs[1] / sum(freqs)
        else:
            p0 = default_p0 or (1.0 / (sum(freqs) + 1))

        # Compute Sampson's Z: for each count `j`, we set Z[j] to the linear
        # interpolation of {i, j, k}, where `i` is the greatest observed count less
        # than `j`, and `k` the smallest observed count greater than `j`.
        i = [0, *freqs_keys[:-1]]
        k = [*freqs_keys[1:], 2 * freqs_keys[-1] - i[-1]]
        z = {j: 2 * freqs_of_freqs[j] / (k - i) for i, j, k in zip(i, freqs_keys, k, strict=False)}

        # Compute a loglinear regression of Z[r] over r. We cast keys and values to
        # a list for the computation with `linalg.lstsq`.
        z_keys = list(z.keys())
        z_values = list(z.values())
        slope, intercept = scipy.linalg.lstsq(
            np.c_[np.log(z_keys), (1,) * len(z_keys)], np.log(z_values)
        )[0]
        if slope > -1.0 and allow_fail:
            raise RuntimeWarning("In SGT, linear regression slope is > -1.0.")

        # Apply Gale and Sampson's "simple" log-linear smoothing method.
        r_smoothed = {}
        use_y = False
        for r in freqs_keys:
            # `y` is the log-linear smoothing.
            y = (
                float(r + 1)
                * np.exp(slope * np.log(r + 1) + intercept)
                / np.exp(slope * np.log(r) + intercept)
            )

            # If we've already started using `y` as the estimate for `r`, then
            # continue doing so; also start doing so if no samples were observed
            # with count equal to `r+1` (following comments and variable names in
            # both Sampson's C implementation and in NLTK, we check at which
            # point we should `switch`)
            if r + 1 not in freqs_of_freqs and not use_y:
                # An unobserved count was reached before we were able to cross
                # the smoothing threshold; this means that assumptions were
                # not met and the results will likely be off.
                if not allow_fail:
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
                width = confidence_level * np.sqrt(
                    float(r + 1) ** 2 * (nr1 / nr**2) * (1.0 + (nr1 / nr))
                )

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
