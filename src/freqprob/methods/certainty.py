"""Certainty-degree probability estimation."""

import math

from freqprob.base import (
    MIN_PROBABILITY,
    FrequencyDistribution,
    Probability,
    ScoringMethod,
    ScoringMethodConfig,
)
from freqprob.cache import cached_computation


class CertaintyDegree(ScoringMethod):
    """Certainty-degree smoothing.

    Reserves probability mass for unseen elements based on the degree of
    certainty that no unobserved samples remain, derived from the number of
    observations relative to the support size. As more of the possible bins are
    observed, the reserved mass shrinks; the remaining mass rescales the
    Maximum-Likelihood estimate of each observed element.

    Args:
        freqdist: Mapping of elements to observed counts.
        bins: Optional total number of possible sample bins for the experiment.
            If ``None`` (the default), it defaults to the number of observed
            types.
        unobs_prob: Reserved probability mass for unobserved states
            (``0.0 <= unobs_prob <= 1.0``). Defaults to ``0.0``.
        logprob: Return log-probabilities if ``True`` (the default), otherwise
            plain probabilities. Counts are corrected to avoid domain errors
            when taking logarithms.

    Examples:
        Observed elements are rescaled by the computed certainty mass, and the
        remainder is reserved for unseen elements:

        >>> from freqprob import CertaintyDegree
        >>> freqdist = {"apple": 8, "banana": 4, "cherry": 2, "date": 1}
        >>> cd = CertaintyDegree(freqdist, logprob=False)
        >>> round(cd("apple"), 3)
        0.515
        >>> round(cd("date"), 3)
        0.064
        >>> round(cd("kiwi"), 3)  # unobserved
        0.035

    Note:
        This is an experimental distribution under development by Tiago
        Tresoldi; it should not be used as the sole or main distribution yet.
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
        config = ScoringMethodConfig(bins=bins, unobs_prob=unobs_prob, logprob=logprob)
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
            # Floor the reserved mass with a fixed epsilon (not self._unobs, which
            # holds the previous output and would be stale on a second fit()).
            unobs_prob = max(unobs_prob or 0.0, MIN_PROBABILITY)
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
