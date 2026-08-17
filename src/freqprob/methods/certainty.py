"""Certainty-degree probability estimation."""

from freqprob.base import (
    Estimates,
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
    def _estimate(self, freqdist: FrequencyDistribution) -> Estimates:
        """Scale the maximum-likelihood estimate by the certainty of full coverage.

        For very large ``n`` the reserved mass underflows, because the estimator
        is then confident that every element that matters has been observed.
        That used to need an explicit epsilon here to keep :func:`math.log` in
        its domain; :meth:`~freqprob.base.ScoringMethod._score_of` now applies
        that floor for every method.

        Args:
            freqdist: Frequency distribution with element counts.

        Returns:
            The discounted relative frequency of each element, and the mass
            reserved for everything unobserved.
        """
        n = sum(freqdist.values())
        b = len(freqdist)
        z = self.config.bins or b
        unobs_prob = self.config.unobs_prob or 0.0

        prob_space = min(1.0 - (b / (z + 1)) ** n, 1.0 - unobs_prob)
        table = {sample: (count / n) * prob_space for sample, count in freqdist.items()}

        return table, 1.0 - prob_space
