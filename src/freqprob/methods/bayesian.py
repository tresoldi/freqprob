"""Bayesian (Dirichlet-prior) smoothing."""

import math
from dataclasses import dataclass

from freqprob.base import (
    FrequencyDistribution,
    ScoringMethod,
    ScoringMethodConfig,
)


@dataclass
class BayesianConfig(ScoringMethodConfig):
    """Configuration for Bayesian smoothing methods.

    Attributes:
        alpha: Dirichlet concentration parameter (``alpha > 0``, default:
            ``1.0``).
        logprob: Whether to return log-probabilities (default: ``True``).
    """

    alpha: float = 1.0


class Bayesian(ScoringMethod):
    """Bayesian smoothing with a Dirichlet prior.

    Estimates probabilities as the posterior mean under a symmetric Dirichlet
    prior, adding ``alpha`` pseudocounts to each possible outcome. Each observed
    element gets ``(c_i + alpha) / (N + V * alpha)`` and any unobserved element
    gets ``alpha / (N + V * alpha)``, where ``N`` is the total count and ``V``
    the support size. This is numerically identical to Lidstone smoothing with
    ``gamma = alpha``, but framed so that ``alpha`` reads as prior belief: use it
    when you want a theoretically principled smoother whose strength you can tune
    from near-MLE (small ``alpha``) to strongly uniform (large ``alpha``).

    Args:
        freqdist: Mapping of elements to observed counts.
        alpha: Dirichlet concentration parameter (``alpha > 0``). ``alpha -> 0``
            approaches MLE, ``alpha = 1`` gives a uniform prior (Laplace
            smoothing), and ``alpha > 1`` prefers more uniform distributions.
            Defaults to ``1.0``.
        logprob: Return log-probabilities if ``True`` (the default), otherwise
            plain probabilities.

    Examples:
        Bayesian smoothing with a uniform prior (``alpha = 1``):

        >>> from freqprob import Bayesian
        >>> freqdist = {"apple": 8, "banana": 4, "cherry": 1}
        >>> bayes = Bayesian(freqdist, alpha=1.0, logprob=False)
        >>> bayes("apple")     # (8+1)/(13+3*1) = 9/16
        0.5625
        >>> bayes("banana")    # (4+1)/(13+3*1) = 5/16
        0.3125
        >>> bayes("unseen")    # 1/(13+3*1) = 1/16
        0.0625

        A larger alpha smooths more strongly toward uniform:

        >>> bayes_smooth = Bayesian(freqdist, alpha=2.0, logprob=False)
        >>> round(bayes_smooth("apple"), 3)    # (8+2)/(13+3*2) = 10/19
        0.526
        >>> round(bayes_smooth("unseen"), 3)   # 2/(13+3*2) = 2/19
        0.105

        A small alpha stays closer to the observed frequencies:

        >>> bayes_minimal = Bayesian(freqdist, alpha=0.1, logprob=False)
        >>> round(bayes_minimal("apple"), 3)   # (8+0.1)/(13+3*0.1) = 8.1/13.3
        0.609

    Note:
        A positive ``alpha`` is required; passing ``alpha <= 0`` raises
        ``ValueError``. Equivalent to `Lidstone` with ``gamma = alpha``.
    """

    __slots__ = ()

    def __init__(
        self, freqdist: FrequencyDistribution, alpha: float = 1.0, logprob: bool = True
    ) -> None:
        """Initialize Bayesian smoothing."""
        if alpha <= 0:
            raise ValueError("Alpha must be positive")

        config = BayesianConfig(alpha=alpha, logprob=logprob)
        super().__init__(config)
        self.name = "Bayesian"
        self.fit(freqdist)

    def _compute_probabilities(self, freqdist: FrequencyDistribution) -> None:
        """Compute Bayesian smoothed probabilities.

        Parameters
        ----------
        freqdist : FrequencyDistribution
            Frequency distribution with element counts
        """
        alpha = self.config.alpha  # type: ignore

        # Compute Dirichlet posterior parameters
        total_count = sum(freqdist.values())
        vocab_size = len(freqdist)
        denominator = total_count + vocab_size * alpha

        if self.logprob:
            # Log-probability computation
            self._prob = {
                element: math.log((count + alpha) / denominator)
                for element, count in freqdist.items()
            }
            self._unobs = math.log(alpha / denominator)
        else:
            # Regular probability computation
            self._prob = {
                element: (count + alpha) / denominator for element, count in freqdist.items()
            }
            self._unobs = alpha / denominator
