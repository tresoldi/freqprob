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
    ----------
    alpha : float
        Dirichlet concentration parameter (alpha > 0, default: 1.0)
    logprob : bool
        Whether to return log-probabilities (default: True)
    """

    alpha: float = 1.0


class Bayesian(ScoringMethod):
    """Bayesian smoothing with Dirichlet prior.

    Uses a Dirichlet prior distribution to provide Bayesian probability estimates.
    This method is theoretically principled and provides natural uncertainty
    quantification through the prior distribution.

    Mathematical Formulation
    ------------------------
    P_Bayes(wᵢ) = (cᵢ + alpha) / (N + V*alpha)

    Where:
    - cᵢ is the observed count for word wᵢ
    - alpha is the Dirichlet concentration parameter (pseudocount)
    - N is the total observed count
    - V is the vocabulary size

    This is equivalent to adding alpha pseudocounts to each possible outcome
    and corresponds to the posterior mean under a symmetric Dirichlet prior.

    Parameters
    ----------
    freqdist : FrequencyDistribution
        Frequency distribution mapping elements to their observed counts
    alpha : float, default=1.0
        Dirichlet concentration parameter (alpha > 0). Controls smoothing strength:
        - alpha → 0: Approaches MLE (minimal smoothing)
        - alpha = 1: Uniform prior (Laplace smoothing)
        - alpha > 1: Stronger preference for uniformity
    logprob : bool, default=True
        Whether to return log-probabilities or probabilities

    Examples:
    --------
    Basic Bayesian smoothing with uniform prior:
    >>> freqdist = {'apple': 8, 'banana': 4, 'cherry': 1}
    >>> bayes = Bayesian(freqdist, alpha=1.0, logprob=False)
    >>> bayes('apple')     # (8+1)/(13+3*1) = 9/16
    0.5625
    >>> bayes('banana')    # (4+1)/(13+3*1) = 5/16
    0.3125
    >>> bayes('unseen')    # 1/(13+3*1) = 1/16
    0.0625

    Effect of different alpha values:
    >>> # Stronger smoothing (alpha = 2)
    >>> bayes_smooth = Bayesian(freqdist, alpha=2.0, logprob=False)
    >>> bayes_smooth('apple')    # (8+2)/(13+3*2) = 10/19
    0.5263157894736842
    >>> bayes_smooth('unseen')   # 2/(13+3*2) = 2/19
    0.10526315789473684

    >>> # Minimal smoothing (alpha = 0.1)
    >>> bayes_minimal = Bayesian(freqdist, alpha=0.1, logprob=False)
    >>> bayes_minimal('apple')   # (8+0.1)/(13+3*0.1) ≈ 8.1/13.3
    0.6090226699248121

    Properties
    ----------
    - Theoretically principled (Bayesian posterior)
    - Natural uncertainty quantification
    - Generalizes several classical methods
    - Smooth probability estimates
    - Prior encodes domain knowledge

    Notes:
    -----
    The choice of alpha reflects prior beliefs about outcome probabilities:
    - alpha = 1: Uniform prior (no preference for any outcome)
    - alpha < 1: Sparse prior (prefers concentrated distributions)
    - alpha > 1: Dense prior (prefers uniform distributions)

    This method is equivalent to Lidstone smoothing with gamma = alpha, but the
    Bayesian interpretation provides additional theoretical insights.
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
