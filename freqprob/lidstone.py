"""Lidstone family probability scoring methods.

This module implements the Lidstone family of additive smoothing methods,
including Lidstone smoothing (with arbitrary gamma), Laplace smoothing (gamma=1),
and Expected Likelihood Estimation (gamma=0.5). These methods add virtual
counts to observed data to handle the zero probability problem.
"""

import math
from dataclasses import dataclass

from .base import FrequencyDistribution, ScoringMethod, ScoringMethodConfig


@dataclass
class LidstoneConfig(ScoringMethodConfig):
    """Configuration for Lidstone smoothing.

    Attributes:
    ----------
    gamma : float
        Additive smoothing parameter (gamma ≥ 0, default: 1.0)
    bins : int | None
        Total number of possible bins/elements (default: vocabulary size)
    logprob : bool
        Whether to return log-probabilities (default: True)
    """

    gamma: float = 1.0

    bins: int | None = None
    logprob: bool = True


@dataclass
class LaplaceConfig(ScoringMethodConfig):
    """Configuration for Laplace smoothing (Lidstone with gamma=1).

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
class ELEConfig(ScoringMethodConfig):
    """Configuration for Expected Likelihood Estimation (Lidstone with gamma=0.5).

    Attributes:
    ----------
    bins : int | None
        Total number of possible bins/elements (default: vocabulary size)
    logprob : bool
        Whether to return log-probabilities (default: True)
    """

    bins: int | None = None

    logprob: bool = True


class Lidstone(ScoringMethod):
    """Lidstone additive smoothing probability distribution.

    Also known as "additive smoothing," this method addresses the zero
    probability problem by adding a virtual count gamma (gamma) to each possible
    element. This is equivalent to assuming a symmetric Dirichlet prior
    with concentration parameter gamma.

    Mathematical Formulation
    ------------------------
    For elements with counts cᵢ, total count N = Σⱼcⱼ, and B bins:

    P(wᵢ) = (cᵢ + gamma) / (N + B*gamma)  for observed elements wᵢ ∈ V
    P(w)  = gamma / (N + B*gamma)         for unobserved elements w ∉ V

    The method effectively adds gamma "virtual counts" to every possible element,
    creating a uniform pseudocount baseline that prevents zero probabilities.

    Parameters
    ----------
    freqdist : FrequencyDistribution
        Frequency distribution mapping elements to their observed counts
    gamma : float
        Additive smoothing parameter (gamma ≥ 0). Common values:
        - gamma = 1.0: Laplace smoothing (uniform prior)
        - gamma = 0.5: Jeffreys prior (Expected Likelihood Estimation)
        - gamma → 0: Approaches MLE
    bins : int | None, default=None
        Total number of possible elements. If None, uses vocabulary size |V|
    logprob : bool, default=True
        Whether to return log-probabilities or probabilities

    Examples:
    --------
    Basic Lidstone smoothing:
    >>> freqdist = {'apple': 3, 'banana': 1}
    >>> lidstone = Lidstone(freqdist, gamma=1.0, logprob=False)
    >>> lidstone('apple')    # (3+1)/(4+2*1) = 4/6
    0.6666666666666666
    >>> lidstone('banana')   # (1+1)/(4+2*1) = 2/6
    0.3333333333333333
    >>> lidstone('cherry')   # 1/(4+2*1) = 1/6
    0.16666666666666666

    Effect of different gamma values:
    >>> # Higher gamma = more smoothing
    >>> smooth = Lidstone(freqdist, gamma=2.0, logprob=False)
    >>> smooth('apple')      # (3+2)/(4+2*2) = 5/8
    0.625
    >>> smooth('cherry')     # 2/(4+2*2) = 2/8
    0.25

    >>> # Lower gamma = less smoothing
    >>> minimal = Lidstone(freqdist, gamma=0.1, logprob=False)
    >>> minimal('apple')     # (3+0.1)/(4+2*0.1) ≈ 3.1/4.2
    0.7380952380952381

    Specifying larger vocabulary:
    >>> lidstone_big = Lidstone(freqdist, gamma=1.0, bins=1000, logprob=False)
    >>> lidstone_big('apple')    # (3+1)/(4+1000*1) = 4/1004
    0.003984063745019921
    >>> lidstone_big('unseen')   # 1/(4+1000*1) = 1/1004
    0.0009960159203980099

    Properties
    ----------
    - Guarantees non-zero probabilities for all elements
    - Reduces to MLE as gamma → 0 and N → ∞
    - Uniform pseudocount distribution when gamma = constant
    - Bayesian interpretation as Dirichlet prior
    - Simple and computationally efficient

    Notes:
    -----
    The choice of gamma represents a bias-variance tradeoff:
    - Small gamma: Low bias but high variance (closer to MLE)
    - Large gamma: Higher bias but lower variance (more uniform)

    When bins > vocabulary size, the method reserves more probability
    mass for potential unseen elements.
    """

    __slots__ = ()

    def __init__(
        self,
        freqdist: FrequencyDistribution,
        gamma: float,
        bins: int | None = None,
        logprob: bool = True,
    ) -> None:
        """Initialize Lidstone smoothing."""
        # Default bins to vocabulary size if not specified
        if bins is None:
            bins = len(freqdist)

        config = LidstoneConfig(gamma=gamma, bins=bins, logprob=logprob)
        super().__init__(config)
        self.name = "Lidstone"
        self.fit(freqdist)

    def _compute_probabilities(self, freqdist: FrequencyDistribution) -> None:
        """Compute Lidstone smoothed probabilities.

        Parameters
        ----------
        freqdist : FrequencyDistribution
            Frequency distribution with element counts
        """
        gamma = self.config.gamma
        bins = self.config.bins

        # Ensure gamma and bins are not None (they are set in __init__)
        assert gamma is not None, "Gamma must be set in LidstoneConfig"
        assert bins is not None, "Bins must be set in LidstoneConfig"

        # Calculate normalization factors
        total_count = sum(freqdist.values())
        denominator = total_count + bins * gamma

        if self.logprob:
            # Log-probability computation
            self._prob = {
                elem: math.log((count + gamma) / denominator) for elem, count in freqdist.items()
            }
            self._unobs = math.log(gamma / denominator)
        else:
            # Regular probability computation
            self._prob = {elem: (count + gamma) / denominator for elem, count in freqdist.items()}
            self._unobs = gamma / denominator


class Laplace(Lidstone):
    """Laplace smoothing probability distribution.

    A special case of Lidstone smoothing with gamma = 1.0, also known as
    "add-one smoothing." This is the most commonly used additive smoothing
    method, corresponding to a uniform Dirichlet prior.

    Mathematical Formulation
    ------------------------
    P(wᵢ) = (cᵢ + 1) / (N + B)    for observed elements wᵢ ∈ V
    P(w)  = 1 / (N + B)           for unobserved elements w ∉ V

    This is equivalent to adding one virtual observation to each possible element.

    Parameters
    ----------
    freqdist : FrequencyDistribution
        Frequency distribution mapping elements to their observed counts
    bins : int | None, default=None
        Total number of possible elements. If None, uses vocabulary size |V|
    logprob : bool, default=True
        Whether to return log-probabilities or probabilities

    Examples:
    --------
    >>> freqdist = {'red': 3, 'blue': 2, 'green': 1}
    >>> laplace = Laplace(freqdist, logprob=False)
    >>> laplace('red')     # (3+1)/(6+3) = 4/9 ≈ 0.444
    0.4444444444444444
    >>> laplace('blue')    # (2+1)/(6+3) = 3/9 ≈ 0.333
    0.3333333333333333
    >>> laplace('yellow')  # 1/(6+3) = 1/9 ≈ 0.111
    0.1111111111111111

    Notes:
    -----
    Laplace smoothing is widely used because:
    - Simple and intuitive (add one to everything)
    - Provides reasonable smoothing for most applications
    - Has nice theoretical properties (uniform prior)
    - Computationally efficient
    """

    __slots__ = ()

    def __init__(
        self, freqdist: FrequencyDistribution, bins: int | None = None, logprob: bool = True
    ) -> None:
        """Initialize Laplace smoothing."""
        # Call parent with gamma=1.0 for Laplace smoothing
        super().__init__(freqdist, gamma=1.0, bins=bins, logprob=logprob)
        self.name = "Laplace"


class ELE(Lidstone):
    """Expected Likelihood Estimation probability distribution.

    A special case of Lidstone smoothing with gamma = 0.5, corresponding to
    the Jeffreys prior for multinomial distributions. This provides a
    compromise between MLE and uniform smoothing.

    Mathematical Formulation
    ------------------------
    P(wᵢ) = (cᵢ + 0.5) / (N + 0.5*B)  for observed elements wᵢ ∈ V
    P(w)  = 0.5 / (N + 0.5*B)         for unobserved elements w ∉ V

    This corresponds to adding half a virtual observation to each element.

    Parameters
    ----------
    freqdist : FrequencyDistribution
        Frequency distribution mapping elements to their observed counts
    bins : int | None, default=None
        Total number of possible elements. If None, uses vocabulary size |V|
    logprob : bool, default=True
        Whether to return log-probabilities or probabilities

    Examples:
    --------
    >>> freqdist = {'cat': 4, 'dog': 2}
    >>> ele = ELE(freqdist, logprob=False)
    >>> ele('cat')     # (4+0.5)/(6+0.5*2) = 4.5/7 ≈ 0.643
    0.6428571428571429
    >>> ele('dog')     # (2+0.5)/(6+0.5*2) = 2.5/7 ≈ 0.357
    0.35714285714285715
    >>> ele('bird')    # 0.5/(6+0.5*2) = 0.5/7 ≈ 0.071
    0.07142857142857142

    Notes:
    -----
    ELE is particularly useful because:
    - Jeffreys prior is non-informative in the Bayesian sense
    - Provides less smoothing than Laplace (gamma=1)
    - Often performs well empirically
    - Reduces overfitting while maintaining sensitivity to data
    """

    __slots__ = ()

    def __init__(
        self, freqdist: FrequencyDistribution, bins: int | None = None, logprob: bool = True
    ) -> None:
        """Initialize Expected Likelihood Estimation."""
        # Call parent with gamma=0.5 for ELE
        super().__init__(freqdist, gamma=0.5, bins=bins, logprob=logprob)
        self.name = "ELE"
