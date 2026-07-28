"""Lidstone family probability scoring methods.

This module implements the Lidstone family of additive smoothing methods,
including Lidstone smoothing (with arbitrary gamma), Laplace smoothing (gamma=1),
and Expected Likelihood Estimation (gamma=0.5). These methods add virtual
counts to observed data to handle the zero probability problem.
"""

import math

from freqprob.base import FrequencyDistribution, ScoringMethod, ScoringMethodConfig


class Lidstone(ScoringMethod):
    """Lidstone additive smoothing probability distribution.

    Additive smoothing tackles the zero-probability problem by adding a virtual
    count ``gamma`` to every possible element before normalizing. For counts
    ``c_i``, total ``N``, and ``B`` bins, each observed element gets
    ``(c_i + gamma) / (N + B * gamma)`` and any unobserved element gets
    ``gamma / (N + B * gamma)``. This is the posterior mean under a symmetric
    Dirichlet prior with concentration ``gamma``. Use it when you need a simple,
    principled way to keep unseen elements from scoring zero, and tune ``gamma``
    to trade off between staying close to the data (small ``gamma``) and a more
    uniform distribution (large ``gamma``).

    Args:
        freqdist: Mapping of elements to observed counts.
        gamma: Additive smoothing parameter (``gamma >= 0``). ``1.0`` gives
            Laplace smoothing, ``0.5`` gives the Jeffreys prior (ELE), and
            ``gamma -> 0`` approaches MLE.
        bins: Total number of possible elements. If ``None`` (the default),
            uses the support size ``|V|``. A larger ``bins`` reserves more mass
            for unseen elements.
        logprob: Return log-probabilities if ``True`` (the default), otherwise
            plain probabilities.

    Examples:
        Add-gamma smoothing gives unseen elements non-zero probability:

        >>> from freqprob import Lidstone
        >>> freqdist = {"apple": 3, "banana": 1}
        >>> lidstone = Lidstone(freqdist, gamma=1.0, logprob=False)
        >>> round(lidstone("apple"), 3)    # (3+1)/(4+2*1) = 4/6
        0.667
        >>> round(lidstone("banana"), 3)   # (1+1)/(4+2*1) = 2/6
        0.333
        >>> round(lidstone("cherry"), 3)   # 1/(4+2*1) = 1/6
        0.167

        A larger gamma pulls probabilities toward uniform:

        >>> smooth = Lidstone(freqdist, gamma=2.0, logprob=False)
        >>> smooth("apple")      # (3+2)/(4+2*2) = 5/8
        0.625
        >>> smooth("cherry")     # 2/(4+2*2) = 2/8
        0.25

        A larger ``bins`` reserves more mass for potential unseen elements:

        >>> lidstone_big = Lidstone(freqdist, gamma=1.0, bins=1000, logprob=False)
        >>> round(lidstone_big("apple"), 4)    # (3+1)/(4+1000) = 4/1004
        0.004
        >>> round(lidstone_big("unseen"), 4)   # 1/(4+1000) = 1/1004
        0.001

    Note:
        The choice of ``gamma`` is a bias-variance tradeoff: small values stay
        close to MLE (low bias, high variance) while large values approach a
        uniform distribution (higher bias, lower variance).
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
        # Default bins to support size if not specified
        if bins is None:
            bins = len(freqdist)

        config = ScoringMethodConfig(gamma=gamma, bins=bins, logprob=logprob)
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
        assert gamma is not None, "Gamma must be set before computing probabilities"
        assert bins is not None, "Bins must be set before computing probabilities"

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
    """Laplace (add-one) smoothing probability distribution.

    The special case of Lidstone smoothing with ``gamma = 1.0``, also called
    "add-one smoothing." Each observed element gets ``(c_i + 1) / (N + B)`` and
    any unobserved element gets ``1 / (N + B)`` — i.e. one virtual observation
    added to every possible element, corresponding to a uniform Dirichlet prior.
    It is the most common additive smoother: reach for it when you want a simple,
    reasonable default that never assigns zero probability.

    Args:
        freqdist: Mapping of elements to observed counts.
        bins: Total number of possible elements. If ``None`` (the default),
            uses the support size ``|V|``.
        logprob: Return log-probabilities if ``True`` (the default), otherwise
            plain probabilities.

    Examples:
        Add one to every count, then normalize:

        >>> from freqprob import Laplace
        >>> freqdist = {"red": 3, "blue": 2, "green": 1}
        >>> laplace = Laplace(freqdist, logprob=False)
        >>> round(laplace("red"), 3)     # (3+1)/(6+3) = 4/9
        0.444
        >>> round(laplace("blue"), 3)    # (2+1)/(6+3) = 3/9
        0.333
        >>> round(laplace("yellow"), 3)  # 1/(6+3) = 1/9
        0.111
    """

    __slots__ = ()

    def __init__(
        self,
        freqdist: FrequencyDistribution,
        bins: int | None = None,
        logprob: bool = True,
    ) -> None:
        """Initialize Laplace smoothing."""
        # Call parent with gamma=1.0 for Laplace smoothing
        super().__init__(freqdist, gamma=1.0, bins=bins, logprob=logprob)
        self.name = "Laplace"


class ELE(Lidstone):
    """Expected Likelihood Estimation probability distribution.

    The special case of Lidstone smoothing with ``gamma = 0.5``, corresponding
    to the Jeffreys prior for multinomial distributions. Each observed element
    gets ``(c_i + 0.5) / (N + 0.5 * B)`` and any unobserved element gets
    ``0.5 / (N + 0.5 * B)`` — half a virtual observation per element. This is a
    middle ground between MLE and Laplace: use it when add-one smoothing feels
    too aggressive but you still want unseen elements handled.

    Args:
        freqdist: Mapping of elements to observed counts.
        bins: Total number of possible elements. If ``None`` (the default),
            uses the support size ``|V|``.
        logprob: Return log-probabilities if ``True`` (the default), otherwise
            plain probabilities.

    Examples:
        Add half a count to each element, then normalize:

        >>> from freqprob import ELE
        >>> freqdist = {"cat": 4, "dog": 2}
        >>> ele = ELE(freqdist, logprob=False)
        >>> round(ele("cat"), 3)     # (4+0.5)/(6+0.5*2) = 4.5/7
        0.643
        >>> round(ele("dog"), 3)     # (2+0.5)/(6+0.5*2) = 2.5/7
        0.357
        >>> round(ele("bird"), 3)    # 0.5/(6+0.5*2) = 0.5/7
        0.071

    Note:
        ELE applies less smoothing than Laplace (``gamma = 1``), so it stays
        closer to the observed frequencies while still avoiding zero
        probabilities.
    """

    __slots__ = ()

    def __init__(
        self,
        freqdist: FrequencyDistribution,
        bins: int | None = None,
        logprob: bool = True,
    ) -> None:
        """Initialize Expected Likelihood Estimation."""
        # Call parent with gamma=0.5 for ELE
        super().__init__(freqdist, gamma=0.5, bins=bins, logprob=logprob)
        self.name = "ELE"
