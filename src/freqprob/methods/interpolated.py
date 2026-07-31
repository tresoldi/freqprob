"""Interpolated (linear-combination) smoothing."""

import math
from dataclasses import dataclass

from freqprob.base import (
    Element,
    FrequencyDistribution,
    LogProbability,
    Probability,
    ScoringMethod,
    ScoringMethodConfig,
)


@dataclass
class InterpolatedConfig(ScoringMethodConfig):
    """Configuration for interpolated smoothing methods.

    Attributes:
    ----------
    lambda_weight : float
        Interpolation weight for higher-order model (0 ≤ λ ≤ 1, default: 0.7)
    logprob : bool
        Whether to return log-probabilities (default: True)
    """

    lambda_weight: float = 0.7


class Interpolated(ScoringMethod):
    """Linear interpolation smoothing between two frequency distributions.

    Blends a higher-order distribution with a lower-order one using a weighted
    average, ``lambda_weight * P_high + (1 - lambda_weight) * P_low``. This
    keeps the specificity of the high-order model while borrowing robustness
    from the smoother low-order model. Two modes are detected automatically
    from the keys:

    - N-gram interpolation, when the high-order keys are longer tuples than the
      low-order keys. The lower-order context is taken as the suffix of each
      high-order n-gram (e.g. bigram ``(w2, w3)`` from trigram ``(w1, w2, w3)``).
    - Same-type interpolation, when both distributions share the same key type
      (strings or equal-length tuples). Keys are matched directly.

    Args:
        high_order_dist: Higher-order frequency distribution mapping elements to
            counts (e.g. trigrams). For n-gram mode its tuple keys must be at
            least as long as those of ``low_order_dist``.
        low_order_dist: Lower-order frequency distribution (e.g. bigrams).
        lambda_weight: Interpolation weight ``0 <= lambda_weight <= 1`` for the
            high-order model. Higher values favor specificity, lower values
            favor smoothing. Defaults to ``0.7``.
        logprob: Return log-probabilities if ``True`` (the default), otherwise
            plain probabilities.

    Examples:
        N-gram interpolation (trigrams backed off to bigrams). The score of
        ``('the', 'big', 'cat')`` is ``0.7 * (3/5) + 0.3 * (5/10)``, and an
        unseen trigram backs off to ``0.3 * (5/10)``:

        >>> trigrams = {("the", "big", "cat"): 3, ("a", "big", "dog"): 2}
        >>> bigrams = {("big", "cat"): 5, ("big", "dog"): 3, ("small", "cat"): 2}
        >>> interp = Interpolated(trigrams, bigrams, lambda_weight=0.7, logprob=False)
        >>> round(interp(("the", "big", "cat")), 3)
        0.57
        >>> round(interp(("unseen", "big", "cat")), 3)
        0.15

        Same-type interpolation between two string distributions. The score of
        ``"word1"`` is ``0.6 * (10/15) + 0.4 * (3/10)``:

        >>> model1 = {"word1": 10, "word2": 5}
        >>> model2 = {"word1": 3, "word3": 7}
        >>> interp = Interpolated(model1, model2, lambda_weight=0.6, logprob=False)
        >>> round(interp("word1"), 3)
        0.52
        >>> round(interp("word2"), 3)
        0.2
        >>> round(interp("word3"), 3)
        0.28

    Note:
        The mode is inferred from the key types, so mixed tuple lengths within a
        single distribution raise an error. Unseen high-order n-grams back off
        to the low-order model, and all probabilities are floored at ``1e-10``
        for numerical stability. The weight ``lambda_weight`` is best tuned on
        held-out data.
    """

    __slots__ = ("_high_order_dist", "_high_order_n", "_low_order_dist", "_low_order_n")

    def __init__(
        self,
        high_order_dist: FrequencyDistribution,
        low_order_dist: FrequencyDistribution,
        lambda_weight: float = 0.7,
        logprob: bool = True,
    ) -> None:
        """Initialize Interpolated smoothing."""
        if not 0 <= lambda_weight <= 1:
            raise ValueError("Lambda weight must be between 0 and 1")

        config = InterpolatedConfig(lambda_weight=lambda_weight, logprob=logprob)
        super().__init__(config)
        self.name = "Interpolated"
        self._high_order_dist = high_order_dist
        self._low_order_dist = low_order_dist

        # Detect n-gram orders
        self._high_order_n = self._detect_order(high_order_dist)
        self._low_order_n = self._detect_order(low_order_dist)

        # Validate n-gram orders if both are tuples
        if (
            self._high_order_n is not None
            and self._low_order_n is not None
            and self._high_order_n < self._low_order_n
        ):
            raise ValueError(
                f"High-order n-gram order ({self._high_order_n}) must be greater than or equal to "
                f"low-order n-gram order ({self._low_order_n}). "
                f"Hint: Swap the order of your distributions to fix this."
            )

        self.fit(high_order_dist)  # Primary distribution

    def _detect_order(self, dist: FrequencyDistribution) -> int | None:
        """Detect n-gram order from distribution keys.

        Returns:
        -------
        int | None
            N-gram order if keys are tuples, None otherwise
        """
        if not dist:
            return None

        # Sample a few keys to detect type
        sample_keys = list(dist.keys())[:10]

        # Check if keys are tuples
        tuple_keys = [k for k in sample_keys if isinstance(k, tuple)]
        if not tuple_keys:
            return None  # Not n-grams (e.g., strings)

        # Get tuple lengths
        lengths = {len(k) for k in tuple_keys}
        if len(lengths) > 1:
            raise ValueError(
                f"Inconsistent tuple lengths in distribution: {lengths}. "
                "All n-grams must have the same order."
            )

        return lengths.pop()

    def _extract_lower_context(self, high_order_ngram: tuple) -> tuple:
        """Extract lower-order context from higher-order n-gram.

        For trigram (w1, w2, w3) with bigram model, extracts (w2, w3).
        For 4-gram (w1, w2, w3, w4) with trigram model, extracts (w2, w3, w4).

        Parameters
        ----------
        high_order_ngram : tuple
            Higher-order n-gram

        Returns:
        -------
        tuple
            Lower-order context (suffix of length low_order_n)
        """
        assert self._low_order_n is not None
        return high_order_ngram[-self._low_order_n :]

    def __call__(self, element: Element) -> Probability | LogProbability:
        """Score a single element with n-gram interpolation support.

        For n-gram interpolation mode, unseen high-order n-grams are handled
        by backing off to the lower-order model.

        Parameters
        ----------
        element : Element
            Element to be scored

        Returns:
        -------
        Probability | LogProbability
            The probability or log-probability of the element
        """
        # Check if already computed
        if element in self._prob:
            return self._prob[element]

        # For n-gram interpolation, compute on-the-fly for unseen high-order n-grams
        is_ngram_interpolation = (
            self._high_order_n is not None
            and self._low_order_n is not None
            and self._high_order_n > self._low_order_n
        )

        if (
            is_ngram_interpolation
            and isinstance(element, tuple)
            and len(element) == self._high_order_n
        ):
            # Extract lower-order context
            low_context = self._extract_lower_context(element)

            # Compute MLE for lower-order context
            low_total = sum(self._low_order_dist.values()) or 1
            low_count = self._low_order_dist.get(low_context, 0)
            low_prob = low_count / low_total if low_count > 0 else 1e-10

            # For unseen high-order n-gram, backoff: (1-λ) * P_low
            lambda_weight = self.config.lambda_weight  # type: ignore[attr-defined]
            interpolated_prob: float = (1 - lambda_weight) * low_prob
            interpolated_prob = max(interpolated_prob, 1e-10)

            if self.logprob:
                return math.log(interpolated_prob)
            return interpolated_prob

        # Fallback to default behavior
        return self._unobs

    def _compute_probabilities(self, freqdist: FrequencyDistribution) -> None:
        """Compute interpolated probabilities.

        Supports two modes:
        1. N-gram interpolation: Different n-gram orders (e.g., trigrams + bigrams)
           - Extracts lower-order context from higher-order n-grams
           - Interpolates: λ * P_high(ngram) + (1-λ) * P_low(context)

        2. Same-type interpolation: Same element types (both strings or same-length tuples)
           - Direct key matching
           - Interpolates: λ * P_high(elem) + (1-λ) * P_low(elem)

        Parameters
        ----------
        freqdist : FrequencyDistribution
            Primary (high-order) frequency distribution
        """
        lambda_weight = self.config.lambda_weight  # type: ignore

        # Compute MLE for both distributions
        high_total = sum(self._high_order_dist.values()) or 1
        low_total = sum(self._low_order_dist.values()) or 1

        high_probs = {
            element: count / high_total for element, count in self._high_order_dist.items()
        }
        low_probs = {element: count / low_total for element, count in self._low_order_dist.items()}

        # Compute unseen probability for low-order model (MLE-based)
        # For MLE: P(unseen) = 0, but we use small value for numerical stability
        low_unseen_prob = 1e-10

        # Determine interpolation mode
        is_ngram_interpolation = (
            self._high_order_n is not None
            and self._low_order_n is not None
            and self._high_order_n > self._low_order_n
        )

        if is_ngram_interpolation:
            # N-gram interpolation mode: extract lower-order context
            for high_ngram in self._high_order_dist:
                # Get high-order probability
                high_prob = high_probs.get(high_ngram, 0)

                # Extract lower-order context and get its probability
                low_context = self._extract_lower_context(high_ngram)  # type: ignore
                low_prob = low_probs.get(low_context, low_unseen_prob)

                # Interpolate
                interpolated_prob = lambda_weight * high_prob + (1 - lambda_weight) * low_prob

                # Ensure minimum probability for numerical stability
                interpolated_prob = max(interpolated_prob, 1e-10)

                if self.logprob:
                    self._prob[high_ngram] = math.log(interpolated_prob)
                else:
                    self._prob[high_ngram] = interpolated_prob
        else:
            # Same-type interpolation mode: direct key matching
            all_elements = set(high_probs.keys()) | set(low_probs.keys())

            for element in all_elements:
                high_prob = high_probs.get(element, 0)
                low_prob = low_probs.get(element, 0)

                interpolated_prob = lambda_weight * high_prob + (1 - lambda_weight) * low_prob

                # Ensure minimum probability for numerical stability
                interpolated_prob = max(interpolated_prob, 1e-10)

                if self.logprob:
                    self._prob[element] = math.log(interpolated_prob)
                else:
                    self._prob[element] = interpolated_prob

        # Set unobserved probability
        # For n-gram interpolation, unseen elements get backed-off to low-order model
        if is_ngram_interpolation:
            # Unseen high-order n-grams backoff to low-order unseen probability
            interpolated_unseen = (1 - lambda_weight) * low_unseen_prob
        else:
            # Same-type: both models contribute to unseen
            interpolated_unseen = lambda_weight * 1e-10 + (1 - lambda_weight) * 1e-10

        interpolated_unseen = max(interpolated_unseen, 1e-10)

        if self.logprob:
            self._unobs = math.log(interpolated_unseen)
        else:
            self._unobs = interpolated_unseen


def _mle(dist: FrequencyDistribution) -> tuple[dict[Element, float], int]:
    """Return the maximum-likelihood distribution and total count of ``dist``."""
    total = sum(dist.values())
    if total == 0:
        return {}, 0
    return {element: count / total for element, count in dist.items()}, total


def _estimate_jm_lambda(
    high_order_dist: FrequencyDistribution,
    low_order_dist: FrequencyDistribution,
    held_out_dist: FrequencyDistribution,
    low_order_n: int | None,
    init_lambda: float,
    iterations: int,
) -> float:
    """Estimate the Jelinek-Mercer weight by EM (deleted interpolation).

    Iteratively reweights ``lambda`` so that the interpolated model
    ``lambda * P_high + (1 - lambda) * P_low`` maximizes the likelihood of the
    held-out data. Returns ``init_lambda`` unchanged when there is no usable
    held-out evidence.
    """
    high_probs, _ = _mle(high_order_dist)
    low_probs, _ = _mle(low_order_dist)

    # Precompute (p_high, p_low, count) for each held-out n-gram once.
    observations: list[tuple[float, float, float]] = []
    for ngram, count in held_out_dist.items():
        if count <= 0:
            continue
        p_high = high_probs.get(ngram, 0.0)
        if low_order_n is not None and isinstance(ngram, tuple) and len(ngram) >= low_order_n:
            context: Element = ngram[-low_order_n:]
        else:
            context = ngram
        p_low = low_probs.get(context, 0.0)
        if p_high <= 0.0 and p_low <= 0.0:
            continue
        observations.append((p_high, p_low, float(count)))

    if not observations:
        return init_lambda

    lam = init_lambda
    for _ in range(iterations):
        expected_high = 0.0
        total = 0.0
        for p_high, p_low, weight in observations:
            denom = lam * p_high + (1 - lam) * p_low
            if denom <= 0.0:
                continue
            responsibility = lam * p_high / denom
            expected_high += weight * responsibility
            total += weight
        if total <= 0.0:
            break
        lam = expected_high / total
    return lam


class JelinekMercer(Interpolated):
    """Jelinek-Mercer smoothing with an EM-estimated interpolation weight.

    Jelinek-Mercer smoothing linearly interpolates a higher-order model with a
    lower-order one, exactly as :class:`Interpolated` does, but chooses the
    interpolation weight ``lambda`` by *deleted interpolation*: given a held-out
    distribution, the weight is fitted with the Expectation-Maximization
    algorithm to maximize the likelihood of that held-out data. Without held-out
    data it behaves as fixed-weight interpolation using ``lambda_weight``.

    Args:
        high_order_dist: Higher-order frequency distribution (e.g. trigrams).
        low_order_dist: Lower-order frequency distribution (e.g. bigrams).
        held_out_dist: Optional held-out frequency distribution used to fit
            ``lambda`` by EM. If ``None`` (the default), the supplied
            ``lambda_weight`` is used unchanged.
        lambda_weight: Interpolation weight ``0 <= lambda_weight <= 1`` for the
            high-order model. Used directly when there is no held-out data, and
            as the EM starting point otherwise. Defaults to ``0.5``.
        em_iterations: Number of EM iterations when fitting ``lambda`` from
            held-out data (a positive integer). Defaults to ``10``.
        logprob: Return log-probabilities if ``True`` (the default), otherwise
            plain probabilities.

    Attributes:
        estimated_lambda: The interpolation weight actually used — either the
            EM-fitted value or ``lambda_weight`` when no held-out data was given.

    Examples:
        With no held-out data the estimated weight is exactly ``lambda_weight``,
        and scoring matches fixed-weight interpolation:

        >>> trigrams = {("the", "big", "cat"): 3, ("a", "big", "dog"): 2}
        >>> bigrams = {("big", "cat"): 5, ("big", "dog"): 3}
        >>> jm = JelinekMercer(trigrams, bigrams, lambda_weight=0.6, logprob=False)
        >>> round(jm.estimated_lambda, 2)
        0.6
        >>> round(jm(("the", "big", "cat")), 3)   # 0.6 * (3/5) + 0.4 * (5/8)
        0.61

        Held-out data that the high-order model predicts well pulls the weight up:

        >>> held_out = {("the", "big", "cat"): 8, ("a", "big", "dog"): 6}
        >>> jm = JelinekMercer(trigrams, bigrams, held_out_dist=held_out, logprob=False)
        >>> jm.estimated_lambda > 0.5
        True

    Note:
        EM fits a single global ``lambda`` (not the per-context bucketed weights
        of the original Jelinek-Mercer scheme). The held-out data should be
        disjoint from the training data for the estimate to be meaningful.
    """

    __slots__ = ("em_iterations", "estimated_lambda")

    def __init__(
        self,
        high_order_dist: FrequencyDistribution,
        low_order_dist: FrequencyDistribution,
        held_out_dist: FrequencyDistribution | None = None,
        lambda_weight: float = 0.5,
        em_iterations: int = 10,
        logprob: bool = True,
    ) -> None:
        """Initialize Jelinek-Mercer smoothing, fitting ``lambda`` if held-out data is given."""
        if not 0 <= lambda_weight <= 1:
            raise ValueError("Lambda weight must be between 0 and 1")
        if em_iterations < 1:
            raise ValueError("em_iterations must be a positive integer")

        low_order_n = self._detect_order(low_order_dist)
        if held_out_dist:
            estimated = _estimate_jm_lambda(
                high_order_dist,
                low_order_dist,
                held_out_dist,
                low_order_n,
                lambda_weight,
                em_iterations,
            )
        else:
            estimated = lambda_weight

        super().__init__(high_order_dist, low_order_dist, lambda_weight=estimated, logprob=logprob)
        self.name = "Jelinek-Mercer"
        self.em_iterations = em_iterations
        self.estimated_lambda = estimated
