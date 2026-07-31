"""Absolute-discounting and Pitman-Yor bigram smoothing.

Both methods discount observed bigram counts and interpolate with a lower-order
unigram model. Absolute discounting subtracts a fixed amount from every count;
Pitman-Yor generalizes it with a second (concentration) parameter and has
absolute discounting as its ``strength = 0`` special case.
"""

from dataclasses import dataclass

from freqprob.base import FrequencyDistribution, ScoringMethodConfig
from freqprob.methods._bigram import BigramBackoff, extract_bigram_stats


@dataclass
class DiscountingConfig(ScoringMethodConfig):
    """Configuration for absolute-discounting / Pitman-Yor smoothing.

    Attributes:
        discount: Amount subtracted from each observed count (the ``d``
            parameter). ``0 <= discount < 1``.
        strength: Pitman-Yor concentration parameter ``theta`` (``theta >
            -discount``). ``0`` recovers absolute discounting.
    """

    discount: float = 0.75
    strength: float = 0.0


class AbsoluteDiscounting(BigramBackoff):
    """Absolute-discounting bigram smoothing.

    Subtracts a fixed discount ``d`` from every observed bigram count and
    redistributes the freed mass to a lower-order unigram model. For a bigram
    ``(context, word)`` with count ``c`` and context total ``C``,

    ``P(word | context) = max(c - d, 0) / C + (d * N1+(context) / C) * P_uni(word)``

    where ``N1+(context)`` is the number of distinct words seen after the
    context and ``P_uni`` is the unigram maximum-likelihood distribution. This is
    the classic Ney-Essen-Kneser discounting scheme and the foundation Kneser-Ney
    builds on — the difference is that Kneser-Ney replaces ``P_uni`` with a
    continuation distribution. Reach for absolute discounting when you want simple,
    well-understood discounting without Kneser-Ney's continuation weighting.

    Args:
        freqdist: Frequency distribution mapping ``(context, word)`` bigrams to
            counts. Non-bigram keys are ignored.
        discount: Absolute discounting parameter ``0 <= d < 1``. Common values
            are ``0.5``-``0.8``. Defaults to ``0.75``.
        logprob: Return log-probabilities if ``True`` (the default), otherwise
            plain probabilities.

    Examples:
        Observed bigrams are discounted and interpolated with the unigram model,
        while unseen bigrams back off to it:

        >>> bigram_counts = {
        ...     ("the", "cat"): 5, ("the", "dog"): 3, ("a", "cat"): 2,
        ...     ("a", "dog"): 1, ("big", "cat"): 1, ("small", "dog"): 1,
        ... }
        >>> ad = AbsoluteDiscounting(bigram_counts, discount=0.75, logprob=False)
        >>> round(ad(("the", "cat")), 3)
        0.647
        >>> round(ad(("a", "cat")), 3)
        0.724
        >>> round(ad(("big", "dog")), 3)      # unseen word in a known context
        0.288
        >>> round(ad(("new", "cat")), 3)      # unseen context backs off to unigram
        0.615

    Note:
        This implementation assumes bigram input. Unlike Kneser-Ney it uses raw
        unigram frequencies for the lower-order model, so a frequent word keeps
        its weight even if it appears in few distinct contexts.
    """

    __slots__ = ()

    def __init__(
        self,
        freqdist: FrequencyDistribution,
        discount: float = 0.75,
        logprob: bool = True,
    ) -> None:
        """Initialize absolute-discounting smoothing."""
        if not 0 <= discount < 1:
            raise ValueError("Discount parameter must be in [0, 1)")

        config = DiscountingConfig(discount=discount, strength=0.0, logprob=logprob)
        super().__init__(config)
        self.name = "Absolute Discounting"
        self.fit(freqdist)

    def _compute_probabilities(self, freqdist: FrequencyDistribution) -> None:
        """Compute absolute-discounting probabilities."""
        discount = self.config.discount  # type: ignore[attr-defined]

        context_total, context_types, context_word, unigram, total = extract_bigram_stats(freqdist)

        lower = {word: count / total for word, count in unigram.items()} if total else {}

        for context, ctotal in context_total.items():
            n1_plus = len(context_types[context])
            self._backoff[context] = discount * n1_plus / ctotal

        for (context, word), count in context_word.items():
            ctotal = context_total[context]
            main = max(count - discount, 0.0) / ctotal
            prob = main + self._backoff[context] * lower.get(word, 0.0)
            self._store_observed((context, word), prob)

        self._unknown_context_weight = 1.0
        self._finalize(lower, len(unigram))


class PitmanYor(BigramBackoff):
    """Pitman-Yor process bigram smoothing.

    A two-parameter generalization of absolute discounting based on the
    Pitman-Yor process. With discount ``d`` and concentration (strength)
    ``theta``, an observed bigram ``(context, word)`` with count ``c``, context
    total ``C``, and ``t`` distinct words seen after the context scores

    ``P(word | context) = (c - d + (theta + d * t) * P_uni(word)) / (theta + C)``

    and a word unseen in the context gets ``(theta + d * t) / (theta + C) *
    P_uni(word)``. The concentration ``theta`` controls how much mass is reserved
    for the base distribution independently of the discount: ``theta = 0``
    recovers absolute discounting, while larger ``theta`` pulls estimates toward
    the unigram model. Pitman-Yor priors match the power-law behavior of natural
    language and are the basis of the hierarchical Pitman-Yor language model,
    which subsumes Kneser-Ney as a special case.

    Args:
        freqdist: Frequency distribution mapping ``(context, word)`` bigrams to
            counts. Non-bigram keys are ignored.
        discount: Discount parameter ``0 <= d < 1``. Defaults to ``0.75``.
        strength: Concentration parameter ``theta > -discount``. Defaults to
            ``1.0``. ``0`` recovers absolute discounting.
        logprob: Return log-probabilities if ``True`` (the default), otherwise
            plain probabilities.

    Examples:
        Larger counts still dominate, but the concentration parameter reserves
        extra mass for the unigram base distribution:

        >>> bigram_counts = {
        ...     ("the", "cat"): 5, ("the", "dog"): 3, ("a", "cat"): 2,
        ...     ("a", "dog"): 1, ("big", "cat"): 1, ("small", "dog"): 1,
        ... }
        >>> py = PitmanYor(bigram_counts, discount=0.75, strength=1.0, logprob=False)
        >>> round(py(("the", "cat")), 3)
        0.643
        >>> round(py(("a", "cat")), 3)
        0.697
        >>> round(py(("big", "dog")), 3)      # unseen word in a known context
        0.337
        >>> round(py(("new", "cat")), 3)      # unseen context backs off to unigram
        0.615

    Note:
        This implementation is a single-level (bigram) Pitman-Yor model using the
        unigram maximum-likelihood distribution as its base measure and the
        number of observed types as the table count. It is a practical
        approximation of the full hierarchical Pitman-Yor process.
    """

    __slots__ = ()

    def __init__(
        self,
        freqdist: FrequencyDistribution,
        discount: float = 0.75,
        strength: float = 1.0,
        logprob: bool = True,
    ) -> None:
        """Initialize Pitman-Yor smoothing."""
        if not 0 <= discount < 1:
            raise ValueError("Discount parameter must be in [0, 1)")
        if strength <= -discount:
            raise ValueError("Strength parameter must be greater than -discount")

        config = DiscountingConfig(discount=discount, strength=strength, logprob=logprob)
        super().__init__(config)
        self.name = "Pitman-Yor"
        self.fit(freqdist)

    def _compute_probabilities(self, freqdist: FrequencyDistribution) -> None:
        """Compute Pitman-Yor probabilities."""
        discount = self.config.discount  # type: ignore[attr-defined]
        strength = self.config.strength  # type: ignore[attr-defined]

        context_total, context_types, context_word, unigram, total = extract_bigram_stats(freqdist)

        lower = {word: count / total for word, count in unigram.items()} if total else {}

        for context, ctotal in context_total.items():
            num_types = len(context_types[context])
            self._backoff[context] = (strength + discount * num_types) / (strength + ctotal)

        for (context, word), count in context_word.items():
            ctotal = context_total[context]
            num_types = len(context_types[context])
            numerator = (count - discount) + (strength + discount * num_types) * lower.get(
                word, 0.0
            )
            prob = numerator / (strength + ctotal)
            self._store_observed((context, word), prob)

        self._unknown_context_weight = 1.0
        self._finalize(lower, len(unigram))
