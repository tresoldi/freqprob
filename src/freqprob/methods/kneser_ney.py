"""Kneser-Ney and Modified Kneser-Ney smoothing."""

from collections import Counter
from dataclasses import dataclass

from freqprob.base import (
    Element,
    Estimates,
    FrequencyDistribution,
    Probability,
    ScoringMethodConfig,
)
from freqprob.methods._bigram import BigramBackoff, continuation_probs, extract_bigram_stats


@dataclass
class KneserNeyConfig(ScoringMethodConfig):
    """Configuration for Kneser-Ney smoothing.

    Attributes:
    ----------
    discount : float
        Absolute discounting parameter (0 < d < 1, default: 0.75)
    logprob : bool
        Whether to return log-probabilities (default: True)
    """

    discount: float = 0.75


class KneserNey(BigramBackoff):
    """Kneser-Ney smoothing probability distribution.

    One of the most effective smoothing methods for language modeling. It
    combines absolute discounting with interpolation and, crucially, weights a
    word by the diversity of contexts it appears in (its continuation
    probability) rather than its raw frequency. Each observed bigram scores
    ``max(c - d, 0) / c(context) + backoff(context) * P_cont(word)``, and an
    unseen bigram backs off to ``backoff(context) * P_cont(word)``.

    This is absolute discounting with the unigram model replaced by the
    continuation distribution, and it shares the back-off machinery with
    :class:`~freqprob.AbsoluteDiscounting` accordingly.

    Args:
        freqdist: Frequency distribution mapping bigrams to their observed
            counts, in the form ``{(context, word): count}``. Non-bigram keys
            are ignored.
        discount: Absolute discounting parameter ``0 < d < 1``. Common values
            are ``0.5``-``0.8``. Defaults to ``0.75``.
        logprob: Return log-probabilities if ``True`` (the default), otherwise
            plain probabilities.

    Examples:
        Observed bigrams are discounted and interpolated with the continuation
        model, while unseen bigrams back off to it through the context's
        back-off weight:

        >>> bigram_counts = {
        ...     ("the", "cat"): 5, ("the", "dog"): 3, ("a", "cat"): 2,
        ...     ("a", "dog"): 1, ("big", "cat"): 1, ("small", "dog"): 1,
        ... }
        >>> kn = KneserNey(bigram_counts, discount=0.75, logprob=False)
        >>> round(kn(("the", "cat")), 3)
        0.625

        An unseen bigram whose word is known scores through the context's
        back-off weight rather than a flat average -- here ``0.75 * P_cont(dog)``,
        since ``big`` was only ever followed by one word:

        >>> round(kn(("big", "dog")), 3)
        0.375

        An unseen context falls back to the continuation probability alone:

        >>> round(kn(("new_context", "cat")), 3)
        0.5

    Note:
        This implementation assumes bigram input. The discount ``d`` is
        typically between ``0.5`` and ``0.8``; ``0.75`` is a robust default.
        Reliable continuation probabilities require sufficient bigram data.
        A word that never appears in the data has no continuation probability,
        so it scores at the floor rather than borrowing an average.
    """

    __slots__ = ()

    def __init__(
        self,
        freqdist: FrequencyDistribution,
        discount: float = 0.75,
        logprob: bool = True,
    ) -> None:
        """Initialize Kneser-Ney smoothing."""
        if not 0 < discount < 1:
            raise ValueError("Discount parameter must be between 0 and 1")

        config = KneserNeyConfig(discount=discount, logprob=logprob)
        super().__init__(config)
        self.name = "Kneser-Ney"
        self.fit(freqdist)

    def _estimate_bigrams(self, freqdist: FrequencyDistribution) -> Estimates:
        """Discount each bigram and interpolate with the continuation distribution.

        Args:
            freqdist: Bigram frequency distribution with ``(context, word)``
                tuples as keys. Non-bigram keys are ignored.

        Returns:
            ``P(word | context)`` for each observed bigram, and the fallback for
            a query that is not a bigram.
        """
        discount = self.config.discount  # type: ignore[attr-defined]

        context_total, context_types, context_word, _unigram, _total = extract_bigram_stats(
            freqdist
        )
        continuation = continuation_probs(context_types)

        for context, ctotal in context_total.items():
            n1_plus = len(context_types[context])
            self._backoff[context] = discount * n1_plus / ctotal

        table: dict[Element, Probability] = {}
        for (context, word), count in context_word.items():
            main = max(count - discount, 0.0) / context_total[context]
            table[(context, word)] = main + self._backoff[context] * continuation.get(word, 0.0)

        return table, self._record_lower(continuation, len(continuation))


class ModifiedKneserNey(BigramBackoff):
    """Modified Kneser-Ney smoothing probability distribution.

    An enhanced Kneser-Ney variant that applies a different discount depending
    on a bigram's count: ``d1`` for singletons, ``d2`` for doubletons, and
    ``d3`` for counts of three or more. Those discounts are estimated
    automatically from the data's count-of-counts statistics, which usually
    makes it more accurate than the fixed-discount version.

    Args:
        freqdist: Frequency distribution mapping bigrams to their observed
            counts, in the form ``{(context, word): count}``. Non-bigram keys
            are ignored.
        logprob: Return log-probabilities if ``True`` (the default), otherwise
            plain probabilities.

    Examples:
        The discount adapts to each bigram's count, and the score interpolates
        the discounted estimate with the continuation model:

        >>> bigram_counts = {
        ...     ("the", "cat"): 5, ("the", "dog"): 3, ("a", "cat"): 2,
        ...     ("a", "dog"): 1, ("big", "cat"): 1, ("small", "dog"): 1,
        ... }
        >>> mkn = ModifiedKneserNey(bigram_counts, logprob=False)
        >>> round(mkn(("the", "cat")), 3)
        0.625
        >>> round(mkn(("a", "cat")), 3)
        0.558

    Note:
        Modified Kneser-Ney is a standard, state-of-the-art classical smoothing
        method for n-gram language models. Because it estimates its discounts
        from the data, it is more robust than fixed-discount methods but needs
        enough varied counts to estimate them reliably.
    """

    __slots__ = ()

    def __init__(self, freqdist: FrequencyDistribution, logprob: bool = True) -> None:
        """Initialize Modified Kneser-Ney smoothing."""
        config = ScoringMethodConfig(logprob=logprob)
        super().__init__(config)
        self.name = "Modified Kneser-Ney"
        self.fit(freqdist)

    def _estimate_bigrams(self, freqdist: FrequencyDistribution) -> Estimates:
        """Discount by count band, then interpolate with the continuation distribution.

        Args:
            freqdist: Bigram frequency distribution with ``(context, word)``
                tuples as keys. Non-bigram keys are ignored.

        Returns:
            ``P(word | context)`` for each observed bigram, and the fallback for
            a query that is not a bigram.
        """
        context_total, context_types, context_word, _unigram, _total = extract_bigram_stats(
            freqdist
        )
        continuation = continuation_probs(context_types)

        # Count frequency of frequencies (n_r = number of bigrams with count r).
        # Counted over the extracted bigrams, not over `freqdist` itself, so a
        # non-bigram key cannot skew the estimated discounts.
        count_frequencies = Counter(context_word.values())
        n1, n2, n3, n4 = (
            count_frequencies.get(1, 0),
            count_frequencies.get(2, 0),
            count_frequencies.get(3, 0),
            count_frequencies.get(4, 0),
        )

        # Estimate discount parameters using Good-Turing inspired formulas
        # Handle edge cases where denominators might be zero
        if n1 + 2 * n2 > 0:
            d1 = 1 - 2 * (n2 / n1) * (n1 / (n1 + 2 * n2)) if n1 > 0 else 0.5
        else:
            d1 = 0.5

        if n2 + 3 * n3 > 0:
            d2 = 2 - 3 * (n3 / n2) * (n2 / (n2 + 3 * n3)) if n2 > 0 else 0.5
        else:
            d2 = 0.5

        if n3 + 4 * n4 > 0:
            d3 = 3 - 4 * (n4 / n3) * (n3 / (n3 + 4 * n4)) if n3 > 0 else 0.5
        else:
            d3 = 0.5

        # Ensure discounts are reasonable
        d1 = max(0.01, min(0.99, d1))
        d2 = max(0.01, min(1.99, d2))
        d3 = max(0.01, min(2.99, d3))

        def discount_for(count: int) -> float:
            """Pick the discount for a count, by band."""
            if count == 1:
                return d1
            return d2 if count == 2 else d3

        # Context-dependent back-off weights: each band contributes its own
        # discount, times how many words in the context fall in that band.
        for context, ctotal in context_total.items():
            counts = [context_word[(context, word)] for word in context_types[context]]
            backoff_mass = sum(discount_for(count) for count in counts)
            self._backoff[context] = backoff_mass / ctotal

        table: dict[Element, Probability] = {}
        for (context, word), count in context_word.items():
            main = max(count - discount_for(count), 0.0) / context_total[context]
            table[(context, word)] = main + self._backoff[context] * continuation.get(word, 0.0)

        return table, self._record_lower(continuation, len(continuation))
