"""Katz and Stupid Back-off bigram smoothing.

Both methods score an observed bigram directly and fall back to a lower-order
unigram model for unseen bigrams. Katz back-off discounts observed counts with
Good-Turing and computes normalized back-off weights so the result is a proper
probability distribution; Stupid Back-off skips normalization for speed and
returns unnormalized scores.
"""

import math
from collections import Counter
from dataclasses import dataclass

from freqprob.base import FrequencyDistribution, ScoringMethodConfig
from freqprob.methods._bigram import BigramBackoff, extract_bigram_stats


@dataclass
class KatzBackoffConfig(ScoringMethodConfig):
    """Configuration for Katz back-off smoothing.

    Attributes:
        k: Count threshold below which Good-Turing discounting is applied.
            Counts greater than ``k`` are treated as reliable and left
            undiscounted (default: ``5``).
    """

    k: int = 5


@dataclass
class StupidBackoffConfig(ScoringMethodConfig):
    """Configuration for Stupid Back-off scoring.

    Attributes:
        alpha: Fixed back-off weight applied at each fall-back to the lower-order
            model (default: ``0.4``).
    """

    alpha: float = 0.4


class KatzBackoff(BigramBackoff):
    """Katz back-off bigram smoothing.

    Katz's method discounts observed bigram counts with Good-Turing and
    redistributes the freed mass to a lower-order unigram model through
    normalized back-off weights. Following Katz (1987), only counts up to a
    threshold ``k`` are discounted (counts above ``k`` are assumed reliable). For
    a bigram ``(context, word)`` with count ``c`` and context total ``C``,

    ``P(word | context) = d_c * c / C``      if ``c > 0``,

    ``P(word | context) = alpha(context) * P_uni(word)``    otherwise,

    where ``d_c`` is the Good-Turing discount ratio for count ``c`` and
    ``alpha(context)`` is chosen so each context's distribution sums to one. It
    is one of the classic n-gram smoothing methods and pairs naturally with
    :class:`~freqprob.SimpleGoodTuring`, which uses the same count-of-counts
    machinery.

    Args:
        freqdist: Frequency distribution mapping ``(context, word)`` bigrams to
            counts. Non-bigram keys are ignored.
        k: Good-Turing discounting is applied to counts ``1 <= c <= k``; larger
            counts are left undiscounted. Defaults to ``5``.
        logprob: Return log-probabilities if ``True`` (the default), otherwise
            plain probabilities.

    Examples:
        Reliable (high) counts are barely discounted, rare counts are discounted
        more, and unseen bigrams back off to the unigram model:

        >>> bigram_counts = {
        ...     ("the", "cat"): 4, ("the", "dog"): 3,
        ...     ("a", "cat"): 2, ("a", "dog"): 2,
        ...     ("my", "cat"): 1, ("my", "dog"): 1, ("my", "bird"): 1,
        ...     ("his", "fish"): 1, ("her", "fish"): 1, ("its", "bird"): 1,
        ... }
        >>> katz = KatzBackoff(bigram_counts, logprob=False)
        >>> round(katz(("the", "cat")), 3)    # reliable count, undiscounted
        0.571
        >>> round(katz(("my", "cat")), 3)     # singleton, Good-Turing discounted
        0.222
        >>> round(katz(("my", "fish")), 3)    # unseen in context, backs off
        0.333

    Note:
        This implementation assumes bigram input. Good-Turing discounting needs a
        spread of low counts (singletons, doubletons, ...) to estimate discounts;
        with too few distinct counts some discounts default to ``1`` (no
        discounting), leaving little or no mass for unseen bigrams.
    """

    __slots__ = ()

    def __init__(
        self,
        freqdist: FrequencyDistribution,
        k: int = 5,
        logprob: bool = True,
    ) -> None:
        """Initialize Katz back-off smoothing."""
        if k < 1:
            raise ValueError("Threshold k must be a positive integer")

        config = KatzBackoffConfig(k=k, logprob=logprob)
        super().__init__(config)
        self.name = "Katz Back-off"
        self.fit(freqdist)

    def _compute_probabilities(self, freqdist: FrequencyDistribution) -> None:
        """Compute Katz back-off probabilities."""
        k = self.config.k  # type: ignore[attr-defined]

        context_total, context_types, context_word, unigram, total = extract_bigram_stats(freqdist)

        lower = {word: count / total for word, count in unigram.items()} if total else {}

        # Good-Turing discount ratios d_r for counts 1..k, following Katz (1987).
        count_of_counts = Counter(context_word.values())
        n1 = count_of_counts.get(1, 0)
        nk1 = count_of_counts.get(k + 1, 0)
        a = (k + 1) * nk1 / n1 if n1 else 0.0

        discount_ratio: dict[int, float] = {}
        for r in range(1, k + 1):
            nr = count_of_counts.get(r, 0)
            nr1 = count_of_counts.get(r + 1, 0)
            if nr and nr1 and (1 - a) != 0:
                r_star = (r + 1) * nr1 / nr
                d_r = (r_star / r - a) / (1 - a)
                discount_ratio[r] = min(max(d_r, 0.0), 1.0)
            else:
                discount_ratio[r] = 1.0  # cannot estimate: leave undiscounted

        def discount(count: int) -> float:
            return discount_ratio.get(count, 1.0) if count <= k else 1.0

        # Observed-bigram probabilities and per-context back-off weights.
        for context, ctotal in context_total.items():
            used = 0.0
            remaining_lower = 1.0
            for word in context_types[context]:
                count = context_word[(context, word)]
                prob = discount(count) * count / ctotal
                self._store_observed((context, word), prob)
                used += prob
                remaining_lower -= lower.get(word, 0.0)

            beta = max(1.0 - used, 0.0)  # mass reserved for unseen words
            if remaining_lower > 1e-12:
                self._backoff[context] = beta / remaining_lower
            else:
                self._backoff[context] = 0.0

        self._unknown_context_weight = 1.0
        self._finalize(lower, len(unigram))


class StupidBackoff(BigramBackoff):
    """Stupid Back-off bigram scoring.

    A deliberately simple, unnormalized scheme introduced by Brants et al. (2007)
    for web-scale language modeling. An observed bigram scores its relative
    frequency; an unseen bigram scores a fixed weight ``alpha`` times the
    lower-order unigram score:

    ``S(word | context) = c(context, word) / c(context)``   if observed,

    ``S(word | context) = alpha * P_uni(word)``    otherwise.

    Because it skips the normalization Katz back-off performs, the scores do not
    form a probability distribution (they do not sum to one), but they are very
    cheap to compute and work well for ranking when data is plentiful.

    Args:
        freqdist: Frequency distribution mapping ``(context, word)`` bigrams to
            counts. Non-bigram keys are ignored.
        alpha: Back-off weight applied when falling back to the unigram model.
            Defaults to ``0.4`` (the value recommended by Brants et al.).
        logprob: Return log-scores if ``True`` (the default), otherwise plain
            scores.

    Examples:
        Observed bigrams score their relative frequency; unseen bigrams are
        penalized by ``alpha`` and scored from the unigram model:

        >>> bigram_counts = {
        ...     ("the", "cat"): 5, ("the", "dog"): 3, ("a", "cat"): 2,
        ...     ("a", "dog"): 1, ("big", "cat"): 1, ("small", "dog"): 1,
        ... }
        >>> sb = StupidBackoff(bigram_counts, logprob=False)
        >>> round(sb(("the", "cat")), 3)      # 5 / 8
        0.625
        >>> round(sb(("a", "dog")), 3)        # 1 / 3
        0.333
        >>> round(sb(("big", "dog")), 3)      # 0.4 * P_uni("dog")
        0.154

    Note:
        Stupid Back-off returns *scores*, not normalized probabilities, so
        metrics that assume a valid distribution (such as perplexity) are not
        meaningful here. Use it for ranking and comparison, where only the
        relative ordering of scores matters.
    """

    __slots__ = ()

    def __init__(
        self,
        freqdist: FrequencyDistribution,
        alpha: float = 0.4,
        logprob: bool = True,
    ) -> None:
        """Initialize Stupid Back-off scoring."""
        if not 0 < alpha <= 1:
            raise ValueError("Back-off weight alpha must be in (0, 1]")

        config = StupidBackoffConfig(alpha=alpha, logprob=logprob)
        super().__init__(config)
        self.name = "Stupid Back-off"
        self.fit(freqdist)

    def _compute_probabilities(self, freqdist: FrequencyDistribution) -> None:
        """Compute Stupid Back-off scores."""
        alpha = self.config.alpha  # type: ignore[attr-defined]

        context_total, _context_types, context_word, unigram, total = extract_bigram_stats(freqdist)

        lower = {word: count / total for word, count in unigram.items()} if total else {}

        for (context, word), count in context_word.items():
            ctotal = context_total[context]
            self._store_observed((context, word), count / ctotal)

        # Every context (seen or unseen) uses the same alpha weight, so leave
        # self._backoff empty and let the unknown-context weight handle both.
        self._unknown_context_weight = alpha

        vocab_size = len(unigram)
        self._lower = lower
        fallback = alpha / vocab_size if vocab_size else 1e-10
        self._unobs = math.log(fallback) if self.logprob else fallback
