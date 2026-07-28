"""Kneser-Ney and Modified Kneser-Ney smoothing."""

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from freqprob.base import (
    FrequencyDistribution,
    ScoringMethod,
    ScoringMethodConfig,
)
from freqprob.cache import cached_computation


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


class KneserNey(ScoringMethod):
    """Kneser-Ney smoothing probability distribution.

    One of the most effective smoothing methods for language modeling. It
    combines absolute discounting with interpolation and, crucially, weights a
    word by the diversity of contexts it appears in (its continuation
    probability) rather than its raw frequency. Each observed bigram scores
    ``max(c - d, 0) / c(context) + backoff(context) * P_cont(word)``, and
    unseen bigrams fall back to the average continuation probability.

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
        model, while unseen bigrams back off to the average continuation
        probability:

        >>> bigram_counts = {
        ...     ("the", "cat"): 5, ("the", "dog"): 3, ("a", "cat"): 2,
        ...     ("a", "dog"): 1, ("big", "cat"): 1, ("small", "dog"): 1,
        ... }
        >>> kn = KneserNey(bigram_counts, discount=0.75, logprob=False)
        >>> round(kn(("the", "cat")), 3)
        0.625
        >>> round(kn(("the", "mouse")), 3)
        0.5
        >>> round(kn(("new_context", "cat")), 3)
        0.5

    Note:
        This implementation assumes bigram input. The discount ``d`` is
        typically between ``0.5`` and ``0.8``; ``0.75`` is a robust default.
        Reliable continuation probabilities require sufficient bigram data.
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

    @cached_computation()
    def _compute_probabilities(self, freqdist: FrequencyDistribution) -> None:
        """Compute Kneser-Ney smoothed probabilities.

        Parameters
        ----------
        freqdist : FrequencyDistribution
            Bigram frequency distribution with (context, word) tuples as keys
        """
        discount = self.config.discount  # type: ignore

        # Separate contexts and words, compute various counts
        contexts: defaultdict[str, int] = defaultdict(int)
        word_continuation_counts: defaultdict[str, int] = defaultdict(int)
        context_types: defaultdict[str, set[str]] = defaultdict(set)
        all_bigram_types = 0

        # Process the frequency distribution
        for element, count in freqdist.items():
            if not isinstance(element, tuple) or len(element) != 2:
                continue  # Skip non-bigram entries
            context, word = element

            contexts[context] += count
            word_continuation_counts[word] += 1  # Count distinct contexts for this word
            context_types[context].add(word)
            all_bigram_types += 1

        # Compute continuation probabilities
        total_continuation_mass = sum(word_continuation_counts.values())
        continuation_probs = {
            word: count / total_continuation_mass
            for word, count in word_continuation_counts.items()
        }

        # Compute backoff weights for each context
        backoff_weights = {}
        for context in contexts:
            num_types = len(context_types[context])
            backoff_weights[context] = discount * num_types / contexts[context]

        # Compute probabilities for observed bigrams
        for element, count in freqdist.items():
            if not isinstance(element, tuple) or len(element) != 2:
                continue
            context, word = element

            # Discounted probability
            discounted_count = max(count - discount, 0)
            context_count = contexts[context]

            # Interpolated probability
            main_prob = discounted_count / context_count
            backoff_prob = backoff_weights[context] * continuation_probs.get(word, 0)
            total_prob = main_prob + backoff_prob

            if self.logprob:
                if total_prob > 0:
                    self._prob[(context, word)] = math.log(total_prob)
                else:
                    self._prob[(context, word)] = math.log(1e-10)  # Avoid log(0)
            else:
                self._prob[(context, word)] = total_prob

        # Set unobserved probability (average continuation probability)
        avg_continuation = 1.0 / len(continuation_probs) if continuation_probs else 1e-10
        if self.logprob:
            self._unobs = math.log(avg_continuation)
        else:
            self._unobs = avg_continuation


class ModifiedKneserNey(ScoringMethod):
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

    @cached_computation()
    def _compute_probabilities(self, freqdist: FrequencyDistribution) -> None:
        """Compute Modified Kneser-Ney smoothed probabilities.

        Parameters
        ----------
        freqdist : FrequencyDistribution
            Bigram frequency distribution with (context, word) tuples as keys
        """
        # Count frequency of frequencies (n_r = number of bigrams with count r)
        count_frequencies = Counter(freqdist.values())
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

        # Separate contexts and words, compute various counts
        contexts: defaultdict[str, int] = defaultdict(int)
        word_continuation_counts: defaultdict[str, int] = defaultdict(int)
        context_types: defaultdict[str, set[str]] = defaultdict(set)

        for element, count in freqdist.items():
            if not isinstance(element, tuple) or len(element) != 2:
                continue
            context, word = element

            contexts[context] += count
            word_continuation_counts[word] += 1
            context_types[context].add(word)

        # Compute continuation probabilities
        total_continuation_mass = sum(word_continuation_counts.values())
        continuation_probs = {
            word: count / total_continuation_mass
            for word, count in word_continuation_counts.items()
        }

        # Compute context-dependent backoff weights
        backoff_weights = {}
        for context in contexts:
            # Count types by frequency
            n1_types = sum(
                1 for word in context_types[context] if freqdist.get((context, word), 0) == 1
            )
            n2_types = sum(
                1 for word in context_types[context] if freqdist.get((context, word), 0) == 2
            )
            n3_plus_types = sum(
                1 for word in context_types[context] if freqdist.get((context, word), 0) >= 3
            )

            # Compute backoff weight
            backoff_mass = (d1 * n1_types + d2 * n2_types + d3 * n3_plus_types) / contexts[context]
            backoff_weights[context] = backoff_mass

        # Compute probabilities for observed bigrams
        for element, count in freqdist.items():
            if not isinstance(element, tuple) or len(element) != 2:
                continue
            context, word = element

            # Choose discount based on count
            if count == 1:
                discount = d1
            elif count == 2:
                discount = d2
            else:
                discount = d3

            # Discounted probability
            discounted_count = max(count - discount, 0)
            context_count = contexts[context]

            # Interpolated probability
            main_prob = discounted_count / context_count
            backoff_prob = backoff_weights[context] * continuation_probs.get(word, 0)
            total_prob = main_prob + backoff_prob

            if self.logprob:
                if total_prob > 0:
                    self._prob[(context, word)] = math.log(total_prob)
                else:
                    self._prob[(context, word)] = math.log(1e-10)
            else:
                self._prob[(context, word)] = total_prob

        # Set unobserved probability
        avg_continuation = 1.0 / len(continuation_probs) if continuation_probs else 1e-10
        if self.logprob:
            self._unobs = math.log(avg_continuation)
        else:
            self._unobs = avg_continuation
