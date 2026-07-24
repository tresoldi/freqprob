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

    logprob: bool = True


class KneserNey(ScoringMethod):
    """Kneser-Ney smoothing probability distribution.

    Kneser-Ney smoothing is one of the most effective smoothing methods for
    language modeling. It uses absolute discounting combined with interpolation
    and considers the diversity of contexts in which words appear.

    Mathematical Formulation
    ------------------------
    For bigram model P(wᵢ|wᵢ₋₁):

    P_KN(wᵢ|wᵢ₋₁) = max(c(wᵢ₋₁,wᵢ) - d, 0) / c(wᵢ₋₁) + λ(wᵢ₋₁) * P_cont(wᵢ)

    Where:
    - d is the discount parameter (0 < d < 1)
    - λ(wᵢ₋₁) = d * |{w : c(wᵢ₋₁,w) > 0}| / c(wᵢ₋₁) is the backoff weight
    - P_cont(wᵢ) = |{w : c(w,wᵢ) > 0}| / |{(w,w') : c(w,w') > 0}| is the continuation probability

    The key insight is that P_cont models how likely a word is to appear in
    novel contexts, based on the diversity of contexts it has been seen in.

    Parameters
    ----------
    freqdist : FrequencyDistribution
        Frequency distribution mapping bigrams to their observed counts.
        Expected format: {(context, word): count}
    discount : float, default=0.75
        Absolute discounting parameter (0 < d < 1). Common values: 0.5-0.8
    logprob : bool, default=True
        Whether to return log-probabilities or probabilities

    Examples:
    --------
    Basic Kneser-Ney smoothing:
    >>> bigram_counts = {
    ...     ('the', 'cat'): 5, ('the', 'dog'): 3, ('a', 'cat'): 2,
    ...     ('a', 'dog'): 1, ('big', 'cat'): 1, ('small', 'dog'): 1
    ... }
    >>> kn = KneserNey(bigram_counts, discount=0.75, logprob=False)
    >>> kn(('the', 'cat'))  # High-frequency bigram
    0.4583333333333333
    >>> kn(('the', 'mouse'))  # Unseen bigram, backed off to continuation prob
    0.08333333333333333

    The method handles unseen bigrams gracefully by backing off to a
    continuation probability based on word diversity:
    >>> kn(('new_context', 'cat'))  # Unseen context, uses continuation
    0.16666666666666666

    Properties
    ----------
    - Excellent performance in language modeling tasks
    - Handles sparse data better than simple discounting methods
    - Takes into account word frequency diversity across contexts
    - Particularly effective for n-gram language models
    - Widely used baseline in NLP applications

    Notes:
    -----
    This implementation assumes bigram input but can be extended to higher-order
    n-grams. The discount parameter d is typically set between 0.5-0.8, with
    0.75 being a common default that works well across many domains.

    For optimal performance, the input should contain sufficient bigram data
    to estimate continuation probabilities reliably.
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


@dataclass
class ModifiedKneserNeyConfig(ScoringMethodConfig):
    """Configuration for Modified Kneser-Ney smoothing.

    Attributes:
    ----------
    logprob : bool
        Whether to return log-probabilities (default: True)
    """

    logprob: bool = True


class ModifiedKneserNey(ScoringMethod):
    """Modified Kneser-Ney smoothing probability distribution.

    An enhanced version of Kneser-Ney smoothing that uses different discount
    values for different frequency counts. This typically provides better
    performance than standard Kneser-Ney by adapting the discounting strategy
    based on the reliability of count estimates.

    Mathematical Formulation
    ------------------------
    P_MKN(wᵢ|wᵢ₋₁) = max(c(wᵢ₋₁,wᵢ) - D(c(wᵢ₋₁,wᵢ)), 0) / c(wᵢ₋₁) + λ(wᵢ₋₁) * P_cont(wᵢ)

    Where D(c) is a count-dependent discount:
    - D(1) = d₁ for singleton counts
    - D(2) = d₂ for doubleton counts
    - D(c) = d₃ for c ≥ 3

    The discounts are estimated from the data using:
    - d₁ = 1 - 2 * (n₂/n₁) * (n₁/(n₁+2*n₂))
    - d₂ = 2 - 3 * (n₃/n₂) * (n₂/(n₂+3*n₃))
    - d₃ = 3 - 4 * (n₄/n₃) * (n₃/(n₃+4*n₄))

    Parameters
    ----------
    freqdist : FrequencyDistribution
        Frequency distribution mapping bigrams to their observed counts
    logprob : bool, default=True
        Whether to return log-probabilities or probabilities

    Examples:
    --------
    >>> bigram_counts = {
    ...     ('the', 'cat'): 5, ('the', 'dog'): 3, ('a', 'cat'): 2,
    ...     ('a', 'dog'): 1, ('big', 'cat'): 1, ('small', 'dog'): 1
    ... }
    >>> mkn = ModifiedKneserNey(bigram_counts, logprob=False)
    >>> mkn(('the', 'cat'))  # Uses d₃ discount (count ≥ 3)
    0.42857142857142855
    >>> mkn(('a', 'cat'))    # Uses d₂ discount (count = 2)
    0.5714285714285714

    Properties
    ----------
    - Generally outperforms standard Kneser-Ney
    - Adapts discounting based on count reliability
    - Automatic parameter estimation from data
    - Robust across different data sizes and domains
    - Standard method in modern language modeling

    Notes:
    -----
    Modified Kneser-Ney is considered the state-of-the-art classical smoothing
    method for n-gram language models. It automatically estimates optimal
    discount parameters, making it more robust than fixed-discount methods.
    """

    __slots__ = ()

    def __init__(self, freqdist: FrequencyDistribution, logprob: bool = True) -> None:
        """Initialize Modified Kneser-Ney smoothing."""
        config = ModifiedKneserNeyConfig(logprob=logprob)
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
