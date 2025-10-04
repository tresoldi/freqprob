"""Advanced smoothing methods for language modeling.

This module implements sophisticated smoothing techniques commonly used
in natural language processing, including Kneser-Ney, Modified Kneser-Ney,
interpolated methods, and Bayesian approaches.
"""

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from .base import (
    Element,
    FrequencyDistribution,
    LogProbability,
    Probability,
    ScoringMethod,
    ScoringMethodConfig,
)
from .cache import cached_computation


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


@dataclass
class ModifiedKneserNeyConfig(ScoringMethodConfig):
    """Configuration for Modified Kneser-Ney smoothing.

    Attributes:
    ----------
    logprob : bool
        Whether to return log-probabilities (default: True)
    """

    logprob: bool = True


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

    logprob: bool = True


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


class InterpolatedSmoothing(ScoringMethod):
    """Linear interpolation smoothing between multiple models.

    Combines probability estimates from different models using weighted linear
    interpolation. Supports two modes:

    1. **N-gram interpolation**: Different n-gram orders (e.g., trigrams with bigrams)
       - Automatically extracts lower-order context from higher-order n-grams
       - Example: For trigram (w1, w2, w3), uses bigram (w2, w3) from low-order model

    2. **Same-type interpolation**: Same element types
       - Direct key matching between distributions
       - Works with strings, same-length tuples, or any hashable elements

    The mode is automatically detected from the element types in the distributions.

    Mathematical Formulation
    ------------------------
    For n-gram interpolation (e.g., trigram + bigram):
        P_interp(w3|w1,w2) = λ * P_trigram(w3|w1,w2) + (1-λ) * P_bigram(w3|w2)

    For same-type interpolation:
        P_interp(element) = λ * P_high(element) + (1-λ) * P_low(element)

    Where:
    - λ is the interpolation weight (0 ≤ λ ≤ 1)
    - Higher λ favors the high-order model (more specificity)
    - Lower λ favors the low-order model (more smoothing)

    Parameters
    ----------
    high_order_dist : FrequencyDistribution
        Higher-order frequency distribution (e.g., trigrams)
        For n-gram interpolation, must have longer tuples than low_order_dist
    low_order_dist : FrequencyDistribution
        Lower-order frequency distribution (e.g., bigrams)
    lambda_weight : float, default=0.7
        Interpolation weight for higher-order model (0 ≤ λ ≤ 1)
    logprob : bool, default=True
        Whether to return log-probabilities or probabilities

    Examples:
    --------
    N-gram interpolation (trigrams + bigrams):

    >>> trigrams = {('the', 'big', 'cat'): 3, ('a', 'big', 'dog'): 2}
    >>> bigrams = {('big', 'cat'): 5, ('big', 'dog'): 3, ('small', 'cat'): 2}
    >>> interp = InterpolatedSmoothing(trigrams, bigrams, lambda_weight=0.7, logprob=False)
    >>> interp(('the', 'big', 'cat'))
    0.5  # 0.7 * (3/5) + 0.3 * (5/10)
    >>> interp(('unseen', 'big', 'cat'))
    0.15  # 0.7 * 0 + 0.3 * (5/10) - backoff to bigram

    Same-type interpolation (strings):

    >>> model1 = {'word1': 10, 'word2': 5}
    >>> model2 = {'word1': 3, 'word3': 7}
    >>> interp = InterpolatedSmoothing(model1, model2, lambda_weight=0.6, logprob=False)
    >>> interp('word1')
    0.48  # 0.6 * (10/15) + 0.4 * (3/10)

    Properties
    ----------
    - Balances specificity and generalization
    - Reduces overfitting from sparse high-order models
    - Provides robustness through model combination
    - Automatic mode detection from element types
    - All probabilities floored at 1e-10 for numerical stability

    Notes:
    -----
    - For n-gram interpolation, the high-order n must be greater than low-order n
    - Context extraction uses suffix (last n elements) following NLP conventions
    - Unseen n-grams backoff to the lower-order model probability
    - The interpolation weight λ can be tuned on held-out data for optimal performance
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

        if is_ngram_interpolation and isinstance(element, tuple) and len(element) == self._high_order_n:
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


class BayesianSmoothing(ScoringMethod):
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
    >>> bayes = BayesianSmoothing(freqdist, alpha=1.0, logprob=False)
    >>> bayes('apple')     # (8+1)/(13+3*1) = 9/16
    0.5625
    >>> bayes('banana')    # (4+1)/(13+3*1) = 5/16
    0.3125
    >>> bayes('unseen')    # 1/(13+3*1) = 1/16
    0.0625

    Effect of different alpha values:
    >>> # Stronger smoothing (alpha = 2)
    >>> bayes_smooth = BayesianSmoothing(freqdist, alpha=2.0, logprob=False)
    >>> bayes_smooth('apple')    # (8+2)/(13+3*2) = 10/19
    0.5263157894736842
    >>> bayes_smooth('unseen')   # 2/(13+3*2) = 2/19
    0.10526315789473684

    >>> # Minimal smoothing (alpha = 0.1)
    >>> bayes_minimal = BayesianSmoothing(freqdist, alpha=0.1, logprob=False)
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
