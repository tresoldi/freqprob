"""Internal machinery shared by the bigram back-off / discounting smoothers.

`AbsoluteDiscounting`, `KatzBackoff`, `StupidBackoff`, and `PitmanYor` all model
a conditional distribution ``P(word | context)`` from a frequency distribution
over ``(context, word)`` bigrams. They differ only in how observed bigrams are
discounted and how much mass is backed off to the lower-order (unigram) model;
the bookkeeping — extracting counts and scoring unseen bigrams by backing off to
the unigram distribution — is identical, and lives here.

Like the Kneser-Ney implementation, any key that is not a length-2 tuple is
ignored, so a distribution may mix bigrams with other bookkeeping entries.
"""

from abc import abstractmethod

from freqprob.base import (
    MIN_PROBABILITY,
    Element,
    Estimates,
    FrequencyDistribution,
    LogProbability,
    Probability,
    ScoringMethod,
    ScoringMethodConfig,
)


def extract_bigram_stats(
    freqdist: FrequencyDistribution,
) -> tuple[
    dict[Element, int],
    dict[Element, set[Element]],
    dict[tuple[Element, Element], int],
    dict[Element, int],
    int,
]:
    """Extract the counts the bigram smoothers need from a distribution.

    Args:
        freqdist: Mapping of ``(context, word)`` bigrams to counts. Non-bigram
            keys are ignored.

    Returns:
        A tuple ``(context_total, context_types, context_word, unigram, total)``
        where ``context_total[c]`` is ``c(context)``, ``context_types[c]`` is the
        set of distinct words following the context, ``context_word[(c, w)]`` is
        ``c(context, word)``, ``unigram[w]`` is the total number of times a word
        appears in the second position, and ``total`` is the grand total of all
        bigram counts.
    """
    context_total: dict[Element, int] = {}
    context_types: dict[Element, set[Element]] = {}
    context_word: dict[tuple[Element, Element], int] = {}
    unigram: dict[Element, int] = {}
    total = 0

    for element, count in freqdist.items():
        if not (isinstance(element, tuple) and len(element) == 2):
            continue
        context, word = element
        context_total[context] = context_total.get(context, 0) + count
        context_types.setdefault(context, set()).add(word)
        key = (context, word)
        context_word[key] = context_word.get(key, 0) + count
        unigram[word] = unigram.get(word, 0) + count
        total += count

    return context_total, context_types, context_word, unigram, total


class BigramBackoff(ScoringMethod):
    """Base class for bigram smoothers that back off to a unigram model.

    Subclasses implement :meth:`_estimate_bigrams`, returning the probability of
    each observed bigram together with the mass reserved for a non-bigram query.
    While doing so they record, through :meth:`_record_lower`, the lower-order
    (unigram) model, and set:

    - ``self._backoff``: a per-context back-off weight applied to the unigram
      probability of a word unseen in that context, and
    - ``self._unknown_context_weight``: the back-off weight used when the whole
      context is unseen (``1.0`` for the discounting methods, ``alpha`` for
      Stupid Back-off).

    This base class handles scoring, including the on-the-fly back-off for
    unseen bigrams (mirroring the approach used by ``Interpolated``).
    """

    __slots__ = ("_backoff", "_lower", "_unknown_context_weight")

    def __init__(self, config: ScoringMethodConfig) -> None:
        """Initialize the shared back-off state."""
        super().__init__(config)
        self._lower: dict[Element, float] = {}
        self._backoff: dict[Element, float] = {}
        self._unknown_context_weight: float = 1.0

    def __call__(self, element: Element) -> Probability | LogProbability:
        """Score a bigram, backing off to the unigram model when unseen.

        Observed bigrams return their stored score. For an unseen bigram
        ``(context, word)`` the score is ``weight * P_lower(word)``, where the
        weight is the context's back-off weight (or
        :attr:`_unknown_context_weight` when the context itself is unseen).
        """
        stored = self._prob.get(element)
        if stored is not None:
            return stored

        if isinstance(element, tuple) and len(element) == 2:
            context, word = element
            lower = self._lower.get(word, 0.0)
            weight = self._backoff.get(context, self._unknown_context_weight)
            return self._score_of(max(weight * lower, MIN_PROBABILITY))

        return self._unobs

    def _estimate(self, freqdist: FrequencyDistribution) -> Estimates:
        """Discard any back-off state from a previous fit, then estimate.

        Args:
            freqdist: Mapping of ``(context, word)`` bigrams to counts.

        Returns:
            Whatever :meth:`_estimate_bigrams` produces.
        """
        self._lower = {}
        self._backoff = {}
        self._unknown_context_weight = 1.0

        return self._estimate_bigrams(freqdist)

    @abstractmethod
    def _estimate_bigrams(self, freqdist: FrequencyDistribution) -> Estimates:
        """Estimate ``P(word | context)`` for each observed bigram.

        Args:
            freqdist: Mapping of ``(context, word)`` bigrams to counts.

        Returns:
            The probability of each observed bigram, and the mass reserved for
            a query that is not a bigram at all.
        """

    def _record_lower(self, lower: dict[Element, float], vocab_size: int) -> Probability:
        """Record the unigram model and report the fallback for non-bigram queries.

        Args:
            lower: The lower-order (unigram) probability of each word.
            vocab_size: The number of distinct words in the unigram model.

        Returns:
            The probability reserved for a query that is not a bigram.
        """
        self._lower = lower
        return 1.0 / vocab_size if vocab_size else MIN_PROBABILITY
