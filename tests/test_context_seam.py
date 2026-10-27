"""One definition of "the context of an n-gram", shared by the estimators that need it.

`methods/_bigram.py` holds the bigram seam: extracting per-context counts, and
scoring an unseen bigram by backing off to a lower-order model. Four estimators
crossed it; Kneser-Ney and Modified Kneser-Ney re-derived the same extraction
themselves, and the interpolation family derived its context slice a third time.

The bigram family now shares one seam, which is what
:class:`~freqprob.AbsoluteDiscounting`'s own documentation always claimed:
Kneser-Ney *is* absolute discounting with the unigram model replaced by the
continuation distribution. These tests hold that claim to its word.

The interpolation family keeps its own seam on purpose: it takes two
distributions of arbitrary order, where the bigram seam derives its lower-order
model from a single distribution of bigrams. Within that family, the context
slice is now defined once, in ``lower_order_context``.
"""

import pytest

from freqprob import (
    AbsoluteDiscounting,
    Interpolated,
    KatzBackoff,
    KneserNey,
    ModifiedKneserNey,
    PitmanYor,
    StupidBackoff,
)
from freqprob.methods._bigram import (
    BigramBackoff,
    continuation_probs,
    extract_bigram_stats,
)
from freqprob.methods.interpolated import lower_order_context

BIGRAMS = {
    ("the", "cat"): 5,
    ("the", "dog"): 3,
    ("a", "cat"): 2,
    ("a", "dog"): 1,
    ("big", "cat"): 1,
    ("small", "dog"): 1,
}

# Contexts with markedly different weights: 'the' is observed 8 times over two
# words, 'big' just once. 'bird' is a known word that follows neither.
UNEVEN_CONTEXTS = {
    ("the", "cat"): 5,
    ("the", "dog"): 3,
    ("a", "cat"): 2,
    ("a", "bird"): 1,
    ("big", "cat"): 1,
}

# Every estimator that models P(word | context) over bigrams.
BIGRAM_ESTIMATORS = [
    ("KneserNey", lambda: KneserNey(BIGRAMS, discount=0.75, logprob=False)),
    ("ModifiedKneserNey", lambda: ModifiedKneserNey(BIGRAMS, logprob=False)),
    ("AbsoluteDiscounting", lambda: AbsoluteDiscounting(BIGRAMS, logprob=False)),
    ("PitmanYor", lambda: PitmanYor(BIGRAMS, strength=1.0, logprob=False)),
    ("KatzBackoff", lambda: KatzBackoff(BIGRAMS, logprob=False)),
    ("StupidBackoff", lambda: StupidBackoff(BIGRAMS, logprob=False)),
]
IDS = [name for name, _ in BIGRAM_ESTIMATORS]
FACTORIES = [factory for _, factory in BIGRAM_ESTIMATORS]


class TestOneBigramSeam:
    """Every bigram estimator crosses `methods/_bigram.py`."""

    @pytest.mark.parametrize("factory", FACTORIES, ids=IDS)
    def test_estimator_sits_on_the_seam(self, factory: object) -> None:
        """Including the two Kneser-Ney variants, which used to route around it."""
        assert isinstance(factory(), BigramBackoff)  # type: ignore[operator]

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: KneserNey(BIGRAMS, discount=0.75, logprob=False),
            lambda: ModifiedKneserNey(BIGRAMS, logprob=False),
            lambda: AbsoluteDiscounting(BIGRAMS, logprob=False),
            lambda: PitmanYor(BIGRAMS, strength=1.0, logprob=False),
        ],
        ids=["KneserNey", "ModifiedKneserNey", "AbsoluteDiscounting", "PitmanYor"],
    )
    def test_discounting_back_off_is_context_sensitive(self, factory: object) -> None:
        """The back-off weight varies by context, so the same word scores differently.

        Before the seam was shared, both Kneser-Ney variants returned one flat
        average here, no matter which context was asked about. Which way the two
        scores order depends on how each method sets its weights; that they
        differ at all is the shared property.
        """
        scorer = factory()  # type: ignore[operator]
        scorer.fit(UNEVEN_CONTEXTS)

        # 'the' is a heavy context (8 observations over 2 words); 'big' is a
        # light one (a single observation). 'bird' follows neither.
        after_the = scorer(("the", "bird"))
        after_big = scorer(("big", "bird"))

        assert after_the > 0.0
        assert after_big > 0.0
        assert after_the != after_big

    def test_stupid_backoff_is_deliberately_flat(self) -> None:
        """Stupid Back-off opts out: one alpha for every context, seen or not.

        It crosses the same seam, but leaves the per-context weights empty and
        lets the unknown-context weight cover both cases.
        """
        scorer = StupidBackoff(UNEVEN_CONTEXTS, alpha=0.4, logprob=False)

        assert scorer._backoff == {}
        assert scorer(("the", "bird")) == pytest.approx(scorer(("big", "bird")))

    @pytest.mark.parametrize("factory", FACTORIES, ids=IDS)
    def test_unknown_context_is_handled(self, factory: object) -> None:
        """A context never seen falls back without raising."""
        scorer = factory()  # type: ignore[operator]

        assert scorer(("never-seen-context", "cat")) > 0.0

    @pytest.mark.parametrize("factory", FACTORIES, ids=IDS)
    def test_non_bigram_query_gets_the_reserved_score(self, factory: object) -> None:
        """A key that is not a bigram at all is not a back-off case."""
        scorer = factory()  # type: ignore[operator]

        assert scorer("not-a-bigram") == scorer.unobserved_score()
        assert scorer(("a", "b", "c")) == scorer.unobserved_score()


class TestKneserNeyIsAbsoluteDiscountingOverContinuations:
    """The relationship the docstrings assert, now enforced by construction."""

    def test_the_two_differ_only_in_their_lower_order_model(self) -> None:
        """Same discount, same back-off weights; only the lower-order model differs."""
        kn = KneserNey(BIGRAMS, discount=0.75, logprob=False)
        ad = AbsoluteDiscounting(BIGRAMS, discount=0.75, logprob=False)

        assert kn._backoff == pytest.approx(ad._backoff)
        assert kn._lower != ad._lower

    def test_kneser_ney_uses_the_continuation_distribution(self) -> None:
        """Its lower-order model counts distinct contexts, not occurrences."""
        kn = KneserNey(BIGRAMS, discount=0.75, logprob=False)
        _, context_types, _, _, _ = extract_bigram_stats(BIGRAMS)

        assert kn._lower == pytest.approx(continuation_probs(context_types))

    def test_absolute_discounting_uses_the_unigram_distribution(self) -> None:
        """Its lower-order model counts occurrences."""
        ad = AbsoluteDiscounting(BIGRAMS, discount=0.75, logprob=False)
        _, _, _, unigram, total = extract_bigram_stats(BIGRAMS)

        expected = {word: count / total for word, count in unigram.items()}
        assert ad._lower == pytest.approx(expected)

    def test_continuation_favors_the_word_seen_in_more_contexts(self) -> None:
        """The property that makes Kneser-Ney worth its complexity."""
        # 'diverse' follows five different contexts once each; 'narrow' follows
        # one context ten times.
        data = {
            ("c1", "diverse"): 1,
            ("c2", "diverse"): 1,
            ("c3", "diverse"): 1,
            ("c4", "diverse"): 1,
            ("c5", "diverse"): 1,
            ("c1", "narrow"): 10,
        }
        kn = KneserNey(data, discount=0.5, logprob=False)
        ad = AbsoluteDiscounting(data, discount=0.5, logprob=False)

        # In a brand-new context, Kneser-Ney prefers the context-diverse word.
        assert kn(("new", "diverse")) > kn(("new", "narrow"))
        # Absolute discounting, going by raw frequency, prefers the other one.
        assert ad(("new", "narrow")) > ad(("new", "diverse"))


class TestContinuationProbs:
    """The shared continuation helper."""

    def test_sums_to_one(self) -> None:
        """It is a distribution over words."""
        _, context_types, _, _, _ = extract_bigram_stats(BIGRAMS)
        probs = continuation_probs(context_types)

        assert sum(probs.values()) == pytest.approx(1.0)

    def test_counts_distinct_contexts_not_occurrences(self) -> None:
        """'cat' and 'dog' each follow three contexts, despite differing counts."""
        _, context_types, _, _, _ = extract_bigram_stats(BIGRAMS)
        probs = continuation_probs(context_types)

        assert probs["cat"] == pytest.approx(0.5)
        assert probs["dog"] == pytest.approx(0.5)

    def test_empty_input_is_empty_output(self) -> None:
        """No bigrams, no continuation distribution."""
        assert continuation_probs({}) == {}


class TestExtractBigramStatsIsTheOnlyExtractor:
    """Non-bigram keys are skipped in one place, not five."""

    @pytest.mark.parametrize("factory", FACTORIES, ids=IDS)
    def test_non_bigram_keys_are_ignored(self, factory: object) -> None:
        """A distribution may mix bigrams with other bookkeeping entries."""
        clean = factory()  # type: ignore[operator]

        polluted_data = {**BIGRAMS, "a-string-key": 7, ("a", "b", "c"): 3}
        polluted = clean.__class__.__new__(clean.__class__)
        polluted.__init__(polluted_data, logprob=False)  # type: ignore[misc]

        for bigram in BIGRAMS:
            assert polluted(bigram) == pytest.approx(clean(bigram))

    def test_stats_agree_with_a_hand_count(self) -> None:
        """The extraction itself, checked directly rather than through an estimator."""
        context_total, context_types, context_word, unigram, total = extract_bigram_stats(BIGRAMS)

        assert context_total["the"] == 8
        assert context_types["the"] == {"cat", "dog"}
        assert context_word[("the", "cat")] == 5
        assert unigram["cat"] == 8
        assert total == 13


class TestInterpolationFamilyContext:
    """One definition of the context slice, shared inside the interpolation family."""

    @pytest.mark.parametrize(
        ("ngram", "low_order_n", "expected"),
        [
            (("w1", "w2", "w3"), 2, ("w2", "w3")),
            (("w1", "w2", "w3", "w4"), 3, ("w2", "w3", "w4")),
            (("w1", "w2"), 2, ("w1", "w2")),
            ("a-string", 2, "a-string"),
            (("w1", "w2"), None, ("w1", "w2")),
        ],
    )
    def test_lower_order_context(
        self, ngram: object, low_order_n: int | None, expected: object
    ) -> None:
        """The context is the suffix; anything too short is its own context."""
        assert lower_order_context(ngram, low_order_n) == expected  # type: ignore[arg-type]

    def test_interpolated_uses_the_shared_slice(self) -> None:
        """The estimator's own accessor and the shared function agree."""
        trigrams = {("the", "big", "cat"): 3, ("a", "big", "dog"): 2}
        bigrams = {("big", "cat"): 5, ("big", "dog"): 3}
        interp = Interpolated(trigrams, bigrams, lambda_weight=0.7, logprob=False)

        for ngram in trigrams:
            assert interp._extract_lower_context(ngram) == lower_order_context(ngram, 2)

    def test_scoring_an_unseen_ngram_needs_no_rescan(self) -> None:
        """The low-order model is built once at fit time, then looked up."""
        trigrams = {("the", "big", "cat"): 3, ("a", "big", "dog"): 2}
        bigrams = {("big", "cat"): 5, ("big", "dog"): 3, ("small", "cat"): 2}
        interp = Interpolated(trigrams, bigrams, lambda_weight=0.7, logprob=False)

        # 0.3 * P_low(('big', 'cat')) = 0.3 * 5/10
        assert interp(("unseen", "big", "cat")) == pytest.approx(0.15)
        # The cached model is the maximum-likelihood low-order distribution.
        assert interp._low_probs[("big", "cat")] == pytest.approx(0.5)
