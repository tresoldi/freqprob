"""Tests for the back-off / discounting smoothers and the entropy estimators.

Covers ``AbsoluteDiscounting``, ``PitmanYor``, ``KatzBackoff``,
``StupidBackoff``, and ``JelinekMercer``, plus the ``sample_coverage``,
``chao_shen_entropy``, and ``nsb_entropy`` functions added to ``metrics``.
"""

import math

import pytest

from freqprob import (
    AbsoluteDiscounting,
    JelinekMercer,
    KatzBackoff,
    PitmanYor,
    StupidBackoff,
    chao_shen_entropy,
    nsb_entropy,
    sample_coverage,
)
from freqprob.base import Element, ScoringMethod

# Bigram data with a spread of low counts, suitable for every method here.
BIGRAM_DATA: dict[Element, int] = {
    ("the", "cat"): 5,
    ("the", "dog"): 3,
    ("the", "bird"): 2,
    ("a", "cat"): 2,
    ("a", "dog"): 1,
    ("a", "bird"): 1,
    ("big", "cat"): 1,
    ("small", "dog"): 1,
    ("my", "fish"): 1,
    ("your", "fish"): 1,
}


# --------------------------------------------------------------------------- #
# Absolute Discounting
# --------------------------------------------------------------------------- #
def test_absolute_discounting_basic() -> None:
    """Observed bigrams get sensible probabilities; unseen ones stay positive."""
    ad = AbsoluteDiscounting(BIGRAM_DATA, discount=0.75, logprob=False)
    p_the_cat = ad(("the", "cat"))
    p_the_dog = ad(("the", "dog"))
    assert 0 < p_the_dog < p_the_cat < 1
    # Unseen word in a known context still backs off to the unigram model.
    assert ad(("the", "fish")) > 0
    # Unseen context backs off to the raw unigram probability of the word.
    assert ad(("brand", "cat")) == pytest.approx(8 / 18)


def test_absolute_discounting_logprob_matches() -> None:
    """Log-probabilities are the logs of the plain probabilities."""
    ad_p = AbsoluteDiscounting(BIGRAM_DATA, logprob=False)
    ad_l = AbsoluteDiscounting(BIGRAM_DATA, logprob=True)
    for query in [("the", "cat"), ("a", "dog"), ("the", "fish"), ("x", "cat")]:
        assert ad_l(query) == pytest.approx(math.log(ad_p(query)))


def test_absolute_discounting_validation() -> None:
    """Discount must be in [0, 1)."""
    with pytest.raises(ValueError, match="Discount"):
        AbsoluteDiscounting(BIGRAM_DATA, discount=1.0)
    with pytest.raises(ValueError, match="Discount"):
        AbsoluteDiscounting(BIGRAM_DATA, discount=-0.1)


# --------------------------------------------------------------------------- #
# Pitman-Yor
# --------------------------------------------------------------------------- #
def test_pitman_yor_reduces_to_absolute_discounting() -> None:
    """With strength=0, Pitman-Yor equals absolute discounting."""
    ad = AbsoluteDiscounting(BIGRAM_DATA, discount=0.75, logprob=False)
    py = PitmanYor(BIGRAM_DATA, discount=0.75, strength=0.0, logprob=False)
    for query in [("the", "cat"), ("a", "dog"), ("the", "fish"), ("z", "bird")]:
        assert py(query) == pytest.approx(ad(query))


def test_pitman_yor_strength_shifts_mass() -> None:
    """A larger strength pulls a frequent-context estimate toward the base."""
    py_low = PitmanYor(BIGRAM_DATA, discount=0.5, strength=0.0, logprob=False)
    py_high = PitmanYor(BIGRAM_DATA, discount=0.5, strength=10.0, logprob=False)
    # 'the cat' is well attested, so more concentration lowers its estimate.
    assert py_high(("the", "cat")) < py_low(("the", "cat"))


def test_pitman_yor_validation() -> None:
    """Discount in [0, 1) and strength > -discount."""
    with pytest.raises(ValueError, match="Discount"):
        PitmanYor(BIGRAM_DATA, discount=1.0)
    with pytest.raises(ValueError, match="Strength"):
        PitmanYor(BIGRAM_DATA, discount=0.5, strength=-0.5)


# --------------------------------------------------------------------------- #
# Katz Back-off
# --------------------------------------------------------------------------- #
def test_katz_backoff_basic() -> None:
    """Observed bigrams are (at most) their MLE; unseen ones back off."""
    katz = KatzBackoff(BIGRAM_DATA, logprob=False)
    # Discounting never increases an observed probability above its MLE.
    assert katz(("the", "cat")) <= 5 / 10 + 1e-9
    assert katz(("the", "cat")) > 0
    # Unseen word in a known context is non-negative and finite.
    assert katz(("the", "fish")) >= 0


def test_katz_backoff_validation() -> None:
    """Threshold k must be a positive integer."""
    with pytest.raises(ValueError, match="k"):
        KatzBackoff(BIGRAM_DATA, k=0)


# --------------------------------------------------------------------------- #
# Stupid Back-off
# --------------------------------------------------------------------------- #
def test_stupid_backoff_scores() -> None:
    """Observed bigrams score their MLE; unseen ones are alpha * unigram."""
    sb = StupidBackoff(BIGRAM_DATA, alpha=0.4, logprob=False)
    assert sb(("the", "cat")) == pytest.approx(5 / 10)
    assert sb(("a", "dog")) == pytest.approx(1 / 4)
    # 'the fish' unseen: alpha * P_uni(fish) = 0.4 * (2 / 18)
    assert sb(("the", "fish")) == pytest.approx(0.4 * (2 / 18))


def test_stupid_backoff_validation() -> None:
    """Alpha must be in (0, 1]."""
    with pytest.raises(ValueError, match="alpha"):
        StupidBackoff(BIGRAM_DATA, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        StupidBackoff(BIGRAM_DATA, alpha=1.5)


# --------------------------------------------------------------------------- #
# Jelinek-Mercer
# --------------------------------------------------------------------------- #
TRIGRAMS: dict[Element, int] = {
    ("the", "big", "cat"): 3,
    ("a", "big", "dog"): 2,
    ("the", "big", "dog"): 1,
}
BIGRAMS: dict[Element, int] = {("big", "cat"): 5, ("big", "dog"): 3, ("small", "cat"): 2}


def test_jelinek_mercer_without_heldout_is_fixed_lambda() -> None:
    """With no held-out data it matches the supplied lambda exactly."""
    jm = JelinekMercer(TRIGRAMS, BIGRAMS, lambda_weight=0.6, logprob=False)
    assert jm.estimated_lambda == pytest.approx(0.6)


def test_jelinek_mercer_em_moves_lambda() -> None:
    """Held-out data that only high-order predicts pushes lambda toward 1."""
    held_out: dict[Element, int] = {("the", "big", "cat"): 10, ("a", "big", "dog"): 8}
    jm = JelinekMercer(TRIGRAMS, BIGRAMS, held_out_dist=held_out, lambda_weight=0.5)
    assert 0.0 <= jm.estimated_lambda <= 1.0
    assert jm.estimated_lambda > 0.5


def test_jelinek_mercer_validation() -> None:
    """Lambda in [0, 1] and positive iteration count."""
    with pytest.raises(ValueError, match="Lambda"):
        JelinekMercer(TRIGRAMS, BIGRAMS, lambda_weight=1.5)
    with pytest.raises(ValueError, match="em_iterations"):
        JelinekMercer(TRIGRAMS, BIGRAMS, em_iterations=0)


# --------------------------------------------------------------------------- #
# Serialization round-trip for the new estimators
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "scorer",
    [
        AbsoluteDiscounting(BIGRAM_DATA, logprob=True),
        PitmanYor(BIGRAM_DATA, logprob=True),
        KatzBackoff(BIGRAM_DATA, logprob=True),
        StupidBackoff(BIGRAM_DATA, logprob=True),
    ],
)
def test_backoff_methods_pickle_round_trip(scorer: ScoringMethod) -> None:
    """Every new bigram scorer survives a save/load round-trip."""
    import pickle

    restored = pickle.loads(pickle.dumps(scorer))
    for query in [("the", "cat"), ("the", "fish"), ("new", "cat")]:
        assert restored(query) == pytest.approx(scorer(query))


# --------------------------------------------------------------------------- #
# sample_coverage
# --------------------------------------------------------------------------- #
def test_sample_coverage() -> None:
    """Coverage is 1 - N1/N and the missing mass is its complement."""
    no_singletons: dict[Element, int] = {"a": 10, "b": 5, "c": 3}
    some_singletons: dict[Element, int] = {"a": 8, "b": 1, "c": 1}
    all_singletons: dict[Element, int] = {"a": 1, "b": 1, "c": 1}
    empty: dict[Element, int] = {}
    assert sample_coverage(no_singletons) == 1.0
    assert sample_coverage(some_singletons) == pytest.approx(0.8)
    assert sample_coverage(empty) == 0.0
    # All singletons -> zero coverage (all mass is missing).
    assert sample_coverage(all_singletons) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# chao_shen_entropy
# --------------------------------------------------------------------------- #
def test_chao_shen_entropy_bounds_and_bias() -> None:
    """Chao-Shen is non-negative and corrects the plug-in bias upward."""
    freqdist: dict[Element, int] = {"a": 8, "b": 1, "c": 1}
    empty: dict[Element, int] = {}
    single: dict[Element, int] = {"a": 1}
    plugin = -sum((c / 10) * math.log(c / 10) for c in freqdist.values())
    cs = chao_shen_entropy(freqdist)
    assert cs > plugin  # bias-corrected estimate is larger
    assert chao_shen_entropy(empty) == 0.0
    # A single observation has zero entropy.
    assert chao_shen_entropy(single) == pytest.approx(0.0, abs=1e-9)


def test_chao_shen_matches_plugin_when_well_sampled() -> None:
    """With no singletons and large counts it approaches the true entropy."""
    freqdist: dict[Element, int] = {"a": 1000, "b": 1000, "c": 1000}
    assert chao_shen_entropy(freqdist) == pytest.approx(math.log(3), abs=1e-3)


# --------------------------------------------------------------------------- #
# nsb_entropy
# --------------------------------------------------------------------------- #
def test_nsb_entropy_limits() -> None:
    """NSB approaches ln K for a well-sampled uniform distribution."""
    uniform: dict[Element, int] = {chr(97 + i): 500 for i in range(4)}
    peaked: dict[Element, int] = {"a": 10000, "b": 1, "c": 1}
    empty: dict[Element, int] = {}
    single: dict[Element, int] = {"a": 5}
    assert nsb_entropy(uniform) == pytest.approx(math.log(4), abs=1e-2)
    # Highly peaked distribution -> near-zero entropy.
    assert nsb_entropy(peaked) < 0.05
    assert nsb_entropy(empty) == 0.0
    assert nsb_entropy(single) == 0.0  # single type -> zero entropy


def test_nsb_entropy_bins_monotonic() -> None:
    """A larger assumed alphabet reserves more entropy for unseen types."""
    freqdist: dict[Element, int] = {"a": 8, "b": 1, "c": 1}
    small = nsb_entropy(freqdist)  # bins defaults to 3 observed types
    large = nsb_entropy(freqdist, bins=100)
    assert large > small
    assert large <= math.log(100)


def test_nsb_entropy_rejects_too_few_bins() -> None:
    """Bins below the observed type count is an error."""
    too_few: dict[Element, int] = {"a": 1, "b": 1, "c": 1}
    with pytest.raises(ValueError, match="bins"):
        nsb_entropy(too_few, bins=2)
