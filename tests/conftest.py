"""Shared test data and the estimator registry.

Every estimator in the public namespace is listed once, here, as an
:class:`EstimatorCase`. `tests/test_estimator_contract.py` runs the whole
contract over the registry, so a new estimator earns that coverage by adding one
row rather than by copying a block of assertions per method.

Golden-value tests stay separate and stay per-method: they pin the mathematics,
which is exactly the part the shared contract cannot check.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from freqprob import (
    ELE,
    MLE,
    AbsoluteDiscounting,
    Bayesian,
    CertaintyDegree,
    Interpolated,
    JelinekMercer,
    KatzBackoff,
    KneserNey,
    Laplace,
    Lidstone,
    ModifiedKneserNey,
    PitmanYor,
    Random,
    SimpleGoodTuring,
    StreamingLaplace,
    StreamingMLE,
    StupidBackoff,
    Uniform,
    WittenBell,
    create_lazy_laplace,
    create_lazy_mle,
)
from freqprob.base import Element, FrequencyDistribution, ScoringMethod

# --------------------------------------------------------------------------
# Shared distributions
# --------------------------------------------------------------------------

#: A plain unigram distribution with a clear frequency ordering and two
#: singletons, so Good-Turing-family methods have an ``N1`` to work with.
UNIGRAMS: FrequencyDistribution = {"a": 8, "b": 5, "c": 3, "d": 2, "e": 1, "f": 1}

#: Bigrams over an uneven set of contexts: ``the`` is heavy (8 observations over
#: two words), ``big`` and ``small`` are single observations.
BIGRAMS: FrequencyDistribution = {
    ("the", "cat"): 5,
    ("the", "dog"): 3,
    ("a", "cat"): 2,
    ("a", "dog"): 1,
    ("big", "cat"): 1,
    ("small", "dog"): 1,
}

#: Trigrams and the bigrams they back off to, for the interpolation family.
TRIGRAMS: FrequencyDistribution = {
    ("the", "big", "cat"): 3,
    ("a", "big", "dog"): 2,
    ("the", "small", "cat"): 1,
}
LOW_ORDER: FrequencyDistribution = {
    ("big", "cat"): 5,
    ("big", "dog"): 3,
    ("small", "cat"): 2,
}

#: An element no distribution above contains.
UNSEEN: Element = "definitely-unseen-element"


@dataclass(frozen=True)
class EstimatorCase:
    """One estimator, with everything the shared contract needs to exercise it.

    Attributes:
        name: Display name, used as the parametrization id.
        make: Builds the estimator; takes ``logprob`` and nothing else.
        fit_data: A distribution the estimator accepts, for re-fit checks.
        ordered: Whether a higher count must score at least as high as a lower
            one. False for ``Uniform`` (which ignores counts), for ``Random``,
            and for the bigram methods, whose scores are conditional on a
            context rather than monotone in the raw count.
        reserves_mass: Whether an unseen element is given non-zero probability.
            False for ``MLE``, which reserves nothing by design.
        conditional: Whether the estimator models ``P(word | context)``, so its
            mass sums to one *per context* rather than over the whole support.
        normalized: Whether the scores are a probability distribution at all.
            False for ``StupidBackoff``, which is a ranking score by design.
    """

    name: str
    make: Callable[[bool], ScoringMethod]
    fit_data: FrequencyDistribution = field(default_factory=lambda: UNIGRAMS)
    ordered: bool = True
    reserves_mass: bool = True
    conditional: bool = False
    normalized: bool = True


ESTIMATOR_CASES: list[EstimatorCase] = [
    # --- baselines -------------------------------------------------------
    EstimatorCase(
        "Uniform",
        lambda logprob: Uniform(UNIGRAMS, unobs_prob=0.1, logprob=logprob),
        ordered=False,
    ),
    EstimatorCase(
        "Random",
        lambda logprob: Random(UNIGRAMS, unobs_prob=0.1, seed=42, logprob=logprob),
        ordered=False,
    ),
    EstimatorCase(
        "MLE",
        lambda logprob: MLE(UNIGRAMS, logprob=logprob),
        reserves_mass=False,
    ),
    # --- additive --------------------------------------------------------
    EstimatorCase(
        "Lidstone", lambda logprob: Lidstone(UNIGRAMS, gamma=1.5, bins=50, logprob=logprob)
    ),
    EstimatorCase("Laplace", lambda logprob: Laplace(UNIGRAMS, bins=50, logprob=logprob)),
    EstimatorCase("ELE", lambda logprob: ELE(UNIGRAMS, bins=50, logprob=logprob)),
    EstimatorCase("Bayesian", lambda logprob: Bayesian(UNIGRAMS, alpha=0.5, logprob=logprob)),
    # --- Good-Turing family ----------------------------------------------
    EstimatorCase("WittenBell", lambda logprob: WittenBell(UNIGRAMS, bins=50, logprob=logprob)),
    EstimatorCase(
        "SimpleGoodTuring",
        lambda logprob: SimpleGoodTuring(UNIGRAMS, bins=50, logprob=logprob),
    ),
    EstimatorCase(
        "CertaintyDegree",
        lambda logprob: CertaintyDegree(UNIGRAMS, bins=50, unobs_prob=0.1, logprob=logprob),
    ),
    # --- bigram family ---------------------------------------------------
    EstimatorCase(
        "KneserNey",
        lambda logprob: KneserNey(BIGRAMS, discount=0.75, logprob=logprob),
        fit_data=BIGRAMS,
        ordered=False,
        conditional=True,
    ),
    EstimatorCase(
        "ModifiedKneserNey",
        lambda logprob: ModifiedKneserNey(BIGRAMS, logprob=logprob),
        fit_data=BIGRAMS,
        ordered=False,
        conditional=True,
    ),
    EstimatorCase(
        "AbsoluteDiscounting",
        lambda logprob: AbsoluteDiscounting(BIGRAMS, discount=0.75, logprob=logprob),
        fit_data=BIGRAMS,
        ordered=False,
        conditional=True,
    ),
    EstimatorCase(
        "PitmanYor",
        lambda logprob: PitmanYor(BIGRAMS, discount=0.5, strength=1.0, logprob=logprob),
        fit_data=BIGRAMS,
        ordered=False,
        conditional=True,
    ),
    EstimatorCase(
        "KatzBackoff",
        lambda logprob: KatzBackoff(BIGRAMS, k=5, logprob=logprob),
        fit_data=BIGRAMS,
        ordered=False,
        conditional=True,
    ),
    EstimatorCase(
        "StupidBackoff",
        lambda logprob: StupidBackoff(BIGRAMS, alpha=0.4, logprob=logprob),
        fit_data=BIGRAMS,
        ordered=False,
        conditional=True,
        normalized=False,
    ),
    # --- interpolation family --------------------------------------------
    EstimatorCase(
        "Interpolated",
        lambda logprob: Interpolated(TRIGRAMS, LOW_ORDER, lambda_weight=0.7, logprob=logprob),
        fit_data=TRIGRAMS,
        ordered=False,
    ),
    EstimatorCase(
        "JelinekMercer",
        lambda logprob: JelinekMercer(TRIGRAMS, LOW_ORDER, held_out_dist=TRIGRAMS, logprob=logprob),
        fit_data=TRIGRAMS,
        ordered=False,
    ),
    # --- performance adapters over the same contract ----------------------
    EstimatorCase(
        "LazyMLE",
        lambda logprob: create_lazy_mle(UNIGRAMS, logprob=logprob),
        reserves_mass=False,
    ),
    EstimatorCase("LazyLaplace", lambda logprob: create_lazy_laplace(UNIGRAMS, logprob=logprob)),
    EstimatorCase(
        "StreamingMLE",
        lambda logprob: StreamingMLE(initial_freqdist=dict(UNIGRAMS), logprob=logprob),
        reserves_mass=False,
    ),
    EstimatorCase(
        "StreamingLaplace",
        lambda logprob: StreamingLaplace(initial_freqdist=dict(UNIGRAMS), bins=50, logprob=logprob),
    ),
]

#: Ids and values, ready for ``pytest.mark.parametrize``.
CASE_IDS = [case.name for case in ESTIMATOR_CASES]


@pytest.fixture(params=ESTIMATOR_CASES, ids=CASE_IDS)
def estimator_case(request: pytest.FixtureRequest) -> EstimatorCase:
    """Yield each registered estimator in turn."""
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(autouse=True)
def _isolate_estimator_caches() -> None:
    """Keep memoized estimates from leaking between tests.

    Several estimators memoize on the frequency distribution, and the registry
    reuses the same distributions across cases, so a stale entry would otherwise
    be visible from one test to the next.
    """
    from freqprob import clear_all_caches

    clear_all_caches()
