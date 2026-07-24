"""Tests for save()/load() serialization of fitted estimators."""

import pickle

import pytest

import freqprob


def _make_scorers() -> dict[str, freqprob.ScoringMethod]:
    """Build one fitted scorer of each representative type."""
    bigrams = {("a", "b"): 3, ("a", "c"): 1, ("b", "c"): 2}
    return {
        "MLE": freqprob.MLE({"a": 2, "b": 1}, logprob=False),
        "Laplace": freqprob.Laplace({"a": 3, "b": 2, "c": 1}, bins=100, logprob=False),
        "KneserNey": freqprob.KneserNey(bigrams, discount=0.5, logprob=False),
        "SimpleGoodTuring": freqprob.SimpleGoodTuring(
            {"a": 3, "b": 2, "c": 1, "d": 1}, logprob=False
        ),
        "Interpolated": freqprob.Interpolated(
            {("a", "b", "c"): 3}, {("b", "c"): 5}, lambda_weight=0.7, logprob=False
        ),
        "Bayesian": freqprob.Bayesian({"a": 2, "b": 1}, alpha=1.0, logprob=False),
    }


@pytest.mark.parametrize("name", list(_make_scorers().keys()))
def test_save_load_round_trip(name: str, tmp_path) -> None:
    """A saved scorer reloads to the same type and produces identical scores."""
    scorer = _make_scorers()[name]
    path = tmp_path / f"{name}.pkl"

    scorer.save(path)
    restored = type(scorer).load(path)

    assert type(restored) is type(scorer)
    for element in [*scorer._prob, "definitely-unseen-element"]:
        assert restored(element) == scorer(element)


def test_load_wrong_type_raises(tmp_path) -> None:
    """Loading via the wrong subclass raises TypeError."""
    path = tmp_path / "m.pkl"
    freqprob.MLE({"a": 1}, logprob=False).save(path)
    with pytest.raises(TypeError):
        freqprob.KneserNey.load(path)


def test_plain_pickle_also_works() -> None:
    """A fitted scorer round-trips through plain pickle as well."""
    scorer = freqprob.Laplace({"a": 3, "b": 1}, bins=10, logprob=False)
    restored = pickle.loads(pickle.dumps(scorer))
    assert restored("a") == scorer("a")
    assert restored("unseen") == scorer("unseen")
