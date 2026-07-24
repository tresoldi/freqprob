"""Tests for the scikit-learn-style fit/predict/score aliases."""

import freqprob


def test_score_matches_call() -> None:
    scorer = freqprob.MLE({"a": 2, "b": 1}, logprob=False)
    for element in ["a", "b", "unseen"]:
        assert scorer.score(element) == scorer(element)


def test_predict_matches_per_element_calls() -> None:
    scorer = freqprob.Laplace({"a": 3, "b": 2, "c": 1}, bins=100, logprob=False)
    elements = ["a", "b", "c", "unseen"]
    assert scorer.predict(elements) == [scorer(e) for e in elements]


def test_predict_preserves_order_and_length() -> None:
    scorer = freqprob.MLE({"x": 1, "y": 1}, logprob=False)
    elements = ["y", "x", "y", "z"]
    result = scorer.predict(elements)
    assert len(result) == len(elements)
    assert result[0] == result[2]  # both "y"


def test_fit_returns_self_for_chaining() -> None:
    # Fresh instance fitted via the documented chaining pattern.
    scorer = freqprob.MLE({})
    assert scorer.fit({"a": 2, "b": 1}) is scorer
    assert scorer.score("a") == scorer("a")
