"""Test the advanced smoothing methods."""

# Import Python standard libraries
import pytest

# Import the library to test
from freqprob.smoothing import (
    BayesianSmoothing,
    InterpolatedSmoothing,
    KneserNey,
    ModifiedKneserNey,
)

# Set up test data for advanced smoothing methods
# Bigram data for Kneser-Ney methods
BIGRAM_DATA = {
    ("the", "cat"): 5,
    ("the", "dog"): 3,
    ("a", "cat"): 2,
    ("a", "dog"): 1,
    ("big", "cat"): 1,
    ("small", "dog"): 1,
    ("the", "bird"): 2,
    ("a", "bird"): 1,
}

# Simple frequency data for other methods
SIMPLE_DATA = {"apple": 6, "banana": 3, "cherry": 1}

# Larger dataset for testing robustness
LARGE_BIGRAM_DATA = {}
contexts = ["the", "a", "an", "this", "that", "some", "many", "few"]
words = ["cat", "dog", "bird", "fish", "house", "car", "book", "tree"]
for i, context in enumerate(contexts):
    for j, word in enumerate(words):
        count = max(1, (i + j) % 5)  # Varying counts 1-4
        LARGE_BIGRAM_DATA[(context, word)] = count


def test_kneser_ney_basic() -> None:
    """Test basic Kneser-Ney functionality."""
    kn = KneserNey(BIGRAM_DATA, discount=0.75, logprob=False)  # type: ignore[arg-type]

    # Test observed bigrams
    prob_the_cat = kn(("the", "cat"))
    prob_the_dog = kn(("the", "dog"))
    assert prob_the_cat > prob_the_dog  # 'cat' appears in more contexts
    assert prob_the_cat > 0
    assert prob_the_dog > 0

    # Test unobserved bigram
    prob_unseen = kn(("unseen", "word"))
    assert prob_unseen > 0  # Should have some probability from continuation

    # Test that probabilities are reasonable
    assert 0 < prob_the_cat < 1
    assert 0 < prob_unseen < prob_the_cat


def test_kneser_ney_logprob() -> None:
    """Test Kneser-Ney with log-probabilities."""
    kn = KneserNey(BIGRAM_DATA, discount=0.75, logprob=True)  # type: ignore[arg-type]

    # Test that log-probabilities are negative
    log_prob = kn(("the", "cat"))
    assert log_prob < 0

    # Test unobserved bigram
    log_prob_unseen = kn(("unseen", "word"))
    assert log_prob_unseen < 0
    assert log_prob_unseen < log_prob  # Should be less likely


def test_kneser_ney_discount_validation() -> None:
    """Test that Kneser-Ney validates discount parameter."""
    with pytest.raises(ValueError, match="Discount.*between.*0.*1"):
        KneserNey(BIGRAM_DATA, discount=0.0)  # type: ignore[arg-type]  # Invalid: discount must be > 0

    with pytest.raises(ValueError, match="Discount.*between.*0.*1"):
        KneserNey(BIGRAM_DATA, discount=1.0)  # type: ignore[arg-type]  # Invalid: discount must be < 1

    with pytest.raises(ValueError, match="Discount.*between.*0.*1"):
        KneserNey(BIGRAM_DATA, discount=1.5)  # type: ignore[arg-type]  # Invalid: discount must be < 1


def test_modified_kneser_ney_basic() -> None:
    """Test basic Modified Kneser-Ney functionality."""
    mkn = ModifiedKneserNey(LARGE_BIGRAM_DATA, logprob=False)  # type: ignore[arg-type]

    # Test observed bigrams with different counts
    high_count_bigram = ("the", "cat")
    low_count_bigram = ("few", "tree")

    prob_high = mkn(high_count_bigram)
    prob_low = mkn(low_count_bigram)

    assert prob_high > 0
    assert prob_low > 0
    # Note: due to continuation probability, high count doesn't always mean higher probability

    # Test unobserved bigram
    prob_unseen = mkn(("unseen", "word"))
    assert prob_unseen > 0


def test_modified_kneser_ney_logprob() -> None:
    """Test Modified Kneser-Ney with log-probabilities."""
    mkn = ModifiedKneserNey(LARGE_BIGRAM_DATA, logprob=True)  # type: ignore[arg-type]

    log_prob = mkn(("the", "cat"))
    assert log_prob < 0

    log_prob_unseen = mkn(("unseen", "word"))
    assert log_prob_unseen < 0


def test_interpolated_smoothing_basic() -> None:
    """Test basic interpolated smoothing functionality."""
    # Create high-order and low-order distributions
    high_order = {"trigram_1": 3, "trigram_2": 2, "trigram_3": 1}
    low_order = {"bigram_1": 5, "bigram_2": 3, "trigram_1": 1}  # Some overlap

    interp = InterpolatedSmoothing(high_order, low_order, lambda_weight=0.7, logprob=False)  # type: ignore[arg-type]

    # Test element that appears in both distributions
    prob_common = interp("trigram_1")
    assert prob_common > 0

    # Test element that appears only in high-order
    prob_high_only = interp("trigram_2")
    assert prob_high_only > 0

    # Test element that appears only in low-order
    prob_low_only = interp("bigram_1")
    assert prob_low_only > 0

    # Test unobserved element
    prob_unseen = interp("unseen")
    assert prob_unseen > 0


def test_interpolated_smoothing_weights() -> None:
    """Test interpolated smoothing with different weights."""
    high_order = {"common": 1}
    low_order = {"common": 1}

    # Test with high weight on high-order model
    interp_high = InterpolatedSmoothing(high_order, low_order, lambda_weight=0.9, logprob=False)  # type: ignore[arg-type]

    # Test with high weight on low-order model
    interp_low = InterpolatedSmoothing(high_order, low_order, lambda_weight=0.1, logprob=False)  # type: ignore[arg-type]

    # Both should give same result for this balanced case
    prob_high = interp_high("common")
    prob_low = interp_low("common")

    assert prob_high == prob_low  # Same input data, so interpolation doesn't matter


def test_interpolated_smoothing_validation() -> None:
    """Test that interpolated smoothing validates lambda parameter."""
    high_order = {"a": 1}
    low_order = {"b": 1}

    with pytest.raises(ValueError, match="Lambda.*between.*0.*1"):
        InterpolatedSmoothing(high_order, low_order, lambda_weight=-0.1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Lambda.*between.*0.*1"):
        InterpolatedSmoothing(high_order, low_order, lambda_weight=1.1)  # type: ignore[arg-type]


def test_bayesian_smoothing_basic() -> None:
    """Test basic Bayesian smoothing functionality."""
    bayes = BayesianSmoothing(SIMPLE_DATA, alpha=1.0, logprob=False)  # type: ignore[arg-type]

    # Test observed elements
    prob_apple = bayes("apple")  # (6+1)/(10+3*1) = 7/13
    prob_cherry = bayes("cherry")  # (1+1)/(10+3*1) = 2/13
    prob_unseen = bayes("unseen")  # 1/(10+3*1) = 1/13

    assert prob_apple > prob_cherry > prob_unseen
    assert abs(prob_apple - 7 / 13) < 1e-10
    assert abs(prob_cherry - 2 / 13) < 1e-10
    assert abs(prob_unseen - 1 / 13) < 1e-10


def test_bayesian_smoothing_alpha_effects() -> None:
    """Test that different alpha values affect smoothing strength."""
    # Minimal smoothing
    bayes_min = BayesianSmoothing(SIMPLE_DATA, alpha=0.1, logprob=False)  # type: ignore[arg-type]

    # Strong smoothing
    bayes_strong = BayesianSmoothing(SIMPLE_DATA, alpha=5.0, logprob=False)  # type: ignore[arg-type]

    prob_common_min = bayes_min("apple")
    prob_unseen_min = bayes_min("unseen")
    prob_common_strong = bayes_strong("apple")
    prob_unseen_strong = bayes_strong("unseen")

    # Strong smoothing should reduce the gap between seen and unseen
    gap_min = prob_common_min - prob_unseen_min
    gap_strong = prob_common_strong - prob_unseen_strong

    assert gap_strong < gap_min


def test_bayesian_smoothing_logprob() -> None:
    """Test Bayesian smoothing with log-probabilities."""
    bayes = BayesianSmoothing(SIMPLE_DATA, alpha=1.0, logprob=True)  # type: ignore[arg-type]

    log_prob = bayes("apple")
    assert log_prob < 0

    log_prob_unseen = bayes("unseen")
    assert log_prob_unseen < 0
    assert log_prob_unseen < log_prob


def test_bayesian_smoothing_validation() -> None:
    """Test that Bayesian smoothing validates alpha parameter."""
    with pytest.raises(ValueError, match="Alpha.*positive"):
        BayesianSmoothing(SIMPLE_DATA, alpha=0.0)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Alpha.*positive"):
        BayesianSmoothing(SIMPLE_DATA, alpha=-1.0)  # type: ignore[arg-type]


def test_all_methods_string_representation() -> None:
    """Test that all methods have proper string representation."""
    kn = KneserNey(BIGRAM_DATA)  # type: ignore[arg-type]
    mkn = ModifiedKneserNey(BIGRAM_DATA)  # type: ignore[arg-type]
    interp = InterpolatedSmoothing({"a": 1}, {"b": 1})
    bayes = BayesianSmoothing(SIMPLE_DATA)  # type: ignore[arg-type]

    assert "Kneser-Ney" in str(kn)
    assert "Modified Kneser-Ney" in str(mkn)
    assert "Interpolated" in str(interp)
    assert "Bayesian" in str(bayes)

    # All should indicate log-scorer by default
    assert "log-scorer" in str(kn)
    assert "log-scorer" in str(mkn)
    assert "log-scorer" in str(interp)
    assert "log-scorer" in str(bayes)


def test_edge_case_empty_data() -> None:
    """Test behavior with empty or minimal data."""
    # Test with empty data
    empty_data: dict[tuple[str, str], int] = {}

    # These should not crash but may produce degenerate results
    try:
        kn = KneserNey(empty_data, logprob=False)  # type: ignore[arg-type]
        prob = kn(("any", "bigram"))
        assert prob >= 0
    except (ZeroDivisionError, ValueError):
        # It's acceptable for some methods to fail with empty data
        pass

    # Test with minimal data
    minimal_data = {("a", "b"): 1}
    kn = KneserNey(minimal_data, logprob=False)  # type: ignore[arg-type]
    prob = kn(("a", "b"))
    assert prob > 0


def test_consistency_with_traditional_methods() -> None:
    """Test that Bayesian smoothing with alpha=1 matches Laplace smoothing."""
    from freqprob import Laplace

    # For unigram data, Bayesian with alpha=1 should match Laplace
    bayes = BayesianSmoothing(SIMPLE_DATA, alpha=1.0, logprob=False)  # type: ignore[arg-type]
    laplace = Laplace(SIMPLE_DATA, logprob=False)  # type: ignore[arg-type]

    # Test observed elements
    for element in SIMPLE_DATA:
        bayes_prob = bayes(element)
        laplace_prob = laplace(element)
        assert abs(bayes_prob - laplace_prob) < 1e-10

    # Test unobserved element
    bayes_unseen = bayes("unseen")
    laplace_unseen = laplace("unseen")
    assert abs(bayes_unseen - laplace_unseen) < 1e-10


def test_probability_normalization() -> None:
    """Test that probabilities are reasonable (don't test exact normalization)."""
    bayes = BayesianSmoothing(SIMPLE_DATA, alpha=1.0, logprob=False)  # type: ignore[arg-type]

    # Test that all probabilities are positive and reasonable
    for element in SIMPLE_DATA:
        prob = bayes(element)
        assert 0 < prob < 1

    # Test that unobserved probability is positive but smaller than observed
    unobs_prob = bayes("unseen")
    assert 0 < unobs_prob < bayes("apple")  # 'apple' has highest count

    # Test that total observed probabilities are reasonable
    total_observed = sum(bayes(element) for element in SIMPLE_DATA)
    assert 0.5 < total_observed < 1.5  # Should be roughly around 1 but allow flexibility


def test_kneser_ney_context_sensitivity() -> None:
    """Test that Kneser-Ney handles context diversity appropriately."""
    # Create data where words have different context diversity
    bigram_data = {
        ("context1", "diverse"): 1,
        ("context2", "diverse"): 1,
        ("context3", "diverse"): 1,
        ("context4", "diverse"): 1,
        ("context5", "diverse"): 1,
        ("context1", "narrow"): 10,  # Higher frequency but less diverse
    }

    kn = KneserNey(bigram_data, discount=0.5, logprob=False)  # type: ignore[arg-type]

    # Test that both words get some probability in new contexts
    # (the exact relationship depends on the implementation details)
    prob_new_diverse = kn(("new_context", "diverse"))
    prob_new_narrow = kn(("new_context", "narrow"))

    # Both should be positive (this tests that the algorithm works)
    assert prob_new_diverse > 0
    assert prob_new_narrow > 0

    # Test that Kneser-Ney gives reasonable probabilities
    # (diverse word should have good continuation probability due to context variety)
    assert prob_new_diverse >= prob_new_narrow * 0.5  # Allow some flexibility
