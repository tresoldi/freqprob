"""Test the library functions."""

# Import Python standard libraries
import itertools
import random
import string
from collections import Counter

# Import 3rd-party libraries
import pytest

# Import the library to test
from freqprob import (
    ELE,
    MLE,
    CertaintyDegree,
    Laplace,
    Lidstone,
    Random,
    SimpleGoodTuring,
    Uniform,
    WittenBell,
)

# mypy: disable-error-code=arg-type


# TODO: add tests with different bin values for the Lidstone family

# Set up a bunch of fake distribution to test the methods.
# We don't use real data as this is only intended to test
# the programming side, and we don't want to distribute a
# datafile only for this testing purposes. We set up a simple
# distribution with character from A to E, and a far more
# complex one (which includes almost all printable characters
# as single states, plus a combination of letters and characters,
# all with a randomly generated frequency)
random.seed(1305)

TEST_OBS1 = Counter(list("ABBCCCDDDDEEEE"))

TEST_SAMPLES = list(string.printable)
TEST_TWO_CHAR_SAMPLES = [
    [char1 + char2 for char2 in string.digits] for char1 in string.ascii_letters[:10]
]
TEST_SAMPLES += itertools.chain.from_iterable(TEST_TWO_CHAR_SAMPLES)

TEST_OBS2 = {
    sample: (random.randint(1, 1000) ** random.randint(1, 3)) + random.randint(1, 100)
    for sample in TEST_SAMPLES
}
TEST_OBS2["x"] = 100
TEST_OBS2["y"] = 100
TEST_OBS2["z"] = 1000


def test_uniform_dist_nolog_noobs() -> None:
    """Test the Uniform distribution without logprob and without unobserved states score."""

    scorer1 = Uniform(TEST_OBS1, logprob=False)
    assert scorer1("A") == scorer1("E")
    assert scorer1("A") > scorer1("F")
    assert scorer1("F") == 0.0

    scorer2 = Uniform(TEST_OBS2, logprob=False)
    assert scorer2("0") == scorer2("~")
    assert scorer2("x") == scorer2("z")
    assert scorer2("0") > scorer2("aaa")


def test_uniform_dist_log_noobs() -> None:
    """Test the Uniform distribution with logprob and without unobserved states score."""

    scorer1 = Uniform(TEST_OBS1)
    assert scorer1("A") == scorer1("E")
    assert scorer1("A") > scorer1("F")
    assert scorer1("F") < 0.0

    scorer2 = Uniform(TEST_OBS2)
    assert scorer2("0") == scorer2("~")
    assert scorer2("x") == scorer2("z")
    assert scorer2("0") > scorer2("aaa")


def test_uniform_dist_nolog_obs() -> None:
    """Test the Uniform distribution without logprob and with unobserved states score."""

    scorer1 = Uniform(TEST_OBS1, 0.1, logprob=False)
    assert scorer1("A") == scorer1("E")
    assert scorer1("A") > scorer1("F")
    assert scorer1("F") == 0.1

    scorer2 = Uniform(TEST_OBS2, 0.1, logprob=False)
    assert scorer2("0") == scorer2("~")
    assert scorer2("x") == scorer2("z")
    assert scorer2("0") < scorer2(
        "aaa"
    )  # This is the only difference with the parallel test, due to unobs


def test_uniform_dist_log_obs() -> None:
    """Test the Uniform distribution with logprob and with unobserved states score."""

    scorer1 = Uniform(TEST_OBS1, 0.1)
    assert scorer1("A") == scorer1("E")
    assert scorer1("A") > scorer1("F")
    assert scorer1("F") < 0.0

    scorer2 = Uniform(TEST_OBS2, 0.1)
    assert scorer2("0") == scorer2("~")
    assert scorer2("x") == scorer2("z")
    assert scorer2("0") < scorer2(
        "aaa"
    )  # This is the only difference with the parallel test, due to unobs


def test_uniform_raises() -> None:
    """Test the Uniform distribution raises the correct errors."""

    with pytest.raises(ValueError, match="reserved.*probability.*between"):
        Uniform(TEST_OBS1, -1.0, logprob=False)

    with pytest.raises(ValueError, match="reserved.*probability.*between"):
        Uniform(TEST_OBS1, 100.0, logprob=False)


def test_random_dist_nolog_noobs() -> None:
    """Test the Random distribution without logprob and without unobserved states score."""

    scorer1 = Random(TEST_OBS1, logprob=False, seed=13)
    assert scorer1("A") == pytest.approx(0.25)
    assert scorer1("B") == pytest.approx(0.25)
    assert scorer1("F") == 0.0

    scorer2 = Random(TEST_OBS2, logprob=False, seed=13)
    assert scorer2("0") == pytest.approx(0.002699279041816705)
    assert scorer2("~") == pytest.approx(0.002919468074814642)
    assert scorer2("aaa") == 0.0


def test_random_dist_log_noobs() -> None:
    """Test the Random distribution with logprob and without unobserved states score."""

    scorer1 = Random(TEST_OBS1, seed=13)
    assert scorer1("A") == pytest.approx(-1.386294)
    assert scorer1("B") == pytest.approx(-1.386294)
    assert scorer1("F") == pytest.approx(-23.025850)

    scorer2 = Random(TEST_OBS2, seed=13)
    assert scorer2("0") == pytest.approx(-5.914770)
    assert scorer2("~") == pytest.approx(-5.836353)
    assert scorer2("aaa") == pytest.approx(-23.025850)


def test_random_dist_nolog_obs() -> None:
    """Test the Random distribution without logprob and with unobserved states score."""

    scorer1 = Random(TEST_OBS1, 0.1, logprob=False, seed=13)
    assert scorer1("A") == pytest.approx(0.225)
    assert scorer1("B") == pytest.approx(0.225)
    assert scorer1("F") == pytest.approx(0.1)

    scorer2 = Random(TEST_OBS2, 0.1, logprob=False, seed=13)
    assert scorer2("0") == pytest.approx(0.00242935)
    assert scorer2("~") == pytest.approx(0.00262752)
    assert scorer2("aaa") == pytest.approx(0.1)


def test_random_dist_log_obs() -> None:
    """Test the Random distribution with logprob and with unobserved states score."""

    scorer1 = Random(TEST_OBS1, 0.1, seed=13)
    assert scorer1("A") == pytest.approx(-1.491654)
    assert scorer1("B") == pytest.approx(-1.491654)
    assert scorer1("F") == pytest.approx(-2.302585)

    scorer2 = Random(TEST_OBS2, 0.1, seed=13)
    assert scorer2("0") == pytest.approx(-6.020131)
    assert scorer2("~") == pytest.approx(-5.941714)
    assert scorer2("aaa") == pytest.approx(-2.302585)


def test_random_raises() -> None:
    """Test the Random distribution raises the correct errors."""

    with pytest.raises(ValueError, match="reserved.*probability.*between"):
        Random(TEST_OBS1, -1.0, logprob=False)

    with pytest.raises(ValueError, match="reserved.*probability.*between"):
        Random(TEST_OBS1, 100.0, logprob=False)


def test_mle_dist_nolog_noobs() -> None:
    """Test the MLE distribution without logprob and without unobserved states score."""

    scorer1 = MLE(TEST_OBS1, logprob=False)
    assert scorer1("A") == pytest.approx(0.07142857142857142)
    assert scorer1("B") == pytest.approx(0.14285714285714285)
    assert scorer1("F") == 0.0

    scorer2 = MLE(TEST_OBS2, logprob=False)
    assert scorer2("0") == pytest.approx(1.5671458650393528e-06)
    assert scorer2("~") == pytest.approx(3.949027753538335e-08)
    assert scorer2("aaa") == 0.0


def test_mle_dist_log_noobs() -> None:
    """Test the MLE distribution with logprob and without unobserved states score."""

    scorer1 = MLE(TEST_OBS1)
    assert scorer1("A") == pytest.approx(-2.639057329715259)
    assert scorer1("B") == pytest.approx(-1.9459101491553132)
    assert scorer1("F") == pytest.approx(-23.025850)

    scorer2 = MLE(TEST_OBS2)
    assert scorer2("0") == pytest.approx(-13.366254513483746)
    assert scorer2("~") == pytest.approx(-17.047211333781075)
    assert scorer2("aaa") == pytest.approx(-23.025850)


def test_mle_dist_nolog_obs() -> None:
    """Test the MLE distribution without logprob and with unobserved states score."""

    scorer1 = MLE(TEST_OBS1, 0.1, logprob=False)
    assert scorer1("A") == pytest.approx(0.06428571428571428)
    assert scorer1("B") == pytest.approx(0.128571428571428565)
    assert scorer1("F") == pytest.approx(0.1)

    scorer2 = MLE(TEST_OBS2, 0.1, logprob=False)
    assert scorer2("0") == pytest.approx(1.4104312785354176e-06)
    assert scorer2("~") == pytest.approx(3.554124978184502e-08)
    assert scorer2("aaa") == pytest.approx(0.1)


def test_mle_dist_log_obs() -> None:
    """Test the MLE distribution with logprob and with unobserved states score."""

    scorer1 = MLE(TEST_OBS1, 0.1)
    assert scorer1("A") == pytest.approx(-2.744417845273085)
    assert scorer1("B") == pytest.approx(-2.05127066471314)
    assert scorer1("F") == pytest.approx(-2.302585)

    scorer2 = MLE(TEST_OBS2, 0.1)
    assert scorer2("0") == pytest.approx(-13.471615029041573)
    assert scorer2("~") == pytest.approx(-17.1525718493389)
    assert scorer2("aaa") == pytest.approx(-2.302585)


def test_mle_raises() -> None:
    """Test the MLE distribution raises the correct errors."""

    with pytest.raises(ValueError, match="reserved.*probability.*between"):
        MLE(TEST_OBS1, -1.0, logprob=False)

    with pytest.raises(ValueError, match="reserved.*probability.*between"):
        MLE(TEST_OBS1, 100.0, logprob=False)


def test_lidstone_dist_nolog_noobs() -> None:
    """Test the Lidstone distribution without logprob and without unobserved states score.

    For testing purposes, the Lidstone distributions for the tests are initialized
    with a gamma of 1.5.
    """

    scorer1 = Lidstone(TEST_OBS1, gamma=1.5, logprob=False)
    assert scorer1("A") == pytest.approx(0.11627906976744186)
    assert scorer1("B") == pytest.approx(0.16279069767441862)
    assert scorer1("F") == pytest.approx(0.06976744186046512)

    scorer2 = Lidstone(TEST_OBS2, gamma=1.5, logprob=False)
    assert scorer2("0") == pytest.approx(1.56722247157142e-06)
    assert scorer2("~") == pytest.approx(3.956690748046825e-08)
    assert scorer2("aaa") == pytest.approx(7.663055031724e-11)


def test_lidstone_dist_log_noobs() -> None:
    """Test the Lidstone distribution with logprob and without unobserved states score.

    For testing purposes, the Lidstone distributions for the tests are initialized
    with a gamma of 1.5.
    """

    scorer1 = Lidstone(TEST_OBS1, gamma=1.5)
    assert scorer1("A") == pytest.approx(-2.151762203259462)
    assert scorer1("B") == pytest.approx(-1.8152899666382492)
    assert scorer1("F") == pytest.approx(-2.662587827025453)

    scorer2 = Lidstone(TEST_OBS2, gamma=1.5)
    assert scorer2("0") == pytest.approx(-13.366205631743904)
    assert scorer2("~") == pytest.approx(-17.045272737737683)
    assert scorer2("aaa") == pytest.approx(-23.292025289486443)


def test_lidstone_dist_nolog_obs() -> None:
    """Test the Lidstone distribution without logprob and with unobserved states score.

    For testing purposes, the Lidstone distributions for the tests are initialized
    with a gamma of 1.5.
    """

    scorer1 = Lidstone(TEST_OBS1, gamma=1.5, logprob=False)
    assert scorer1("A") == pytest.approx(0.11627906976744186)
    assert scorer1("B") == pytest.approx(0.16279069767441862)
    assert scorer1("F") == pytest.approx(0.06976744186046512)

    scorer2 = Lidstone(TEST_OBS2, gamma=1.5, logprob=False)
    assert scorer2("0") == pytest.approx(1.56722247157142e-06)
    assert scorer2("~") == pytest.approx(3.956690748046825e-08)
    assert scorer2("aaa") == pytest.approx(7.663055031724e-11)


def test_lidstone_dist_log_obs() -> None:
    """Test the Lidstone distribution with logprob and with unobserved states score.

    For testing purposes, the Lidstone distributions for the tests are initialized
    with a gamma of 1.5.
    """

    scorer1 = Lidstone(TEST_OBS1, gamma=1.5)
    assert scorer1("A") == pytest.approx(-2.151762203259462)
    assert scorer1("B") == pytest.approx(-1.8152899666382492)
    assert scorer1("F") == pytest.approx(-2.662587827025453)

    scorer2 = Lidstone(TEST_OBS2, gamma=1.5)
    assert scorer2("0") == pytest.approx(-13.366205631743904)
    assert scorer2("~") == pytest.approx(-17.045272737737683)
    assert scorer2("aaa") == pytest.approx(-23.292025289486443)


def test_lidstone_raises() -> None:
    """Test the Lidstone distribution raises the correct errors."""

    with pytest.raises(ValueError, match="bins.*positive"):
        Lidstone(TEST_OBS1, gamma=1.0, bins=0)

    with pytest.raises(ValueError, match="Gamma.*non-negative"):
        Lidstone(TEST_OBS1, gamma=-1.0)


def test_laplace_dist_nolog_noobs() -> None:
    """Test the Laplace distribution without logprob and without unobserved states score."""

    scorer1 = Laplace(TEST_OBS1, logprob=False)
    assert scorer1("A") == pytest.approx(0.10526315789473684)
    assert scorer1("B") == pytest.approx(0.15789473684210525)
    assert scorer1("F") == pytest.approx(0.05263157894736842)

    scorer2 = Laplace(TEST_OBS2, logprob=False)
    assert scorer2("0") == pytest.approx(1.5671969360609918e-06)
    assert scorer2("~") == pytest.approx(3.954136416570094e-08)
    assert scorer2("aaa") == pytest.approx(5.1087033805815165e-11)


def test_laplace_dist_log_noobs() -> None:
    """Test the Laplace distribution with logprob and without unobserved states score."""

    scorer1 = Laplace(TEST_OBS1)
    assert scorer1("A") == pytest.approx(-2.2512917986064953)
    assert scorer1("B") == pytest.approx(-1.845826690498331)
    assert scorer1("F") == pytest.approx(-2.9444389791664407)

    scorer2 = Laplace(TEST_OBS2)
    assert scorer2("0") == pytest.approx(-13.366221925358195)
    assert scorer2("~") == pytest.approx(-17.045918518896176)
    assert scorer2("aaa") == pytest.approx(-23.697490392485903)


def test_laplace_dist_nolog_obs() -> None:
    """Test the Laplace distribution without logprob and with unobserved states score."""

    scorer1 = Laplace(TEST_OBS1, logprob=False)
    assert scorer1("A") == pytest.approx(0.10526315789473684)
    assert scorer1("B") == pytest.approx(0.15789473684210525)
    assert scorer1("F") == pytest.approx(0.05263157894736842)

    scorer2 = Laplace(TEST_OBS2, logprob=False)
    assert scorer2("0") == pytest.approx(1.5671969360609918e-06)
    assert scorer2("~") == pytest.approx(3.954136416570094e-08)
    assert scorer2("aaa") == pytest.approx(5.1087033805815165e-11)


def test_laplace_dist_log_obs() -> None:
    """Test the Laplace distribution with logprob and with unobserved states score."""

    scorer1 = Laplace(TEST_OBS1)
    assert scorer1("A") == pytest.approx(-2.2512917986064953)
    assert scorer1("B") == pytest.approx(-1.845826690498331)
    assert scorer1("F") == pytest.approx(-2.9444389791664407)

    scorer2 = Laplace(TEST_OBS2)
    assert scorer2("0") == pytest.approx(-13.366221925358195)
    assert scorer2("~") == pytest.approx(-17.045918518896176)
    assert scorer2("aaa") == pytest.approx(-23.697490392485903)


def test_laplace_raises() -> None:
    """Test the Laplace distribution raises the correct errors."""

    with pytest.raises(ValueError, match="bins.*positive"):
        Laplace(TEST_OBS1, bins=-1)


def test_ele_dist_nolog_noobs() -> None:
    """Test the ELE distribution without logprob and without unobserved states score."""

    scorer1 = ELE(TEST_OBS1, logprob=False)
    assert scorer1("A") == pytest.approx(0.09090909090909091)
    assert scorer1("B") == pytest.approx(0.15151515151515152)
    assert scorer1("F") == pytest.approx(0.030303030303030304)

    scorer2 = ELE(TEST_OBS2, logprob=False)
    assert scorer2("0") == pytest.approx(1.5671714005503027e-06)
    assert scorer2("~") == pytest.approx(3.951582085067264e-08)
    assert scorer2("aaa") == pytest.approx(2.5543517033401835e-11)


def test_ele_dist_log_noobs() -> None:
    """Test the ELE distribution with logprob and without unobserved states score."""

    scorer1 = ELE(TEST_OBS1)
    assert scorer1("A") == pytest.approx(-2.3978952727983707)
    assert scorer1("B") == pytest.approx(-1.8870696490323797)
    assert scorer1("F") == pytest.approx(-3.4965075614664802)

    scorer2 = ELE(TEST_OBS2)
    assert scorer2("0") == pytest.approx(-13.36623821923814)
    assert scorer2("~") == pytest.approx(-17.046564717364078)
    assert scorer2("aaa") == pytest.approx(-24.390637567937144)


def test_ele_dist_nolog_obs() -> None:
    """Test the ELE distribution without logprob and with unobserved states score."""

    scorer1 = ELE(TEST_OBS1, logprob=False)
    assert scorer1("A") == pytest.approx(0.09090909090909091)
    assert scorer1("B") == pytest.approx(0.15151515151515152)
    assert scorer1("F") == pytest.approx(0.030303030303030304)

    scorer2 = ELE(TEST_OBS2, logprob=False)
    assert scorer2("0") == pytest.approx(1.5671714005503027e-06)
    assert scorer2("~") == pytest.approx(3.951582085067264e-08)
    assert scorer2("aaa") == pytest.approx(2.5543517033401835e-11)


def test_ele_dist_log_obs() -> None:
    """Test the ELE distribution with logprob and with unobserved states score."""

    scorer1 = ELE(TEST_OBS1)
    assert scorer1("A") == pytest.approx(-2.3978952727983707)
    assert scorer1("B") == pytest.approx(-1.8870696490323797)
    assert scorer1("F") == pytest.approx(-3.4965075614664802)

    scorer2 = ELE(TEST_OBS2)
    assert scorer2("0") == pytest.approx(-13.36623821923814)
    assert scorer2("~") == pytest.approx(-17.046564717364078)
    assert scorer2("aaa") == pytest.approx(-24.390637567937144)


def test_ele_raises() -> None:
    """Test the ELE distribution raises the correct errors."""

    with pytest.raises(ValueError, match="bins.*positive"):
        ELE(TEST_OBS1, bins=-1)


def test_wb_dist_nolog_noobs() -> None:
    """Test the Witten-Bell distribution without logprob and without unobserved states score."""

    scorer1 = WittenBell(TEST_OBS1, logprob=False)
    assert scorer1("A") == pytest.approx(0.05263157894736842)
    assert scorer1("B") == pytest.approx(0.10526315789473684)
    assert scorer1("F") == pytest.approx(0.2631578947368421)

    scorer2 = WittenBell(TEST_OBS2, logprob=False)
    assert scorer2("0") == pytest.approx(1.5671458490271861e-06)
    assert scorer2("~") == pytest.approx(3.9490277131895124e-08)
    assert scorer2("aaa") == pytest.approx(1.0217406761163034e-08)


def test_wb_dist_log_noobs() -> None:
    """Test the Witten-Bell distribution with logprob and without unobserved states score."""

    scorer1 = WittenBell(TEST_OBS1)
    assert scorer1("A") == pytest.approx(-2.9444389791664407)
    assert scorer1("B") == pytest.approx(-2.2512917986064953)
    assert scorer1("F") == pytest.approx(-1.3350010667323402)

    scorer2 = WittenBell(TEST_OBS2)
    assert scorer2("0") == pytest.approx(-13.366254523601153)
    assert scorer2("~") == pytest.approx(-17.047211343898482)
    assert scorer2("aaa") == pytest.approx(-18.399173025937866)


def test_wb_dist_nolog_obs() -> None:
    """Test the Witten-Bell distribution without logprob and with unobserved states score."""

    scorer1 = WittenBell(TEST_OBS1, logprob=False)
    assert scorer1("A") == pytest.approx(0.05263157894736842)
    assert scorer1("B") == pytest.approx(0.10526315789473684)
    assert scorer1("F") == pytest.approx(0.2631578947368421)

    scorer2 = WittenBell(TEST_OBS2, logprob=False)
    assert scorer2("0") == pytest.approx(1.5671458490271861e-06)
    assert scorer2("~") == pytest.approx(3.9490277131895124e-08)
    assert scorer2("aaa") == pytest.approx(1.0217406761163034e-08)


def test_wb_dist_log_obs() -> None:
    """Test the Witten-Bell distribution with logprob and with unobserved states score."""

    scorer1 = WittenBell(TEST_OBS1)
    assert scorer1("A") == pytest.approx(-2.9444389791664407)
    assert scorer1("B") == pytest.approx(-2.2512917986064953)
    assert scorer1("F") == pytest.approx(-1.3350010667323402)

    scorer2 = WittenBell(TEST_OBS2)
    assert scorer2("0") == pytest.approx(-13.366254523601153)
    assert scorer2("~") == pytest.approx(-17.047211343898482)
    assert scorer2("aaa") == pytest.approx(-18.399173025937866)


def test_wb_raises() -> None:
    """Test the Witten-Bell distribution raises the correct errors."""


def test_cb_dist_nolog_noobs() -> None:
    """Test the Certainty Degree distribution without logprob and without unobserved states score."""

    scorer1 = CertaintyDegree(TEST_OBS1, logprob=False)
    assert scorer1("A") == pytest.approx(0.06586524529838218)
    assert scorer1("B") == pytest.approx(0.13173049059676437)
    assert scorer1("F") == pytest.approx(0.07788656582264941)

    scorer2 = CertaintyDegree(TEST_OBS2, logprob=False)
    assert scorer2("0") == pytest.approx(1.5671458490271861e-06)
    assert scorer2("~") == pytest.approx(3.9490277131895124e-08)
    assert scorer2("aaa") == pytest.approx(-0.0)


def test_cb_dist_log_noobs() -> None:
    """Test the Certainty Degree distribution with logprob and without unobserved states score."""

    scorer1 = CertaintyDegree(TEST_OBS1)
    assert scorer1("A") == pytest.approx(-2.7201443620473227)
    assert scorer1("B") == pytest.approx(-2.0269971814873777)
    assert scorer1("F") == pytest.approx(-2.552501795115364)

    scorer2 = CertaintyDegree(TEST_OBS2)
    assert scorer2("0") == pytest.approx(-13.366254523601153)
    assert scorer2("~") == pytest.approx(-17.047211343898482)
    assert scorer2("aaa") == pytest.approx(-23.02585084720009)


def test_cb_dist_nolog_obs() -> None:
    """Test the Certainty Degree distribution without logprob and with unobserved states score."""

    scorer1 = CertaintyDegree(TEST_OBS1, logprob=False)
    assert scorer1("A") == pytest.approx(0.06586524529838218)
    assert scorer1("B") == pytest.approx(0.13173049059676437)
    assert scorer1("F") == pytest.approx(0.07788656582264941)

    scorer2 = CertaintyDegree(TEST_OBS2, logprob=False)
    assert scorer2("0") == pytest.approx(1.5671458490271861e-06)
    assert scorer2("~") == pytest.approx(3.9490277131895124e-08)
    assert scorer2("aaa") == pytest.approx(-0.0)


def test_cb_dist_log_obs() -> None:
    """Test the Certainty Degree distribution with logprob and with unobserved states score."""

    scorer1 = CertaintyDegree(TEST_OBS1)
    assert scorer1("A") == pytest.approx(-2.7201443620473227)
    assert scorer1("B") == pytest.approx(-2.0269971814873777)
    assert scorer1("F") == pytest.approx(-2.552501795115364)

    scorer2 = CertaintyDegree(TEST_OBS2)
    assert scorer2("0") == pytest.approx(-13.366254523601153)
    assert scorer2("~") == pytest.approx(-17.047211343898482)
    assert scorer2("aaa") == pytest.approx(-23.02585084720009)


def test_cb_raises() -> None:
    """Test the Certainty Degree distribution raises the correct errors."""


def test_sgt_dist_nolog_noobs() -> None:
    """Test the Simple Good-Turing distribution without logprob and without unobserved states score."""

    # The second distribution does not have enough data for confident results,
    # but it has enough to stress-test the method
    with pytest.raises(RuntimeWarning):
        scorer = SimpleGoodTuring(TEST_OBS2, logprob=False, allow_fail=False)

    # Test with allow_fail=True to demonstrate SGT can handle poor data gracefully
    scorer = SimpleGoodTuring(TEST_OBS2, logprob=False, allow_fail=True)
    # Just verify it returns reasonable probabilities
    assert scorer("0") > 0
    assert scorer("~") > 0
    assert scorer("aaa") > 0


def test_sgt_dist_log_noobs() -> None:
    """Test the Simple Good-Turing distribution with logprob and without unobserved states score."""

    # The second distribution does not have enough data for confident results,
    # but it has enough to stress-test the method
    with pytest.raises(RuntimeWarning):
        scorer = SimpleGoodTuring(TEST_OBS2, allow_fail=False)

    # Test with allow_fail=True to demonstrate SGT can handle poor data gracefully
    scorer = SimpleGoodTuring(TEST_OBS2, allow_fail=True)
    # Just verify it returns reasonable log probabilities (negative values)
    assert scorer("0") < 0
    assert scorer("~") < 0
    assert scorer("aaa") < 0


def test_sgt_dist_nolog_obs() -> None:
    """Test the Simple Good-Turing distribution without logprob and with unobserved states score."""

    # The second distribution does not have enough data for confident results,
    # but it has enough to stress-test the method
    with pytest.raises(RuntimeWarning):
        scorer = SimpleGoodTuring(TEST_OBS2, logprob=False, allow_fail=False)

    # Test with allow_fail=True to demonstrate SGT can handle poor data gracefully
    scorer = SimpleGoodTuring(TEST_OBS2, logprob=False, allow_fail=True)
    # Just verify it returns reasonable probabilities
    assert scorer("0") > 0
    assert scorer("~") > 0
    assert scorer("aaa") > 0


def test_sgt_dist_log_obs() -> None:
    """Test the Simple Good-Turing distribution with logprob and with unobserved states score."""

    # The second distribution does not have enough data for confident results,
    # but it has enough to stress-test the method
    with pytest.raises(RuntimeWarning):
        scorer = SimpleGoodTuring(TEST_OBS2, allow_fail=False)

    # Test with allow_fail=True to demonstrate SGT can handle poor data gracefully
    scorer = SimpleGoodTuring(TEST_OBS2, allow_fail=True)
    # Just verify it returns reasonable log probabilities (negative values)
    assert scorer("0") < 0
    assert scorer("~") < 0
    assert scorer("aaa") < 0


def test_sgt_raises() -> None:
    """Test the Simple Good-Turing distribution raises the correct errors."""

    with pytest.raises(RuntimeWarning):
        SimpleGoodTuring(TEST_OBS1)
