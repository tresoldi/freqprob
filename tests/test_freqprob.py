"""
Test the library functions.
"""

# Import Python standard libraries
import itertools
import random
import string
from collections import Counter

# Import 3rd-party libraries
import pytest

# Import the library to test
from freqprob import Uniform

# Set up a bunch of fake distribution to test the methods.
# We don't use real data as this is only intended to test
# the programming side, and we don't want to distribute a
# datafile only for this testing purposes. We set up a simple
# distribution with character from A to E, and a far more
# complex one (which includes almost all printable characters
# as single states, plus a combination of letters and characters,
# all with a randomly generated frequency)
random.seed(1305)

TEST_OBS1 = Counter([char for char in "ABBCCCDDDDEEEE"])

TEST_SAMPLES = [char for char in string.printable]
TEST_TWO_CHAR_SAMPLES = [[char1 + char2 for char2 in string.digits] for char1 in string.ascii_letters[:10]]
TEST_SAMPLES += itertools.chain.from_iterable(TEST_TWO_CHAR_SAMPLES)

TEST_OBS2 = {
    sample: (random.randint(1, 1000) ** random.randint(1, 3)) + random.randint(1, 100) for sample in TEST_SAMPLES
}
TEST_OBS2["x"] = 100
TEST_OBS2["y"] = 100
TEST_OBS2["z"] = 1000


def test_uniform_dist_nolog_noobs():
    """
    Test the Uniform distribution without logprob and without unobserved states score.
    """

    scorer1 = Uniform(TEST_OBS1, logprob=False)
    assert scorer1("A") == scorer1("E")
    assert scorer1("A") > scorer1("F")
    assert scorer1("F") == 0.0

    scorer2 = Uniform(TEST_OBS2, logprob=False)
    assert scorer2("0") == scorer2("~")
    assert scorer2("x") == scorer2("z")
    assert scorer2("0") > scorer2("aaa")


def test_uniform_dist_log_noobs():
    """
    Test the Uniform distribution with logprob and without unobserved states score.
    """

    scorer1 = Uniform(TEST_OBS1)
    assert scorer1("A") == scorer1("E")
    assert scorer1("A") > scorer1("F")
    assert scorer1("F") < 0.0

    scorer2 = Uniform(TEST_OBS2)
    assert scorer2("0") == scorer2("~")
    assert scorer2("x") == scorer2("z")
    assert scorer2("0") > scorer2("aaa")


def test_uniform_dist_nolog_obs():
    """
    Test the Uniform distribution without logprob and with unobserved states score.
    """

    scorer1 = Uniform(TEST_OBS1, 0.1, logprob=False)
    assert scorer1("A") == scorer1("E")
    assert scorer1("A") > scorer1("F")
    assert scorer1("F") == 0.1

    scorer2 = Uniform(TEST_OBS2, 0.1, logprob=False)
    assert scorer2("0") == scorer2("~")
    assert scorer2("x") == scorer2("z")
    assert scorer2("0") < scorer2("aaa")  # This is the only difference with the parallel test, due to unobs


def test_uniform_dist_log_obs():
    """
    Test the Uniform distribution with logprob and with unobserved states score.
    """

    scorer1 = Uniform(TEST_OBS1, 0.1)
    assert scorer1("A") == scorer1("E")
    assert scorer1("A") > scorer1("F")
    assert scorer1("F") < 0.0

    scorer2 = Uniform(TEST_OBS2, 0.1)
    assert scorer2("0") == scorer2("~")
    assert scorer2("x") == scorer2("z")
    assert scorer2("0") < scorer2("aaa")  # This is the only difference with the parallel test, due to unobs


def test_uniform_raises():
    """
    Test the Uniform distribution raises the correct errors.
    """

    with pytest.raises(ValueError):
        Uniform(TEST_OBS1, -1.0, logprob=False)

    with pytest.raises(ValueError):
        Uniform(TEST_OBS1, 100.0, logprob=False)
