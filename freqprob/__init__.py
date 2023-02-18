"""
freqprob __init__.py
"""

# Version of the ngesh package
__version__ = "0.1.0"  # Remember to sync with setup.py
__author__ = "Tiago Tresoldi"
__email__ = "tiago.tresoldi@lingfil.uu.se"


# Import from local modules
from .freqprob import ELE, MLE, Laplace, Lidstone, Random, Uniform, WittenBell

# Build the namespace
__all__ = ["Uniform", "Random", "MLE", "Lidstone", "Laplace", "ELE", "WittenBell"]
