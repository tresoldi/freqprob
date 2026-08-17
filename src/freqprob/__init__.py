"""FreqProb: frequency-based probability estimation library.

The public namespace is the library's compatibility promise, so it holds the
domain — the estimators, the metrics you evaluate them with, and the text
helpers that build a frequency distribution — plus the handful of performance
adapters that a caller reaches for directly.

Development and diagnostic tooling is deliberately *not* re-exported. It is
still there, one import deeper, where its being infrastructure is visible:

    from freqprob.performance.profiling import MemoryProfiler
    from freqprob.performance.streaming import StreamingDataProcessor
"""

# Version of the freqprob package
__version__ = "0.7.0"
__author__ = "Tiago Tresoldi"
__email__ = "freqprob@tresoldi.org"


# Base type (useful for isinstance checks and type annotations)
from .base import ScoringMethod

# Core estimators, grouped by family under methods/
from .cache import clear_all_caches, get_cache_stats
from .methods.additive import ELE, Laplace, Lidstone
from .methods.backoff import KatzBackoff, StupidBackoff
from .methods.baselines import MLE, Random, Uniform
from .methods.bayesian import Bayesian
from .methods.certainty import CertaintyDegree
from .methods.discounting import AbsoluteDiscounting, PitmanYor
from .methods.goodturing import SimpleGoodTuring, WittenBell
from .methods.interpolated import Interpolated, JelinekMercer
from .methods.kneser_ney import KneserNey, ModifiedKneserNey

# Model-evaluation metrics and (NLP) text helpers
from .metrics import (
    chao_shen_entropy,
    cross_entropy,
    kl_divergence,
    model_comparison,
    nsb_entropy,
    perplexity,
    sample_coverage,
)

# Performance adapters over the estimator contract, and the compact
# representations a large distribution can be held in.
from .performance.lazy import create_lazy_laplace, create_lazy_mle
from .performance.memory_efficient import (
    CompressedFrequencyDistribution,
    QuantizedProbabilityTable,
    SparseFrequencyDistribution,
)
from .performance.streaming import (
    StreamingFrequencyDistribution,
    StreamingLaplace,
    StreamingMLE,
)
from .performance.vectorized import BatchScorer, VectorizedScorer
from .text import generate_ngrams, ngram_frequency, word_frequency

# Build the namespace
__all__ = [
    "ELE",
    "MLE",
    "AbsoluteDiscounting",
    "BatchScorer",
    "Bayesian",
    "CertaintyDegree",
    "CompressedFrequencyDistribution",
    "Interpolated",
    "JelinekMercer",
    "KatzBackoff",
    "KneserNey",
    "Laplace",
    "Lidstone",
    "ModifiedKneserNey",
    "PitmanYor",
    "QuantizedProbabilityTable",
    "Random",
    "ScoringMethod",
    "SimpleGoodTuring",
    "SparseFrequencyDistribution",
    "StreamingFrequencyDistribution",
    "StreamingLaplace",
    "StreamingMLE",
    "StupidBackoff",
    "Uniform",
    "VectorizedScorer",
    "WittenBell",
    "chao_shen_entropy",
    "clear_all_caches",
    "create_lazy_laplace",
    "create_lazy_mle",
    "cross_entropy",
    "generate_ngrams",
    "get_cache_stats",
    "kl_divergence",
    "model_comparison",
    "ngram_frequency",
    "nsb_entropy",
    "perplexity",
    "sample_coverage",
    "word_frequency",
]
