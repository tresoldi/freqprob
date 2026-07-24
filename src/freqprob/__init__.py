"""FreqProb: frequency-based probability estimation library."""

# Version of the freqprob package
__version__ = "0.4.0"
__author__ = "Tiago Tresoldi"
__email__ = "tiago.tresoldi@lingfil.uu.se"


# Core estimators, grouped by family under methods/
from .cache import clear_all_caches, get_cache_stats
from .methods.additive import ELE, Laplace, Lidstone
from .methods.baselines import MLE, Random, Uniform
from .methods.bayesian import Bayesian
from .methods.certainty import CertaintyDegree
from .methods.goodturing import SimpleGoodTuring, WittenBell
from .methods.interpolated import Interpolated
from .methods.kneser_ney import KneserNey, ModifiedKneserNey

# Model-evaluation metrics and (NLP) text helpers
from .metrics import cross_entropy, kl_divergence, model_comparison, perplexity

# Performance infrastructure (lazy/streaming/vectorized/memory/profiling)
from .performance.lazy import (
    LazyBatchScorer,
    LazyScoringMethod,
    create_lazy_laplace,
    create_lazy_mle,
)
from .performance.memory_efficient import (
    CompressedFrequencyDistribution,
    QuantizedProbabilityTable,
    SparseFrequencyDistribution,
    create_compressed_distribution,
    create_sparse_distribution,
)
from .performance.profiling import DistributionMemoryAnalyzer, MemoryMonitor, MemoryProfiler
from .performance.streaming import (
    StreamingDataProcessor,
    StreamingFrequencyDistribution,
    StreamingLaplace,
    StreamingMLE,
)
from .performance.vectorized import BatchScorer, VectorizedScorer, create_vectorized_batch_scorer
from .text import generate_ngrams, ngram_frequency, word_frequency

# Build the namespace
__all__ = [
    "ELE",
    "MLE",
    "BatchScorer",
    "Bayesian",
    "CertaintyDegree",
    "CompressedFrequencyDistribution",
    "DistributionMemoryAnalyzer",
    "Interpolated",
    "KneserNey",
    "Laplace",
    "LazyBatchScorer",
    "LazyScoringMethod",
    "Lidstone",
    "MemoryMonitor",
    "MemoryProfiler",
    "ModifiedKneserNey",
    "QuantizedProbabilityTable",
    "Random",
    "SimpleGoodTuring",
    "SparseFrequencyDistribution",
    "StreamingDataProcessor",
    "StreamingFrequencyDistribution",
    "StreamingLaplace",
    "StreamingMLE",
    "Uniform",
    "VectorizedScorer",
    "WittenBell",
    "clear_all_caches",
    "create_compressed_distribution",
    "create_lazy_laplace",
    "create_lazy_mle",
    "create_sparse_distribution",
    "create_vectorized_batch_scorer",
    "cross_entropy",
    "generate_ngrams",
    "get_cache_stats",
    "kl_divergence",
    "model_comparison",
    "ngram_frequency",
    "perplexity",
    "word_frequency",
]
