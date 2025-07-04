"""
freqprob __init__.py
"""

# Version of the ngesh package
__version__ = "0.1.0"  # Remember to sync with setup.py
__author__ = "Tiago Tresoldi"
__email__ = "tiago.tresoldi@lingfil.uu.se"


# Import from local modules
from .basic import MLE, Random, Uniform
from .lidstone import ELE, Laplace, Lidstone
from .advanced import CertaintyDegree, SimpleGoodTuring, WittenBell
from .smoothing import BayesianSmoothing, InterpolatedSmoothing, KneserNey, ModifiedKneserNey
from .utils import (
    generate_ngrams, word_frequency, ngram_frequency, 
    perplexity, cross_entropy, kl_divergence, model_comparison
)
from .cache import clear_all_caches, get_cache_stats
from .vectorized import VectorizedScorer, BatchScorer, create_vectorized_batch_scorer
from .lazy import LazyScoringMethod, LazyBatchScorer, create_lazy_mle, create_lazy_laplace
from .streaming import (
    StreamingFrequencyDistribution, StreamingMLE, StreamingLaplace, 
    StreamingDataProcessor
)
from .memory_efficient import (
    CompressedFrequencyDistribution, SparseFrequencyDistribution, 
    QuantizedProbabilityTable, create_compressed_distribution, create_sparse_distribution
)
from .profiling import MemoryProfiler, DistributionMemoryAnalyzer, MemoryMonitor

# Build the namespace
__all__ = [
    "Uniform",
    "Random",
    "MLE",
    "Lidstone",
    "Laplace",
    "ELE",
    "WittenBell",
    "CertaintyDegree",
    "SimpleGoodTuring",
    "KneserNey",
    "ModifiedKneserNey",
    "InterpolatedSmoothing",
    "BayesianSmoothing",
    "generate_ngrams",
    "word_frequency",
    "ngram_frequency",
    "perplexity",
    "cross_entropy",
    "kl_divergence",
    "model_comparison",
    "clear_all_caches",
    "get_cache_stats",
    "VectorizedScorer",
    "BatchScorer",
    "create_vectorized_batch_scorer",
    "LazyScoringMethod",
    "LazyBatchScorer",
    "create_lazy_mle",
    "create_lazy_laplace",
    "StreamingFrequencyDistribution",
    "StreamingMLE", 
    "StreamingLaplace",
    "StreamingDataProcessor",
    "CompressedFrequencyDistribution",
    "SparseFrequencyDistribution",
    "QuantizedProbabilityTable",
    "create_compressed_distribution",
    "create_sparse_distribution",
    "MemoryProfiler",
    "DistributionMemoryAnalyzer",
    "MemoryMonitor",
]
