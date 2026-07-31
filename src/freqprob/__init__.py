"""FreqProb: frequency-based probability estimation library."""

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
    "AbsoluteDiscounting",
    "BatchScorer",
    "Bayesian",
    "CertaintyDegree",
    "CompressedFrequencyDistribution",
    "DistributionMemoryAnalyzer",
    "Interpolated",
    "JelinekMercer",
    "KatzBackoff",
    "KneserNey",
    "Laplace",
    "LazyBatchScorer",
    "LazyScoringMethod",
    "Lidstone",
    "MemoryMonitor",
    "MemoryProfiler",
    "ModifiedKneserNey",
    "PitmanYor",
    "QuantizedProbabilityTable",
    "Random",
    "ScoringMethod",
    "SimpleGoodTuring",
    "SparseFrequencyDistribution",
    "StreamingDataProcessor",
    "StreamingFrequencyDistribution",
    "StreamingLaplace",
    "StreamingMLE",
    "StupidBackoff",
    "Uniform",
    "VectorizedScorer",
    "WittenBell",
    "chao_shen_entropy",
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
    "nsb_entropy",
    "perplexity",
    "sample_coverage",
    "word_frequency",
]
