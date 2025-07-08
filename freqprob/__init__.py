"""FreqProb: frequency-based probability estimation library."""

# Version of the freqprob package
__version__ = "0.2.0"
__author__ = "Tiago Tresoldi"
__email__ = "tiago.tresoldi@lingfil.uu.se"


from .advanced import CertaintyDegree, SimpleGoodTuring, WittenBell

# Import from local modules
from .basic import MLE, Random, Uniform
from .cache import clear_all_caches, get_cache_stats
from .lazy import LazyBatchScorer, LazyScoringMethod, create_lazy_laplace, create_lazy_mle
from .lidstone import ELE, Laplace, Lidstone
from .memory_efficient import (
    CompressedFrequencyDistribution,
    QuantizedProbabilityTable,
    SparseFrequencyDistribution,
    create_compressed_distribution,
    create_sparse_distribution,
)
from .profiling import DistributionMemoryAnalyzer, MemoryMonitor, MemoryProfiler
from .smoothing import BayesianSmoothing, InterpolatedSmoothing, KneserNey, ModifiedKneserNey
from .streaming import (
    StreamingDataProcessor,
    StreamingFrequencyDistribution,
    StreamingLaplace,
    StreamingMLE,
)
from .utils import (
    cross_entropy,
    generate_ngrams,
    kl_divergence,
    model_comparison,
    ngram_frequency,
    perplexity,
    word_frequency,
)
from .vectorized import BatchScorer, VectorizedScorer, create_vectorized_batch_scorer

# Validation and profiling tools (optional import)
try:
    from .validation import (
        BenchmarkSuite,
        PerformanceMetrics,
        PerformanceProfiler,
        ValidationResult,
        ValidationSuite,
        compare_method_performance,
        profile_method_performance,
        quick_validate_method,
    )

    HAS_VALIDATION = True
except ImportError:
    HAS_VALIDATION = False

# Build the namespace
__all__ = [
    "ELE",
    "MLE",
    "BatchScorer",
    "BayesianSmoothing",
    "CertaintyDegree",
    "CompressedFrequencyDistribution",
    "DistributionMemoryAnalyzer",
    "InterpolatedSmoothing",
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

# Add validation tools to __all__ if available
if HAS_VALIDATION:
    __all__ += [
        "BenchmarkSuite",
        "PerformanceMetrics",
        "PerformanceProfiler",
        "ValidationResult",
        "ValidationSuite",
        "compare_method_performance",
        "profile_method_performance",
        "quick_validate_method",
    ]
