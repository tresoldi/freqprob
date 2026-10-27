"""What the top-level namespace promises.

`__all__` is the library's compatibility surface, and nothing asserted anything
about it: names could be added, removed, or left unreachable without a test
noticing. It had grown to fifty names, of which twenty-one were performance and
diagnostic infrastructure rather than the domain — including one class that
could not be constructed through the public API at all, because the config type
its constructor required was not exported.

These tests hold the surface to three rules: every promised name resolves, every
name is reachable using only what is exported, and development tooling stays one
import deeper.
"""

import importlib
import inspect

import pytest

import freqprob
from freqprob.base import ScoringMethod

#: The estimators. These are the library's reason to exist.
ESTIMATORS = {
    "AbsoluteDiscounting",
    "Bayesian",
    "CertaintyDegree",
    "ELE",
    "Interpolated",
    "JelinekMercer",
    "KatzBackoff",
    "KneserNey",
    "Laplace",
    "Lidstone",
    "MLE",
    "ModifiedKneserNey",
    "PitmanYor",
    "Random",
    "SimpleGoodTuring",
    "StreamingLaplace",
    "StreamingMLE",
    "StupidBackoff",
    "Uniform",
    "WittenBell",
}

#: Development and diagnostic tooling, deliberately not re-exported.
DEMOTED = {
    "freqprob.performance.profiling": [
        "MemoryProfiler",
        "MemoryMonitor",
        "DistributionMemoryAnalyzer",
    ],
    "freqprob.performance.streaming": ["StreamingDataProcessor"],
    "freqprob.performance.lazy": ["LazyScoringMethod"],
    "freqprob.cache": ["ComputationCache"],
}

#: Names removed as shallow or unreachable. Each was a pass-through, a false
#: promise, or had no caller anywhere in the library.
REMOVED = [
    "LazyBatchScorer",
    "create_vectorized_batch_scorer",
    "create_compressed_distribution",
    "create_sparse_distribution",
    "memory_usage_comparison",
    "MemoizedProperty",
]


class TestEveryPromisedNameResolves:
    """`__all__` and the module agree."""

    @pytest.mark.parametrize("name", sorted(freqprob.__all__))
    def test_name_is_importable(self, name: str) -> None:
        """A name in ``__all__`` is an attribute of the package."""
        assert hasattr(freqprob, name), f"{name} is promised but missing"

    def test_all_has_no_duplicates(self) -> None:
        """A name promised twice is a merge accident; ruff RUF022 keeps it sorted."""
        assert len(freqprob.__all__) == len(set(freqprob.__all__))

    def test_star_import_matches_all(self) -> None:
        """``from freqprob import *`` yields exactly the promised names."""
        namespace: dict[str, object] = {}
        exec("from freqprob import *", namespace)
        exported = {k for k in namespace if not k.startswith("__")}

        assert exported == set(freqprob.__all__)


class TestTheSurfaceIsTheDomain:
    """The namespace reads as a library about frequency counts."""

    def test_every_estimator_is_exported(self) -> None:
        """No estimator is hidden away."""
        assert set(freqprob.__all__) >= ESTIMATORS

    def test_estimators_are_the_largest_group(self) -> None:
        """Infrastructure no longer outnumbers the domain."""
        exported = set(freqprob.__all__)
        infrastructure = {
            name
            for name in exported
            if inspect.getmodule(getattr(freqprob, name))
            and "performance" in (inspect.getmodule(getattr(freqprob, name)).__name__ or "")
        }

        assert len(infrastructure) < len(ESTIMATORS)

    def test_exported_estimators_share_one_contract(self) -> None:
        """Every exported estimator is a ``ScoringMethod``."""
        for name in ESTIMATORS:
            assert issubclass(getattr(freqprob, name), ScoringMethod)


class TestEveryExportIsReachable:
    """A promised name can be used with only what is promised."""

    def test_no_export_needs_an_unexported_type_to_construct(self) -> None:
        """``LazyScoringMethod`` was exported but needed the unexported config type."""
        exported = set(freqprob.__all__)

        for name in exported:
            obj = getattr(freqprob, name)
            if not inspect.isclass(obj) or not issubclass(obj, ScoringMethod):
                continue
            if inspect.isabstract(obj):
                continue  # exported for isinstance and annotations, not construction
            required = [
                parameter
                for parameter in inspect.signature(obj.__init__).parameters.values()
                if parameter.name != "self" and parameter.default is inspect.Parameter.empty
            ]
            for parameter in required:
                annotation = str(parameter.annotation)
                assert "ScoringMethodConfig" not in annotation, (
                    f"{name} requires an unexported type to construct"
                )


class TestDemotedToolingStaysReachable:
    """Demoting a name removes the promise, not the capability."""

    @pytest.mark.parametrize(
        ("module_name", "attribute"),
        [(module, attr) for module, attrs in DEMOTED.items() for attr in attrs],
    )
    def test_demoted_name_is_importable_from_its_module(
        self, module_name: str, attribute: str
    ) -> None:
        """It moved one import deeper, where being infrastructure is visible."""
        module = importlib.import_module(module_name)
        assert hasattr(module, attribute)

    @pytest.mark.parametrize("name", sorted({attr for attrs in DEMOTED.values() for attr in attrs}))
    def test_demoted_name_is_not_re_exported(self, name: str) -> None:
        """And is no longer part of the compatibility promise."""
        assert name not in freqprob.__all__


class TestRemovedNamesAreGone:
    """A deleted pass-through does not linger in the namespace."""

    @pytest.mark.parametrize("name", REMOVED)
    def test_removed_name_is_absent(self, name: str) -> None:
        """Nothing re-introduces it by accident."""
        assert name not in freqprob.__all__
        assert not hasattr(freqprob, name)

    def test_the_named_constructors_replace_the_factories(self) -> None:
        """The `create_*` factories duplicated a constructor; `from_counts` is it."""
        compact = freqprob.CompressedFrequencyDistribution.from_counts({"a": 10, "b": 5})
        sparse = freqprob.SparseFrequencyDistribution.from_counts({"a": 10})

        assert compact.get_count("a") == 10
        assert compact.get_vocabulary_size() == 2
        assert sparse.get_count("a") == 10
        assert sparse.get_count("never-added") == 0
