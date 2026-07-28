"""Execute every docstring example in the package as a doctest.

The API reference is generated from docstrings (mkdocstrings), so their code
examples are user-facing documentation. Running them here in the normal test
suite guarantees they stay correct against the installed code — an example can
never silently rot the way the old hand-written docs did.

This collects and runs the ``>>>`` examples in every ``freqprob`` submodule.
For a readable per-line report when something fails, run:

    python -m pytest src/ --doctest-modules
"""

from __future__ import annotations

import doctest
import importlib
import pkgutil

import pytest

import freqprob


def _iter_module_names() -> list[str]:
    """Return the import paths of every submodule under ``freqprob``."""
    names = [freqprob.__name__]
    for info in pkgutil.walk_packages(freqprob.__path__, prefix=f"{freqprob.__name__}."):
        names.append(info.name)
    return sorted(names)


@pytest.mark.parametrize("module_name", _iter_module_names())
def test_module_docstring_examples(module_name: str) -> None:
    """Every ``>>>`` example in the module must run and match its output."""
    module = importlib.import_module(module_name)
    results = doctest.testmod(module, verbose=False, report=False)
    assert results.failed == 0, (
        f"{results.failed} doctest example(s) failed in {module_name}. "
        f"Reproduce with: python -m pytest --doctest-modules "
        f"{module_name.replace('.', '/')}.py"
    )
