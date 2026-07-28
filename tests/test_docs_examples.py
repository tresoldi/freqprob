"""Execute every ``python`` code block in the prose docs.

The README and the docs site are full of runnable examples. This extracts each
fenced ``python`` block from those Markdown files and executes it, so an example
that references a renamed or non-existent API fails the test suite instead of
shipping. (This is the failure mode that rotted the old hand-written docs.)

Conventions for doc authors:
* Each ``python`` block must run on its own in a fresh namespace — import what
  it needs. Blocks do not share state.
* A block that is deliberately illustrative / not meant to run can opt out with
  a first-line marker comment: ``# docs-test: skip``.

For a specific failure, the test id names the file and the 1-based block index.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DOC_FILES = [
    _REPO_ROOT / "README.md",
    _REPO_ROOT / "docs" / "index.md",
    _REPO_ROOT / "docs" / "USER_GUIDE.md",
]

_FENCE = re.compile(r"^```python\b[^\n]*\n(.*?)^```", re.MULTILINE | re.DOTALL)


def _extract_blocks(path: Path) -> list[str]:
    """Return the ``python`` fenced code blocks in a Markdown file."""
    if not path.exists():
        return []
    return _FENCE.findall(path.read_text(encoding="utf-8"))


def _collect() -> list[tuple[str, int, str]]:
    """Collect (file-label, 1-based index, code) for every runnable block."""
    cases: list[tuple[str, int, str]] = []
    for path in _DOC_FILES:
        for i, block in enumerate(_extract_blocks(path), start=1):
            if "# docs-test: skip" in block.splitlines()[0:1] or "# docs-test: skip" in block:
                continue
            cases.append((path.relative_to(_REPO_ROOT).as_posix(), i, block))
    return cases


_CASES = _collect()


@pytest.mark.parametrize(
    ("doc_file", "index", "code"),
    _CASES,
    ids=[f"{f}#block{i}" for f, i, _ in _CASES],
)
def test_doc_code_block_runs(doc_file: str, index: int, code: str) -> None:
    """Each python block in the docs must execute without raising."""
    namespace: dict[str, object] = {"__name__": "__doc_example__"}
    try:
        exec(compile(code, f"{doc_file}#block{index}", "exec"), namespace)
    except Exception as exc:
        pytest.fail(f"{doc_file} block {index} failed to run: {type(exc).__name__}: {exc}")
