"""Run the documentation's examples, so they stay correct as the code changes.

Every ``pycon`` block in ``docs/**/*.md`` is executed as a doctest, and its
printed results must match the page. The blocks of one page share a namespace
and run in order, so a page can build on its earlier examples, as a reader
would in an interactive session. A page whose examples need matplotlib is
skipped when it is not installed. The README has its own test,
``tests/test_readme.py``.
"""

import doctest
import importlib.util
import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
BLOCK = re.compile(r"^```pycon\n(.*?)^```", flags=re.MULTILINE | re.DOTALL)
PAGES = sorted(
    page for page in DOCS.rglob("*.md") if BLOCK.search(page.read_text(encoding="utf-8"))
)


def test_docs_have_examples() -> None:
    """The documentation keeps runnable examples."""
    assert PAGES, "docs/ should contain at least one ```pycon example"


@pytest.mark.parametrize("page", PAGES, ids=lambda page: str(page.relative_to(DOCS)))
def test_docs_page_examples(page: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each page's examples run and print exactly what the page shows."""
    text = page.read_text(encoding="utf-8")
    if "matplotlib" in text and importlib.util.find_spec("matplotlib") is None:
        pytest.skip("matplotlib is not installed")

    # Examples that write files do so in a fresh temporary directory.
    monkeypatch.chdir(tmp_path)

    namespace: dict[str, object] = {}
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    parser = doctest.DocTestParser()
    for index, match in enumerate(BLOCK.finditer(text), start=1):
        line = text.count("\n", 0, match.start(1))
        name = f"{page.relative_to(DOCS)} block {index}"
        test = parser.get_doctest(match.group(1), namespace, name, str(page), line)
        runner.run(test, clear_globs=False)
        namespace = test.globs  # DocTest copies its globals; carry them to the next block

    assert runner.failures == 0, f"{page.relative_to(DOCS)}: see the doctest report above"
