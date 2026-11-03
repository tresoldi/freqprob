# Installation

FreqProb supports Python 3.10 to 3.12 and installs from PyPI:

```bash
pip install freqprob
```

It depends only on NumPy and SciPy. The `memory` extra adds `psutil`, used by
the memory profiling tools in `freqprob.performance.profiling`:

```bash
pip install "freqprob[memory]"
```

To check the installation and the version:

```pycon
>>> import freqprob
>>> isinstance(freqprob.__version__, str)
True
```

Report the version when you [cite FreqProb](../citing.md): results can change
between versions, as the
[Changelog](https://github.com/tresoldi/freqprob/blob/main/CHANGELOG.md) and the
[Migration guide](https://github.com/tresoldi/freqprob/blob/main/MIGRATION.md)
explain.

## From source

To work on FreqProb itself, clone the repository and install it in editable
mode with the development tools:

```bash
git clone https://github.com/tresoldi/freqprob.git
cd freqprob
pip install -e ".[dev]"
pytest
```

Next: the [Tutorial](tutorial.md).
