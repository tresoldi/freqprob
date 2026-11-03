# How to cite

If FreqProb contributes to work you publish, please cite the software and the
version you used. Results can change between versions (see the
[Changelog](https://github.com/tresoldi/freqprob/blob/main/CHANGELOG.md)), so
the version number is what lets a reader reproduce your numbers. Check it with:

```python
import freqprob

print(freqprob.__version__)
```

## Citing the software

**APA**

> Tresoldi, T. (2026). *FreqProb: A Python library for probability smoothing and
> frequency-based estimation* (Version 0.8.0) [Computer software].
> https://freqprob.tresoldi.org/

**BibTeX** (biblatex, or any style that supports `@software`)

```bibtex
@software{tresoldi_freqprob,
  author  = {Tresoldi, Tiago},
  title   = {{FreqProb}: A {Python} library for probability smoothing and
             frequency-based estimation},
  year    = {2026},
  version = {0.8.0},
  url     = {https://freqprob.tresoldi.org/}
}
```

**BibTeX** (classic styles without `@software`, such as `plain` or most journal
templates)

```bibtex
@misc{tresoldi_freqprob,
  author       = {Tresoldi, Tiago},
  title        = {{FreqProb}: A {Python} library for probability smoothing and
                  frequency-based estimation},
  year         = {2026},
  note         = {Version 0.8.0},
  howpublished = {\url{https://freqprob.tresoldi.org/}}
}
```

Replace the version and year with those of the release you used. The citation
metadata also lives in
[`CITATION.cff`](https://github.com/tresoldi/freqprob/blob/main/CITATION.cff);
GitHub's **Cite this repository** button on the repository page builds APA and
BibTeX entries from it.

## Citing the methods

FreqProb implements published methods. When a result depends on a specific
estimator, such as Simple Good–Turing or the Chao–Shen entropy estimator, cite
the method's original paper as well as the software. The
[References](references.md) page lists the source for each method and provides
them all as a BibTeX file.
