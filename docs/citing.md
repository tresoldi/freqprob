# How to cite

If FreqProb contributes to work you publish, please cite the software and the
version you used. Results can change between versions (see the
[Changelog](https://github.com/tresoldi/freqprob/blob/main/CHANGELOG.md)), so
the version number is what lets a reader reproduce your numbers. Check it with:

```python
import freqprob

print(freqprob.__version__)
```

## DOIs

Every release of FreqProb from 0.8.1 onwards is archived on
[Zenodo](https://doi.org/10.5281/zenodo.23917393) with its own DOI:

- **A specific version:** cite the DOI of the release you used, so readers get
  exactly that code. Version 0.8.1 is
  [10.5281/zenodo.23917394](https://doi.org/10.5281/zenodo.23917394). Later versions are listed under "Versions" on the
  Zenodo record.
- **All versions:** [10.5281/zenodo.23917393](https://doi.org/10.5281/zenodo.23917393) always resolves to the latest
  release. Use it when you refer to FreqProb in general rather than to results
  from a particular version.

Releases before 0.8.1 have no DOI; cite them by version number and URL.

## Citing the software

**APA**

> Tresoldi, T. (2026). *FreqProb: A Python library for probability smoothing and
> frequency-based estimation* (Version 0.8.1) [Computer software]. Zenodo.
> https://doi.org/10.5281/zenodo.23917394

**BibTeX** (biblatex, or any style that supports `@software`)

```bibtex
@software{tresoldi_freqprob,
  author  = {Tresoldi, Tiago},
  title   = {{FreqProb}: A {Python} library for probability smoothing and
             frequency-based estimation},
  year    = {2026},
  version = {0.8.1},
  doi     = {10.5281/zenodo.23917394},
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
  note         = {Version 0.8.1},
  doi          = {10.5281/zenodo.23917394},
  howpublished = {\url{https://doi.org/10.5281/zenodo.23917394}}
}
```

Replace the version, year and DOI with those of the release you used. The citation
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
