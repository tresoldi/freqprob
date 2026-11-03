# References

FreqProb implements established methods from the statistics, natural language
processing and ecology literatures. This page lists the primary sources for each
method, so that work using FreqProb can cite the methods as well as the
software. All entries are also available as a BibTeX file:
[`references.bib`](references.bib).

To cite FreqProb itself, see [How to cite](citing.md).

## Additive smoothing

`Laplace`, `Lidstone` and `ELE` add a pseudo-count to every possible outcome:
1 for Laplace, an arbitrary `gamma` for Lidstone, and 0.5 for the Expected
Likelihood Estimate.

- Laplace, P.-S. (1814). *Essai philosophique sur les probabilités*. Paris:
  Courcier.
- Lidstone, G. J. (1920). Note on the general case of the Bayes–Laplace formula
  for inductive or a posteriori probabilities. *Transactions of the Faculty of
  Actuaries*, 8, 182–192.
- Manning, C. D., & Schütze, H. (1999). *Foundations of Statistical Natural
  Language Processing*. Cambridge, MA: MIT Press. (Section 6.2 covers Laplace,
  Lidstone and the ELE.)

## Good–Turing and Witten–Bell

`SimpleGoodTuring` follows Gale and Sampson's simplification of the Good–Turing
estimator, and `sample_coverage` is the Good–Turing estimate of sample coverage.
`WittenBell` reserves mass in proportion to the number of distinct types seen.

- Good, I. J. (1953). The population frequencies of species and the estimation
  of population parameters. *Biometrika*, 40(3–4), 237–264.
  [doi:10.1093/biomet/40.3-4.237](https://doi.org/10.1093/biomet/40.3-4.237)
- Gale, W. A., & Sampson, G. (1995). Good–Turing frequency estimation without
  tears. *Journal of Quantitative Linguistics*, 2(3), 217–237.
  [doi:10.1080/09296179508590051](https://doi.org/10.1080/09296179508590051)
- Witten, I. H., & Bell, T. C. (1991). The zero-frequency problem: Estimating
  the probabilities of novel events in adaptive text compression. *IEEE
  Transactions on Information Theory*, 37(4), 1085–1094.
  [doi:10.1109/18.87000](https://doi.org/10.1109/18.87000)

## Discounting and back-off for n-grams

`AbsoluteDiscounting`, `KneserNey`, `ModifiedKneserNey`, `PitmanYor`,
`KatzBackoff` and `StupidBackoff`.

- Ney, H., Essen, U., & Kneser, R. (1994). On structuring probabilistic
  dependences in stochastic language modelling. *Computer Speech & Language*,
  8(1), 1–38. [doi:10.1006/csla.1994.1001](https://doi.org/10.1006/csla.1994.1001)
  (absolute discounting)
- Kneser, R., & Ney, H. (1995). Improved backing-off for M-gram language
  modeling. In *Proceedings of ICASSP-95* (Vol. 1, pp. 181–184).
  [doi:10.1109/ICASSP.1995.479394](https://doi.org/10.1109/ICASSP.1995.479394)
  (Kneser–Ney)
- Chen, S. F., & Goodman, J. (1999). An empirical study of smoothing techniques
  for language modeling. *Computer Speech & Language*, 13(4), 359–393.
  [doi:10.1006/csla.1999.0128](https://doi.org/10.1006/csla.1999.0128)
  (modified Kneser–Ney, and a comparison of most methods here)
- Pitman, J., & Yor, M. (1997). The two-parameter Poisson–Dirichlet
  distribution derived from a stable subordinator. *The Annals of Probability*,
  25(2), 855–900. [doi:10.1214/aop/1024404422](https://doi.org/10.1214/aop/1024404422)
- Teh, Y. W. (2006). A hierarchical Bayesian language model based on
  Pitman–Yor processes. In *Proceedings of COLING/ACL 2006* (pp. 985–992).
  [doi:10.3115/1220175.1220299](https://doi.org/10.3115/1220175.1220299)
  (Pitman–Yor smoothing for language models)
- Katz, S. M. (1987). Estimation of probabilities from sparse data for the
  language model component of a speech recognizer. *IEEE Transactions on
  Acoustics, Speech, and Signal Processing*, 35(3), 400–401.
  [doi:10.1109/TASSP.1987.1165125](https://doi.org/10.1109/TASSP.1987.1165125)
- Brants, T., Popat, A. C., Xu, P., Och, F. J., & Dean, J. (2007). Large
  language models in machine translation. In *Proceedings of EMNLP-CoNLL 2007*
  (pp. 858–867). [ACL Anthology D07-1090](https://aclanthology.org/D07-1090/)
  (Stupid Back-off)

## Interpolation and Bayesian smoothing

`Interpolated`, `JelinekMercer` and `Bayesian`.

- Jelinek, F., & Mercer, R. L. (1980). Interpolated estimation of Markov source
  parameters from sparse data. In E. S. Gelsema & L. N. Kanal (Eds.), *Pattern
  Recognition in Practice* (pp. 381–397). Amsterdam: North-Holland.
- MacKay, D. J. C., & Peto, L. C. B. (1995). A hierarchical Dirichlet language
  model. *Natural Language Engineering*, 1(3), 289–308.
  [doi:10.1017/S1351324900000218](https://doi.org/10.1017/S1351324900000218)

## Entropy, coverage and evaluation

`chao_shen_entropy`, `nsb_entropy`, `cross_entropy`, `kl_divergence` and
`perplexity`.

- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System
  Technical Journal*, 27(3), 379–423.
  [doi:10.1002/j.1538-7305.1948.tb01338.x](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x)
- Kullback, S., & Leibler, R. A. (1951). On information and sufficiency. *The
  Annals of Mathematical Statistics*, 22(1), 79–86.
  [doi:10.1214/aoms/1177729694](https://doi.org/10.1214/aoms/1177729694)
- Chao, A., & Shen, T.-J. (2003). Nonparametric estimation of Shannon's index of
  diversity when there are unseen species in sample. *Environmental and
  Ecological Statistics*, 10(4), 429–443.
  [doi:10.1023/A:1026096204727](https://doi.org/10.1023/A:1026096204727)
- Nemenman, I., Shafee, F., & Bialek, W. (2002). Entropy and inference,
  revisited. In *Advances in Neural Information Processing Systems 14*
  (pp. 471–478). MIT Press. [arXiv:physics/0108025](https://arxiv.org/abs/physics/0108025)
- Wolpert, D. H., & Wolf, D. R. (1995). Estimating functions of probability
  distributions from a finite set of samples. *Physical Review E*, 52(6),
  6841–6854. [doi:10.1103/PhysRevE.52.6841](https://doi.org/10.1103/PhysRevE.52.6841)
  (the posterior-mean entropy used inside the NSB estimator)

## Methods without a separate reference

`MLE`, `Uniform` and `Random` are standard baselines. `CertaintyDegree` is an
experimental method with no published reference; its behaviour is documented in
the [API Reference](reference.md).

## Software

FreqProb's estimators are checked against NLTK's implementations, and FreqProb
is built on NumPy and SciPy.

- Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with
  Python*. Sebastopol, CA: O'Reilly Media.
- Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array
  programming with NumPy. *Nature*, 585(7825), 357–362.
  [doi:10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2)
- Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0:
  Fundamental algorithms for scientific computing in Python. *Nature Methods*,
  17(3), 261–272. [doi:10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2)
