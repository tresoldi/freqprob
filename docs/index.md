# FreqProb

**Turn frequency counts into probability estimates.**

FreqProb converts a mapping of elements to observed counts into smoothed
probabilities that handle unseen elements sensibly. It is a general-purpose
statistical tool — natural language processing is one consumer among many
(information retrieval, ecology, genomics, categorical analytics, ML features).

## Install

```bash
pip install freqprob
```

## A first example

```python
import freqprob

counts = {"the": 100, "cat": 50, "dog": 30, "bird": 10}

# Add-one (Laplace) smoothing over a vocabulary of 10,000 possible elements
laplace = freqprob.Laplace(counts, bins=10_000, logprob=False)

laplace("cat")       # 0.0050  — an observed element
laplace("elephant")  # 0.0001  — an unseen element still gets non-zero mass
```

Every estimator follows the same contract: construct it with a frequency
distribution, then call it to score an element.

## Where to next

- **[User Guide](USER_GUIDE.md)** — concepts, choosing a method, worked examples.
- **[API Reference](reference.md)** — every public class and function, generated
  from the source.

## Choosing a method

| Method | Good for | Key parameter |
|--------|----------|---------------|
| `MLE` | raw relative frequencies (no smoothing) | — |
| `Laplace` / `Lidstone` / `ELE` | simple additive smoothing | `bins`, `gamma` |
| `SimpleGoodTuring` | heavy-tailed count data | `p_value` |
| `KneserNey` / `ModifiedKneserNey` | n-gram language models | `discount` |
| `Bayesian` | Dirichlet-prior smoothing | `alpha` |
| `Interpolated` | combining models of different orders | `lambda_weight` |
