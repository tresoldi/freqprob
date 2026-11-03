# Save, load and refit

## Save and load

A fitted estimator can be written to a file and restored without refitting.
Serialization uses `pickle`, so only load files you trust:

```pycon
>>> import freqprob
>>> model = freqprob.Laplace({"a": 5, "b": 3}, bins=100, logprob=False)
>>> model.save("model.pkl")
>>> restored = freqprob.Laplace.load("model.pkl")
>>> restored("a") == model("a")
True
```

`load` checks that the file holds an estimator of the class it is called on.

## Refit

`fit` refits an estimator to new counts, keeping its parameters, and returns
it, so calls can be chained:

```pycon
>>> model = freqprob.MLE({"a": 3, "b": 1}, logprob=False)
>>> model.fit({"a": 1, "b": 1})("a")
0.5
```

`revision` increases at every refit, so code that caches scores can tell when
to rebuild them.

## The scikit-learn-style API

Alongside calling the estimator, `score` scores one element and `predict`
scores many:

```pycon
>>> model.score("a") == model("a")
True
>>> model.predict(["a", "b", "c"])
[0.5, 0.5, 0.0]
```
