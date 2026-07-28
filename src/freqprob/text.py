"""Text preprocessing helpers: n-gram generation and frequency counting.

These are convenience functions for the common (NLP) case where elements are
tokens; the estimator core itself is domain-neutral.
"""

from collections import Counter


def generate_ngrams(text: str | list[str], n: int) -> list[tuple[str, ...]]:
    """Generate n-grams from text.

    A string input is treated as a sequence of characters, while a list input is
    treated as a sequence of tokens. Each n-gram is returned as a tuple.

    Args:
        text: Input text, either a string (split into characters) or a list of
            tokens.
        n: Size of the n-grams to generate. Must be positive.

    Returns:
        A list of n-gram tuples. Empty if the input has fewer than ``n`` items.

    Raises:
        ValueError: If ``n`` is not positive.

    Examples:
        Character n-grams from a string:

        >>> from freqprob import generate_ngrams
        >>> generate_ngrams("abcd", 2)
        [('a', 'b'), ('b', 'c'), ('c', 'd')]

        Token n-grams from a list:

        >>> generate_ngrams(["hello", "world", "test"], 2)
        [('hello', 'world'), ('world', 'test')]
    """
    tokens = list(text) if isinstance(text, str) else text

    if n <= 0:
        raise ValueError("n must be positive")

    if len(tokens) < n:
        return []

    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def word_frequency(text: str | list[str], normalize: bool = False) -> dict[str, int | float]:
    """Compute word frequency from text.

    A string input is split on whitespace into words; a list input is used as-is.
    Counts are returned by default, or relative frequencies when ``normalize`` is
    set.

    Args:
        text: Input text, either a whitespace-delimited string or a list of
            tokens.
        normalize: If ``True``, return relative frequencies that sum to ``1.0``
            instead of raw counts. Defaults to ``False``.

    Returns:
        A mapping of each word to its frequency (an ``int`` count, or a ``float``
        relative frequency when ``normalize`` is ``True``).

    Examples:
        Raw counts:

        >>> from freqprob import word_frequency
        >>> word_frequency("hello world hello")
        {'hello': 2, 'world': 1}

        Normalized frequencies:

        >>> freq = word_frequency(["hello", "world", "hello"], normalize=True)
        >>> round(freq["hello"], 4)
        0.6667
    """
    tokens = text.split() if isinstance(text, str) else text

    freq_dict = Counter(tokens)

    if normalize:
        total = sum(freq_dict.values())
        return {word: count / total for word, count in freq_dict.items()}

    return dict(freq_dict)


def ngram_frequency(
    text: str | list[str], n: int, normalize: bool = False
) -> dict[tuple[str, ...], int | float]:
    """Compute n-gram frequency from text.

    Generates n-grams with :func:`generate_ngrams` and counts them. Counts are
    returned by default, or relative frequencies when ``normalize`` is set.

    Args:
        text: Input text, either a string (split into characters) or a list of
            tokens.
        n: Size of the n-grams to count. Must be positive.
        normalize: If ``True``, return relative frequencies that sum to ``1.0``
            instead of raw counts. Defaults to ``False``.

    Returns:
        A mapping of each n-gram tuple to its frequency (an ``int`` count, or a
        ``float`` relative frequency when ``normalize`` is ``True``).

    Examples:
        >>> from freqprob import ngram_frequency
        >>> freq = ngram_frequency(["a", "b", "a", "b"], 2)
        >>> freq[("a", "b")]
        2
        >>> len(freq)
        2
    """
    ngrams = generate_ngrams(text, n)
    freq_dict = Counter(ngrams)

    if normalize:
        total = sum(freq_dict.values())
        return {ngram: count / total for ngram, count in freq_dict.items()}

    return dict(freq_dict)
