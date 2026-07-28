"""Text preprocessing helpers: n-gram generation and frequency counting.

These are convenience functions for the common (NLP) case where elements are
tokens; the estimator core itself is domain-neutral.
"""

from collections import Counter


def generate_ngrams(text: str | list[str], n: int) -> list[tuple[str, ...]]:
    """Generate n-grams from text.

    Parameters
    ----------
    text : Union[str, List[str]]
        Input text as string or list of tokens
    n : int
        Size of n-grams to generate

    Returns:
    -------
    List[Tuple[str, ...]]
        List of n-gram tuples

    Examples:
    --------
    >>> generate_ngrams("hello world", 2)
    [('h', 'e'), ('e', 'l'), ('l', 'l'), ('l', 'o'), ('o', ' '), (' ', 'w'), ('w', 'o'), ('o', 'r'), ('r', 'l'), ('l', 'd')]

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

    Parameters
    ----------
    text : Union[str, List[str]]
        Input text as string or list of tokens
    normalize : bool, default=False
        If True, return relative frequencies instead of counts

    Returns:
    -------
    Dict[str, Union[int, float]]
        Dictionary mapping words to their frequencies

    Examples:
    --------
    >>> word_frequency("hello world hello")
    {'hello': 2, 'world': 1}

    >>> word_frequency(["hello", "world", "hello"], normalize=True)
    {'hello': 0.6666666666666666, 'world': 0.3333333333333333}
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

    Parameters
    ----------
    text : Union[str, List[str]]
        Input text as string or list of tokens
    n : int
        Size of n-grams to generate
    normalize : bool, default=False
        If True, return relative frequencies instead of counts

    Returns:
    -------
    Dict[Tuple[str, ...], Union[int, float]]
        Dictionary mapping n-grams to their frequencies

    Examples:
    --------
    >>> ngram_frequency("hello world", 2)
    {('h', 'e'): 1, ('e', 'l'): 1, ('l', 'l'): 1, ('l', 'o'): 1, ('o', ' '): 1, (' ', 'w'): 1, ('w', 'o'): 1, ('o', 'r'): 1, ('r', 'l'): 1, ('l', 'd'): 1}
    """
    ngrams = generate_ngrams(text, n)
    freq_dict = Counter(ngrams)

    if normalize:
        total = sum(freq_dict.values())
        return {ngram: count / total for ngram, count in freq_dict.items()}

    return dict(freq_dict)
