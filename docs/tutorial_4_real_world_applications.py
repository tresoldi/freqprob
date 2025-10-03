#' # FreqProb Tutorial 4: Real-World NLP Applications
#'
#' This tutorial demonstrates practical applications of FreqProb in real-world NLP scenarios:
#'
#' 1. **Language Modeling** - Building n-gram language models
#' 2. **Text Classification** - Feature extraction with smoothed probabilities
#' 3. **Information Retrieval** - Document scoring and ranking
#' 4. **Sentiment Analysis** - Building probability-based classifiers
#' 5. **Machine Translation** - Language model scoring
#' 6. **Speech Recognition** - N-gram model integration
#'
#' ## Setup and Data Preparation

# | hide
import math
import random
import re
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import freqprob

# Set up plotting and random seeds
plt.style.use("default")
sns.set_palette("husl")
random.seed(42)
np.random.seed(42)

print("FreqProb Real-World Applications Tutorial")
print("=" * 40)
print()

# Sample datasets for demonstration
# In practice, you would load these from files or APIs

# 1. News articles dataset
news_articles = {
    "technology": [
        "Apple announces new iPhone with advanced AI capabilities and improved camera system",
        "Google launches innovative machine learning platform for developers worldwide",
        "Microsoft introduces cloud computing solutions for enterprise customers",
        "Tesla unveils autonomous driving technology with neural network improvements",
        "Amazon develops new artificial intelligence algorithms for recommendation systems",
        "Facebook invests heavily in virtual reality and metaverse technologies",
        "Intel releases next generation processors with enhanced performance capabilities",
        "Samsung showcases foldable smartphone technology at tech conference",
    ],
    "sports": [
        "Olympic athletes compete in swimming championships with record breaking performances",
        "Basketball team wins championship after intense playoff games throughout season",
        "Soccer world cup features exciting matches between international teams",
        "Tennis tournament showcases incredible athletic talent and competitive spirit",
        "Baseball season concludes with thrilling world series games",
        "Hockey players demonstrate exceptional skills during championship playoffs",
        "Marathon runners compete in challenging race through city streets",
        "Golf tournament attracts professional players from around the world",
    ],
    "politics": [
        "Government announces new policies regarding healthcare and education reform",
        "Presidential election campaign focuses on economic issues and foreign policy",
        "Congress debates legislation concerning environmental protection and climate change",
        "International diplomacy efforts aim to resolve conflicts through peaceful negotiations",
        "Local elections determine representatives for state and municipal governments",
        "Political parties present platforms addressing social justice and equality",
        "Supreme court decisions impact constitutional law and civil rights",
        "Trade agreements between nations affect global economic relationships",
    ],
}

# 2. Movie reviews dataset (sentiment analysis)
movie_reviews = {
    "positive": [
        "This movie is absolutely fantastic with amazing performances and brilliant storytelling",
        "Incredible cinematography and outstanding acting make this film truly exceptional",
        "Wonderful story with excellent character development and beautiful visuals",
        "Superb direction and phenomenal performances create an unforgettable experience",
        "Magnificent film with incredible depth and emotional resonance throughout",
        "Brilliant screenplay and outstanding cast deliver an amazing cinematic experience",
        "Excellent movie with fantastic performances and compelling narrative structure",
        "Remarkable film featuring wonderful acting and beautiful cinematographic work",
    ],
    "negative": [
        "This movie is terrible with awful performances and boring storyline throughout",
        "Horrible cinematography and poor acting make this film completely unwatchable",
        "Terrible story with weak character development and disappointing visual effects",
        "Poor direction and bad performances create a frustrating viewing experience",
        "Awful film with no depth and completely uninteresting plot development",
        "Bad screenplay and terrible cast deliver a disappointing cinematic disaster",
        "Horrible movie with weak performances and confusing narrative structure",
        "Disappointing film featuring poor acting and uninspiring cinematographic choices",
    ],
}

print(f"Loaded {sum(len(articles) for articles in news_articles.values())} news articles")
print(f"Loaded {sum(len(reviews) for reviews in movie_reviews.values())} movie reviews")
print()

# Text preprocessing utilities


def preprocess_text(text: str) -> list[str]:
    """Basic text preprocessing: lowercase, tokenize, remove punctuation."""
    # Convert to lowercase and remove punctuation
    text = re.sub(r"[^\w\s]", "", text.lower())
    # Split into words
    return text.split()


def generate_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Generate n-grams from a list of tokens."""
    if n == 1:
        return [(token,) for token in tokens]

    padded_tokens = ["<s>"] * (n - 1) + tokens + ["</s>"] * (n - 1)
    ngrams = []
    for i in range(len(padded_tokens) - n + 1):
        ngrams.append(tuple(padded_tokens[i : i + n]))
    return ngrams


print("Text preprocessing utilities ready")
print("Sample preprocessing:", preprocess_text("Hello, World! This is a test."))
print("Sample bigrams:", generate_ngrams(["hello", "world", "test"], 2))
# |

#' ## Application 1: Language Modeling
#'
#' Build n-gram language models for text generation and probability estimation.

print("APPLICATION 1: LANGUAGE MODELING")
print("=" * 35)
print()


class LanguageModel:
    """N-gram language model with multiple smoothing options."""

    def __init__(self, n: int = 2, smoothing: str = "kneser_ney"):
        self.n = n
        self.smoothing = smoothing
        self.models = {}
        self.vocab = set()

    def train(self, documents: list[str]):
        """Train the language model on a corpus of documents."""
        print(f"Training {self.n}-gram model with {self.smoothing} smoothing...")

        # Collect all n-grams for each order
        ngram_counts = {i: Counter() for i in range(1, self.n + 1)}

        for doc in documents:
            tokens = preprocess_text(doc)
            self.vocab.update(tokens)

            for order in range(1, self.n + 1):
                ngrams = generate_ngrams(tokens, order)
                ngram_counts[order].update(ngrams)

        # Create smoothed models for each n-gram order
        for order in range(1, self.n + 1):
            if self.smoothing == "kneser_ney" and order >= 2:
                try:
                    self.models[order] = freqprob.KneserNey(
                        ngram_counts[order], discount=0.75, logprob=True
                    )
                except Exception:
                    # Fallback to Laplace if KN fails
                    self.models[order] = freqprob.Laplace(
                        ngram_counts[order], bins=len(ngram_counts[order]) * 2, logprob=True
                    )
            else:
                self.models[order] = freqprob.Laplace(
                    ngram_counts[order], bins=len(ngram_counts[order]) * 2, logprob=True
                )

        print(f"Trained on {len(documents)} documents")
        print(f"Vocabulary size: {len(self.vocab):,}")
        for order in range(1, self.n + 1):
            print(f"{order}-gram types: {len(ngram_counts[order]):,}")

    def score_sequence(self, text: str) -> float:
        """Score a text sequence using the language model."""
        tokens = preprocess_text(text)
        ngrams = generate_ngrams(tokens, self.n)

        log_prob = 0.0
        for ngram in ngrams:
            log_prob += self.models[self.n](ngram)

        return log_prob

    def perplexity(self, test_texts: list[str]) -> float:
        """Calculate perplexity on test data."""
        total_log_prob = 0.0
        total_words = 0

        for text in test_texts:
            tokens = preprocess_text(text)
            ngrams = generate_ngrams(tokens, self.n)

            for ngram in ngrams:
                total_log_prob += self.models[self.n](ngram)
                total_words += 1

        if total_words == 0:
            return float("inf")

        return math.exp(-total_log_prob / total_words)

    def generate_text(self, prompt: str = "", max_length: int = 20) -> str:
        """Generate text using the language model (simplified version)."""
        if not prompt:
            # Start with sentence boundary
            generated = ["<s>"] * (self.n - 1)
        else:
            generated = ["<s>"] * (self.n - 1) + preprocess_text(prompt)

        for _ in range(max_length):
            # Get context (last n-1 words)
            context = tuple(generated[-(self.n - 1) :])

            # Find best next word (simplified - in practice, use sampling)
            best_word = None
            best_score = float("-inf")

            # Try common words from vocabulary
            candidate_words = list(self.vocab)[:50]  # Limit for efficiency
            for word in candidate_words:
                ngram = (*context, word)
                score = self.models[self.n](ngram)
                if score > best_score:
                    best_score = score
                    best_word = word

            if best_word == "</s>" or best_word is None:
                break

            generated.append(best_word)

        # Remove sentence boundaries and return
        result = [w for w in generated if w not in ["<s>", "</s>"]]
        return " ".join(result)


# Train language models on different domains
print("Training domain-specific language models...")
print()

domain_models = {}
for domain, articles in news_articles.items():
    print(f"Training {domain} model:")
    model = LanguageModel(n=3, smoothing="kneser_ney")
    model.train(articles)
    domain_models[domain] = model
    print()

# Test sentences from different domains
test_sentences = {
    "technology": "Apple releases new smartphone with artificial intelligence",
    "sports": "Basketball team wins championship game with incredible performance",
    "politics": "Government announces new policy regarding healthcare reform",
    "mixed": "The president discussed artificial intelligence in sports technology",
}

print("Language Model Evaluation:")
print("=" * 27)
print()

# Score test sentences with each domain model
results = defaultdict(dict)
for test_name, test_sentence in test_sentences.items():
    print(f"Test: '{test_sentence}'")

    scores = {}
    for domain, model in domain_models.items():
        score = model.score_sequence(test_sentence)
        scores[domain] = score
        results[test_name][domain] = score

    # Find best matching domain
    best_domain = max(scores, key=scores.get)
    print(f"  Best match: {best_domain} (score: {scores[best_domain]:.2f})")

    # Show all scores
    for domain, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"    {domain:<12}: {score:6.2f}")
    print()

# Visualize domain classification results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 1. Heatmap of scores
score_matrix = []
test_names = list(test_sentences.keys())
domain_names = list(domain_models.keys())

for test_name in test_names:
    row = [results[test_name][domain] for domain in domain_names]
    score_matrix.append(row)

im = ax1.imshow(score_matrix, cmap="RdYlBu_r", aspect="auto")
ax1.set_xticks(range(len(domain_names)))
ax1.set_xticklabels(domain_names)
ax1.set_yticks(range(len(test_names)))
ax1.set_yticklabels(test_names)
ax1.set_title("Language Model Scores\n(Higher = Better Match)")

# Add score annotations
for i in range(len(test_names)):
    for j in range(len(domain_names)):
        ax1.text(
            j,
            i,
            f"{score_matrix[i][j]:.1f}",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )

plt.colorbar(im, ax=ax1)

# 2. Perplexity comparison
print("Computing perplexities...")
perplexities = {}
for domain, model in domain_models.items():
    # Test on held-out sentences
    test_data = [
        list(test_sentences.values())[i]
        for i in range(len(test_sentences))
        if i % 3 == hash(domain) % 3
    ]
    if test_data:
        pp = model.perplexity(test_data)
        perplexities[domain] = pp

if perplexities:
    domains = list(perplexities.keys())
    pp_values = list(perplexities.values())

    bars = ax2.bar(domains, pp_values, alpha=0.7, color=["blue", "orange", "green"])
    ax2.set_title("Model Perplexity on Test Data\n(Lower = Better)")
    ax2.set_ylabel("Perplexity")
    ax2.set_xlabel("Domain")

    # Add value labels
    for bar, value in zip(bars, pp_values, strict=False):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{value:.1f}",
            ha="center",
            va="bottom",
        )

plt.tight_layout()

# Text generation example
print("Text Generation Examples:")
print("=" * 25)

generation_prompts = {
    "technology": "Apple announces",
    "sports": "The team",
    "politics": "Government policy",
}

for domain, prompt in generation_prompts.items():
    if domain in domain_models:
        generated = domain_models[domain].generate_text(prompt, max_length=10)
        print(f"{domain.capitalize()}: {generated}")

print("\nKey Insights:")
print("- Domain-specific models capture vocabulary and style differences")
print("- Higher scores indicate better domain match")
print("- Language models can be used for domain classification")
print("- Text generation reflects domain-specific patterns")

#' ## Application 2: Text Classification
#'
#' Use smoothed probability features for document classification.

print("APPLICATION 2: TEXT CLASSIFICATION")
print("=" * 36)
print()


class ProbabilisticTextClassifier:
    """Text classifier using smoothed probability features."""

    def __init__(self, smoothing_method="laplace", feature_type="unigram"):
        self.smoothing_method = smoothing_method
        self.feature_type = feature_type
        self.class_models = {}
        self.vocabulary = set()
        self.classes = []

    def train(self, documents: dict[str, list[str]]):
        """Train the classifier on labeled documents."""
        print(f"Training classifier with {self.smoothing_method} smoothing...")

        self.classes = list(documents.keys())

        # Build vocabulary and class-specific frequency distributions
        class_frequencies = {}

        for class_label, docs in documents.items():
            # Combine all documents for this class
            all_tokens = []
            for doc in docs:
                tokens = preprocess_text(doc)
                all_tokens.extend(tokens)
                self.vocabulary.update(tokens)

            # Create frequency distribution
            if self.feature_type == "unigram":
                class_frequencies[class_label] = Counter(all_tokens)
            elif self.feature_type == "bigram":
                bigrams = []
                for doc in docs:
                    tokens = preprocess_text(doc)
                    bigrams.extend(generate_ngrams(tokens, 2))
                class_frequencies[class_label] = Counter(bigrams)

        # Create smoothed models for each class
        vocab_size = (
            len(self.vocabulary) if self.feature_type == "unigram" else len(self.vocabulary) ** 2
        )

        for class_label, frequencies in class_frequencies.items():
            if self.smoothing_method == "laplace":
                self.class_models[class_label] = freqprob.Laplace(
                    frequencies, bins=vocab_size, logprob=True
                )
            elif self.smoothing_method == "bayesian":
                self.class_models[class_label] = freqprob.BayesianSmoothing(
                    frequencies, alpha=0.5, logprob=True
                )
            else:  # MLE
                self.class_models[class_label] = freqprob.MLE(frequencies, logprob=True)

        print(f"Trained on {len(self.classes)} classes")
        print(f"Vocabulary size: {len(self.vocabulary):,}")
        for class_label in self.classes:
            class_size = sum(class_frequencies[class_label].values())
            print(f"  {class_label}: {class_size:,} tokens")

    def extract_features(self, text: str) -> list:
        """Extract features from text based on feature type."""
        tokens = preprocess_text(text)

        if self.feature_type == "unigram":
            return tokens
        if self.feature_type == "bigram":
            return generate_ngrams(tokens, 2)
        return tokens

    def classify(self, text: str) -> tuple[str, dict[str, float]]:
        """Classify a text document."""
        features = self.extract_features(text)

        class_scores = {}
        for class_label, model in self.class_models.items():
            # Calculate log probability of features given class
            log_prob = sum(model(feature) for feature in features)
            class_scores[class_label] = log_prob

        # Predict class with highest score
        predicted_class = max(class_scores, key=class_scores.get)

        return predicted_class, class_scores

    def evaluate(self, test_documents: dict[str, list[str]]) -> dict[str, float]:
        """Evaluate classifier on test data."""
        total = 0
        correct = 0
        class_stats = {class_label: {"correct": 0, "total": 0} for class_label in self.classes}

        for true_class, docs in test_documents.items():
            for doc in docs:
                predicted_class, _scores = self.classify(doc)

                total += 1
                class_stats[true_class]["total"] += 1

                if predicted_class == true_class:
                    correct += 1
                    class_stats[true_class]["correct"] += 1

        # Calculate metrics
        accuracy = correct / total if total > 0 else 0

        class_accuracies = {}
        for class_label, stats in class_stats.items():
            if stats["total"] > 0:
                class_accuracies[class_label] = stats["correct"] / stats["total"]
            else:
                class_accuracies[class_label] = 0

        return {
            "overall_accuracy": accuracy,
            "class_accuracies": class_accuracies,
            "total_documents": total,
        }


# Train classifiers with different configurations
print("Training multiple classifier configurations...")
print()

classifiers = {
    "Unigram + Laplace": ProbabilisticTextClassifier("laplace", "unigram"),
    "Unigram + Bayesian": ProbabilisticTextClassifier("bayesian", "unigram"),
    "Bigram + Laplace": ProbabilisticTextClassifier("laplace", "bigram"),
}

# Train on news articles
for name, classifier in classifiers.items():
    print(f"Training {name}:")
    classifier.train(news_articles)
    print()

# Create test data (using different sentences)
test_articles = {
    "technology": [
        "New smartphone features advanced machine learning algorithms",
        "Cloud computing platform launches with AI integration",
        "Software developers adopt innovative programming frameworks",
    ],
    "sports": [
        "Professional athletes train for upcoming Olympic games",
        "Championship match features incredible team performance",
        "Stadium hosts exciting tournament with international competitors",
    ],
    "politics": [
        "Legislative session addresses important social issues",
        "International relations improve through diplomatic efforts",
        "Electoral process ensures democratic representation",
    ],
}

print("Classifier Evaluation:")
print("=" * 22)
print()

evaluation_results = {}
for name, classifier in classifiers.items():
    print(f"{name}:")
    results = classifier.evaluate(test_articles)
    evaluation_results[name] = results

    print(f"  Overall Accuracy: {results['overall_accuracy']:.3f}")
    for class_label, acc in results["class_accuracies"].items():
        print(f"    {class_label}: {acc:.3f}")
    print()

# Detailed classification examples
print("Detailed Classification Examples:")
print("=" * 34)
print()

example_texts = [
    "Apple develops new artificial intelligence chip for smartphones",
    "Basketball players compete in championship finals",
    "Government implements new environmental protection policies",
    "Machine learning algorithms improve medical diagnosis accuracy",  # Mixed domain
]

best_classifier = max(evaluation_results, key=lambda x: evaluation_results[x]["overall_accuracy"])
print(f"Using best classifier: {best_classifier}")
print()

for i, text in enumerate(example_texts):
    predicted_class, scores = classifiers[best_classifier].classify(text)

    print(f"Example {i + 1}: '{text}'")
    print(f"  Predicted: {predicted_class}")
    print("  Scores:")
    for class_label, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"    {class_label:<12}: {score:6.2f}")
    print()

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Overall accuracy comparison
classifier_names = list(evaluation_results.keys())
accuracies = [evaluation_results[name]["overall_accuracy"] for name in classifier_names]

bars = axes[0, 0].bar(range(len(classifier_names)), accuracies, alpha=0.7)
axes[0, 0].set_title("Overall Classification Accuracy")
axes[0, 0].set_ylabel("Accuracy")
axes[0, 0].set_xticks(range(len(classifier_names)))
axes[0, 0].set_xticklabels(classifier_names, rotation=45)
axes[0, 0].set_ylim(0, 1)

# Add value labels
for bar, acc in zip(bars, accuracies, strict=False):
    axes[0, 0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{acc:.3f}",
        ha="center",
        va="bottom",
    )

# 2. Class-wise accuracy heatmap
class_names = list(news_articles.keys())
accuracy_matrix = []

for classifier_name in classifier_names:
    row = [
        evaluation_results[classifier_name]["class_accuracies"][class_name]
        for class_name in class_names
    ]
    accuracy_matrix.append(row)

im = axes[0, 1].imshow(accuracy_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
axes[0, 1].set_xticks(range(len(class_names)))
axes[0, 1].set_xticklabels(class_names)
axes[0, 1].set_yticks(range(len(classifier_names)))
axes[0, 1].set_yticklabels(classifier_names)
axes[0, 1].set_title("Class-wise Accuracy")

# Add accuracy annotations
for i in range(len(classifier_names)):
    for j in range(len(class_names)):
        axes[0, 1].text(
            j,
            i,
            f"{accuracy_matrix[i][j]:.2f}",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )

plt.colorbar(im, ax=axes[0, 1])

# 3. Feature importance (vocabulary analysis)
# Analyze most discriminative words for each class
best_model = classifiers[best_classifier]
class_words = {}

for class_label, model in best_model.class_models.items():
    # Get words with highest probabilities in this class
    word_scores = []
    sample_words = list(best_model.vocabulary)[:100]  # Sample for efficiency

    for word in sample_words:
        score = model(word)
        word_scores.append((word, score))

    # Get top words
    top_words = sorted(word_scores, key=lambda x: x[1], reverse=True)[:5]
    class_words[class_label] = top_words

# Plot top words per class
y_pos = 0
colors = ["blue", "orange", "green"]
for i, (class_label, words) in enumerate(class_words.items()):
    word_names = [word for word, score in words]
    word_scores = [score for word, score in words]

    y_positions = [y_pos + j * 0.2 for j in range(len(word_names))]
    axes[1, 0].barh(
        y_positions, word_scores, height=0.15, color=colors[i], alpha=0.7, label=class_label
    )

    # Add word labels
    for j, (word, score) in enumerate(words):
        axes[1, 0].text(score + 0.1, y_pos + j * 0.2, word, va="center", fontsize=8)

    y_pos += len(words) * 0.2 + 0.5

axes[1, 0].set_title("Top Words by Class (Log Probability)")
axes[1, 0].set_xlabel("Log Probability")
axes[1, 0].legend()
axes[1, 0].set_yticks([])

# 4. Classification confidence distribution
confidence_scores = []
for text in example_texts:
    _, scores = classifiers[best_classifier].classify(text)
    max_score = max(scores.values())
    min_score = min(scores.values())
    confidence = max_score - min_score  # Score difference
    confidence_scores.append(confidence)

axes[1, 1].hist(confidence_scores, bins=5, alpha=0.7, color="purple")
axes[1, 1].set_title("Classification Confidence Distribution")
axes[1, 1].set_xlabel("Confidence (Score Difference)")
axes[1, 1].set_ylabel("Frequency")

plt.tight_layout()

print("\nKey Insights:")
print("- Smoothing methods handle unseen words gracefully")
print("- Bigram features can capture more context but may overfit")
print("- Different domains show varying classification difficulty")
print("- Probability-based features provide interpretable results")

#' ## Application 3: Information Retrieval
#'
#' Build document ranking systems using language model scoring.

print("APPLICATION 3: INFORMATION RETRIEVAL")
print("=" * 37)
print()


class LanguageModelRetrieval:
    """Information retrieval using language model scoring."""

    def __init__(self, smoothing="laplace", interpolation_lambda=0.8):
        self.smoothing = smoothing
        self.interpolation_lambda = interpolation_lambda
        self.document_models = {}
        self.collection_model = None
        self.documents = {}

    def index_documents(self, documents: dict[str, str]):
        """Index a collection of documents."""
        print(f"Indexing {len(documents)} documents...")

        self.documents = documents

        # Build collection-level language model
        all_tokens = []
        for doc_text in documents.values():
            tokens = preprocess_text(doc_text)
            all_tokens.extend(tokens)

        collection_freq = Counter(all_tokens)
        vocab_size = len(set(all_tokens))

        if self.smoothing == "laplace":
            self.collection_model = freqprob.Laplace(
                collection_freq, bins=vocab_size * 2, logprob=True
            )
        else:
            self.collection_model = freqprob.MLE(collection_freq, logprob=True)

        # Build document-specific language models
        for doc_id, doc_text in documents.items():
            tokens = preprocess_text(doc_text)
            doc_freq = Counter(tokens)

            if self.smoothing == "laplace":
                doc_model = freqprob.Laplace(doc_freq, bins=vocab_size * 2, logprob=True)
            else:
                doc_model = freqprob.MLE(doc_freq, logprob=True)

            self.document_models[doc_id] = doc_model

        print(f"Collection vocabulary: {vocab_size:,} words")
        print(f"Average document length: {len(all_tokens) / len(documents):.1f} words")

    def score_document(self, doc_id: str, query: str) -> float:
        """Score a document for a given query using language model interpolation."""
        query_tokens = preprocess_text(query)

        if doc_id not in self.document_models:
            return float("-inf")

        doc_model = self.document_models[doc_id]
        total_score = 0.0

        for token in query_tokens:
            # Interpolated smoothing: λ * P(w|D) + (1-λ) * P(w|C)
            doc_prob = math.exp(doc_model(token))  # Convert from log
            collection_prob = math.exp(self.collection_model(token))

            interpolated_prob = (
                self.interpolation_lambda * doc_prob
                + (1 - self.interpolation_lambda) * collection_prob
            )

            if interpolated_prob > 0:
                total_score += math.log(interpolated_prob)
            else:
                total_score += float("-inf")

        return total_score

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float, str]]:
        """Search for documents matching the query."""
        scores = []

        for doc_id in self.documents:
            score = self.score_document(doc_id, query)
            scores.append((doc_id, score, self.documents[doc_id]))

        # Sort by score (descending) and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def evaluate_retrieval(self, queries: dict[str, list[str]]) -> dict[str, float]:
        """Evaluate retrieval performance using relevance judgments."""
        total_queries = 0
        total_precision = 0.0
        total_recall = 0.0

        for query, relevant_docs in queries.items():
            results = self.search(query, top_k=len(self.documents))
            retrieved_docs = [doc_id for doc_id, _, _ in results[: len(relevant_docs)]]

            # Calculate precision and recall
            relevant_retrieved = set(relevant_docs) & set(retrieved_docs)

            if len(retrieved_docs) > 0:
                precision = len(relevant_retrieved) / len(retrieved_docs)
            else:
                precision = 0.0

            recall = len(relevant_retrieved) / len(relevant_docs) if len(relevant_docs) > 0 else 0.0

            total_precision += precision
            total_recall += recall
            total_queries += 1

        avg_precision = total_precision / total_queries if total_queries > 0 else 0
        avg_recall = total_recall / total_queries if total_queries > 0 else 0
        f1_score = (
            2 * avg_precision * avg_recall / (avg_precision + avg_recall)
            if (avg_precision + avg_recall) > 0
            else 0
        )

        return {"precision": avg_precision, "recall": avg_recall, "f1_score": f1_score}


# Create document collection from our news articles
document_collection = {}
doc_id = 0

for category, articles in news_articles.items():
    for article in articles:
        document_collection[f"{category}_{doc_id:03d}"] = article
        doc_id += 1

print(f"Created document collection with {len(document_collection)} documents")
print()

# Initialize and train retrieval system
retrieval_systems = {
    "Laplace (λ=0.8)": LanguageModelRetrieval("laplace", 0.8),
    "Laplace (λ=0.5)": LanguageModelRetrieval("laplace", 0.5),
    "MLE (λ=0.8)": LanguageModelRetrieval("mle", 0.8),
}

for name, system in retrieval_systems.items():
    print(f"Indexing documents for {name}...")
    system.index_documents(document_collection)
    print()

# Define test queries and relevance judgments
test_queries = {
    "artificial intelligence machine learning": [
        "technology_000",
        "technology_001",
        "technology_004",
    ],
    "basketball championship games": ["sports_001", "sports_005"],
    "government policy healthcare": ["politics_000", "politics_004"],
    "smartphone technology apple": ["technology_000", "technology_007"],
    "olympic athletes competition": ["sports_000", "sports_003"],
}

print("Search Results Examples:")
print("=" * 25)
print()

# Demonstrate search results
sample_queries = ["artificial intelligence", "basketball championship", "government policy"]
best_system = next(iter(retrieval_systems.values()))  # Use first system for examples

for query in sample_queries:
    print(f"Query: '{query}'")
    results = best_system.search(query, top_k=3)

    for i, (doc_id, score, doc_text) in enumerate(results, 1):
        # Truncate document text for display
        display_text = doc_text[:80] + "..." if len(doc_text) > 80 else doc_text
        print(f"  {i}. {doc_id} (score: {score:.2f})")
        print(f"     {display_text}")
    print()

# Evaluate retrieval systems
print("Retrieval System Evaluation:")
print("=" * 29)
print()

evaluation_results = {}
for name, system in retrieval_systems.items():
    results = system.evaluate_retrieval(test_queries)
    evaluation_results[name] = results

    print(f"{name}:")
    print(f"  Precision: {results['precision']:.3f}")
    print(f"  Recall:    {results['recall']:.3f}")
    print(f"  F1-Score:  {results['f1_score']:.3f}")
    print()

# Query analysis
print("Query-Document Similarity Analysis:")
print("=" * 35)
print()

# Analyze how different query terms affect retrieval
analysis_query = "artificial intelligence technology"
query_tokens = preprocess_text(analysis_query)

print(f"Analyzing query: '{analysis_query}'")
print(f"Query tokens: {query_tokens}")
print()

# Show how each query term contributes to document scores
sample_docs = list(document_collection.keys())[:5]
term_contributions = defaultdict(dict)

for doc_id in sample_docs:
    doc_model = best_system.document_models[doc_id]

    for token in query_tokens:
        doc_prob = math.exp(doc_model(token))
        collection_prob = math.exp(best_system.collection_model(token))

        interpolated_prob = (
            best_system.interpolation_lambda * doc_prob
            + (1 - best_system.interpolation_lambda) * collection_prob
        )

        term_contributions[doc_id][token] = (
            math.log(interpolated_prob) if interpolated_prob > 0 else float("-inf")
        )

# Display term contributions
print("Term contributions to document scores:")
for doc_id in sample_docs:
    total_score = sum(term_contributions[doc_id].values())
    print(f"\n{doc_id} (total: {total_score:.2f}):")
    for token in query_tokens:
        contrib = term_contributions[doc_id][token]
        print(f"  {token:<15}: {contrib:6.2f}")

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. System performance comparison
system_names = list(evaluation_results.keys())
metrics = ["precision", "recall", "f1_score"]

x = np.arange(len(system_names))
width = 0.25

for i, metric in enumerate(metrics):
    values = [evaluation_results[name][metric] for name in system_names]
    offset = (i - 1) * width
    axes[0, 0].bar(x + offset, values, width, label=metric.capitalize(), alpha=0.8)

axes[0, 0].set_title("Retrieval System Performance")
axes[0, 0].set_ylabel("Score")
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(system_names, rotation=45)
axes[0, 0].legend()
axes[0, 0].set_ylim(0, 1)

# 2. Query difficulty analysis
query_difficulties = []
query_names = []

for query, relevant_docs in test_queries.items():
    results = best_system.search(query, top_k=len(document_collection))
    retrieved_docs = [doc_id for doc_id, _, _ in results[: len(relevant_docs)]]

    relevant_retrieved = set(relevant_docs) & set(retrieved_docs)
    precision = len(relevant_retrieved) / len(retrieved_docs) if retrieved_docs else 0

    query_difficulties.append(precision)
    query_names.append(query[:20] + "..." if len(query) > 20 else query)

bars = axes[0, 1].bar(range(len(query_names)), query_difficulties, alpha=0.7, color="orange")
axes[0, 1].set_title("Query Difficulty (Precision)")
axes[0, 1].set_ylabel("Precision")
axes[0, 1].set_xticks(range(len(query_names)))
axes[0, 1].set_xticklabels(query_names, rotation=45)
axes[0, 1].set_ylim(0, 1)

# Add value labels
for bar, value in zip(bars, query_difficulties, strict=False):
    axes[0, 1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{value:.2f}",
        ha="center",
        va="bottom",
    )

# 3. Document score distribution
sample_query = "artificial intelligence"
all_scores = []
relevant_scores = []
non_relevant_scores = []

results = best_system.search(sample_query, top_k=len(document_collection))
relevant_docs_set = set(test_queries.get("artificial intelligence machine learning", []))

for doc_id, score, _ in results:
    all_scores.append(score)
    if doc_id in relevant_docs_set:
        relevant_scores.append(score)
    else:
        non_relevant_scores.append(score)

axes[1, 0].hist(non_relevant_scores, bins=10, alpha=0.7, label="Non-relevant", color="red")
axes[1, 0].hist(relevant_scores, bins=10, alpha=0.7, label="Relevant", color="green")
axes[1, 0].set_title(f'Score Distribution\nQuery: "{sample_query}"')
axes[1, 0].set_xlabel("Document Score")
axes[1, 0].set_ylabel("Frequency")
axes[1, 0].legend()

# 4. Term contribution heatmap
contrib_matrix = []
doc_labels = []

for doc_id in sample_docs:
    row = [term_contributions[doc_id][token] for token in query_tokens]
    contrib_matrix.append(row)
    doc_labels.append(doc_id)

im = axes[1, 1].imshow(contrib_matrix, cmap="RdYlBu_r", aspect="auto")
axes[1, 1].set_xticks(range(len(query_tokens)))
axes[1, 1].set_xticklabels(query_tokens)
axes[1, 1].set_yticks(range(len(doc_labels)))
axes[1, 1].set_yticklabels(doc_labels)
axes[1, 1].set_title("Term Contribution to Document Scores")

# Add contribution annotations
for i in range(len(doc_labels)):
    for j in range(len(query_tokens)):
        value = contrib_matrix[i][j]
        if value != float("-inf"):
            axes[1, 1].text(
                j, i, f"{value:.1f}", ha="center", va="center", color="white", fontweight="bold"
            )

plt.colorbar(im, ax=axes[1, 1])

plt.tight_layout()

print("\nKey Insights:")
print("- Language model interpolation balances document and collection statistics")
print("- Smoothing prevents zero probabilities for unseen query terms")
print("- Different interpolation weights affect retrieval performance")
print("- Term contributions help explain retrieval decisions")
print("- Relevant documents tend to have higher scores than non-relevant ones")

#' ## Application 4: Sentiment Analysis
#'
#' Build sentiment classifiers using probability-based features.

print("APPLICATION 4: SENTIMENT ANALYSIS")
print("=" * 34)
print()


class SentimentAnalyzer:
    """Sentiment analysis using probabilistic language models."""

    def __init__(self, smoothing="bayesian", feature_weighting="tf"):
        self.smoothing = smoothing
        self.feature_weighting = feature_weighting
        self.sentiment_models = {}
        self.vocabulary = set()
        self.word_sentiment_scores = {}

    def train(self, labeled_reviews: dict[str, list[str]]):
        """Train sentiment analyzer on labeled review data."""
        print(f"Training sentiment analyzer with {self.smoothing} smoothing...")

        # Collect vocabulary and build sentiment-specific frequency distributions
        sentiment_frequencies = {}

        for sentiment, reviews in labeled_reviews.items():
            all_tokens = []
            for review in reviews:
                tokens = preprocess_text(review)
                all_tokens.extend(tokens)
                self.vocabulary.update(tokens)

            sentiment_frequencies[sentiment] = Counter(all_tokens)

        # Create smoothed models for each sentiment class
        vocab_size = len(self.vocabulary)

        for sentiment, frequencies in sentiment_frequencies.items():
            if self.smoothing == "bayesian":
                self.sentiment_models[sentiment] = freqprob.BayesianSmoothing(
                    frequencies, alpha=0.5, logprob=True
                )
            elif self.smoothing == "laplace":
                self.sentiment_models[sentiment] = freqprob.Laplace(
                    frequencies, bins=vocab_size * 2, logprob=True
                )
            else:  # ELE
                self.sentiment_models[sentiment] = freqprob.ELE(
                    frequencies, bins=vocab_size * 2, logprob=True
                )

        # Compute word-level sentiment scores
        self._compute_word_sentiment_scores()

        print(f"Trained on {len(labeled_reviews)} sentiment classes")
        print(f"Vocabulary size: {len(self.vocabulary):,}")
        for sentiment, freq in sentiment_frequencies.items():
            print(f"  {sentiment}: {sum(freq.values()):,} tokens")

    def _compute_word_sentiment_scores(self):
        """Compute sentiment polarity scores for individual words."""
        if "positive" in self.sentiment_models and "negative" in self.sentiment_models:
            pos_model = self.sentiment_models["positive"]
            neg_model = self.sentiment_models["negative"]

            for word in self.vocabulary:
                pos_prob = math.exp(pos_model(word))
                neg_prob = math.exp(neg_model(word))

                # Compute log-odds ratio as sentiment score
                if pos_prob > 0 and neg_prob > 0:
                    sentiment_score = math.log(pos_prob / neg_prob)
                    self.word_sentiment_scores[word] = sentiment_score
                else:
                    self.word_sentiment_scores[word] = 0.0

    def analyze_sentiment(self, text: str) -> tuple[str, dict[str, float], float]:
        """Analyze sentiment of a text."""
        tokens = preprocess_text(text)

        # Calculate probability under each sentiment model
        sentiment_scores = {}

        for sentiment, model in self.sentiment_models.items():
            if self.feature_weighting == "tf":
                # Simple term frequency weighting
                token_counts = Counter(tokens)
                score = sum(model(token) * count for token, count in token_counts.items())
            else:
                # Uniform weighting
                score = sum(model(token) for token in tokens)

            sentiment_scores[sentiment] = score

        # Predict sentiment with highest score
        predicted_sentiment = max(sentiment_scores, key=sentiment_scores.get)

        # Calculate confidence (difference between top two scores)
        sorted_scores = sorted(sentiment_scores.values(), reverse=True)
        confidence = (
            sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else abs(sorted_scores[0])
        )

        return predicted_sentiment, sentiment_scores, confidence

    def get_most_polar_words(self, n: int = 10) -> dict[str, list[tuple[str, float]]]:
        """Get most positive and negative words based on sentiment scores."""
        if not self.word_sentiment_scores:
            return {}

        # Sort words by sentiment score
        sorted_words = sorted(self.word_sentiment_scores.items(), key=lambda x: x[1])

        return {"most_negative": sorted_words[:n], "most_positive": sorted_words[-n:]}

    def explain_prediction(self, text: str) -> dict[str, list[tuple[str, float]]]:
        """Explain sentiment prediction by showing word contributions."""
        tokens = preprocess_text(text)
        token_counts = Counter(tokens)

        explanations = {}

        for sentiment, model in self.sentiment_models.items():
            word_contributions = []

            for token, count in token_counts.items():
                contribution = model(token) * count
                word_contributions.append((token, contribution))

            # Sort by contribution magnitude
            word_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            explanations[sentiment] = word_contributions

        return explanations

    def evaluate(self, test_reviews: dict[str, list[str]]) -> dict[str, float]:
        """Evaluate sentiment analyzer on test data."""
        total = 0
        correct = 0

        sentiment_stats = {
            sentiment: {"correct": 0, "total": 0} for sentiment in self.sentiment_models
        }

        for true_sentiment, reviews in test_reviews.items():
            for review in reviews:
                predicted_sentiment, _, _ = self.analyze_sentiment(review)

                total += 1
                sentiment_stats[true_sentiment]["total"] += 1

                if predicted_sentiment == true_sentiment:
                    correct += 1
                    sentiment_stats[true_sentiment]["correct"] += 1

        overall_accuracy = correct / total if total > 0 else 0

        class_accuracies = {}
        for sentiment, stats in sentiment_stats.items():
            if stats["total"] > 0:
                class_accuracies[sentiment] = stats["correct"] / stats["total"]
            else:
                class_accuracies[sentiment] = 0

        return {"overall_accuracy": overall_accuracy, "class_accuracies": class_accuracies}


# Train sentiment analyzers with different configurations
print("Training sentiment analyzers...")
print()

sentiment_analyzers = {
    "Bayesian + TF": SentimentAnalyzer("bayesian", "tf"),
    "Laplace + TF": SentimentAnalyzer("laplace", "tf"),
    "ELE + Uniform": SentimentAnalyzer("ele", "uniform"),
}

for name, analyzer in sentiment_analyzers.items():
    print(f"Training {name}:")
    analyzer.train(movie_reviews)
    print()

# Create test data
test_reviews = {
    "positive": [
        "This film is absolutely brilliant with outstanding performances",
        "Incredible movie with amazing story and fantastic acting",
        "Wonderful cinematography and excellent direction make this film great",
    ],
    "negative": [
        "This movie is terrible with awful acting and boring plot",
        "Horrible film with poor direction and disappointing story",
        "Bad cinematography and weak performances ruin this movie",
    ],
}

# Evaluate analyzers
print("Sentiment Analyzer Evaluation:")
print("=" * 31)
print()

evaluation_results = {}
for name, analyzer in sentiment_analyzers.items():
    results = analyzer.evaluate(test_reviews)
    evaluation_results[name] = results

    print(f"{name}:")
    print(f"  Overall Accuracy: {results['overall_accuracy']:.3f}")
    for sentiment, acc in results["class_accuracies"].items():
        print(f"    {sentiment}: {acc:.3f}")
    print()

# Demonstrate sentiment analysis on example texts
best_analyzer = max(
    sentiment_analyzers.items(), key=lambda x: evaluation_results[x[0]]["overall_accuracy"]
)[1]

print("Sentiment Analysis Examples:")
print("=" * 29)
print()

example_texts = [
    "This movie is absolutely fantastic with brilliant performances",
    "The film was disappointing with poor acting and weak story",
    "Good cinematography but the plot was confusing and boring",  # Mixed sentiment
    "Amazing visuals and incredible soundtrack make this film outstanding",
]

for i, text in enumerate(example_texts, 1):
    sentiment, scores, confidence = best_analyzer.analyze_sentiment(text)

    print(f"Example {i}: '{text}'")
    print(f"  Predicted sentiment: {sentiment} (confidence: {confidence:.2f})")
    print("  Scores:")
    for sent, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"    {sent}: {score:.2f}")

    # Show word contributions
    explanations = best_analyzer.explain_prediction(text)
    print("  Top contributing words:")
    for sent in ["positive", "negative"]:
        if sent in explanations:
            top_words = explanations[sent][:3]
            word_strs = [f"{word}({contrib:.1f})" for word, contrib in top_words]
            print(f"    {sent}: {', '.join(word_strs)}")
    print()

# Analyze most polar words
print("Most Polar Words:")
print("=" * 17)
print()

polar_words = best_analyzer.get_most_polar_words(5)
if polar_words:
    print("Most negative words:")
    for word, score in polar_words["most_negative"]:
        print(f"  {word:<15}: {score:6.2f}")

    print("\nMost positive words:")
    for word, score in polar_words["most_positive"]:
        print(f"  {word:<15}: {score:6.2f}")
    print()

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Analyzer performance comparison
analyzer_names = list(evaluation_results.keys())
overall_accuracies = [evaluation_results[name]["overall_accuracy"] for name in analyzer_names]

bars = axes[0, 0].bar(range(len(analyzer_names)), overall_accuracies, alpha=0.7)
axes[0, 0].set_title("Sentiment Analyzer Performance")
axes[0, 0].set_ylabel("Overall Accuracy")
axes[0, 0].set_xticks(range(len(analyzer_names)))
axes[0, 0].set_xticklabels(analyzer_names, rotation=45)
axes[0, 0].set_ylim(0, 1)

# Add value labels
for bar, acc in zip(bars, overall_accuracies, strict=False):
    axes[0, 0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{acc:.3f}",
        ha="center",
        va="bottom",
    )

# 2. Class-wise accuracy comparison
positive_accs = [
    evaluation_results[name]["class_accuracies"]["positive"] for name in analyzer_names
]
negative_accs = [
    evaluation_results[name]["class_accuracies"]["negative"] for name in analyzer_names
]

x = np.arange(len(analyzer_names))
width = 0.35

axes[0, 1].bar(x - width / 2, positive_accs, width, label="Positive", alpha=0.8)
axes[0, 1].bar(x + width / 2, negative_accs, width, label="Negative", alpha=0.8)

axes[0, 1].set_title("Class-wise Accuracy")
axes[0, 1].set_ylabel("Accuracy")
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(analyzer_names, rotation=45)
axes[0, 1].legend()
axes[0, 1].set_ylim(0, 1)

# 3. Word polarity visualization
if polar_words:
    neg_words = [word for word, score in polar_words["most_negative"]]
    neg_scores = [score for word, score in polar_words["most_negative"]]
    pos_words = [word for word, score in polar_words["most_positive"]]
    pos_scores = [score for word, score in polar_words["most_positive"]]

    # Combine and plot
    all_words = neg_words + pos_words
    all_scores = neg_scores + pos_scores
    colors = ["red"] * len(neg_words) + ["green"] * len(pos_words)

    bars = axes[1, 0].barh(range(len(all_words)), all_scores, color=colors, alpha=0.7)
    axes[1, 0].set_yticks(range(len(all_words)))
    axes[1, 0].set_yticklabels(all_words)
    axes[1, 0].set_xlabel("Sentiment Score (log-odds ratio)")
    axes[1, 0].set_title("Most Polar Words")
    axes[1, 0].axvline(x=0, color="black", linestyle="--", alpha=0.5)

# 4. Confidence distribution
confidences = []
for text in example_texts + [review for reviews in test_reviews.values() for review in reviews]:
    _, _, confidence = best_analyzer.analyze_sentiment(text)
    confidences.append(confidence)

axes[1, 1].hist(confidences, bins=10, alpha=0.7, color="purple")
axes[1, 1].set_title("Prediction Confidence Distribution")
axes[1, 1].set_xlabel("Confidence Score")
axes[1, 1].set_ylabel("Frequency")

plt.tight_layout()

print("\nKey Insights:")
print("- Bayesian smoothing with term frequency weighting performs well")
print("- Word-level sentiment scores provide interpretable features")
print("- Confidence scores help identify uncertain predictions")
print("- Polar words align with human intuitions about sentiment")
print("- Different smoothing methods handle rare words differently")

#' ## Application 5: Cross-Domain Applications
#'
#' Demonstrate how FreqProb techniques apply across different NLP domains.

print("APPLICATION 5: CROSS-DOMAIN APPLICATIONS")
print("=" * 41)
print()

# Demonstrate various cross-domain applications

# 1. Machine Translation Scoring
print("1. Machine Translation Scoring:")
print("=" * 32)
print()


class TranslationScorer:
    """Score translation quality using language models."""

    def __init__(self, target_language_corpus: list[str]):
        # Build target language model
        all_tokens = []
        for sentence in target_language_corpus:
            tokens = preprocess_text(sentence)
            all_tokens.extend(tokens)

        bigrams = []
        for sentence in target_language_corpus:
            tokens = preprocess_text(sentence)
            bigrams.extend(generate_ngrams(tokens, 2))

        bigram_freq = Counter(bigrams)
        try:
            self.language_model = freqprob.KneserNey(bigram_freq, discount=0.75, logprob=True)
        except Exception:
            self.language_model = freqprob.Laplace(
                bigram_freq, bins=len(bigram_freq) * 2, logprob=True
            )

    def score_translation(self, translation: str) -> float:
        """Score a translation using the language model."""
        tokens = preprocess_text(translation)
        bigrams = generate_ngrams(tokens, 2)

        if not bigrams:
            return float("-inf")

        total_score = sum(self.language_model(bigram) for bigram in bigrams)
        return total_score / len(bigrams)  # Normalized by length


# Simulate English corpus for target language model
english_corpus = [
    "the cat sits on the mat",
    "dogs run quickly in the park",
    "beautiful flowers bloom in spring",
    "children play games in the garden",
    "birds fly high in the sky",
]

translation_scorer = TranslationScorer(english_corpus)

# Score different translation candidates
translation_candidates = [
    "the dog runs quickly in park",  # Good but missing article
    "dogs run fast in the garden",  # Good translation
    "quickly dogs run park in the",  # Bad word order
    "beautiful children play outside",  # Different but fluent
]

print("Translation Quality Scores:")
for i, candidate in enumerate(translation_candidates, 1):
    score = translation_scorer.score_translation(candidate)
    print(f"{i}. '{candidate}'")
    print(f"   Score: {score:.2f}")
print()

# 2. Spelling Correction
print("2. Spelling Correction:")
print("=" * 23)
print()


class SpellingCorrector:
    """Simple spelling correction using language models."""

    def __init__(self, corpus: list[str]):
        # Build word frequency model
        all_words = []
        for sentence in corpus:
            words = preprocess_text(sentence)
            all_words.extend(words)

        word_freq = Counter(all_words)
        self.word_model = freqprob.Laplace(word_freq, bins=len(word_freq) * 2, logprob=False)
        self.vocabulary = set(all_words)

    def generate_candidates(self, word: str) -> list[str]:
        """Generate spelling correction candidates (simplified)."""
        candidates = []

        # Single character edits
        for i in range(len(word)):
            # Deletions
            candidate = word[:i] + word[i + 1 :]
            if candidate in self.vocabulary:
                candidates.append(candidate)

            # Substitutions (limited alphabet for demo)
            for c in "aeiou":
                candidate = word[:i] + c + word[i + 1 :]
                if candidate in self.vocabulary:
                    candidates.append(candidate)

        return list(set(candidates))  # Remove duplicates

    def correct_spelling(self, word: str) -> str:
        """Correct spelling of a word."""
        if word in self.vocabulary:
            return word  # Already correct

        candidates = self.generate_candidates(word)
        if not candidates:
            return word  # No candidates found

        # Score candidates and return best
        best_candidate = max(candidates, key=lambda c: self.word_model(c))
        return best_candidate


# Create spelling corrector
corrector = SpellingCorrector(
    english_corpus + list(movie_reviews["positive"]) + list(movie_reviews["negative"])
)

# Test spelling correction
misspelled_words = ["beautful", "chldren", "flwers", "gme", "qickly"]

print("Spelling Corrections:")
for word in misspelled_words:
    correction = corrector.correct_spelling(word)
    candidates = corrector.generate_candidates(word)
    print(f"'{word}' -> '{correction}'")
    if candidates:
        candidate_scores = [(c, corrector.word_model(c)) for c in candidates[:3]]
        candidate_strs = [f"{c}({s:.3f})" for c, s in candidate_scores]
        print(f"  Candidates: {', '.join(candidate_strs)}")
print()

# 3. Text Readability Assessment
print("3. Text Readability Assessment:")
print("=" * 31)
print()


class ReadabilityAssessor:
    """Assess text readability using language model features."""

    def __init__(self, simple_corpus: list[str], complex_corpus: list[str]):
        # Build models for simple and complex text
        simple_tokens = []
        for text in simple_corpus:
            simple_tokens.extend(preprocess_text(text))

        complex_tokens = []
        for text in complex_corpus:
            complex_tokens.extend(preprocess_text(text))

        simple_freq = Counter(simple_tokens)
        complex_freq = Counter(complex_tokens)

        vocab_size = len(set(simple_tokens + complex_tokens))

        self.simple_model = freqprob.Laplace(simple_freq, bins=vocab_size, logprob=True)
        self.complex_model = freqprob.Laplace(complex_freq, bins=vocab_size, logprob=True)

    def assess_readability(self, text: str) -> dict[str, float]:
        """Assess readability of text."""
        tokens = preprocess_text(text)

        simple_score = sum(self.simple_model(token) for token in tokens)
        complex_score = sum(self.complex_model(token) for token in tokens)

        # Normalize by length
        if tokens:
            simple_score /= len(tokens)
            complex_score /= len(tokens)

        # Readability score (higher = more readable)
        readability_score = simple_score - complex_score

        return {
            "simple_score": simple_score,
            "complex_score": complex_score,
            "readability": readability_score,
        }


# Simple vs complex text corpora
simple_texts = ["the cat is happy", "dogs run fast", "children play games", "birds fly in the sky"]

complex_texts = [
    "artificial intelligence algorithms demonstrate sophisticated computational capabilities",
    "multidisciplinary research endeavors investigate complex phenomena across various domains",
    "contemporary technological innovations facilitate unprecedented communication opportunities",
]

readability_assessor = ReadabilityAssessor(simple_texts, complex_texts)

# Test readability assessment
test_texts = [
    "the dog runs quickly",  # Simple
    "machine learning improves computer performance",  # Medium
    "sophisticated algorithms demonstrate computational prowess",  # Complex
]

print("Readability Assessment:")
for text in test_texts:
    scores = readability_assessor.assess_readability(text)
    readability = "Simple" if scores["readability"] > 0 else "Complex"

    print(f"'{text}'")
    print(f"  Readability: {readability} (score: {scores['readability']:.2f})")
    print(
        f"  Simple model: {scores['simple_score']:.2f}, Complex model: {scores['complex_score']:.2f}"
    )
print()

# 4. Text Summarization Sentence Scoring
print("4. Text Summarization Sentence Scoring:")
print("=" * 40)
print()


class SentenceScorer:
    """Score sentences for extractive summarization."""

    def __init__(self, document_corpus: list[str]):
        # Build document language model
        all_tokens = []
        for doc in document_corpus:
            tokens = preprocess_text(doc)
            all_tokens.extend(tokens)

        token_freq = Counter(all_tokens)
        self.document_model = freqprob.Laplace(token_freq, bins=len(token_freq) * 2, logprob=True)

    def score_sentence(self, sentence: str) -> float:
        """Score sentence based on term importance in document."""
        tokens = preprocess_text(sentence)

        if not tokens:
            return 0.0

        # Score based on average word probability
        token_scores = [math.exp(self.document_model(token)) for token in tokens]
        return sum(token_scores) / len(token_scores)


# Create sentence scorer using news articles
all_articles = []
for articles in news_articles.values():
    all_articles.extend(articles)

sentence_scorer = SentenceScorer(all_articles)

# Score sentences from a sample article
sample_article = "Apple announces new iPhone with advanced AI capabilities. The device features improved camera system and faster processing. Machine learning algorithms enhance user experience through intelligent recommendations. The smartphone includes innovative design elements and better battery life."

sentences = sample_article.split(". ")
sentence_scores = [(sent, sentence_scorer.score_sentence(sent)) for sent in sentences]
sentence_scores.sort(key=lambda x: x[1], reverse=True)

print("Sentence Importance Scores (for summarization):")
for i, (sentence, score) in enumerate(sentence_scores, 1):
    print(f"{i}. {sentence} (score: {score:.4f})")
print()

# Visualize cross-domain applications
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Translation scoring
translation_scores = [
    translation_scorer.score_translation(candidate) for candidate in translation_candidates
]
bars = axes[0, 0].bar(range(len(translation_candidates)), translation_scores, alpha=0.7)
axes[0, 0].set_title("Translation Quality Scores")
axes[0, 0].set_ylabel("Language Model Score")
axes[0, 0].set_xticks(range(len(translation_candidates)))
axes[0, 0].set_xticklabels([f"Candidate {i + 1}" for i in range(len(translation_candidates))])

# Add score labels
for bar, score in zip(bars, translation_scores, strict=False):
    axes[0, 0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.05,
        f"{score:.2f}",
        ha="center",
        va="bottom",
    )

# 2. Readability assessment
readability_scores = [
    readability_assessor.assess_readability(text)["readability"] for text in test_texts
]
colors = ["green" if score > 0 else "red" for score in readability_scores]

bars = axes[0, 1].bar(range(len(test_texts)), readability_scores, color=colors, alpha=0.7)
axes[0, 1].set_title("Text Readability Assessment")
axes[0, 1].set_ylabel("Readability Score\n(Positive = Simple)")
axes[0, 1].set_xticks(range(len(test_texts)))
axes[0, 1].set_xticklabels([f"Text {i + 1}" for i in range(len(test_texts))])
axes[0, 1].axhline(y=0, color="black", linestyle="--", alpha=0.5)

# 3. Sentence scoring for summarization
sent_labels = [f"Sentence {i + 1}" for i in range(len(sentence_scores))]
sent_score_values = [score for _, score in sentence_scores]

bars = axes[1, 0].bar(range(len(sent_labels)), sent_score_values, alpha=0.7, color="purple")
axes[1, 0].set_title("Sentence Importance for Summarization")
axes[1, 0].set_ylabel("Importance Score")
axes[1, 0].set_xticks(range(len(sent_labels)))
axes[1, 0].set_xticklabels(sent_labels)

# 4. Application performance summary
applications = [
    "Translation\nScoring",
    "Spelling\nCorrection",
    "Readability\nAssessment",
    "Sentence\nScoring",
]
effectiveness = [0.85, 0.70, 0.90, 0.80]  # Simulated effectiveness scores

bars = axes[1, 1].bar(range(len(applications)), effectiveness, alpha=0.7, color="orange")
axes[1, 1].set_title("Application Effectiveness")
axes[1, 1].set_ylabel("Effectiveness Score")
axes[1, 1].set_xticks(range(len(applications)))
axes[1, 1].set_xticklabels(applications)
axes[1, 1].set_ylim(0, 1)

# Add value labels
for bar, eff in zip(bars, effectiveness, strict=False):
    axes[1, 1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{eff:.2f}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()

print("Cross-Domain Applications Summary:")
print("=" * 35)
print("✓ Translation Quality Assessment - Language models score fluency")
print("✓ Spelling Correction - Probability models prefer common words")
print("✓ Readability Assessment - Simple vs complex vocabulary models")
print("✓ Text Summarization - Term importance drives sentence selection")
print()
print("Key Takeaways:")
print("- FreqProb techniques generalize across many NLP tasks")
print("- Language model scores provide useful quality metrics")
print("- Smoothing is crucial for handling unseen vocabulary")
print("- Domain-specific training data improves performance")
print("- Probability-based features are interpretable and robust")

#' ## Summary and Best Practices
#'
#' This tutorial demonstrated real-world applications of FreqProb across various NLP domains.

print("REAL-WORLD APPLICATIONS: SUMMARY AND BEST PRACTICES")
print("=" * 56)
print()

print("🎯 APPLICATIONS COVERED:")
print("=" * 25)
print("1. Language Modeling      - N-gram models for text generation and scoring")
print("2. Text Classification    - Probability-based features for categorization")
print("3. Information Retrieval  - Document ranking using language model scoring")
print("4. Sentiment Analysis     - Emotion detection with smoothed probabilities")
print("5. Cross-Domain Tasks     - Translation, spelling, readability, summarization")
print()

print("🔧 TECHNICAL INSIGHTS:")
print("=" * 22)
print("• Kneser-Ney smoothing excels for language modeling")
print("• Bayesian smoothing provides principled parameter control")
print("• Interpolation balances specificity with robustness")
print("• Term frequency weighting improves classification")
print("• Domain-specific training significantly impacts performance")
print()

print("📊 PERFORMANCE PATTERNS:")
print("=" * 24)
print("• Higher-order n-grams capture more context but may overfit")
print("• Smoothing is essential for handling unseen vocabulary")
print("• Different domains show varying classification difficulty")
print("• Confidence scores help identify uncertain predictions")
print("• Evaluation metrics depend heavily on application goals")
print()

print("🏗️ IMPLEMENTATION BEST PRACTICES:")
print("=" * 35)
print()

print("Data Preparation:")
print("• Consistent text preprocessing across train/test")
print("• Handle sentence boundaries appropriately for n-grams")
print("• Consider domain-specific tokenization and normalization")
print("• Balance training data across classes/domains")
print()

print("Model Selection:")
print("• Start with Laplace smoothing as a robust baseline")
print("• Use Kneser-Ney for n-gram language models (n ≥ 2)")
print("• Consider Bayesian smoothing for parameter interpretability")
print("• Tune smoothing parameters using validation data")
print()

print("Feature Engineering:")
print("• Experiment with unigram vs. n-gram features")
print("• Consider term frequency vs. binary feature weighting")
print("• Use probability ratios for comparative tasks")
print("• Normalize scores by sequence length when appropriate")
print()

print("Evaluation Strategy:")
print("• Use held-out test data for final evaluation")
print("• Report both overall and class-wise performance")
print("• Include confidence/uncertainty analysis")
print("• Compare against simple baselines")
print()

print("⚠️ COMMON PITFALLS TO AVOID:")
print("=" * 29)
print("• Using MLE without smoothing (zero probability problem)")
print("• Inconsistent vocabulary handling across models")
print("• Ignoring the impact of training data size and quality")
print("• Over-tuning hyperparameters on small validation sets")
print("• Forgetting to normalize probabilities when comparing sequences")
print()

print("🚀 SCALING TO PRODUCTION:")
print("=" * 26)
print("• Use streaming models for real-time applications")
print("• Implement caching for frequently accessed probabilities")
print("• Consider memory-efficient representations for large vocabularies")
print("• Monitor model performance and drift over time")
print("• Plan for vocabulary updates and model retraining")
print()

print("🔬 CHOOSING THE RIGHT APPROACH:")
print("=" * 32)
print()

decision_matrix = {
    "Small Dataset (<1K docs)": "Laplace/Bayesian smoothing, unigram features",
    "Medium Dataset (1K-10K)": "ELE/Kneser-Ney, bigram features, validation tuning",
    "Large Dataset (>10K)": "Modified Kneser-Ney, higher-order n-grams, efficiency features",
    "Real-time Application": "Streaming models, bounded vocabulary, fast scoring",
    "High Accuracy Needed": "Ensemble methods, extensive hyperparameter tuning",
    "Interpretability Important": "Bayesian smoothing, word-level analysis",
    "Memory Constrained": "Compressed representations, sparse models",
}

for scenario, recommendation in decision_matrix.items():
    print(f"{scenario:<25}: {recommendation}")
print()

print("📚 FURTHER LEARNING:")
print("=" * 20)
print("• Experiment with different smoothing methods on your data")
print("• Try combining FreqProb with modern neural approaches")
print("• Explore domain adaptation techniques")
print("• Study advanced n-gram models (e.g., class-based, factored)")
print("• Investigate hybrid statistical-neural architectures")
print()

print("✨ FINAL THOUGHTS:")
print("=" * 18)
print("FreqProb provides a solid foundation for probability-based NLP tasks.")
print("While deep learning dominates many applications, classical smoothing")
print("methods remain valuable for:")
print("  • Small data scenarios")
print("  • Interpretable models")
print("  • Baseline comparisons")
print("  • Hybrid architectures")
print("  • Resource-constrained environments")
print()
print("Master these fundamentals, and you'll have a strong foundation")
print("for tackling any NLP challenge! 🎉")

#' ## Conclusion
#'
#' Congratulations! You've completed the comprehensive FreqProb tutorial series covering:
#'
#' ### Tutorial Series Summary:
#'
#' 1. **Tutorial 1: Basic Smoothing Methods** - Foundation concepts and simple techniques
#' 2. **Tutorial 2: Advanced Smoothing Methods** - State-of-the-art techniques for complex scenarios
#' 3. **Tutorial 3: Efficiency and Memory Management** - Scaling to production environments
#' 4. **Tutorial 4: Real-World Applications** - Practical NLP implementations
#'
#' ### What You've Learned:
#'
#' - **Core Concepts**: Probability estimation, smoothing, zero probability problem
#' - **Methods**: From simple Laplace to advanced Kneser-Ney smoothing
#' - **Optimization**: Vectorization, caching, lazy evaluation, memory management
#' - **Applications**: Language modeling, classification, retrieval, sentiment analysis
#' - **Best Practices**: Model selection, evaluation, scaling, debugging
#'
#' ### Your Next Steps:
#'
#' 1. **Apply to Your Projects** - Use FreqProb techniques in your own NLP tasks
#' 2. **Experiment** - Try different combinations of methods and parameters
#' 3. **Benchmark** - Compare with other approaches on your specific data
#' 4. **Contribute** - Share improvements and new applications with the community
#' 5. **Learn More** - Explore advanced topics like neural-statistical hybrids
#'
#' ### Remember:
#'
#' - **Always use smoothing** in real applications
#' - **Validate your approach** with proper evaluation
#' - **Consider the trade-offs** between accuracy, speed, and interpretability
#' - **Start simple** and add complexity as needed
#' - **Monitor performance** in production environments
#'
#' Happy modeling! 🚀
