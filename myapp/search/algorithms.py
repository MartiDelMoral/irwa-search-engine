import math
from collections import Counter, defaultdict
from typing import Any

from myapp.search.objects import Document
from myapp.search.text_preprocessing import clean_text

InvertedIndex = dict[str, dict[str, int]]
RANKING_METHODS = ("tfidf", "bm25", "custom", "word2vec")


def build_inverted_index(
    corpus: dict[str, Document],
) -> tuple[InvertedIndex, dict[str, int]]:
    index: defaultdict[str, dict[str, int]] = defaultdict(dict)
    document_lengths = {}
    for pid, document in corpus.items():
        tokens = [
            token
            for field_tokens in document.processed_fields.values()
            for token in field_tokens
        ]
        document_lengths[pid] = len(tokens)
        for term, frequency in Counter(tokens).items():
            index[term][pid] = frequency
    return dict(index), document_lengths


def build_tfidf_document_norms(
    index: InvertedIndex, document_count: int
) -> dict[str, float]:
    squared_norms: defaultdict[str, float] = defaultdict(float)
    for postings in index.values():
        idf = math.log((document_count + 1) / (len(postings) + 1)) + 1
        for pid, frequency in postings.items():
            squared_norms[pid] += (frequency * idf) ** 2
    return {pid: math.sqrt(value) for pid, value in squared_norms.items()}


def _matching_documents(
    query: str, index: InvertedIndex
) -> tuple[list[str], set[str]]:
    terms = clean_text(query)
    if not terms:
        return [], set()

    postings = [index.get(term, {}) for term in terms]
    if any(not term_postings for term_postings in postings):
        return terms, set()
    matching = set(postings[0])
    for term_postings in postings[1:]:
        matching.intersection_update(term_postings)
    return terms, matching


def search_tfidf(
    query: str,
    corpus: dict[str, Document],
    index: InvertedIndex,
    document_lengths: dict[str, int],
    conjunctive: bool = True,
    limit: int | None = None,
    document_norms: dict[str, float] | None = None,
) -> list[tuple[str, float]]:
    """Rank matches by cosine similarity of TF-IDF vectors."""
    terms, matching = _matching_documents(query, index)
    if not terms:
        return []
    if conjunctive:
        if not matching:
            return []
    else:
        matching = set().union(
            *(index[term].keys() for term in set(terms) if term in index)
        )
        if not matching:
            return []

    document_count = len(corpus)
    query_frequencies = Counter(terms)
    query_weights = {}
    for term, frequency in query_frequencies.items():
        if term not in index:
            continue
        df = len(index[term])
        idf = math.log((document_count + 1) / (df + 1)) + 1
        query_weights[term] = frequency * idf
    query_norm = math.sqrt(sum(weight**2 for weight in query_weights.values()))

    if document_norms is None:
        document_norms = build_tfidf_document_norms(index, document_count)
    scores = {}
    for pid in matching:
        dot_product = sum(
            query_weights[term] * index[term][pid]
            * (math.log((document_count + 1) / (len(index[term]) + 1)) + 1)
            for term in query_weights
            if pid in index[term]
        )
        denominator = query_norm * document_norms.get(pid, 0)
        if denominator:
            scores[pid] = dot_product / denominator
    return _rank(scores, limit)


def search_bm25(
    query: str,
    corpus: dict[str, Document],
    index: InvertedIndex,
    document_lengths: dict[str, int],
    limit: int | None = None,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[str, float]]:
    terms, matching = _matching_documents(query, index)
    if not matching:
        return []

    document_count = len(corpus)
    average_length = (
        sum(document_lengths.values()) / document_count if document_count else 0
    )
    scores: defaultdict[str, float] = defaultdict(float)
    for term in set(terms):
        postings = index[term]
        df = len(postings)
        idf = math.log(1 + (document_count - df + 0.5) / (df + 0.5))
        for pid in matching:
            frequency = postings[pid]
            length = document_lengths.get(pid, 0)
            length_ratio = length / average_length if average_length else 0
            denominator = frequency + k1 * (1 - b + b * length_ratio)
            if denominator:
                scores[pid] += (
                    idf * frequency * (k1 + 1) / denominator
                )
    return _rank(scores, limit)


def _numeric(value: Any, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return min(max(number / maximum, 0.0), 1.0)


def search_custom(
    query: str,
    corpus: dict[str, Document],
    index: InvertedIndex,
    document_lengths: dict[str, int],
    limit: int | None = None,
) -> list[tuple[str, float]]:
    """Combine BM25 relevance with rating, discount, and stock availability."""
    text_scores = dict(search_bm25(query, corpus, index, document_lengths))
    if not text_scores:
        return []

    maximum_text_score = max(text_scores.values())
    scores = {}
    for pid, text_score in text_scores.items():
        document = corpus[pid]
        rating = _numeric(document.average_rating, 5)
        discount = _numeric(document.discount, 100)
        in_stock = 0.0 if document.out_of_stock else 1.0
        normalized_text = text_score / maximum_text_score if maximum_text_score else 0.0
        scores[pid] = (
            0.70 * normalized_text
            + 0.15 * rating
            + 0.10 * discount
            + 0.05 * in_stock
        )
    return _rank(scores, limit)


def train_word2vec(
    corpus: dict[str, Document],
) -> tuple[Any | None, dict[str, Any]]:
    """Train deterministic document-corpus embeddings and average each document."""
    from gensim.models import Word2Vec
    import numpy as np

    document_tokens = {
        pid: [
            token
            for field_tokens in document.processed_fields.values()
            for token in field_tokens
        ]
        for pid, document in corpus.items()
    }
    sentences = [tokens for tokens in document_tokens.values() if tokens]
    if not sentences:
        return None, {}

    model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=1,
        workers=1,
        sg=1,
        seed=42,
        epochs=10,
    )
    document_vectors = {
        pid: np.mean([model.wv[token] for token in tokens], axis=0)
        for pid, tokens in document_tokens.items()
        if tokens
    }
    return model, document_vectors


def search_word2vec(
    query: str,
    index: InvertedIndex,
    model: Any,
    document_vectors: dict[str, Any],
    limit: int | None = 20,
) -> list[tuple[str, float]]:
    """Return conjunctive matches ranked by cosine similarity of mean vectors."""
    terms, matching = _matching_documents(query, index)
    if not matching or model is None:
        return []
    query_vectors = [model.wv[term] for term in terms if term in model.wv]
    if len(query_vectors) != len(terms):
        return []

    import numpy as np

    query_vector = np.mean(query_vectors, axis=0)
    query_norm = float(np.linalg.norm(query_vector))
    if not query_norm:
        return []

    scores = {}
    for pid in matching:
        document_vector = document_vectors.get(pid)
        if document_vector is None:
            continue
        denominator = query_norm * float(np.linalg.norm(document_vector))
        if denominator:
            scores[pid] = float(np.dot(query_vector, document_vector) / denominator)
    return _rank(scores, limit)


def _rank(scores: dict[str, float], limit: int | None) -> list[tuple[str, float]]:
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ranked if limit is None else ranked[:limit]
