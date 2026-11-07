import math
from collections import Counter, defaultdict
from typing import Optional

from myapp.search.objects import Document
from myapp.search.text_preprocessing import clean_text

InvertedIndex = dict[str, dict[str, int]]


def build_inverted_index(
    corpus: dict[str, Document],
) -> tuple[InvertedIndex, dict[str, int]]:
    """Build a term-frequency inverted index from preprocessed product fields."""
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


def search_tfidf(
    query: str,
    corpus: dict[str, Document],
    index: InvertedIndex,
    document_lengths: dict[str, int],
    conjunctive: bool = True,
    limit: Optional[int] = None,
) -> list[tuple[str, float]]:
    """Return document IDs ranked by length-normalized TF-IDF score."""
    query_terms = clean_text(query)
    if not query_terms:
        return []

    term_postings = [index.get(term, {}) for term in query_terms]
    if conjunctive and any(not postings for postings in term_postings):
        return []

    document_count = len(corpus)
    scores: defaultdict[str, float] = defaultdict(float)
    for term, postings in zip(query_terms, term_postings):
        document_frequency = len(postings)
        if not document_frequency:
            continue

        inverse_document_frequency = (
            math.log((document_count + 1) / (document_frequency + 1)) + 1
        )
        for pid, term_frequency in postings.items():
            scores[pid] += term_frequency * inverse_document_frequency

    if conjunctive:
        scores = defaultdict(
            float,
            {
                pid: score
                for pid, score in scores.items()
                if all(pid in postings for postings in term_postings)
            },
        )

    for pid in scores:
        length = document_lengths.get(pid, 0)
        if length:
            scores[pid] /= length

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ranked if limit is None else ranked[:limit]
