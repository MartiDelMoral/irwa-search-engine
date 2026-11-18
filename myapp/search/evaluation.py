import csv
import math
from pathlib import Path
from typing import Mapping

from myapp.search.algorithms import (
    RANKING_METHODS,
    build_inverted_index,
    build_tfidf_document_norms,
    search_bm25,
    search_custom,
    search_tfidf,
    search_word2vec,
    train_word2vec,
)
from myapp.search.objects import Document

VALIDATION_QUERIES = {
    "1": "women full sleeve sweatshirt cotton",
    "2": "men slim jeans blue",
}

OWN_QUERIES = {
    "1": "men black polo t shirt",
    "2": "women solid track pants",
    "3": "men casual checkered shirt",
    "4": "women round neck printed t shirt",
    "5": "men grey full sleeve sweatshirt",
}


def precision_at_k(relevances: list[int], k: int) -> float:
    return sum(relevances[:k]) / k if k > 0 else 0.0


def recall_at_k(relevances: list[int], k: int, total_relevant: int) -> float:
    return sum(relevances[:k]) / total_relevant if total_relevant else 0.0


def average_precision_at_k(
    relevances: list[int], k: int, total_relevant: int
) -> float:
    if k <= 0 or total_relevant <= 0:
        return 0.0

    relevant_seen = 0
    precision_sum = 0.0
    for rank, relevant in enumerate(relevances[:k], start=1):
        if relevant:
            relevant_seen += 1
            precision_sum += relevant_seen / rank

    return precision_sum / min(k, total_relevant)


def f1_at_k(relevances: list[int], k: int, total_relevant: int) -> float:
    precision = precision_at_k(relevances, k)
    recall = recall_at_k(relevances, k, total_relevant)
    return (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def reciprocal_rank(relevances: list[int]) -> float:
    for rank, relevant in enumerate(relevances, start=1):
        if relevant:
            return 1 / rank
    return 0.0


def ndcg_at_k(relevances: list[int], k: int, total_relevant: int) -> float:
    if k <= 0 or total_relevant <= 0:
        return 0.0

    dcg = sum(
        relevant / math.log2(rank + 1)
        for rank, relevant in enumerate(relevances[:k], start=1)
    )
    ideal_dcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, min(k, total_relevant) + 1)
    )
    return dcg / ideal_dcg if ideal_dcg else 0.0


def load_labels(path: str | Path) -> dict[str, dict[str, int]]:
    """Load binary judgments as query ID -> document ID -> relevance."""
    labels: dict[str, dict[str, int]] = {}
    with open(path, "r", encoding="utf-8", newline="") as label_file:
        for row in csv.DictReader(label_file):
            query_id = row["query_id"]
            label = int(row["labels"])
            if label not in (0, 1):
                raise ValueError(f"Expected a binary label, got {label}.")
            labels.setdefault(query_id, {})[row["pid"]] = label
    return labels


def evaluate(
    queries: Mapping[str, str],
    labels: Mapping[str, Mapping[str, int]],
    corpus: dict[str, Document],
    ks: tuple[int, ...] = (5, 10),
    ranking_method: str = "tfidf",
) -> dict[str, dict[str, float]]:
    """Evaluate a full ranking method against the supplied judgments."""
    if ranking_method not in RANKING_METHODS:
        raise ValueError(f"Unknown ranking method: {ranking_method}")

    index, document_lengths = build_inverted_index(corpus)
    document_norms = (
        build_tfidf_document_norms(index, len(corpus))
        if ranking_method == "tfidf"
        else None
    )
    word2vec_model, word2vec_vectors = (
        train_word2vec(corpus) if ranking_method == "word2vec" else (None, {})
    )
    results = {}

    for query_id, query in queries.items():
        judged = labels.get(query_id, {})
        if ranking_method == "tfidf":
            ranking = search_tfidf(
                query,
                corpus,
                index,
                document_lengths,
                document_norms=document_norms,
            )
        elif ranking_method == "bm25":
            ranking = search_bm25(query, corpus, index, document_lengths)
        elif ranking_method == "custom":
            ranking = search_custom(query, corpus, index, document_lengths)
        else:
            ranking = search_word2vec(
                query, index, word2vec_model, word2vec_vectors, limit=None
            )
        # Unjudged documents count as non-relevant without changing their rank.
        relevances = [judged.get(pid, 0) for pid, _ in ranking]
        total_relevant = sum(judged.values())

        query_metrics = {}
        for k in ks:
            query_metrics[f"P@{k}"] = precision_at_k(relevances, k)
            query_metrics[f"R@{k}"] = recall_at_k(
                relevances, k, total_relevant
            )
            query_metrics[f"AP@{k}"] = average_precision_at_k(
                relevances, k, total_relevant
            )
            query_metrics[f"F1@{k}"] = f1_at_k(
                relevances, k, total_relevant
            )
            query_metrics[f"NDCG@{k}"] = ndcg_at_k(
                relevances, k, total_relevant
            )

        query_metrics["AP"] = average_precision_at_k(
            relevances, len(relevances), total_relevant
        )
        query_metrics["RR"] = reciprocal_rank(relevances)
        results[query_id] = query_metrics

    return results


def mean_average_precision(
    results: Mapping[str, Mapping[str, float]],
) -> float:
    return (
        sum(metrics["AP"] for metrics in results.values()) / len(results)
        if results
        else 0.0
    )


def mean_reciprocal_rank(
    results: Mapping[str, Mapping[str, float]],
) -> float:
    return (
        sum(metrics["RR"] for metrics in results.values()) / len(results)
        if results
        else 0.0
    )


def _print_numeric_results(results: Mapping[str, Mapping[str, float]]) -> None:
    metric_order = (
        "P@5", "R@5", "AP@5", "F1@5", "NDCG@5",
        "P@10", "R@10", "AP@10", "F1@10", "NDCG@10",
        "AP", "RR",
    )
    for query_id, metrics in results.items():
        values = [query_id, *(f"{metrics[key]:.3f}" for key in metric_order)]
        print(" ".join(values))
    print(
        f"{mean_average_precision(results):.3f} "
        f"{mean_reciprocal_rank(results):.3f}"
    )


if __name__ == "__main__":
    repository_root = Path(__file__).resolve().parents[2]
    corpus_path = repository_root / "data" / "fashion_products_dataset.json"
    labels_path = repository_root / "data" / "validation_labels.csv"

    from myapp.search.load_corpus import load_corpus

    validation_corpus = load_corpus(corpus_path)
    validation_labels = load_labels(labels_path)
    for ranking_method in RANKING_METHODS:
        print(f"\nRanking method: {ranking_method}")
        _print_numeric_results(
            evaluate(
                VALIDATION_QUERIES,
                validation_labels,
                validation_corpus,
                ranking_method=ranking_method,
            )
        )
