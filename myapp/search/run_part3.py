from pathlib import Path

from myapp.search.evaluation import OWN_QUERIES
from myapp.search.load_corpus import load_corpus
from myapp.search.search_engine import SearchEngine

METHOD_NAMES = {
    "tfidf": "TF-IDF cosine",
    "bm25": "BM25",
    "custom": "Custom product score",
    "word2vec": "Word2Vec cosine",
}


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    corpus = load_corpus(
        repository_root / "data" / "fashion_products_dataset.json"
    )
    engine = SearchEngine(num_results=20)

    for query_id, query in OWN_QUERIES.items():
        print(f"\nQuery {query_id}: {query}")
        for method, method_name in METHOD_NAMES.items():
            print(f"\n{method_name} - top 20")
            results = engine.search(query, query_id, corpus, method)
            for rank, result in enumerate(results, start=1):
                print(
                    f"{rank:2}. {result.pid} | {result.ranking:.4f} | "
                    f"{result.title}"
                )


if __name__ == "__main__":
    main()
