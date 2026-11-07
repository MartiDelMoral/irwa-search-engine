from myapp.search.algorithms import build_inverted_index, search_tfidf
from myapp.search.objects import Document, ResultItem


class SearchEngine:
    """Search the product corpus with conjunctive TF-IDF retrieval."""

    def __init__(self, num_results: int = 20):
        self.num_results = num_results
        self._indexed_corpus: dict[str, Document] | None = None
        self._index = {}
        self._document_lengths = {}

    def search(
        self,
        search_query: str,
        search_id: str,
        corpus: dict[str, Document],
    ) -> list[ResultItem]:
        if self._indexed_corpus is not corpus:
            self._index, self._document_lengths = build_inverted_index(corpus)
            self._indexed_corpus = corpus

        ranked_documents = search_tfidf(
            search_query,
            corpus,
            self._index,
            self._document_lengths,
            conjunctive=True,
            limit=self.num_results,
        )

        return [
            ResultItem(
                pid=pid,
                title=corpus[pid].title,
                description=corpus[pid].description,
                url=f"doc_details?pid={pid}&search_id={search_id}&param2=2",
                ranking=score,
            )
            for pid, score in ranked_documents
        ]
