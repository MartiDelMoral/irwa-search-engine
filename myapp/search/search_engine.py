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
from myapp.search.objects import Document, ResultItem


class SearchEngine:
    """Search a corpus with selectable conjunctive ranking methods."""

    def __init__(self, num_results: int = 20):
        self.num_results = num_results
        self._indexed_corpus: dict[str, Document] | None = None
        self._index = {}
        self._document_lengths = {}
        self._document_norms = {}
        self._word2vec_model = None
        self._word2vec_vectors = {}

    def search(
        self,
        search_query: str,
        search_id: str,
        corpus: dict[str, Document],
        ranking_method: str = "tfidf",
    ) -> list[ResultItem]:
        if ranking_method not in RANKING_METHODS:
            raise ValueError(f"Unknown ranking method: {ranking_method}")

        if self._indexed_corpus is not corpus:
            self._index, self._document_lengths = build_inverted_index(corpus)
            self._document_norms = build_tfidf_document_norms(
                self._index, len(corpus)
            )
            self._indexed_corpus = corpus
            self._word2vec_model = None
            self._word2vec_vectors = {}

        if ranking_method == "tfidf":
            ranked_documents = search_tfidf(
                search_query,
                corpus,
                self._index,
                self._document_lengths,
                limit=self.num_results,
                document_norms=self._document_norms,
            )
        elif ranking_method == "bm25":
            ranked_documents = search_bm25(
                search_query,
                corpus,
                self._index,
                self._document_lengths,
                limit=self.num_results,
            )
        elif ranking_method == "custom":
            ranked_documents = search_custom(
                search_query,
                corpus,
                self._index,
                self._document_lengths,
                limit=self.num_results,
            )
        else:
            if self._word2vec_model is None:
                self._word2vec_model, self._word2vec_vectors = train_word2vec(
                    corpus
                )
            ranked_documents = search_word2vec(
                search_query,
                self._index,
                self._word2vec_model,
                self._word2vec_vectors,
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
