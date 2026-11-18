import unittest
from types import SimpleNamespace

import numpy as np

from myapp.search.algorithms import (
    build_inverted_index,
    build_tfidf_document_norms,
    search_bm25,
    search_custom,
    search_tfidf,
    search_word2vec,
)
from myapp.search.evaluation import evaluate


class RankingAlgorithmTests(unittest.TestCase):
    def setUp(self):
        self.corpus = {
            "a": SimpleNamespace(
                processed_fields={"title": ["blue", "shoe"]},
                average_rating=5,
                discount=20,
                out_of_stock=False,
            ),
            "b": SimpleNamespace(
                processed_fields={"title": ["blue", "shoe"]},
                average_rating=2,
                discount=0,
                out_of_stock=True,
            ),
            "c": SimpleNamespace(
                processed_fields={"title": ["blue", "boot"]},
                average_rating=5,
                discount=50,
                out_of_stock=False,
            ),
        }
        self.index, self.lengths = build_inverted_index(self.corpus)

    def test_text_rankers_require_all_query_terms(self):
        norms = build_tfidf_document_norms(self.index, len(self.corpus))
        tfidf = search_tfidf(
            "blue shoe", self.corpus, self.index, self.lengths,
            document_norms=norms,
        )
        bm25 = search_bm25("blue shoe", self.corpus, self.index, self.lengths)

        self.assertEqual({pid for pid, _ in tfidf}, {"a", "b"})
        self.assertEqual({pid for pid, _ in bm25}, {"a", "b"})
        self.assertEqual(
            search_bm25("blue unknown", self.corpus, self.index, self.lengths),
            [],
        )
        or_results = search_tfidf(
            "blue unknown",
            self.corpus,
            self.index,
            self.lengths,
            conjunctive=False,
            document_norms=norms,
        )
        self.assertEqual({pid for pid, _ in or_results}, {"a", "b", "c"})

    def test_custom_score_uses_product_fields_after_text_match(self):
        results = search_custom("blue shoe", self.corpus, self.index, self.lengths)

        self.assertEqual([pid for pid, _ in results], ["a", "b"])
        self.assertGreater(results[0][1], results[1][1])

    def test_word2vec_cosine_ranks_only_conjunctive_matches(self):
        vectors = {
            "a": np.array([1.0, 1.0]),
            "b": np.array([1.0, 1.0]),
            "c": np.array([1.0, 0.0]),
        }
        model = SimpleNamespace(
            wv={"blue": np.array([1.0, 0.0]), "shoe": np.array([0.0, 1.0])}
        )
        results = search_word2vec(
            "blue shoe", self.index, model, vectors, limit=20
        )

        self.assertEqual({pid for pid, _ in results}, {"a", "b"})
        self.assertAlmostEqual(results[0][1], 1.0)

    def test_evaluation_accepts_each_ranking_method(self):
        labels = {"q": {"a": 1, "b": 0, "c": 1}}
        for method in ("tfidf", "bm25", "custom", "word2vec"):
            with self.subTest(method=method):
                result = evaluate(
                    {"q": "blue shoe"},
                    labels,
                    self.corpus,
                    ks=(2,),
                    ranking_method=method,
                )
                self.assertIn("P@2", result["q"])

        with self.assertRaises(ValueError):
            evaluate({}, labels, self.corpus, ranking_method="unknown")


if __name__ == "__main__":
    unittest.main()
