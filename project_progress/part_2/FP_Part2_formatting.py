import csv
import json
import math
import re
from collections import Counter, defaultdict

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


# NLTK needs the English stop-word list for the text-cleaning step.
nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
STEMMER = SnowballStemmer("english")


def clean_text(text):
    """Lowercase text, split it into tokens, remove stop words and stem them."""
    if not text:
        return []

    # Keep letters and numbers, and use whitespace/punctuation as separators.
    tokens = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

    return [
        STEMMER.stem(token)
        for token in tokens
        if token not in STOP_WORDS
    ]


def load_data(file_path):
    """
    Read the product JSON file and keep the available product information.

    The searchable text combines the product's title, brand, category,
    sub-category, description, product details and seller.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = {}
    for item in data:
        pid = str(item.get("pid", item.get("_id", "")))
        if not pid:
            continue

        # Use empty strings for text fields that are missing or set to None.
        title = item.get("title", "") or ""
        desc = item.get("description", "") or ""
        brand = item.get("brand", "") or ""
        category = item.get("category", "") or ""
        sub_category = item.get("sub_category", "") or ""
        product_details = item.get("product_details", "") or ""
        seller = item.get("seller", "") or ""

        searchable_text = (
            f"{title} {brand} {category} {sub_category} "
            f"{desc} {product_details} {seller}"
        )

        # Keep the original product fields as well as the text used for search.
        docs[pid] = {
            "pid": pid,
            "title": title,
            "description": desc,
            "brand": brand,
            "category": category,
            "sub_category": sub_category,
            "product_details": product_details,
            "seller": seller,
            "out_of_stock": item.get("out_of_stock", False),
            "selling_price": item.get("selling_price", None),
            "discount": item.get("discount", None),
            "actual_price": item.get("actual_price", None),
            "average_rating": item.get("average_rating", None),
            "url": item.get("url", ""),
            "searchable_text": searchable_text,
        }

    return docs


def build_inverted_index(docs):
    """Build the term-to-document index and record each document's token count."""
    index = defaultdict(dict)  # {term: {pid: term frequency}}
    doc_lengths = {}

    for pid, doc in docs.items():
        tokens = clean_text(doc["searchable_text"])
        doc_lengths[pid] = len(tokens)
        counts = Counter(tokens)

        for term, freq in counts.items():
            index[term][pid] = freq

    return index, doc_lengths


def search_tfidf(query, index, docs, doc_lengths, conjunctive=False):
    """
    Search the collection using TF-IDF.

    By default, a document can match any query term (OR). If conjunctive is
    True, it must contain every query term (AND).
    """
    query_tokens = clean_text(query)
    if not query_tokens:
        return []

    N = len(docs)
    scores = defaultdict(float)

    for term in query_tokens:
        if term in index:
            df = len(index[term])
            idf = math.log((N + 1) / (df + 1)) + 1
            for pid, tf in index[term].items():
                scores[pid] += tf * idf

    # For AND search, discard documents that are missing even one query term.
    if conjunctive:
        scores = {
            pid: sc
            for pid, sc in scores.items()
            if all(pid in index.get(t, {}) for t in query_tokens)
        }

    # Keep the existing document-length normalisation.
    for pid in scores:
        if doc_lengths[pid] > 0:
            scores[pid] /= doc_lengths[pid]

    ranked_results = []
    for pid, score in sorted(
        scores.items(), key=lambda x: x[1], reverse=True
    ):
        doc_data = docs[pid].copy()
        doc_data["score"] = score
        ranked_results.append(doc_data)

    return ranked_results


# ----------------------------------------------------------------------
# PART 2: EVALUATION METRICS
#
# Each metric uses rels, the binary relevance labels in ranking order
# (rank 1 first), and R, the total number of relevant documents.
# ----------------------------------------------------------------------


def precision_at_k(rels, k):
    return sum(rels[:k]) / k


def recall_at_k(rels, k, R):
    return sum(rels[:k]) / R if R else 0.0


def f1_at_k(rels, k, R):
    p, r = precision_at_k(rels, k), recall_at_k(rels, k, R)
    return 2 * p * r / (p + r) if p + r else 0.0


def average_precision_at_k(rels, k, R):
    hits, total = 0, 0.0
    for i, rel in enumerate(rels[:k], start=1):
        if rel:
            hits += 1
            total += hits / i  # Precision at each rank containing a relevant result.

    return total / min(k, R) if R else 0.0


def reciprocal_rank(rels):
    for i, rel in enumerate(rels, start=1):
        if rel:
            return 1 / i
    return 0.0


def ndcg_at_k(rels, k, R):
    dcg = sum(
        rel / math.log2(i + 1)
        for i, rel in enumerate(rels[:k], start=1)
    )
    idcg = sum(
        1 / math.log2(i + 1)
        for i in range(1, min(k, R) + 1)
    )

    return dcg / idcg if idcg else 0.0


def load_labels(path):
    """Read a labels CSV into the form {query_id: {pid: 0 or 1}}."""
    labels = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["query_id"]][row["pid"]] = int(row["labels"])

    return labels


def evaluate(queries, labels, index, docs, doc_lengths, ks=(5, 10)):
    """
    Runs each query with AND + TF-IDF and compare the judged results with
    the relevance labels. Unjudged documents are skipped. Relevant documents
    that aren't retrieved still count in R and so they have lower recall.
    """
    results = {}

    for qid, query in queries.items():
        judged = labels[qid]
        ranking = search_tfidf(
            query, index, docs, doc_lengths, conjunctive=True
        )
        rels = [
            judged[d["pid"]]
            for d in ranking
            if d["pid"] in judged
        ]
        R = sum(judged.values())

        res = {}
        for k in ks:
            res[f"P@{k}"] = precision_at_k(rels, k)
            res[f"R@{k}"] = recall_at_k(rels, k, R)
            res[f"AP@{k}"] = average_precision_at_k(rels, k, R)
            res[f"F1@{k}"] = f1_at_k(rels, k, R)
            res[f"NDCG@{k}"] = ndcg_at_k(rels, k, R)

        res["AP"] = average_precision_at_k(rels, len(rels), R)
        res["RR"] = reciprocal_rank(rels)
        results[qid] = res

    return results


def print_results(title, results):
    print(f"\n{title}\n" + "=" * 50)

    for qid, res in results.items():
        metrics = "  ".join(f"{m}={v:.3f}" for m, v in res.items())
        print(f"Query {qid}: {metrics}")

    print(
        f"MAP={sum(r['AP'] for r in results.values()) / len(results):.3f}  "
        f"MRR={sum(r['RR'] for r in results.values()) / len(results):.3f}"
    )


VALIDATION_QUERIES = {
    "1": "women full sleeve sweatshirt cotton",
    "2": "men slim jeans blue",
}

# Five test queries created for Part 1, step 2, using frequent terms.
OWN_QUERIES = {
    "1": "men black polo t shirt",
    "2": "women solid track pants",
    "3": "men casual checkered shirt",
    "4": "women round neck printed t shirt",
    "5": "men grey full sleeve sweatshirt",
}


if __name__ == "__main__":
    file_path = "fashion_products_dataset.json"

    # Load the product collection and build its inverted index.
    docs = load_data(file_path)
    index, doc_lengths = build_inverted_index(docs)

    # original example query using the default OR behaviour.
    test_query = "women full sleeve sweatshirt cotton"
    results = search_tfidf(test_query, index, docs, doc_lengths)

    print(f"Top 3 Results for Query: '{test_query}'\n" + "=" * 50)
    for doc in results[:3]:
        print(f"PID: {doc['pid']}")
        print(f"Title: {doc['title']}")
        print(f"Brand: {doc['brand']} | Price: {doc['selling_price']}")
        print(f"Score: {doc['score']:.4f}")
        print("-" * 50)

    # Part 2: compare the number of results from OR and AND retrieval.
    and_results = search_tfidf(
        test_query, index, docs, doc_lengths, conjunctive=True
    )
    print(
        f"\nAND query returned {len(and_results)} docs "
        f"(OR returned {len(results)})"
    )

    # Evaluate the two supplied validation queries.
    val_labels = load_labels("validation_labels.csv")
    print_results(
        "Validation queries",
        evaluate(VALIDATION_QUERIES, val_labels, index, docs, doc_lengths),
    )

    # Evaluat our five queries using our own relevance considerations.
    own_labels = load_labels("own_queries_labels.csv")
    print_results(
        "Our own queries",
        evaluate(OWN_QUERIES, own_labels, index, docs, doc_lengths),
    )
