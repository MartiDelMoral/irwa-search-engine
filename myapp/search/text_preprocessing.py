import re
from typing import Any, Iterator, Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from myapp.search.objects import Document


def _load_stop_words() -> set[str]:
    try:
        return set(stopwords.words("english"))
    except LookupError:
        if not nltk.download("stopwords", quiet=True):
            raise RuntimeError("NLTK English stop words are unavailable.")
        try:
            return set(stopwords.words("english"))
        except LookupError as error:
            raise RuntimeError("NLTK English stop words could not be loaded.") from error


STOP_WORDS = _load_stop_words()
STEMMER = SnowballStemmer("english")


def clean_text(text: Optional[str]) -> list[str]:
    if not text:
        return []

    tokens = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    return [STEMMER.stem(token) for token in tokens if token not in STOP_WORDS]


def _text_values(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, nested_value in value.items():
            yield str(key)
            yield from _text_values(nested_value)
    elif isinstance(value, (list, tuple)):
        for nested_value in value:
            yield from _text_values(nested_value)
    elif value is not None:
        yield str(value)


def preprocess_document(document: Document) -> dict[str, list[str]]:
    product_details = " ".join(_text_values(document.product_details or {}))
    return {
        "title": clean_text(document.title),
        "description": clean_text(document.description),
        "brand": clean_text(document.brand),
        "category": clean_text(document.category),
        "sub_category": clean_text(document.sub_category),
        "product_details": clean_text(product_details),
        "seller": clean_text(document.seller),
    }
