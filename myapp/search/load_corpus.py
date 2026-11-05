import pandas as pd

from myapp.search.objects import Document
from myapp.search.text_preprocessing import preprocess_document
from typing import Dict


def load_corpus(path) -> Dict[str, Document]:
    """
    Load file and transform to dictionary with each document as an object for easier treatment when needed for displaying
     in results, stats, etc.
    :param path:
    :return:
    """
    df = pd.read_json(path)
    corpus = _build_corpus(df)
    return corpus

def _build_corpus(df: pd.DataFrame) -> Dict[str, Document]:
    """
    Build corpus from dataframe
    :param df:
    :return:
    """
    corpus = {}
    for _, row in df.iterrows():
        doc = Document(**row.to_dict())
        doc = doc.model_copy(update={"processed_fields": preprocess_document(doc)})
        corpus[doc.pid] = doc
    return corpus
