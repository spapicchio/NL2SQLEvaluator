import re
from decimal import Decimal
from pathlib import Path
from typing import Optional, Any

import bm25s

_ws_collapse = re.compile(r"\s+")


def _looks_numeric(val: Any) -> bool:
    """Return True if val is numeric or a numeric-looking string (incl. negatives, decimals)."""
    if isinstance(val, (int, float, Decimal)):
        return True
    if val is None:
        return False
    s = str(val).strip()
    if s == "":
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def _normalize(s: str) -> str:
    """Lowercase, collapse whitespace, trim, and cap length to 40 chars."""
    s = str(s).strip()
    s = _ws_collapse.sub(" ", s).lower()
    return s[:40]


def _process_corpus(corpus: list[str]) -> list[str]:
    # Filter out numeric-looking items and normalize
    corpus = [_normalize(v) for v in corpus if v != "" and not _looks_numeric(v)]
    # Deduplicate after normalization
    corpus = list({v for v in corpus if v})
    return corpus


def create_bm25_index(corpus: list[str], path_for_bm25_index: Optional[str] = None):
    # Create the BM25 model and index the corpus
    path = Path(path_for_bm25_index) if path_for_bm25_index is not None else None
    if path is not None and path.exists():
        retriever = bm25s.BM25.load(path, load_corpus=True)

    else:
        corpus = _process_corpus(corpus)
        if len(corpus) == 0:
            return None
        retriever = bm25s.BM25(corpus=corpus)
        tokenized = bm25s.tokenize(corpus)
        if len(tokenized.vocab) == 0:
            return None
        retriever.index(tokenized)
        if path is not None:
            # Save the index to disk
            path.mkdir(parents=True, exist_ok=True)
            retriever.save(path)
    return retriever


def retrieve_from_bm25_index(query: str, retriever: bm25s.BM25, top_k=2):
    # Load the BM25 model and index the corpus
    if top_k <= 0:
        raise ValueError("k must be a positive integer")
    top_k = top_k if top_k <= retriever.scores['num_docs'] else retriever.scores['num_docs']
    query = _process_corpus([query])[0]
    # Query the corpus and get top-k results
    docs, scores = retriever.retrieve(bm25s.tokenize(query), k=top_k)
    # Let's see what we got!
    docs = [doc['text'] if isinstance(doc, dict) else doc for doc in docs[0]]
    scores = scores[0]
    return docs


if __name__ == '__main__':
    corpus = ["a cat is a feline and likes to purr",
              "a dog is the human's best friend and loves to play",
              "a bird is a beautiful animal that can fly",
              "a fish is a creature that lives in water and swims"]

    retriever = create_bm25_index(corpus, path_for_bm25_index=None)
    retrieve_from_bm25_index(query='The cat is a lovely pet', retriever=retriever, top_k=2)
