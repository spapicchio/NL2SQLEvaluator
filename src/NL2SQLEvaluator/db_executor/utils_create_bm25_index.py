from pathlib import Path
from typing import Optional

import bm25s


def create_bm25_index(corpus: list[str], path_for_bm25_index: Optional[str] = None):
    # Create the BM25 model and index the corpus
    path = Path(path_for_bm25_index) if path_for_bm25_index is not None else None
    if path is not None and path.exists():
        retriever = bm25s.BM25.load(path, load_corpus=True)

    else:
        retriever = bm25s.BM25(corpus=corpus)
        retriever.index(bm25s.tokenize(corpus))
        if path is not None:
            # Save the index to disk
            path.mkdir(parents=True, exist_ok=True)
            retriever.save(path)
    return retriever


def retrieve_from_bm25_index(query: str, retriever, top_k=2):
    # Load the BM25 model and index the corpus
    if top_k <= 0:
        raise ValueError("k must be a positive integer")
    if top_k > len(corpus):
        raise ValueError("k must be less than or equal to the number of documents in the corpus")
    # Query the corpus and get top-k results
    docs, scores = retriever.retrieve(bm25s.tokenize(query), k=top_k)
    # Let's see what we got!
    print(docs)
    docs = [doc['text'] if isinstance(doc, dict) else doc for doc in docs[0]]
    scores = scores[0]
    for doc, score in zip(docs, scores):
        print(f"Rank (score: {score}): {docs}")


if __name__ == '__main__':
    corpus = ["a cat is a feline and likes to purr",
              "a dog is the human's best friend and loves to play",
              "a bird is a beautiful animal that can fly",
              "a fish is a creature that lives in water and swims"]

    retriever = create_bm25_index(corpus, path_for_bm25_index=None)
    retrieve_from_bm25_index(query='The cat is a lovely pet', retriever=retriever, top_k=2)
