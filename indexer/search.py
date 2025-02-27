import os
import pickle
import json
from utils import tokenize, stem
from math import log, sqrt

INDEX_PATH = os.path.join(os.path.dirname(__file__), '..', 'index', 'inverted_index.pkl')
DOC_MAP_PATH = os.path.join(os.path.dirname(__file__), '..', 'index', 'doc_id_map.json')

def load_term_postings(term):
    """Loads postings for a specific term from disk without loading the entire index."""
    with open(INDEX_PATH, 'rb') as f:
        while True:
            try:
                term_index = pickle.load(f)
                if term in term_index:
                    return term_index[term]
            except EOFError:
                break  # Reached end of file
    
    return []  # Term not found

def search(query):
    """Handles AND queries by retrieving and intersecting postings."""
    query_tokens = stem(tokenize(query))
    relevant_docs = None

    for token in query_tokens:
        postings = load_term_postings(token)
        docs_with_term = {entry["doc_id"] for entry in postings}
        relevant_docs = docs_with_term if relevant_docs is None else relevant_docs & docs_with_term

    if not relevant_docs:
        return []

    # Load document ID to URL mapping
    with open(DOC_MAP_PATH, 'r') as f:
        doc_id_map = json.load(f)

    return [url for url, doc_id in doc_id_map.items() if doc_id in relevant_docs][:5]

if __name__ == "__main__":
    queries = [
        "Iftekhar Ahmed",
        "machine learning",
        "ACM",
        "master of software engineering"
    ]

    for query in queries:
        results = search(query)
        print(f"Top results for '{query}':")
        for url in results:
            print(url)
        print()
