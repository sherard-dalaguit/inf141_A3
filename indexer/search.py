import os
import pickle
import json
from utils import tokenize, stem
from math import log, sqrt

INDEX_PATH = os.path.join(os.path.dirname(__file__), '..', 'index', 'inverted_index.pkl')
DOC_MAP_PATH = os.path.join(os.path.dirname(__file__), '..', 'index', 'doc_id_map.json')


def load_index():
    """Loads the entire inverted index from disk."""
    if not os.path.exists(INDEX_PATH):
        print("Error: Inverted index file not found.")
        return {}
    
    with open(INDEX_PATH, 'rb') as f:
        return pickle.load(f)


def load_doc_map():
    """Loads the document ID to URL mapping and ensures doc_id lookup works correctly."""
    if not os.path.exists(DOC_MAP_PATH):
        print("Error: Document map file not found.")
        return {}

    with open(DOC_MAP_PATH, 'r') as f:
        raw_map = json.load(f)

    # Invert dictionary to make doc_id the key
    return {str(v): k for k, v in raw_map.items()}


def compute_tf_idf(term, postings, doc_frequencies, total_docs):
    """Computes TF-IDF scores for documents containing the term."""
    scores = {}
    idf = log((total_docs + 1) / (doc_frequencies.get(term, 1) + 1)) + 1  # IDF calculation
    
    for entry in postings:
        doc_id = entry.get("doc_id")
        term_count = entry.get("count", 0)  # Use 0 if 'count' is missing
        
        if term_count > 0:
            tf = 1 + log(term_count)  # Term Frequency (TF)
        else:
            tf = 0
        
        scores[doc_id] = tf * idf  # TF-IDF score
    
    return scores



def search(query):
    """Handles ranked search using TF-IDF scoring."""
    index = load_index()
    doc_map = load_doc_map()
    total_docs = len(doc_map)
    
    if not index or not doc_map:
        return []
    
    query_tokens = stem(tokenize(query))
    doc_scores = {}
    doc_frequencies = {term: len(index.get(term, [])) for term in query_tokens}
    
    for token in query_tokens:
        if token in index:
            term_scores = compute_tf_idf(token, index[token], doc_frequencies, total_docs)
            for doc_id, score in term_scores.items():
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score  # Accumulate scores
    
    ranked_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:5]

    # Debugging: Print found doc_ids before mapping
    print(f"Retrieved doc_ids for query '{query}':", [doc_id for doc_id, _ in ranked_results])

    # Ensure doc_id lookup matches data type
    return [doc_map.get(str(doc_id), f"Unknown URL for doc_id {doc_id}") for doc_id, _ in ranked_results]


if __name__ == "__main__":
    while True:
        query = input("Enter a query (or type 'stop' to exit): ")
        if query.lower() == 'stop':
            break
        
        results = search(query)
        print(f"Top results for '{query}':")
        for url in results:
            print(url)
        print()
