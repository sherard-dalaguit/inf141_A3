import os
import pickle
import json
import tkinter as tk
from utils import tokenize, stem
from math import log
from tkinter import scrolledtext, messagebox

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
        term_count = entry.get("term_freq", 0)  # Use 0 if 'term_freq' is missing
        
        if term_count > 0:
            tf = 1 + log(term_count)  # Term Frequency (TF)
        else:
            tf = 0
        
        scores[doc_id] = tf * idf  # TF-IDF score
    
    return scores


def search(query, index, doc_map):
    """Handles ranked search using TF-IDF scoring using preloaded index and doc_map."""
    total_docs = len(doc_map)

    if not index or not doc_map:
        return []

    query_tokens = stem(tokenize(query))
    if not query_tokens:
        return []

    # For each token, get the set of doc IDs that contain it
    postings_sets = []
    for token in query_tokens:
        if token in index:
            doc_ids = {entry["doc_id"] for entry in index[token]}
            postings_sets.append(doc_ids)
        else:
            return []

    valid_doc_ids = set.intersection(*postings_sets) if postings_sets else set()
    if not valid_doc_ids:
        return []

    doc_scores = {doc_id: 0 for doc_id in valid_doc_ids}
    doc_frequencies = {term: len(index.get(term, [])) for term in query_tokens}

    for token in query_tokens:
        if token in index:
            filtered_postings = [entry for entry in index[token] if entry["doc_id"] in valid_doc_ids]
            term_scores = compute_tf_idf(token, filtered_postings, doc_frequencies, total_docs)
            for doc_id, score in term_scores.items():
                doc_scores[doc_id] += score  # Accumulate scores

    ranked_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"Retrieved doc_ids for query '{query}':", [doc_id for doc_id, _ in ranked_results])

    return [doc_map.get(str(doc_id), f"Unknown URL for doc_id {doc_id}") for doc_id, _ in ranked_results]


class SearchGUI(tk.Tk):
    def __init__(self, index, doc_map):
        super().__init__()
        self.title("IN4MATX 141 Search Engine")
        self.geometry("600x400")
        self.index = index
        self.doc_map = doc_map
        self.create_widgets()

    def create_widgets(self):
        self.label = tk.Label(self, text="Enter your query (or type 'stop' to exit):")
        self.label.pack(pady=10)

        self.query_entry = tk.Entry(self, width=50)
        self.query_entry.pack(pady=5)

        self.search_button = tk.Button(self, text="Search", command=self.perform_search)
        self.search_button.pack(pady=5)

        self.results_box = scrolledtext.ScrolledText(self, wrap=tk.WORD, width=70, height=15)
        self.results_box.pack(pady=10)

    def perform_search(self):
        """
        Executes the search query entered by the user and displays the results.

        This function retrieves the query from the input field, performs the search
        using the preloaded index and document map, and displays the top results
        in the results box. If the query is 'stop', the application will close.

        Raises:
            messagebox.showwarning: If the query is empty.
        """
        query = self.query_entry.get().strip()
        if query.lower() == 'stop':
            self.destroy()
            return
        if not query:
            messagebox.showwarning("Empty Query", "Please enter a query.")
            return
        results = search(query, self.index, self.doc_map)
        self.results_box.delete("1.0", tk.END)
        if results:
            self.results_box.insert(tk.END, f"Top Results for {query}:\n")
            for index, url in enumerate(results):
                self.results_box.insert(tk.END, f"{index + 1}) " + url + "\n")
        else:
            self.results_box.insert(tk.END, "No results found.")


if __name__ == "__main__":
    index = load_index()
    doc_map = load_doc_map()
    app = SearchGUI(index, doc_map)
    app.mainloop()
