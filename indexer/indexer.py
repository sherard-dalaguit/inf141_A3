# this file will contain code that reads & processes the data, and builds the inverted index
# TODO: create a doc_id_map to assign each file/document a numeric ID
# TODO: traverse the DEV data directory and read each HTML/JSON file
# TODO: parse each file's content (clean HTML, tokenize, stem, etc.) using helper functions from utils.py
# TODO: update the in-memory inverted index with the tokens for each document
# TODO: track how many documents are processed
# TODO: track how many unique tokens have been encountered
# TODO: (optional) for large datasets, write partial indexes to disk and merge them later
# TODO: after building the full index, serialize it to disk in the index/ directory
# TODO: compute and print the total size of the index on disk
