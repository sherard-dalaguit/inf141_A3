import os
import json
import pickle
from collections import defaultdict
from typing import Dict, List
from utils import tokenize, stem, extract_text

DOC_ID_MAP: Dict[str, int] = {}
INVERTED_INDEX: Dict[str, List[Dict[str, int]]] = defaultdict(list)
DOCUMENT_COUNT: int = 0
UNIQUE_TOKENS = set()


def traverse_data_directory(data_dir: str):
	"""
	Traverses the DEV data directory and processes each file to build the inverted index.

	Args:
		data_dir (str): The path to the data directory containing the files to be indexed.
	"""
	global DOCUMENT_COUNT
	for root, _, files in os.walk(data_dir):
		for file in files:
			if file.endswith('.json'):
				file_path = os.path.join(root, file)
				doc_id = len(DOC_ID_MAP)
				process_file_content(file_path, doc_id)
				DOCUMENT_COUNT += 1


def process_file_content(file_path: str, doc_id: int):
	"""
	Processes the content of a file, extracts text, tokenizes, stems, and updates the inverted index.

	Args:
		file_path (str): The path to the file to be processed.
		doc_id (int): The document ID assigned to the file.
	"""
	with open(file_path, 'r', encoding='utf-8') as file:
		if file_path.endswith('.json'):
			data = json.load(file)
			content = data.get('content', '')
			text = extract_text(content)

		# maps the document URL to the document ID
		document_url = data["url"]
		DOC_ID_MAP[document_url] = doc_id

		tokens = tokenize(text)
		stemmed_tokens = stem(tokens)
		update_inverted_index(stemmed_tokens, doc_id)


def update_inverted_index(tokens: List[str], doc_id: int):
	"""
	Updates the in-memory inverted index with the tokens from a document.

	Args:
		tokens (List[str]): The list of tokens extracted from the document.
		doc_id (int): The document ID of the document being processed.
	"""
	term_freq = defaultdict(int)
	for token in tokens:
		term_freq[token] += 1
		UNIQUE_TOKENS.add(token)

	for token, freq in term_freq.items():
		INVERTED_INDEX[token].append({'doc_id': doc_id, 'term_freq': freq})


def serialize_index(output_dir: str):
	"""
	Serializes the inverted index to disk and prints index statistics.

	Args:
		output_dir (str): The directory where the serialized index will be saved.
	"""
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	index_path = os.path.join(output_dir, 'inverted_index.pkl')
	with open(index_path, 'wb') as file:
		pickle.dump(INVERTED_INDEX, file)

	index_size = os.path.getsize(index_path) / 1024  # size in kilobytes

	# generate index statistics for report
	report_lines = []
	report_lines.append(f"Total documents: {DOCUMENT_COUNT}")
	report_lines.append(f"Total unique tokens: {len(UNIQUE_TOKENS)}")
	report_lines.append(f"Index size: {index_size:.2f} KB")

	report = "\n".join(report_lines)
	report_path = os.path.join(output_dir, 'report.txt')
	with open(report_path, 'w') as file:
		file.write(report)

	print(f"Report saved to: {report_path}")


if __name__ == "__main__":
	data_dir = os.path.join(os.path.dirname(__file__), '..', 'DEV')
	output_dir = os.path.join(os.path.dirname(__file__), '..', 'index')

	traverse_data_directory(data_dir)
	serialize_index(output_dir)
