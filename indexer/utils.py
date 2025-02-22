import re
from typing import List
from bs4 import BeautifulSoup


def extract_text(content: str) -> str:
	"""
	Extracts and cleans text from HTML content.

	Args:
		content (str): The HTML content to be cleaned.

	Returns:
		str: The cleaned text extracted from the HTML content.
	"""
	soup = BeautifulSoup(content, 'html.parser')

	# remove script and style elements
	for script_or_style in soup(['script', 'style']):
		script_or_style.decompose()

	text = soup.get_text()

	# replace multiple spaces with a single space
	text = re.sub(r'\s+', ' ', text)
	text = text.strip()

	return text


def tokenize(text: str) -> List[str]:
	"""
	Tokenizes the input text into a list of lowercase words.

	Args:
		text (str): The text to be tokenized.

	Returns:
		List[str]: A list of tokens (words) extracted from the text.
	"""
	return re.findall(r'\b\w+\b', text.lower())


def stem(tokens: List[str]) -> List[str]:
	"""
	Applies stemming to a list of tokens using the Porter stemmer.

	Args:
		tokens (List[str]): The list of tokens to be stemmed.

	Returns:
		List[str]: A list of stemmed tokens.
	"""
	from nltk.stem import PorterStemmer
	stemmer = PorterStemmer()
	return [stemmer.stem(token) for token in tokens]
