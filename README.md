informatics 141: assignment 3 milestone 1

presumed file structure

assignment3/
├── data/
│   └── ... (the HTML / JSON files you’re indexing) ...
├── indexer/
│   ├── indexer.py          (code that reads & processes the data, builds the inverted index)
│   └── utils.py            (optional, helper functions: tokenization, stemming, etc.)
├── search/                 (you won’t need this for MS#1, but you’ll need it for MS#2 & MS#3)
│   └── search.py           (code that loads the index and performs queries)
├── index/                  (directory to store your on-disk index files, if you choose)
├── README.md               (optional, or you can just have a short readme in your report)
└── report.pdf              (the short report for milestone #1)