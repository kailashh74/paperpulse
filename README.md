# PaperPulse

**PaperPulse** is a research intelligence platform that helps users explore, understand, and track AI research papers through hybrid retrieval, confidence-aware responses, and live fallback search.

Unlike traditional chatbot-style RAG systems, PaperPulse is designed to prioritize retrieval quality, source transparency, and honest responses when evidence is insufficient.

---

## Features

### Hybrid Retrieval

Combines:

* Semantic Search (FAISS + MiniLM embeddings)
* BM25 Keyword Search
* Reciprocal Rank Fusion (RRF)

This enables PaperPulse to retrieve both semantically similar papers and exact keyword matches.

### Confidence-Aware Responses

Every query passes through a confidence routing layer.

* High-confidence queries are answered directly from the indexed research corpus.
* Low-confidence queries trigger a live Semantic Scholar fallback search.
* Out-of-domain queries are identified and handled gracefully.

### Weekly Research Ingestion

PaperPulse continuously updates its knowledge base by:

* Fetching new papers from arXiv
* Tracking multiple AI research categories
* Avoiding duplicate indexing using hash-based deduplication
* Automatically updating the retrieval index

### Follow-Up Question Support

The system detects follow-up questions such as:

* "Explain in simpler terms"
* "Tell me more"
* "Give an example"

and reuses previously retrieved sources to maintain context.

### Research Digest

Generates structured summaries of recent papers:

* What the paper does
* Why it matters
* Prerequisites required to understand it

### Topic Pulse

Tracks research activity across AI topics such as:

* LLMs
* Agents
* Computer Vision
* Multimodal AI
* Retrieval-Augmented Generation (RAG)

and allows users to directly explore recent developments.

### Source-Backed Answers

Every response includes:

* Retrieved source papers
* Confidence information
* Transparent grounding

---

## System Architecture

PaperPulse follows the pipeline below:

arXiv API
→ Weekly Ingestion
→ Hash Deduplication
→ MiniLM Embeddings
→ FAISS Index

Parallel:

Metadata
→ BM25 Index

User Query
→ Hybrid Retrieval (BM25 + Semantic)
→ Confidence Check

High Confidence:
→ Answer from Indexed Papers

Low Confidence:
→ Semantic Scholar Live Search

Retrieved Context
→ Llama 3.3 70B (Groq)
→ Final Response

---

## Tech Stack

### Retrieval

* FAISS
* BM25 (rank-bm25)
* Sentence Transformers
* all-MiniLM-L6-v2

### LLM

* Llama 3.3 70B
* Groq API

### Data Sources

* arXiv API
* Semantic Scholar API

### Frontend

* Streamlit

### Language

* Python

---

## Evaluation

PaperPulse was evaluated using a benchmark of 20 natural-language research questions spanning:

* Large Language Models
* Agents
* Retrieval-Augmented Generation
* Computer Vision
* Multimodal AI
* Safety and Alignment
* Graph Learning
* Federated Learning

Evaluation corpus:

* 16,647 indexed AI research papers

Results:

| Metric    | Score |
| --------- | ----- |
| Recall@5  | 75%   |
| Recall@10 | 85%   |
| MRR       | 0.62  |

These metrics evaluate the ability of the retrieval system to surface relevant research papers for realistic user queries.

---

## Example Queries

* Summarize recent AI papers and explain why they matter
* Explain the architecture of Vision Transformers
* How can language models use external knowledge sources?
* What are the latest developments in LLM agents?
* Explain AI applications in healthcare

---

## Installation

Clone the repository:

```bash
git clone https://github.com/kailashh74/paperpulse.git
cd paperpulse
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

---

## Future Improvements

* Cross-encoder reranking
* Improved topic trend analytics
* Citation graph exploration
* Advanced evaluation benchmarks
* Multi-paper synthesis workflows

---

## Author

Kailash S

B.Tech Artificial Intelligence & Data Science (Medical Engineering)

Amrita Vishwa Vidyapeetham

---

