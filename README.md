# 🚀 HybridMind: Intent-Aware & Explainable Hybrid Retrieval System

> An end-to-end **Intent-Aware Hybrid Retrieval System** that combines **LLMs, Knowledge Graphs, BM25, Semantic Search, Reciprocal Rank Fusion (RRF), and Cross-Encoder Re-ranking** to deliver accurate and explainable retrieval results.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![OpenSearch](https://img.shields.io/badge/OpenSearch-BM25-orange)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-green)
![Neo4j](https://img.shields.io/badge/Neo4j-Knowledge_Graph-blue)
![Gemini](https://img.shields.io/badge/LLM-Gemini_2.5_Flash-purple)
![License](https://img.shields.io/badge/License-MIT-success)

---

# 📌 Overview

Traditional keyword search often fails to understand user intent, while purely semantic search may ignore important lexical matches.

**HybridMind** addresses this problem by combining:

- 🔍 BM25 Keyword Search (OpenSearch)
- 🧠 Dense Semantic Retrieval (FAISS)
- 🌐 Knowledge Graph Expansion (Neo4j)
- 🤖 LLM-based Intent Detection (Gemini 2.5 Flash)
- ⚡ Reciprocal Rank Fusion (RRF)
- 🎯 Cross-Encoder Re-ranking
- 📖 Explainable Retrieval

The result is an **end-to-end explainable hybrid retrieval system** capable of understanding user intent and producing highly relevant search results.

---

# 🏗️ System Architecture

> Add the architecture diagram below after uploading it to the repository.

```text
                     User Query
                          │
                          ▼
                 Gemini 2.5 Flash
          (Intent + Entity Extraction)
                          │
                          ▼
                 Structured JSON
                          │
          ┌───────────────┴────────────────┐
          ▼                                ▼
  Knowledge Graph                 Graph Relationship Retrieval
          │                      (Uses Structured JSON directly)
          ▼                                │
  Graph Query Expansion                     │
          │                                │
          ▼                                │
     Expanded Query                        │
          │                                │
     ┌────┴─────┐                          │
     ▼          ▼                          │
   BM25      Semantic                      │
     │          │                          │
     └────┬─────┘                          │
          ▼                                │
          RRF                              │
          │                                │
          └──────────────┬─────────────────┘
                         ▼
          Graph-aware Relevance Scoring
                         │
                         ▼
        Final Hybrid Ranking (RRF + Graph)
                         │
                         ▼
              Cross Encoder Re-ranker
                         │
                         ▼
                 Final Ranked Results
```

---

# ✨ Features

- Intent-aware retrieval using Gemini 2.5 Flash
- Entity extraction from natural language queries
- Knowledge Graph based query expansion
- BM25 keyword retrieval with OpenSearch
- Dense semantic retrieval using FAISS
- Reciprocal Rank Fusion (RRF)
- Graph-aware relevance scoring
- Cross-Encoder re-ranking
- Explainable search results
- Modular architecture for easy extension
- Comprehensive evaluation framework

---

# 🛠️ Tech Stack

| Component | Technology |
|------------|------------|
| Programming Language | Python |
| LLM | Google Gemini 2.5 Flash |
| Keyword Search | OpenSearch (BM25) |
| Semantic Search | FAISS |
| Embedding Model | all-MiniLM-L6-v2 |
| Knowledge Graph | Neo4j |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Re-ranking | Cross Encoder |
| Data Processing | Pandas, NumPy |
| Evaluation | Custom Evaluation Framework |

---

# 📂 Project Structure

```text
HybridMind/
│
├── api/
├── data/
│   ├── raw/
│   ├── cleaned/
│   └── chunks/
│
├── embeddings/
├── faiss/
├── graph/
├── llm/
├── opensearch/
├── reranker/
├── retrieval/
├── evaluation/
├── validation/
│
├── pipeline.py
├── chunk_data.py
├── generate_embeddings.py
├── requirements.txt
└── README.md
```

---

# 🔄 Retrieval Pipeline

### Step 1

User enters a natural language query.

Example:

```
Looking for a Python Backend Developer with FastAPI experience in Pune.
```

---

### Step 2

Gemini performs:

- Intent Detection
- Entity Extraction

Example output:

```json
{
  "intent":"profile_search",
  "skills":["Python","FastAPI"],
  "location":"Pune",
  "role":"Backend Developer"
}
```

---

### Step 3

Knowledge Graph expands the query.

Example:

```
Python
↓

FastAPI
↓

REST API

↓

Docker

↓

PostgreSQL

↓

Microservices
```

---

### Step 4

Parallel Retrieval

- BM25 (OpenSearch)
- Semantic Search (FAISS)

---

### Step 5

RRF combines both rankings.

---

### Step 6

Knowledge Graph computes relationship scores.

---

### Step 7

Cross Encoder performs final re-ranking.

---

### Step 8

Explainable results are returned.

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/VISHALMEENAMEENA/Intent-Aware-_Retrieval-_System-.git
cd Intent-Aware-_Retrieval-_System-
```

Create virtual environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

Linux/Mac

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ⚙️ Environment Variables

Create

```
api/llm.env
```

Example

```
GEMINI_API=YOUR_API_KEY

NEO4J_URI=bolt://localhost:7687

NEO4J_USERNAME=neo4j

NEO4J_PASSWORD=your_password

OPENSEARCH_HOST=http://localhost:9200
```

---

# ▶️ Running the Pipeline

```bash
python pipeline.py --query "Looking for a Python Backend Developer with FastAPI experience in Pune."
```

---

# 📊 Evaluation

The repository contains an evaluation framework for:

- Precision@K
- Recall@K
- MRR
- nDCG
- MAP
- Graph Expansion Precision
- RRF Evaluation
- Cross Encoder Evaluation

Run:

```bash
python evaluation/evaluate_pipeline.py
```

---

# 📈 Example Technologies Used

- Google Gemini 2.5 Flash
- OpenSearch
- Neo4j
- FAISS
- Sentence Transformers
- Cross Encoder
- Reciprocal Rank Fusion
- Knowledge Graphs
- Python

---

# 🔮 Future Improvements

- ColBERT Re-ranking
- Multi-hop Graph Reasoning
- Personalized Retrieval
- Multi-modal Retrieval
- Graph Neural Networks
- Agentic Query Planning
- Distributed Vector Search

---

# 👨‍💻 Author

**Vishal Meena**

B.Tech, Mathematics & Computing  
Indian Institute of Technology Mandi

GitHub:

https://github.com/VISHALMEENAMEENA

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
