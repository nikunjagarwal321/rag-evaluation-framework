# Retrieval-Augmented Generation (RAG) Evaluation Framework

Retrieval-Augmented Generation (RAG) enhances large language models by incorporating external knowledge dynamically during inference. This allows models to stay up-to-date, provide source attribution, and generate more faithful outputs.  

Our project introduces a **systematic evaluation framework** for RAG systems, focusing on the key components that impact performance:  
- Retriever architectures (sparse and dense retrievers)  
- Document chunking strategies  
- Re-ranking mechanisms  
- Retrieval depth  

By analyzing these components in isolation, the framework provides insights into trade-offs between accuracy, latency, and resource usage. This helps optimize RAG systems for knowledge-intensive applications like open-domain question answering, fact verification, and conversational AI.  

The framework aims to bridge the gap between academic benchmarks and practical deployment, offering actionable guidance for designing efficient and reliable RAG pipelines.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Components](#components)
- [Evaluation Metrics](#evaluation-metrics)
- [Running Experiments](#running-experiments)
- [Project Structure](#project-structure)

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-evaluation-framework
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Key Dependencies

- `rank-bm25`: BM25 sparse retrieval
- `sentence-transformers`: Dense retrieval and embedding models
- `faiss-cpu`: Efficient similarity search
- `pandas`: Data analysis and results management
- `matplotlib`: Visualization

## Quick Start

The main evaluation script is `test.py`:

```bash
python test.py --dataset ../triviaqa-unfiltered/unfiltered-web-dev.json \
               --max_docs 300 \
               --max_qa 100 \
               --chunker fixed \
               --retrievers bm25,dpr \
               --rerankers none \
               --top_k_list 5,10
```

For interactive exploration, use `main_experiment.ipynb`.

## Components

### Chunkers

The framework supports three chunking strategies:

**Fixed Chunker**: Splits documents into fixed-size token chunks. Useful for consistent chunk sizes and when working with models that have strict input length requirements.

**Overlapping Chunker**: Creates overlapping windows to preserve context across chunk boundaries. Helps maintain semantic coherence when information spans multiple chunks.

**Semantic Chunker**: Splits documents at sentence boundaries while respecting character limits. Preserves sentence integrity and improves semantic coherence of chunks.

### Retrievers

**BM25 Retriever**: Sparse retrieval using TF-IDF based BM25 algorithm. Fast and effective for keyword-based queries, requiring no neural network.

**Dense Passage Retriever (DPR)**: Dense retrieval using sentence transformer embeddings with FAISS indexing. Provides better semantic understanding than BM25 but requires more computational resources.

**ColBERT Retriever**: Late-interaction retrieval using ColBERT architecture. Enables fine-grained token-level interactions, balancing accuracy and efficiency.

**Hybrid Retriever**: Combines sparse (BM25) and dense (DPR) retrieval scores. Leverages strengths of both approaches through configurable weighting.

### Rerankers

**No Reranking**: Passes through retrieved documents without modification.

**Embedding Reranker**: Reranks using cosine similarity between query and document embeddings.

**Cross-Encoder Reranker**: Uses a cross-encoder model to jointly score query-document pairs. Most accurate but slower, best for final ranking stage.

## Evaluation Metrics

The framework computes the following metrics:

1. **Exact Match (EM)**: Binary indicator if the gold answer appears in any retrieved passage
2. **Precision@k**: Fraction of retrieved passages containing the answer
3. **Recall@k**: Binary indicator if the answer is found
4. **Mean Reciprocal Rank (MRR)**: Reciprocal of the rank of the first relevant passage
5. **Latency**: Average retrieval time in milliseconds

## Running Experiments

### Command-Line Interface

```bash
python test.py \
    --dataset <path-to-triviaqa> \
    --max_docs 300 \
    --max_qa 100 \
    --chunker all \
    --retrievers bm25,dpr,colbert,hybrid \
    --rerankers none,embed,cross \
    --top_k_list 5,10 \
    --output_dir results \
    --seed 42
```

### Arguments

- `--dataset`: Path to TriviaQA JSON file
- `--max_docs`: Maximum number of documents to load (default: 300)
- `--max_qa`: Maximum number of QA pairs to evaluate (default: 100)
- `--chunker`: Chunker to use: `fixed`, `overlap`, `semantic`, or `all` (default: `all`)
- `--retrievers`: Comma-separated list: `bm25,dpr,colbert,hybrid` (default: `bm25,dpr`)
- `--rerankers`: Comma-separated list: `none,embed,cross` (default: `none`)
- `--top_k_list`: Comma-separated k values (default: `5,10`)
- `--output_dir`: Directory for results (default: `results`)
- `--seed`: Random seed for reproducibility (default: 42)

### Output Files

Each experiment generates:
- Summary CSV files with aggregated metrics
- Details JSON files with per-query results
- Metrics plots and comparison visualizations
- Combined results CSV for all experiments

## References

For detailed methodology and experimental results, see `CS598_Final_Project_Report.pdf`.
