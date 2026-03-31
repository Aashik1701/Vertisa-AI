# 🏛️ Vertisa AI: Complete Project Knowledge Base

This document serves as the comprehensive, top-to-bottom technical and conceptual guide for the **Vertisa AI** project. It details the precise data flow, architecture choices, enhancements, and the codebase structure.

---

## 1. Project Overview & Motivation

**The Problem:** Standard Retrieval-Augmented Generation (RAG) systems fail on legal documents because they slice text into arbitrary token counts (e.g., 512 tokens). This often cuts complex legal clauses perfectly in half, causing the AI to miss vital context (like exceptions, conditions, or definitions). Furthermore, standard RAG lacks confidence metrics, leading to dangerous legal hallucinations.

**The Solution:** Vertisa AI is a specialized legal pipeline. It introduces **Clause-Aware Adaptive Chunking**, **Hybrid Retrieval** (Semantic + Exact Match), and a **Self-Reflection Agent** to guarantee precision, contextual integrity, and verifiable confidence when answering legal queries.

---

## 2. The Core Architecture: Data Flow

When a user uploads a PDF and asks a question, the data moves through the following pipeline:

### Phase 1: Ingestion & Parsing
1. **PDF Upload:** The user uploads a PDF document (e.g., a commercial contract or privacy policy).
2. **Text Extraction:** Using `pypdf`, the system parses the binary PDF into raw text strings, preserving page numbers for potential citation.

### Phase 2: Processing & Indexing
1. **Clause-Aware Chunking (Enhancement #1):**
   - Instead of breaking text every $X$ words, the `ClauseAwareChunker` uses a series of complex Regular Expressions (Regex).
   - It searches for natural legal boundaries such as `SECTION 1`, `ARTICLE IV`, `WHEREAS`, `CONFIDENTIALITY`, `TERMINATION`, etc.
   - It separates the document exactly at these boundaries so that a single "chunk" contains one unbroken, logically complete legal clause.
2. **Semantic Embedding (Dense):**
   - Each clause is passed through the `nlpaueb/legal-bert-base-uncased` language model via `sentence-transformers`.
   - This translates the English text into high-dimensional vector representations optimized specifically for legal terminology.
   - These vectors are stored in **FAISS** (Facebook AI Similarity Search) for blazing-fast similarity retrieval.
3. **Keyword Indexing (Sparse):**
   - Simultaneously, the clauses are tokenized and indexed using **BM25 (Okapi)**, assigning statistical weights to rare/specific words.

### Phase 3: Retrieval
1. **Query Expansion:** The user's plain-English question is expanded (e.g., "NDA" becomes "non-disclosure agreement") to maximize search capture.
2. **Hybrid Search:**
   - **Dense Search (60% weight):** FAISS searches for vectors mathematically closest to the question's vector. This finds *conceptual* matches.
   - **Sparse Search (40% weight):** BM25 searches for exact keyword overlaps. This finds *literal* matches (crucial for exact legal definitions or dollar amounts).
   - The scores are normalized and combined to pull the absolute Top-K (usually Top 3) most relevant legal clauses.

### Phase 4: Generation & Reflection
1. **LLM Generation:**
   - The Top-3 retrieved clauses and the user's question are packaged into a strict prompt.
   - The prompt explicitly instructs the LLM (`Llama-3.1-8B-Instant` via Groq) to *only* use the provided text and to actively cite the specific clause it uses.
   - The LLM streams out a plain English answer.
2. **Self-Reflection Agent (Enhancement #2):**
   - The system intercepts the answer before showing it to the user.
   - A *second* LLM call is made. The LLM is given the original question, the generated answer, and the source text.
   - It is asked to grade its own work and output a JSON object containing a `confidence` score (0-100) and a `has_hallucination` boolean flag.
   - **Auto-Refinement:** If the confidence falls below 60%, the system automatically rewrites the question and triggers Phase 3 again to find better context.

### Phase 5: UI Delivery
1. **Streamlit Rendering:** The final answer is displayed gracefully with a colored confidence bar (Green for >75%, Yellow for 50-74%, Red for <50%).
2. **Source Citation:** The exact legal clauses used to generate the answer are attached below in expandable UI boxes so lawyers can independently verify the AI's claims.

---

## 3. Technology Stack

- **Frontend / Application Logic:** [Streamlit](https://streamlit.io/)
- **LLM API Engine:** [Groq](https://groq.com/) (Using `llama-3.1-8b-instant` for ultra-fast, high-quality open-source inference)
- **Vector Database:** [FAISS-CPU](https://github.com/facebookresearch/faiss)
- **Embedding Model:** [HuggingFace `sentence-transformers`](https://huggingface.co/nlpaueb/legal-bert-base-uncased) (Legal-BERT)
- **Sparse Retrieval:** `rank_bm25`
- **PDF Processing:** `pypdf`
- **Data Manipulation & Evaluation:** `pandas`, `numpy`, `rouge-score`, `nltk` (for BLEU/METEOR)

---

## 4. Evaluation & Benchmarking (The Notebook)

To prove this methodology is superior to standard RAG, the project contains an extensive automated evaluation pipeline (`notebooks/Vertisa AI_RAG.ipynb`).

### The Datasets (Enhancement #3: Generalizability)
The system is automatically evaluated against three distinct HuggingFace legal datasets to prove it works across multiple domains of law:
1. **CUAD:** Commercial contracts (highly complex logic).
2. **MAUD:** Merger Agreements (M&A deal documents).
3. **PrivacyQA:** Software and App Privacy Policies.

### The Methodology Comparison
The notebook runs the identical questions and documents through two distinct pipelines:
- **Baseline ("Fixed"):** Cuts the document blindly every 512 tokens.
- **Ours ("Clause"):** Uses our intelligent legal-boundary regex chunker.

### Automated Metrics
Both answers are compared to the "Golden" reference answers in the datasets using standard NLP metrics:
- **ROUGE-1 / ROUGE-2 / ROUGE-L:** Measures exact word overlaps, bigram overlaps, and the longest common subsequence to prove the AI captured the same facts as the human reference.
- **METEOR & BLEU:** Standard translation metrics adapted to ensure the semantic alignment and fluency of the generated answer matches the ground truth.

**Results Output:** The final averages are saved directly to `results/results_summary.csv` and visualized using `matplotlib` / `seaborn` graphs stored in the `/graphs` directory. Our metrics conclusively prove that Clause-Aware chunking vastly outperforms blind token chunking natively in legal text.

---

## 5. Directory Structure Overview

```text
/PROJECT/RAG langchain/
├── app.py                  # The core Streamlit UI and execution logic
├── requirements.txt        # Exact library versions required to run the pipeline
├── README.md               # Quick-start documentation for Github/Users
├── notebooks/
│   └── Vertisa AI_RAG.ipynb # The testing, benchmarking, and development laboratory
├── results/                
│   ├── results_summary.csv        # Aggregated performance comparisons
│   ├── results_clause_method.csv  # Raw data/answers from our advanced logic
│   └── results_fixed_method.csv   # Raw data/answers from the flawed baseline logic
├── graphs/                 
│   ├── graph_rouge_comparison.png # Visual proof of ROUGE superiority
│   └── graph_meteor_bleu.png      # Semantic metric visualizer
└── docs/                   
    └── Project_Knowledge_Base.md  # (This file) Complete technical documentation
```
