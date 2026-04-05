# 🏛️ Vertisa AI: Complete Project Knowledge Base

This document serves as the comprehensive, top-to-bottom technical and conceptual guide for the **Vertisa AI** project. It details the precise data flow, architecture choices, enhancements, and the codebase structure.

---

## 1. Project Overview & Motivation

**The Problem:** Standard Retrieval-Augmented Generation (RAG) systems fail on legal documents because they slice text into arbitrary token counts (e.g., 512 tokens). This often cuts complex legal clauses perfectly in half, causing the AI to miss vital context (like exceptions, conditions, or definitions). Furthermore, standard RAG lacks confidence metrics, leading to dangerous legal hallucinations.

**The Solution:** Vertisa AI is a specialized legal pipeline. It introduces **Clause-Aware Adaptive Chunking**, **Hybrid Retrieval** (Semantic + Exact Match), a **Reference-Aware Recursive GraphRAG Resolver**, a **Self-Reflection Agent**, an **Answer Tone Adjuster**, an optional **Red-Flag Risk Scanner**, a **Baseline-vs-GraphRAG Retrieval Comparison Panel**, **GraphRAG Expansion CSV Logging**, a **Claim-Evidence Audit + Auto-Repair loop**, **Grounding Trend CSV Telemetry**, and a **Multi-Agent Legal Debate View** to guarantee precision, contextual integrity, and transparent legal reasoning when answering legal queries.

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
4. **Red-Flag Risk Scan (Enhancement #2):**
   - After indexing, the app can sample the uploaded document and call Groq with a legal-risk prompt.
   - It returns a maximum of three concise risk alerts (for example: broad indemnity, unilateral termination, aggressive liability terms).
   - Users can keep it on auto mode or run it manually to reduce token usage.
5. **Graph Construction for Legal Cross-References (Enhancement #3):**
   - During indexing, the app runs a second pass to detect each chunk's identity (`Section 4.1`, `Clause 9.3`, etc.) and all explicit references (`notwithstanding Section 4.1`, `subject to Section 8`).
   - A directed `networkx.DiGraph` is built where an edge means one clause references or modifies another.
   - Override-heavy links are tagged so contradiction resolution can prioritize them at answer time.

### Phase 3: Retrieval
1. **Question Input + Tone Selection:** The user asks in plain English and optionally selects answer tone: `ELI5`, `Standard`, or `Strict Legalese`.
2. **Hybrid Search:**
   - **Dense Search (60% weight):** FAISS searches for vectors mathematically closest to the question's vector. This finds *conceptual* matches.
   - **Sparse Search (40% weight):** BM25 searches for exact keyword overlaps. This finds *literal* matches (crucial for exact legal definitions or dollar amounts).
   - The scores are normalized and combined to pull the absolute Top-K (usually Top 3) most relevant legal clauses.

### Phase 4: Retrieval Expansion (GraphRAG)
1. **Optional Multi-Hop Expansion:**
   - If GraphRAG is enabled, the system starts from the initial hybrid-retrieved Top-K chunk indices.
   - It traverses outgoing references (what this clause points to) and incoming references (what points to or overrides this clause).
   - This catches exception clauses that standard retrieval may miss.
2. **Contradiction Resolution Prompting:**
   - The expanded clause set is sent to Groq with a strict hierarchy prompt.
   - The model is instructed to resolve overrides (`notwithstanding`, `subject to`) and explicitly state naive misreadings.
3. **Operational Metrics Logging:**
   - Each answered question appends GraphRAG expansion metrics to `results/graphrag_expansion_log.csv`.
   - Logged fields include hops requested, baseline vs expanded clause counts, extra clauses added, override detection, graph trigger flags, and confidence.

### Phase 5: Generation, Evidence Audit & Reflection
1. **LLM Generation:**
   - If GraphRAG is enabled, generation uses the expanded graph-aware clause set; otherwise it uses the direct Top-3 retrieval.
   - The prompt explicitly instructs the LLM (`Llama-3.1-8B-Instant` via Groq) to *only* use the provided text and to actively cite the specific clause it uses.
   - Tone controls (`ELI5`, `Standard`, `Strict Legalese`) are injected into the prompt regardless of retrieval mode.
   - The LLM streams out a plain English answer.
2. **Claim-Evidence Audit + Auto-Repair:**
   - The generated answer is decomposed into atomic claims by a dedicated audit prompt.
   - Each claim is labeled as `SUPPORTED`, `WEAK`, or `UNSUPPORTED` against the retrieved clause context.
   - If unsupported claims are detected, the app triggers an automatic repair pass to regenerate a stricter grounded answer.
   - Repair is accepted only when grounding metrics improve (fewer unsupported claims or higher support rate without regressions).
3. **Self-Reflection Agent (Enhancement #3):**
   - The system intercepts the answer before showing it to the user.
   - A *second* LLM call is made. The LLM is given the original question, the **final** (possibly repaired) answer, and the source text.
   - It is asked to grade its own work and output a JSON object containing a `confidence` score (0-100) and a `has_hallucination` boolean flag.
   - The confidence score and reasoning are shown in the UI so users can quickly judge answer reliability.

### Phase 5B: Grounding Telemetry Logging
1. **Claim Grounding Metrics (Per Query):**
   - Every query appends grounding telemetry to `results/claim_grounding_log.csv`.
   - Logged fields include claim support rate, unsupported claim counts, repair attempted/success flags, and final grounded confidence.
2. **Trend Analytics:**
   - The UI reads recent rows and renders a confidence trend across runs, enabling paper-ready temporal analysis.

### Phase 5C: Multi-Agent Legal Debate (Enhancement #5)
1. **Parallel Persona Calls:**
   - Using the same retrieved legal context, the app sends three concurrent Groq calls:
     - `Plaintiff` (aggressive claimant interpretation)
     - `Defense` (literal protective interpretation)
     - `Judge` (balanced likely ruling)
2. **Synthesis for Legal Insight:**
   - The three outputs are rendered side-by-side so users can compare adversarial legal interpretations.
   - This makes ambiguity explicit and improves explainability for legal review.

### Phase 6: UI Delivery
1. **Streamlit Rendering:** The final answer is displayed gracefully with a colored confidence bar (Green for >75%, Yellow for 50-74%, Red for <50%).
2. **Source Citation:** The exact legal clauses used to generate the answer are attached below in expandable UI boxes so lawyers can independently verify the AI's claims.
3. **Risk Panel:** Red-flag findings are shown as a high-visibility alert card under indexing status.
4. **Retrieval Comparison Panel:** A side-by-side baseline Top-K vs GraphRAG-expanded clause panel is rendered for GraphRAG-enabled queries.
5. **GraphRAG Panel:** Graph node/edge counts, override counts, and expansion alerts are shown to explain why additional clauses were pulled.
6. **Evidence Locker Panel:** Claim support rate, unsupported claim counts, auto-repair outcome, and grounded-confidence trend are rendered for each query.
7. **Debate Panel:** Plaintiff, Defense, and Judge analyses are displayed in three columns.

---

## 3. Technology Stack

- **Frontend / Application Logic:** [Streamlit](https://streamlit.io/)
- **LLM API Engine:** [Groq](https://groq.com/) (Using `llama-3.1-8b-instant` for ultra-fast, high-quality open-source inference)
- **Vector Database:** [FAISS-CPU](https://github.com/facebookresearch/faiss)
- **Embedding Model:** [HuggingFace `sentence-transformers`](https://huggingface.co/nlpaueb/legal-bert-base-uncased) (Legal-BERT)
- **Sparse Retrieval:** `rank_bm25`
- **Graph Runtime (GraphRAG):** `networkx` (via `f2.py`)
- **Parallel Debate Runtime:** `asyncio` + thread pool executor (via `f1.py`)
- **PDF Processing:** `pypdf`
- **Data Manipulation & Evaluation:** `pandas`, `numpy`, `rouge-score`, `nltk` (for BLEU/METEOR)

---

## 4. Evaluation & Benchmarking (The Notebook)

To prove this methodology is superior to standard RAG, the project contains an extensive automated evaluation pipeline (`notebooks/Vertisa_AI.ipynb`).

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
├── app.py                  # The core Streamlit UI and execution logic
├── f1.py                   # Multi-agent legal debate engine (3 concurrent Groq calls)
├── f2.py                   # Reference-aware recursive GraphRAG engine (cross-reference graph + contradiction resolver)
├── requirements.txt        # Exact library versions required to run the pipeline
├── README.md               # Quick-start documentation for Github/Users
├── notebooks/
│   └── Vertisa_AI.ipynb     # The testing, benchmarking, and development laboratory
├── results/                
│   ├── results_summary.csv        # Aggregated performance comparisons
│   ├── results_clause_method.csv  # Raw data/answers from our advanced logic
│   ├── results_fixed_method.csv   # Raw data/answers from the flawed baseline logic
│   ├── graphrag_expansion_log.csv # Per-question GraphRAG expansion telemetry
│   └── claim_grounding_log.csv    # Per-question claim grounding + auto-repair telemetry
├── graphs/                 
│   ├── graph_rouge_comparison.png # Visual proof of ROUGE superiority
│   └── graph_meteor_bleu.png      # Semantic metric visualizer
└── docs/                   
    └── Project_Knowledge_Base.md  # (This file) Complete technical documentation
```
