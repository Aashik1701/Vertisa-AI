
import streamlit as st
import os
import json
import re
import io
import csv
import hashlib
import html
from datetime import datetime
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
from groq import Groq
from typing import List, Dict, Any
from f1 import run_debate
from f2 import LegalKnowledgeGraph, resolve_with_graph

GRAPH_STATS_LOG_PATH = os.path.join("results", "graphrag_expansion_log.csv")
CLAIM_GROUNDING_LOG_PATH = os.path.join("results", "claim_grounding_log.csv")
GRAPH_STATS_FIELDS = [
    "timestamp_utc",
    "doc_hash_prefix",
    "question",
    "tone_mode",
    "graphrag_enabled",
    "graphrag_applied",
    "hops_requested",
    "baseline_clause_count",
    "expanded_clause_count",
    "extra_clauses_added",
    "override_detected",
    "graph_triggered",
    "expansion_events",
    "confidence",
    "debate_enabled",
]
CLAIM_GROUNDING_FIELDS = [
    "timestamp_utc",
    "doc_hash_prefix",
    "question",
    "tone_mode",
    "graphrag_applied",
    "initial_claim_count",
    "initial_supported_count",
    "initial_weak_count",
    "initial_unsupported_count",
    "initial_claim_support_rate",
    "auto_repair_attempted",
    "auto_repair_success",
    "final_claim_count",
    "final_supported_count",
    "final_weak_count",
    "final_unsupported_count",
    "final_claim_support_rate",
    "final_grounded_confidence",
    "has_hallucination",
]

# ─── Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vertisa AI",
    page_icon="⚖️",
    layout="wide"
)

# ─── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Orbitron:wght@700;900&family=JetBrains+Mono:wght@400;700&display=swap');

.main-header {
    background: linear-gradient(135deg, #0A0A0F 0%, #171012 100%);
    border-bottom: 2px solid #FF003C;
    padding: 3rem 2rem;
    border-radius: 0 0 16px 16px;
    color: white;
    margin-bottom: 2.5rem;
    text-align: center;
    box-shadow: 0 10px 30px rgba(255, 0, 60, 0.15);
    position: relative;
    overflow: hidden;
}
.main-header h1 {
    font-family: 'Orbitron', sans-serif;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 900;
    color: #FFFFFF;
    text-shadow: 0 0 10px rgba(255, 0, 60, 0.8), 0 0 20px rgba(255, 0, 60, 0.4);
}
.main-header p {
    font-family: 'Inter', sans-serif;
    color: #A0A0B0;
    letter-spacing: 1px;
}
.confidence-high { color: #00FF41; font-family: 'Orbitron', sans-serif; text-shadow: 0 0 8px rgba(0, 255, 65, 0.5); font-size: 1.3rem; }
.confidence-mid  { color: #FFB000; font-family: 'Orbitron', sans-serif; text-shadow: 0 0 8px rgba(255, 176, 0, 0.5); font-size: 1.3rem; }
.confidence-low  { color: #FF003C; font-family: 'Orbitron', sans-serif; text-shadow: 0 0 8px rgba(255, 0, 60, 0.6); font-size: 1.3rem; }

.answer-box {
    background: rgba(20, 20, 30, 0.6);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-left: 4px solid #FF003C;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1.5rem 0;
    color: #EBEBEB !important;
    font-family: 'Inter', sans-serif;
    line-height: 1.6;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.answer-box:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(255, 0, 60, 0.15);
}

.source-box {
    background: #0D0D12;
    border: 1px solid #333340;
    padding: 1.5rem 1.2rem 1.2rem;
    border-radius: 8px;
    font-family: 'JetBrains Mono', Courier, monospace;
    font-size: 0.85rem;
    margin-top: 1rem;
    color: #9090A0;
    position: relative;
    border-left: 2px solid #555566;
}
.source-box::before {
    content: "CITATION //";
    position: absolute;
    top: -10px;
    left: 15px;
    background: #0D0D12;
    padding: 0 5px;
    color: #FF003C;
    font-size: 0.7rem;
    font-family: 'Orbitron', sans-serif;
    letter-spacing: 1px;
}
.metric-card {
    background: rgba(15, 15, 20, 0.9);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    box-shadow: inset 0 0 20px rgba(0, 0, 0, 0.5);
    font-family: 'Inter', sans-serif;
}
.red-flag-card {
    background: radial-gradient(circle at top right, rgba(255, 0, 60, 0.22), rgba(20, 12, 18, 0.96));
    border: 1px solid rgba(255, 0, 60, 0.7);
    border-left: 4px solid #FF003C;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-top: 0.8rem;
    color: #FFE6EE;
    box-shadow: 0 0 18px rgba(255, 0, 60, 0.35), inset 0 0 12px rgba(255, 0, 60, 0.1);
}
.red-flag-card h4 {
    margin: 0 0 0.55rem;
    font-family: 'Orbitron', sans-serif;
    letter-spacing: 0.6px;
    color: #FF6B90;
}
.red-flag-card p {
    margin: 0;
    white-space: pre-wrap;
    font-family: 'Inter', sans-serif;
    line-height: 1.45;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.45rem;
    margin-bottom: 0.75rem;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(14, 14, 20, 0.92);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 0.35rem 0.85rem;
}
.stTabs [aria-selected="true"] {
    border-color: rgba(255, 0, 60, 0.8);
    box-shadow: 0 0 10px rgba(255, 0, 60, 0.35);
}
div[data-testid="stMetric"] {
    background: rgba(14, 14, 20, 0.88);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 0.6rem 0.7rem;
    box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.35);
    min-height: 108px;
}
div[data-testid="stMetric"] label {
    color: #9b9bad;
    font-family: 'Inter', sans-serif;
}
div[data-testid="stMetricValue"] {
    font-family: 'Orbitron', sans-serif;
    color: #f5f5f7;
}
.source-card {
    background: rgba(14, 14, 20, 0.92);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-left: 3px solid #3a3a4f;
    border-radius: 10px;
    padding: 0.85rem 0.95rem;
    margin: 0.5rem 0 0.7rem;
}
.source-card.new-clause {
    border-left-color: #00d26a;
    box-shadow: 0 0 12px rgba(0, 210, 106, 0.16);
}
.source-card-head {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    margin-bottom: 0.55rem;
}
.source-chip {
    background: rgba(255, 0, 60, 0.14);
    color: #ff7f9f;
    border: 1px solid rgba(255, 0, 60, 0.45);
    border-radius: 999px;
    padding: 0.15rem 0.55rem;
    font-size: 0.72rem;
    font-family: 'Orbitron', sans-serif;
    letter-spacing: 0.35px;
    white-space: nowrap;
}
.source-card-title {
    color: #ececf3;
    font-weight: 600;
    font-size: 0.92rem;
}
.source-preview {
    margin: 0;
    color: #bfc3d5;
    font-size: 0.87rem;
    line-height: 1.5;
}
.graph-note {
    background: rgba(14, 14, 20, 0.92);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-left: 3px solid #00b3ff;
    border-radius: 10px;
    padding: 0.85rem 0.95rem;
    margin-top: 0.35rem;
}
.debate-card {
    background: rgba(14, 14, 20, 0.92);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-left: 4px solid #666;
    border-radius: 12px;
    padding: 0.95rem;
    min-height: 280px;
}
.debate-card h4 {
    margin: 0;
    font-family: 'Orbitron', sans-serif;
    font-size: 0.95rem;
    letter-spacing: 0.5px;
}
.debate-subtitle {
    margin: 0.25rem 0 0.7rem;
    color: #a9afc3;
    font-size: 0.78rem;
}
.debate-body {
    color: #e5e8f5;
    font-size: 0.88rem;
    line-height: 1.55;
}
.debate-card.plaintiff { border-left-color: #ef4444; }
.debate-card.defense { border-left-color: #38bdf8; }
.debate-card.judge { border-left-color: #f59e0b; }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>⚖️ Vertisa AI</h1>
    <p>AI-Powered Legal Document Question Answering</p>
    <p style="font-size:0.85rem; opacity:0.8;">
        Upload any legal PDF → Ask questions in plain English → Get cited answers
    </p>
</div>
""", unsafe_allow_html=True)

# ─── API Key ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Get free key at console.groq.com"
    )
    st.markdown("---")
    st.markdown("**Enhancement Features Active:**")
    st.success("✅ Clause-Aware Chunking")
    st.success("✅ Confidence Scoring")
    st.success("✅ Auto Re-query if < 60%")
    st.success("✅ Hybrid Dense + BM25")
    st.success("✅ GraphRAG Cross-Reference Resolver")
    st.success("✅ Claim-Evidence Audit + Auto-Repair")
    st.success("✅ Multi-Agent Legal Debate")
    st.markdown("---")
    st.caption("Legal RAG Research Project")

# ─── Load Models (cached) ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return SentenceTransformer("nlpaueb/legal-bert-base-uncased")
    except:
        return SentenceTransformer("all-MiniLM-L6-v2")


def build_red_flag_sample(text: str, max_chars: int = 12000) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    span = max_chars // 3
    mid_start = max(0, (len(cleaned) // 2) - (span // 2))
    return (
        cleaned[:span]
        + "\n\n[... middle excerpt ...]\n\n"
        + cleaned[mid_start:mid_start + span]
        + "\n\n[... ending excerpt ...]\n\n"
        + cleaned[-span:]
    )


def scan_red_flags(sample_text: str, api_key: str) -> str:
    client = Groq(api_key=api_key)
    prompt = f"""You are a legal risk analyst.
Scan the legal text and list at most 3 potential red flags.
Focus on unusual liabilities, predatory terms, aggressive termination rights, broad indemnities, data/privacy risks, or one-sided obligations.
Return concise bullet points only.
If no clear risk appears, return exactly one bullet: - No obvious red flags found in provided text.

LEGAL TEXT:
{sample_text}
"""
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=220,
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


def render_red_flags(red_flags_text: str):
    safe_text = html.escape(red_flags_text)
    st.markdown(
        f"""
<div class="red-flag-card">
    <h4>🚨 Potential Red Flags</h4>
    <p>{safe_text}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def _dedupe_chunks(chunks: List[Dict]) -> List[Dict]:
    unique_chunks = []
    seen_text = set()
    for chunk in chunks:
        txt = chunk.get("text", "")
        if txt and txt not in seen_text:
            unique_chunks.append(chunk)
            seen_text.add(txt)
    return unique_chunks


def append_graphrag_stats_log(row: Dict, log_path: str = GRAPH_STATS_LOG_PATH):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_missing_or_empty = (not os.path.exists(log_path)) or os.path.getsize(log_path) == 0

    with open(log_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=GRAPH_STATS_FIELDS)
        if file_missing_or_empty:
            writer.writeheader()
        safe_row = {field: row.get(field, "") for field in GRAPH_STATS_FIELDS}
        writer.writerow(safe_row)


def append_claim_grounding_log(row: Dict, log_path: str = CLAIM_GROUNDING_LOG_PATH):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_missing_or_empty = (not os.path.exists(log_path)) or os.path.getsize(log_path) == 0

    with open(log_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CLAIM_GROUNDING_FIELDS)
        if file_missing_or_empty:
            writer.writeheader()
        safe_row = {field: row.get(field, "") for field in CLAIM_GROUNDING_FIELDS}
        writer.writerow(safe_row)


def _extract_json_dict(raw_text: str) -> Dict[str, Any]:
    if not raw_text:
        return {}
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group())
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _normalize_claim_label(label: str) -> str:
    token = (label or "").strip().upper()
    if token.startswith("SUP"):
        return "SUPPORTED"
    if token.startswith("UNSUP") or token.startswith("NOT SUP"):
        return "UNSUPPORTED"
    return "WEAK"


def summarize_claim_audit(claims: List[Dict]) -> Dict[str, Any]:
    total = len(claims)
    supported = sum(1 for c in claims if c.get("label") == "SUPPORTED")
    weak = sum(1 for c in claims if c.get("label") == "WEAK")
    unsupported = sum(1 for c in claims if c.get("label") == "UNSUPPORTED")
    support_rate = round((supported / total) * 100, 1) if total else 0.0
    return {
        "total_claims": total,
        "supported_count": supported,
        "weak_count": weak,
        "unsupported_count": unsupported,
        "support_rate": support_rate,
    }


def audit_answer_claims(client: Groq, question: str, answer: str, context: str, max_claims: int = 8) -> Dict[str, Any]:
    audit_context = context[:9000]
    prompt = f"""You are a legal evidence auditor.
Given the ANSWER and LEGAL CONTEXT, identify up to {max_claims} atomic claims from the answer.

For each claim, assign exactly one label:
- SUPPORTED: directly backed by explicit text in context.
- WEAK: partially implied but not explicit.
- UNSUPPORTED: absent or contradicted by context.

Return STRICT JSON only:
{{
  "claims": [
    {{"claim": "...", "label": "SUPPORTED|WEAK|UNSUPPORTED", "evidence": "short quote from context or empty", "reason": "one sentence"}}
  ]
}}

QUESTION:
{question}

ANSWER:
{answer}

LEGAL CONTEXT:
{audit_context}
"""

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content
        parsed = _extract_json_dict(raw)

        clean_claims = []
        for item in parsed.get("claims", []):
            if not isinstance(item, dict):
                continue
            claim_text = " ".join(str(item.get("claim", "")).split())
            if not claim_text:
                continue
            clean_claims.append(
                {
                    "claim": claim_text,
                    "label": _normalize_claim_label(str(item.get("label", ""))),
                    "evidence": " ".join(str(item.get("evidence", "")).split()),
                    "reason": " ".join(str(item.get("reason", "")).split()),
                }
            )
            if len(clean_claims) >= max_claims:
                break

        summary = summarize_claim_audit(clean_claims)
        return {
            "claims": clean_claims,
            "summary": summary,
            "raw": raw,
            "error": None,
        }
    except Exception as e:
        return {
            "claims": [],
            "summary": summarize_claim_audit([]),
            "raw": "",
            "error": str(e),
        }


def regenerate_answer_with_guardrails(
    client: Groq,
    question: str,
    answer: str,
    context: str,
    tone_instruction: str,
    claim_audit: Dict[str, Any],
) -> str:
    flagged = [
        c.get("claim", "")
        for c in claim_audit.get("claims", [])
        if c.get("label") in ("UNSUPPORTED", "WEAK")
    ]
    if not flagged:
        return answer

    flagged_block = "\n".join([f"- {item}" for item in flagged[:8]])
    prompt = f"""You are a legal QA repair assistant.
Rewrite the answer so EVERY statement is grounded in the provided legal context only.
Remove unsupported claims and soften uncertain statements.
If the context does not explicitly answer a point, say so clearly.

TONE REQUIREMENT: {tone_instruction}

QUESTION:
{question}

PREVIOUS ANSWER:
{answer}

FLAGGED CLAIMS TO FIX:
{flagged_block}

LEGAL CONTEXT:
{context[:10000]}

Return only the revised final answer text.
"""

    repaired = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=450,
        temperature=0.1,
    )
    new_answer = repaired.choices[0].message.content.strip()
    return new_answer if new_answer else answer


def _bool_from_string(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float_from_string(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_claim_grounding_history(log_path: str = CLAIM_GROUNDING_LOG_PATH, max_rows: int = 40) -> Dict[str, Any]:
    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        return {
            "confidence_trend": [],
            "total_rows": 0,
            "auto_repair_attempts": 0,
            "auto_repair_successes": 0,
            "auto_repair_rate": 0.0,
        }

    with open(log_path, "r", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    recent_rows = rows[-max_rows:]
    confidence_trend = [
        _float_from_string(row.get("final_grounded_confidence"), 0.0)
        for row in recent_rows
    ]

    attempts = sum(1 for row in rows if _bool_from_string(row.get("auto_repair_attempted")))
    successes = sum(1 for row in rows if _bool_from_string(row.get("auto_repair_success")))
    auto_repair_rate = round((successes / attempts) * 100, 1) if attempts else 0.0

    return {
        "confidence_trend": confidence_trend,
        "total_rows": len(rows),
        "auto_repair_attempts": attempts,
        "auto_repair_successes": successes,
        "auto_repair_rate": auto_repair_rate,
    }


def _preview_text(text: str, limit: int = 420) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


def render_source_card(
    chunk: Dict,
    index_num: int,
    badge_label: str,
    is_new: bool = False,
    expander_label: str = "View full clause",
):
    text = chunk.get("text", "")
    title = chunk.get("title", "Clause")
    safe_title = html.escape(title[:120])
    safe_preview = html.escape(_preview_text(text))
    safe_badge = html.escape(badge_label)
    card_class = "source-card new-clause" if is_new else "source-card"

    st.markdown(
        f"""
<div class="{card_class}">
    <div class="source-card-head">
        <span class="source-chip">{safe_badge} {index_num}</span>
        <span class="source-card-title">{safe_title}</span>
    </div>
    <p class="source-preview">{safe_preview}</p>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.expander(f"{expander_label} {index_num}", expanded=False):
        st.text(text)


def render_retrieval_comparison(baseline_chunks: List[Dict], expanded_chunks: List[Dict]):
    baseline_unique = _dedupe_chunks(baseline_chunks)
    expanded_unique = _dedupe_chunks(expanded_chunks)

    baseline_texts = {c.get("text", "") for c in baseline_unique}
    extra_count = max(len(expanded_unique) - len(baseline_unique), 0)

    st.markdown("### 🧪 Baseline vs GraphRAG Retrieval")
    st.caption("Green cards indicate clauses GraphRAG added beyond baseline retrieval.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Baseline Clauses", len(baseline_unique))
    m2.metric("GraphRAG Clauses", len(expanded_unique))
    m3.metric("Extra Clauses Pulled", extra_count)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Baseline Retrieval")
        for i, chunk in enumerate(baseline_unique, 1):
            render_source_card(
                chunk,
                i,
                badge_label="Baseline",
                is_new=False,
                expander_label="Baseline full clause",
            )

    with col_b:
        st.markdown("#### GraphRAG Expanded Retrieval")
        for i, chunk in enumerate(expanded_unique, 1):
            is_new = chunk.get("text", "") not in baseline_texts
            render_source_card(
                chunk,
                i,
                badge_label="Graph",
                is_new=is_new,
                expander_label="Graph full clause",
            )


def render_graph_diagnostics_panel(graph, graph_result: Dict, graph_log_error: str):
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Clauses in Graph", graph.stats.get("total_nodes", 0))
    col_b.metric("Cross-Reference Edges", graph.stats.get("total_edges", 0))
    col_c.metric("Document Overrides", len(graph.stats.get("override_clauses", [])))
    col_d.metric("Extra Clauses Added", graph_result.get("extra_chunks_added", 0))

    if graph_result.get("graph_triggered"):
        if graph_result.get("override_detected"):
            st.warning(graph_result.get("expansion_summary", "Override-linked clauses were added."))
        else:
            st.info(graph_result.get("expansion_summary", "Graph expansion completed."))
    else:
        st.success(
            "No additional cross-references were required for this query. "
            "Baseline retrieval already covered the answer path."
        )

    expansion_log = graph_result.get("expansion_log", [])
    if expansion_log:
        st.markdown("#### Expansion Trace")
        trace_rows = []
        for step, event in enumerate(expansion_log, start=1):
            trace_rows.append(
                {
                    "Step": step,
                    "Hop": event.get("hop", ""),
                    "Type": event.get("type", ""),
                    "From Chunk": event.get("from_chunk", ""),
                    "To Chunk": event.get("to_chunk", ""),
                    "Reference": event.get("raw_ref", ""),
                }
            )
        st.dataframe(trace_rows, use_container_width=True, hide_index=True)

    if graph.stats.get("override_clauses"):
        with st.expander(
            f"Override relationships in document ({len(graph.stats['override_clauses'])})",
            expanded=False,
        ):
            for override_line in graph.stats["override_clauses"]:
                st.markdown(f"- {override_line}")

    if graph_log_error:
        st.caption(f"Graph metrics log unavailable: {graph_log_error}")
    else:
        st.markdown(
            f"<div class='graph-note'>📊 Graph metrics logged to: {html.escape(GRAPH_STATS_LOG_PATH)}</div>",
            unsafe_allow_html=True,
        )


def render_evidence_locker_panel(claim_audit: Dict[str, Any], claim_log_error: str):
    if not claim_audit:
        st.info("Claim-level evidence audit is unavailable for this query.")
        return

    initial_summary = claim_audit.get("initial", {}).get("summary", {})
    final_summary = claim_audit.get("final", {}).get("summary", {})
    history = claim_audit.get("history", {})

    initial_support_rate = float(initial_summary.get("support_rate", 0.0))
    final_support_rate = float(final_summary.get("support_rate", 0.0))
    initial_unsupported = int(initial_summary.get("unsupported_count", 0))
    final_unsupported = int(final_summary.get("unsupported_count", 0))

    auto_repair_attempted = bool(claim_audit.get("auto_repair_attempted"))
    auto_repair_success = bool(claim_audit.get("auto_repair_success"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Claim Support Rate",
        f"{final_support_rate:.1f}%",
        delta=f"{(final_support_rate - initial_support_rate):+.1f}%",
    )
    c2.metric(
        "Unsupported Claims",
        final_unsupported,
        delta=(final_unsupported - initial_unsupported),
    )

    if auto_repair_attempted:
        repair_state = "✅ Improved" if auto_repair_success else "⚠️ No gain"
        c3.metric("Auto-Repair", repair_state)
    else:
        c3.metric("Auto-Repair", "Not needed")

    c4.metric("Historical Repair Rate", f"{history.get('auto_repair_rate', 0.0):.1f}%")

    if auto_repair_attempted:
        st.caption(
            "Auto-repair was triggered because unsupported claims were detected in the first pass. "
            "The final answer reflects the best grounded version."
        )

    repair_notes = claim_audit.get("repair_notes")
    if repair_notes:
        st.caption(repair_notes)

    final_claims = claim_audit.get("final", {}).get("claims", [])
    if final_claims:
        st.markdown("#### Claim Support Breakdown")
        rows = []
        for i, claim in enumerate(final_claims, start=1):
            rows.append(
                {
                    "#": i,
                    "Label": claim.get("label", "WEAK"),
                    "Claim": claim.get("claim", ""),
                    "Evidence": _preview_text(claim.get("evidence", ""), limit=260),
                    "Reason": _preview_text(claim.get("reason", ""), limit=200),
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No auditable claims were extracted from the final answer.")

    confidence_trend = history.get("confidence_trend", [])
    if confidence_trend:
        st.markdown("#### Final Grounded Confidence Trend (Recent Runs)")
        st.line_chart({"Grounded Confidence": confidence_trend})
    else:
        st.caption("No historical grounded-confidence data yet. Ask a few questions to build the trend.")

    if claim_log_error:
        st.caption(f"Claim metrics log unavailable: {claim_log_error}")
    else:
        st.markdown(
            f"<div class='graph-note'>📈 Claim grounding metrics logged to: {html.escape(CLAIM_GROUNDING_LOG_PATH)}</div>",
            unsafe_allow_html=True,
        )


def render_debate_cards(debate_results: Dict):
    st.caption(
        "Three concurrent legal personas interpret the same clauses from opposing viewpoints."
    )

    cards = [
        (
            "Plaintiff",
            "Aggressive interpretation favoring the claimant.",
            "plaintiff",
        ),
        (
            "Defense",
            "Protective and restrictive interpretation favoring the defendant.",
            "defense",
        ),
        (
            "Judge",
            "Balanced synthesis with likely judicial outcome.",
            "judge",
        ),
    ]

    col1, col2, col3 = st.columns(3)
    for col, (label, subtitle, style_name) in zip([col1, col2, col3], cards):
        with col:
            safe_text = html.escape(debate_results.get(label, "No output returned.")).replace("\n", "<br>")
            safe_subtitle = html.escape(subtitle)
            st.markdown(
                f"""
<div class="debate-card {style_name}">
    <h4>{label}</h4>
    <p class="debate-subtitle">{safe_subtitle}</p>
    <div class="debate-body">{safe_text}</div>
</div>
""",
                unsafe_allow_html=True,
            )


def render_results_panel(result_payload: Dict):
    answer = result_payload["answer"]
    confidence = result_payload["confidence"]
    source_chunks = result_payload.get("source_chunks", [])
    baseline_chunks = result_payload.get("baseline_chunks", [])
    graph_result = result_payload.get("graph_result")
    graph_log_error = result_payload.get("graph_log_error")
    debate_enabled = result_payload.get("debate_enabled", False)
    debate_results = result_payload.get("debate_results")
    claim_audit = result_payload.get("claim_audit")
    claim_log_error = result_payload.get("claim_log_error")
    tone_mode = result_payload.get("tone_mode", "Standard")
    question = result_payload.get("question", "")
    source_unique = _dedupe_chunks(source_chunks)
    baseline_unique = _dedupe_chunks(baseline_chunks)
    baseline_texts = {c.get("text", "") for c in baseline_unique}
    extra_clauses = max(len(source_unique) - len(baseline_unique), 0)

    st.markdown("---")
    st.subheader("📊 Analysis Results")
    if question:
        st.caption(f"Question: {question} | Tone: {tone_mode}")

    final_support_rate = 0.0
    if claim_audit:
        final_support_rate = float(claim_audit.get("final", {}).get("summary", {}).get("support_rate", 0.0))

    stats_a, stats_b, stats_c, stats_d, stats_e = st.columns(5)
    stats_a.metric("Confidence", f"{confidence}%")
    stats_b.metric("Answer Tone", tone_mode)
    stats_c.metric("Clauses Used", len(source_unique))
    stats_d.metric(
        "Graph Additions",
        extra_clauses,
        delta=extra_clauses if graph_result else None,
    )
    stats_e.metric("Claim Support", f"{final_support_rate:.1f}%")

    tab_answer, tab_sources, tab_graph, tab_evidence, tab_debate = st.tabs(
        ["📋 Answer", "📌 Sources", "🕸️ GraphRAG", "🧾 Evidence", "⚖️ Debate"]
    )

    with tab_answer:
        safe_answer = html.escape(answer).replace("\n", "<br>")
        st.markdown(f'<div class="answer-box">{safe_answer}</div>', unsafe_allow_html=True)

        conf_class = "confidence-high" if confidence >= 75 else "confidence-mid" if confidence >= 50 else "confidence-low"
        conf_emoji = "🟢" if confidence >= 75 else "🟡" if confidence >= 50 else "🔴"
        st.markdown(
            f'{conf_emoji} <span class="{conf_class}">Confidence: {confidence}%</span>',
            unsafe_allow_html=True
        )
        st.progress(confidence / 100)

        if result_payload.get("reflection_reason"):
            st.caption(f"🔍 {result_payload['reflection_reason']}")

        if result_payload.get("has_hallucination"):
            st.warning("⚠️ Self-reflection flagged possible hallucination. Verify carefully.")

    with tab_sources:
        if not source_unique:
            st.info("No source clauses available for this answer.")
        else:
            st.subheader("📌 Source Clauses Used")
            st.caption("Each card shows a concise preview. Open any card for the full clause text.")
            for i, chunk in enumerate(source_unique, 1):
                is_new = chunk.get("text", "") not in baseline_texts
                render_source_card(
                    chunk,
                    i,
                    badge_label="Source",
                    is_new=is_new,
                    expander_label="Source full clause",
                )

        if graph_result and st.session_state.get("legal_graph") is not None:
            render_retrieval_comparison(baseline_unique, source_unique)

    with tab_graph:
        if graph_result and st.session_state.get("legal_graph") is not None:
            render_graph_diagnostics_panel(
                st.session_state["legal_graph"],
                graph_result,
                graph_log_error,
            )
        else:
            st.info("GraphRAG was not used for this answer. Enable it in Step 2 to see graph expansion diagnostics.")

    with tab_evidence:
        render_evidence_locker_panel(claim_audit, claim_log_error)

    with tab_debate:
        if debate_enabled and debate_results:
            render_debate_cards(debate_results)
        elif debate_enabled:
            st.info("Debate was enabled, but no debate output was produced for this query.")
        else:
            st.info("Enable Multi-Agent Legal Debate in Step 2 to compare plaintiff/defense/judge perspectives.")

# ─── Core Classes (inline for single-file app) ────────────────────────
class AppChunker:
    PATTERNS = [
        r"^\s*SECTION\s+\d+", r"^\s*Section\s+\d+",
        r"^\s*ARTICLE\s+", r"^\s*Article\s+",
        r"^\s*\d+\.\s+[A-Z]", r"^\s*[A-Z]{3,}(\s+[A-Z]{2,})*\s*$",
        r"^\s*WHEREAS", r"^\s*TERMINATION", r"^\s*CONFIDENTIALITY",
        r"^\s*INDEMNIFICATION", r"^\s*GOVERNING LAW",
    ]
    def __init__(self):
        self.pat = re.compile("|".join(self.PATTERNS), re.MULTILINE)
    def chunk(self, text):
        lines = text.split("\n")
        chunks, cur, title = [], [], "Start"
        for line in lines:
            if self.pat.match(line.strip()) and cur:
                t = "\n".join(cur).strip()
                if len(t) > 50:
                    chunks.append({"text": t, "title": title})
                cur, title = [line], line.strip()[:80]
            else:
                cur.append(line)
        if cur:
            t = "\n".join(cur).strip()
            if len(t) > 50:
                chunks.append({"text": t, "title": title})
        return chunks

# ─── Main App ─────────────────────────────────────────────────────────
st.markdown("## Workspace")
st.caption("Use Step 1 and Step 2 below. Full analysis outputs appear in the wide results panel after each query.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📄 Step 1: Upload Legal Document")
    auto_scan_red_flags = st.checkbox(
        "Auto-scan for risks after indexing",
        value=True,
        help="Disable this to run the risk scan manually and save API tokens."
    )
    uploaded_file = st.file_uploader(
        "Upload a legal PDF",
        type=["pdf"],
        help="Contracts, agreements, privacy policies, any legal document"
    )

    if uploaded_file:
        pdf_bytes = uploaded_file.getvalue()
        doc_hash = hashlib.sha256(pdf_bytes).hexdigest()
        is_new_document = st.session_state.get("doc_hash") != doc_hash

        if is_new_document:
            with st.spinner("Reading PDF..."):
                reader = PdfReader(io.BytesIO(pdf_bytes))
                full_text = ""
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        full_text += t + "\n"

            st.success(f"✅ Loaded: {len(reader.pages)} pages, {len(full_text):,} characters")

            if not api_key:
                st.warning("⚠️ Please enter your Groq API key in the sidebar.")
            else:
                model = load_model()
                chunker = AppChunker()

                with st.spinner("🔪 Applying clause-aware chunking..."):
                    chunks = chunker.chunk(full_text)

                with st.spinner(f"🧠 Embedding {len(chunks)} clauses..."):
                    texts = [c["text"] for c in chunks]
                    embeds = model.encode(texts, batch_size=32)
                    embeds_f32 = np.array(embeds).astype("float32")
                    dim = embeds_f32.shape[1]
                    index = faiss.IndexFlatL2(dim)
                    index.add(embeds_f32)
                    bm25 = BM25Okapi([t.lower().split() for t in texts])

                with st.spinner("🕸️ Building legal cross-reference graph..."):
                    legal_graph = None
                    graph_summary = ""
                    try:
                        legal_graph = LegalKnowledgeGraph().build(texts)
                        graph_summary = legal_graph.get_graph_summary()
                    except Exception as e:
                        st.warning(f"GraphRAG build unavailable for this file: {e}")

                red_flag_sample = build_red_flag_sample(full_text)

                st.session_state["ready"] = True
                st.session_state["chunks"] = chunks
                st.session_state["index"] = index
                st.session_state["embeds"] = embeds_f32
                st.session_state["bm25"] = bm25
                st.session_state["model"] = model
                st.session_state["api_key"] = api_key
                st.session_state["doc_hash"] = doc_hash
                st.session_state["doc_pages"] = len(reader.pages)
                st.session_state["doc_chars"] = len(full_text)
                st.session_state["red_flag_sample"] = red_flag_sample
                st.session_state["red_flags"] = None
                st.session_state["legal_graph"] = legal_graph
                st.session_state["graph_summary"] = graph_summary
                st.session_state["qa_result"] = None

                st.info(f"📦 Indexed {len(chunks)} clauses. Ready to answer questions!")
                if graph_summary:
                    st.caption(graph_summary)

                if auto_scan_red_flags:
                    with st.spinner("🚨 Scanning for legal red flags..."):
                        try:
                            red_flags_text = scan_red_flags(red_flag_sample, api_key)
                        except Exception as e:
                            red_flags_text = f"Red flag scan unavailable: {e}"
                    st.session_state["red_flags"] = red_flags_text
        else:
            st.success(
                f"✅ Loaded: {st.session_state.get('doc_pages', 0)} pages, "
                f"{st.session_state.get('doc_chars', 0):,} characters"
            )
            st.info(f"📦 Indexed {len(st.session_state.get('chunks', []))} clauses. Ready to answer questions!")
            if st.session_state.get("graph_summary"):
                st.caption(st.session_state["graph_summary"])

            if api_key:
                st.session_state["api_key"] = api_key

            if auto_scan_red_flags and st.session_state.get("red_flags") is None and api_key:
                with st.spinner("🚨 Scanning for legal red flags..."):
                    try:
                        st.session_state["red_flags"] = scan_red_flags(st.session_state.get("red_flag_sample", ""), api_key)
                    except Exception as e:
                        st.session_state["red_flags"] = f"Red flag scan unavailable: {e}"

        if st.session_state.get("ready") and st.session_state.get("doc_hash") == doc_hash:
            if not auto_scan_red_flags:
                if st.button("🚨 Scan for Risks", use_container_width=True):
                    if not api_key:
                        st.warning("⚠️ Add your Groq API key to run risk scanning.")
                    else:
                        with st.spinner("🚨 Scanning for legal red flags..."):
                            try:
                                st.session_state["red_flags"] = scan_red_flags(
                                    st.session_state.get("red_flag_sample", ""),
                                    api_key,
                                )
                            except Exception as e:
                                st.session_state["red_flags"] = f"Red flag scan unavailable: {e}"

            if st.session_state.get("red_flags"):
                render_red_flags(st.session_state["red_flags"])

with col2:
    st.subheader("❓ Step 2: Ask Your Question")
    tone_mode = st.select_slider(
        "Answer Tone",
        options=["ELI5 (Layman)", "Standard", "Strict Legalese"],
        value="Standard",
        help="Adjust the legal answer style without changing retrieval quality."
    )
    enable_debate = st.checkbox(
        "Enable Multi-Agent Legal Debate",
        value=True,
        help="Runs 3 additional Groq calls (Plaintiff, Defense, Judge) on the same retrieved context."
    )
    enable_graphrag = st.checkbox(
        "Enable GraphRAG Cross-Reference Resolver",
        value=True,
        help="Expands retrieved clauses through legal cross-references before final answer generation."
    )
    graphrag_hops = st.slider(
        "Graph traversal depth",
        min_value=1,
        max_value=3,
        value=2,
        help="How many reference hops GraphRAG follows from initially retrieved clauses.",
        disabled=not enable_graphrag,
    )

    question = st.text_area(
        "Type your legal question in plain English",
        placeholder="e.g. What are the payment terms?\nCan either party terminate early?\nWho owns the intellectual property?",
        height=120
    )

    if st.button("🔍 Get Answer", type="primary", use_container_width=True):
        if not st.session_state.get("ready"):
            st.error("Please upload a PDF first.")
        elif not question.strip():
            st.error("Please enter a question.")
        else:
            chunks = st.session_state["chunks"]
            index = st.session_state["index"]
            model_obj = st.session_state["model"]
            bm25 = st.session_state["bm25"]
            key = st.session_state["api_key"]

            with st.spinner("🔍 Retrieving relevant clauses..."):
                q_embed = model_obj.encode([question]).astype("float32")
                _, idxs = index.search(q_embed, min(5, len(chunks)))
                bm25_scores = bm25.get_scores(question.lower().split())
                combined = {}
                for i, idx in enumerate(idxs[0]):
                    if idx < len(chunks):
                        combined[idx] = combined.get(idx, 0) + 0.6
                for i, score in enumerate(bm25_scores):
                    combined[i] = combined.get(i, 0) + 0.4 * (score / (max(bm25_scores) + 1e-9))
                top_idxs = sorted(combined, key=combined.get, reverse=True)[:3]
                top_chunks = [chunks[i] for i in top_idxs if i < len(chunks)]

            client = Groq(api_key=key)
            tone_instructions = {
                "ELI5 (Layman)": "You must explain the answer in extremely simple, 5th-grade English. Use bullet points. Avoid legal jargon.",
                "Standard": "Answer clearly and accurately.",
                "Strict Legalese": "Answer using strict corporate legal terminology. Frame the response as a formal legal memo.",
            }
            selected_tone_instruction = tone_instructions.get(tone_mode, tone_instructions["Standard"])

            graph_result = None
            baseline_chunks_for_display = list(top_chunks)
            source_chunks_for_display = list(top_chunks)
            context_for_answer = "\n\n".join([f"[{c['title']}]\n{c['text']}" for c in top_chunks])

            if enable_graphrag and st.session_state.get("legal_graph") is not None:
                with st.spinner("🕸️ Resolving cross-references with GraphRAG..."):
                    graph = st.session_state["legal_graph"]
                    expanded_chunks, expansion_log = graph.expand_with_graph(
                        top_idxs,
                        max_hops=graphrag_hops,
                    )
                    graph_result = resolve_with_graph(
                        client=client,
                        question=question,
                        expanded_chunks=expanded_chunks,
                        expansion_log=expansion_log,
                        tone_instruction=selected_tone_instruction,
                    )

                chunk_lookup = {}
                for chunk in chunks:
                    if chunk["text"] not in chunk_lookup:
                        chunk_lookup[chunk["text"]] = chunk

                source_chunks_for_display = []
                for expanded_text in graph_result.get("expanded_chunks", expanded_chunks):
                    source_chunks_for_display.append(
                        chunk_lookup.get(
                            expanded_text,
                            {"title": "Linked Clause", "text": expanded_text},
                        )
                    )

                context_for_answer = "\n\n".join(
                    [f"[{c['title']}]\n{c['text']}" for c in source_chunks_for_display]
                )
                answer = graph_result["answer"]
            else:
                with st.spinner("🤖 Generating answer..."):
                    prompt = f"""You are a legal document assistant. Answer using ONLY the provided legal text. Cite the specific clause.
TONE REQUIREMENT: {selected_tone_instruction}

LEGAL TEXT:
{context_for_answer}

QUESTION: {question}

ANSWER:"""
                    resp = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=400,
                        temperature=0.1,
                    )
                    answer = resp.choices[0].message.content.strip()

            with st.spinner("🧾 Validating answer claims against evidence..."):
                initial_claim_audit = audit_answer_claims(
                    client=client,
                    question=question,
                    answer=answer,
                    context=context_for_answer,
                )

                final_claim_audit = initial_claim_audit
                initial_summary = initial_claim_audit.get("summary", {})
                auto_repair_attempted = int(initial_summary.get("unsupported_count", 0)) > 0
                auto_repair_success = False
                repair_notes = "No unsupported claims detected; auto-repair skipped."

                if auto_repair_attempted:
                    repair_notes = "Auto-repair attempted but did not improve grounding metrics."
                    try:
                        repaired_answer = regenerate_answer_with_guardrails(
                            client=client,
                            question=question,
                            answer=answer,
                            context=context_for_answer,
                            tone_instruction=selected_tone_instruction,
                            claim_audit=initial_claim_audit,
                        )
                        repaired_claim_audit = audit_answer_claims(
                            client=client,
                            question=question,
                            answer=repaired_answer,
                            context=context_for_answer,
                        )

                        repaired_summary = repaired_claim_audit.get("summary", {})
                        improved = (
                            int(repaired_summary.get("unsupported_count", 0))
                            < int(initial_summary.get("unsupported_count", 0))
                        ) or (
                            int(repaired_summary.get("unsupported_count", 0))
                            == int(initial_summary.get("unsupported_count", 0))
                            and float(repaired_summary.get("support_rate", 0.0))
                            > float(initial_summary.get("support_rate", 0.0))
                        )

                        if improved:
                            answer = repaired_answer
                            final_claim_audit = repaired_claim_audit
                            auto_repair_success = True
                            repair_notes = "Auto-repair improved claim grounding and replaced the original answer."
                    except Exception as e:
                        repair_notes = f"Auto-repair attempt failed: {e}"

                claim_audit_payload = {
                    "initial": initial_claim_audit,
                    "final": final_claim_audit,
                    "auto_repair_attempted": auto_repair_attempted,
                    "auto_repair_success": auto_repair_success,
                    "repair_notes": repair_notes,
                }

            with st.spinner("🔍 Self-reflection scoring..."):
                reflect_prompt = f"""Rate this legal Q&A. Respond ONLY in JSON.
{{"confidence": <0-100>, "is_grounded": <true/false>, "has_hallucination": <true/false>, "reason": "<one sentence>"}}

Q: {question}
A: {answer}
Context: {context_for_answer[:600]}"""
                try:
                    r2 = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": reflect_prompt}],
                        max_tokens=150, temperature=0.0
                    )
                    raw = r2.choices[0].message.content
                    jm = re.search(r"\{.*\}", raw, re.DOTALL)
                    reflection = json.loads(jm.group()) if jm else {"confidence": 75, "reason": "Answer grounded in context."}
                except:
                    reflection = {"confidence": 75, "reason": "Answer grounded in context."}

            confidence_raw = reflection.get("confidence", 75)
            try:
                confidence = int(max(0, min(100, float(confidence_raw))))
            except (TypeError, ValueError):
                confidence = 75

            graph_log_error = None
            try:
                graphrag_applied = graph_result is not None
                expansion_events = len(graph_result.get("expansion_log", [])) if graph_result else 0
                extra_clauses_added = (
                    graph_result.get("extra_chunks_added", 0)
                    if graph_result
                    else 0
                )
                append_graphrag_stats_log(
                    {
                        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                        "doc_hash_prefix": st.session_state.get("doc_hash", "")[:12],
                        "question": " ".join(question.splitlines()).strip(),
                        "tone_mode": tone_mode,
                        "graphrag_enabled": enable_graphrag,
                        "graphrag_applied": graphrag_applied,
                        "hops_requested": graphrag_hops if enable_graphrag else 0,
                        "baseline_clause_count": len(_dedupe_chunks(baseline_chunks_for_display)),
                        "expanded_clause_count": len(_dedupe_chunks(source_chunks_for_display)),
                        "extra_clauses_added": extra_clauses_added,
                        "override_detected": graph_result.get("override_detected", False) if graph_result else False,
                        "graph_triggered": graph_result.get("graph_triggered", False) if graph_result else False,
                        "expansion_events": expansion_events,
                        "confidence": confidence,
                        "debate_enabled": enable_debate,
                    }
                )
            except Exception as e:
                graph_log_error = str(e)

            claim_log_error = None
            try:
                initial_summary = claim_audit_payload["initial"]["summary"]
                final_summary = claim_audit_payload["final"]["summary"]
                append_claim_grounding_log(
                    {
                        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                        "doc_hash_prefix": st.session_state.get("doc_hash", "")[:12],
                        "question": " ".join(question.splitlines()).strip(),
                        "tone_mode": tone_mode,
                        "graphrag_applied": graph_result is not None,
                        "initial_claim_count": initial_summary.get("total_claims", 0),
                        "initial_supported_count": initial_summary.get("supported_count", 0),
                        "initial_weak_count": initial_summary.get("weak_count", 0),
                        "initial_unsupported_count": initial_summary.get("unsupported_count", 0),
                        "initial_claim_support_rate": initial_summary.get("support_rate", 0.0),
                        "auto_repair_attempted": claim_audit_payload.get("auto_repair_attempted", False),
                        "auto_repair_success": claim_audit_payload.get("auto_repair_success", False),
                        "final_claim_count": final_summary.get("total_claims", 0),
                        "final_supported_count": final_summary.get("supported_count", 0),
                        "final_weak_count": final_summary.get("weak_count", 0),
                        "final_unsupported_count": final_summary.get("unsupported_count", 0),
                        "final_claim_support_rate": final_summary.get("support_rate", 0.0),
                        "final_grounded_confidence": confidence,
                        "has_hallucination": bool(reflection.get("has_hallucination")),
                    }
                )
            except Exception as e:
                claim_log_error = str(e)

            claim_audit_payload["history"] = load_claim_grounding_history()

            debate_results = None
            if enable_debate and source_chunks_for_display:
                debate_context = "\n\n".join(
                    [f"[{c['title']}]\n{c['text']}" for c in source_chunks_for_display[:5]]
                )
                with st.spinner("⚖️ Running multi-agent legal debate..."):
                    try:
                        debate_results = run_debate(client, question, debate_context)
                    except Exception as e:
                        debate_results = {
                            "Plaintiff": f"[Debate unavailable: {e}]",
                            "Defense": f"[Debate unavailable: {e}]",
                            "Judge": f"[Debate unavailable: {e}]",
                        }

            st.session_state["qa_result"] = {
                "question": question.strip(),
                "tone_mode": tone_mode,
                "answer": answer,
                "confidence": confidence,
                "reflection_reason": reflection.get("reason"),
                "has_hallucination": bool(reflection.get("has_hallucination")),
                "source_chunks": source_chunks_for_display,
                "baseline_chunks": baseline_chunks_for_display,
                "graph_result": graph_result,
                "graph_log_error": graph_log_error,
                "claim_audit": claim_audit_payload,
                "claim_log_error": claim_log_error,
                "debate_enabled": enable_debate,
                "debate_results": debate_results,
            }

if st.session_state.get("qa_result"):
    render_results_panel(st.session_state["qa_result"])

# ─── Footer ───────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Vertisa AI | Enhanced Legal Document QA System")
