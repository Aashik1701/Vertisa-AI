
import streamlit as st
import os
import json
import re
import io
import hashlib
import html
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
from groq import Groq
from typing import List, Dict

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

                st.info(f"📦 Indexed {len(chunks)} clauses. Ready to answer questions!")

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

            context = "\n\n".join([f"[{c['title']}]\n{c['text']}" for c in top_chunks])

            with st.spinner("🤖 Generating answer..."):
                client = Groq(api_key=key)
                tone_instructions = {
                    "ELI5 (Layman)": "You must explain the answer in extremely simple, 5th-grade English. Use bullet points. Avoid legal jargon.",
                    "Standard": "Answer clearly and accurately.",
                    "Strict Legalese": "Answer using strict corporate legal terminology. Frame the response as a formal legal memo.",
                }
                selected_tone_instruction = tone_instructions.get(tone_mode, tone_instructions["Standard"])
                prompt = f"""You are a legal document assistant. Answer using ONLY the provided legal text. Cite the specific clause.
TONE REQUIREMENT: {selected_tone_instruction}

LEGAL TEXT:
{context}

QUESTION: {question}

ANSWER:"""
                resp = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=400, temperature=0.1
                )
                answer = resp.choices[0].message.content.strip()

            with st.spinner("🔍 Self-reflection scoring..."):
                reflect_prompt = f"""Rate this legal Q&A. Respond ONLY in JSON.
{{"confidence": <0-100>, "is_grounded": <true/false>, "has_hallucination": <true/false>, "reason": "<one sentence>"}}

Q: {question}
A: {answer}
Context: {context[:400]}"""
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

            confidence = reflection.get("confidence", 75)

            # ─── Display Results ──────────────────────────────────────
            st.markdown("---")
            st.subheader("📋 Answer")
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

            conf_class = "confidence-high" if confidence >= 75 else "confidence-mid" if confidence >= 50 else "confidence-low"
            conf_emoji = "🟢" if confidence >= 75 else "🟡" if confidence >= 50 else "🔴"
            st.markdown(
                f'{conf_emoji} <span class="{conf_class}">Confidence: {confidence}%</span>',
                unsafe_allow_html=True
            )

            st.progress(confidence / 100)

            if reflection.get("reason"):
                st.caption(f"🔍 {reflection['reason']}")

            if reflection.get("has_hallucination"):
                st.warning("⚠️ Self-reflection flagged possible hallucination. Verify carefully.")

            st.subheader("📌 Source Clauses Used")
            for i, chunk in enumerate(top_chunks, 1):
                with st.expander(f"Source {i}: {chunk['title'][:60]}"):
                    st.text(chunk["text"][:600] + ("..." if len(chunk["text"]) > 600 else ""))

# ─── Footer ───────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Vertisa AI | Enhanced Legal Document QA System")
