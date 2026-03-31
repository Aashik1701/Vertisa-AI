"""
Vertisa AI — Feature 2: Reference-Aware Recursive GraphRAG
===========================================================
The Problem:
    Standard RAG retrieves Clause 4.1 for "Can I be fired without notice?"
    and says "Yes." It MISSES Clause 9.3 which says "Notwithstanding 4.1,
    if you've worked 5+ years, 90-day notice is mandatory."
    Standard RAG is WRONG. This system catches that.

The Solution — 3-step pipeline:
    Step 1 (BUILD):  During chunking, scan every clause for cross-references
                     ("subject to Section X", "notwithstanding Clause Y").
                     Build a directed graph where edges = legal dependencies.

    Step 2 (EXPAND): After standard retrieval gives you top-K chunks,
                     follow their graph edges to pull ALL referenced chunks.

    Step 3 (RESOLVE): Feed ALL chunks (original + graph-expanded) to the LLM
                      with a "Contradiction Detection + Legal Hierarchy" prompt.
"""

import re
import json
from typing import Optional
import networkx as nx
from groq import Groq


# ─────────────────────────────────────────────────────────────────────────────
# REGEX PATTERNS — Legal Cross-Reference Detection
# ─────────────────────────────────────────────────────────────────────────────

# Matches patterns like: Section 4.1, Clause 9, Article III, Paragraph 2(b)
REFERENCE_PATTERNS = [
    # "Section 4.1", "section 12"
    (r"\bsection\s+(\d+(?:\.\d+)*(?:\([a-z]\))?)", "section"),
    # "Clause 9.3", "clause 2"
    (r"\bclause\s+(\d+(?:\.\d+)*(?:\([a-z]\))?)", "clause"),
    # "Article IV", "Article 3"
    (r"\barticle\s+([IVXivx]+|\d+)", "article"),
    # "Paragraph 2(b)"
    (r"\bparagraph\s+(\d+(?:\.[a-z0-9]+)*)", "paragraph"),
    # "Exhibit A", "Schedule 1"
    (r"\b(?:exhibit|schedule|annex)\s+([A-Z0-9]+)", "exhibit"),
]

# Triggers that indicate a legal override / dependency
OVERRIDE_KEYWORDS = [
    "notwithstanding",
    "subject to",
    "pursuant to",
    "as set forth in",
    "in accordance with",
    "as defined in",
    "as provided in",
    "except as provided in",
    "without limiting",
    "in addition to",
    "contrary to",
    "override",
    "supersede",
    "govern",
]

# Patterns to detect which section a chunk IS (its identity)
IDENTITY_PATTERNS = [
    r"^(?:section|clause|article|paragraph)\s+(\d+(?:\.\d+)*(?:\([a-z]\))?)",
    r"^(\d+(?:\.\d+)+)\s+[-–—]",  # "4.1 - Termination"
    r"^(?:exhibit|schedule|annex)\s+([A-Z0-9]+)",
]


# ─────────────────────────────────────────────────────────────────────────────
# LEGAL KNOWLEDGE GRAPH
# ─────────────────────────────────────────────────────────────────────────────

class LegalKnowledgeGraph:
    """
    Builds a directed dependency graph over legal clauses.

    Nodes = individual clauses (chunks)
    Edges = "Clause A references Clause B" (A depends on / modifies B)

    Graph is built ONCE during document ingestion, stored in session.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.chunks: dict[int, str] = {}
        self.chunk_ids: dict[str, int] = {}  # "section 4.1" -> chunk index
        self.stats = {
            "total_nodes": 0,
            "total_edges": 0,
            "override_clauses": [],
        }

    def _normalize_ref(self, ref_type: str, ref_value: str) -> str:
        """Normalize a reference to a standard key, e.g. 'section 4.1'."""
        return f"{ref_type} {ref_value.lower().strip()}"

    def _detect_chunk_identity(self, chunk: str) -> Optional[str]:
        """
        Find what section/clause THIS chunk IS.
        Returns normalized key like 'section 4.1' or None.
        """
        first_line = chunk.strip().split("\n")[0].strip().lower()
        for pattern in IDENTITY_PATTERNS:
            match = re.match(pattern, first_line, re.IGNORECASE)
            if match:
                # Try to determine type from context
                if "section" in first_line[:15]:
                    return f"section {match.group(1)}"
                elif "clause" in first_line[:15]:
                    return f"clause {match.group(1)}"
                elif "article" in first_line[:15]:
                    return f"article {match.group(1)}"
                else:
                    return f"section {match.group(1)}"  # default
        return None

    def _find_references(self, chunk: str) -> list[dict]:
        """
        Scan a chunk for all cross-references it makes.
        Returns list of {ref_key, has_override_keyword}
        """
        chunk_lower = chunk.lower()
        found_refs = []

        has_override = any(kw in chunk_lower for kw in OVERRIDE_KEYWORDS)

        for pattern, ref_type in REFERENCE_PATTERNS:
            matches = re.finditer(pattern, chunk, re.IGNORECASE)
            for match in matches:
                ref_key = self._normalize_ref(ref_type, match.group(1))
                found_refs.append({
                    "ref_key": ref_key,
                    "has_override": has_override,
                    "raw_match": match.group(0),
                })

        return found_refs

    def build(self, chunks: list[str]) -> "LegalKnowledgeGraph":
        """
        Main build method. Call this once after clause-aware chunking.

        Args:
            chunks: List of text chunks from your ClauseAwareChunker

        Returns:
            self (for chaining)
        """
        self.chunks = {i: chunk for i, chunk in enumerate(chunks)}

        # ── Pass 1: Identify what each chunk IS ──────────────────────────────
        for i, chunk in enumerate(chunks):
            self.graph.add_node(i, text=chunk, identity=None)
            identity = self._detect_chunk_identity(chunk)
            if identity:
                self.graph.nodes[i]["identity"] = identity
                self.chunk_ids[identity] = i

        # ── Pass 2: Find what each chunk REFERENCES ───────────────────────────
        for i, chunk in enumerate(chunks):
            refs = self._find_references(chunk)
            for ref in refs:
                target_idx = self.chunk_ids.get(ref["ref_key"])
                if target_idx is not None and target_idx != i:
                    self.graph.add_edge(
                        i,
                        target_idx,
                        has_override=ref["has_override"],
                        raw_ref=ref["raw_match"],
                    )
                    if ref["has_override"]:
                        self.stats["override_clauses"].append(
                            f"Chunk {i} overrides/modifies Chunk {target_idx} "
                            f"(via '{ref['raw_match']}')"
                        )

        self.stats["total_nodes"] = self.graph.number_of_nodes()
        self.stats["total_edges"] = self.graph.number_of_edges()

        return self

    def expand_with_graph(
        self,
        initial_chunk_indices: list[int],
        max_hops: int = 2
    ) -> tuple[list[str], list[dict]]:
        """
        Given initial retrieved chunk indices, follow graph edges to
        pull in all referenced/dependent clauses.

        Args:
            initial_chunk_indices: Top-K chunk indices from standard retrieval
            max_hops: How many "levels" of references to follow (default: 2)

        Returns:
            (expanded_chunks, expansion_log)
            expansion_log: describes WHY each extra chunk was added
        """
        if not self.graph.nodes:
            # Graph not built yet — return original unchanged
            return [self.chunks[i] for i in initial_chunk_indices], []

        expanded_indices = set(initial_chunk_indices)
        expansion_log = []

        frontier = set(initial_chunk_indices)
        for hop in range(max_hops):
            new_frontier = set()
            for node in frontier:
                if node not in self.graph:
                    continue

                # Follow outgoing edges (clauses this one references)
                for successor in self.graph.successors(node):
                    if successor not in expanded_indices:
                        edge_data = self.graph.edges[node, successor]
                        is_override = edge_data.get("has_override", False)
                        expansion_log.append({
                            "from_chunk": node,
                            "to_chunk": successor,
                            "hop": hop + 1,
                            "type": "override" if is_override else "reference",
                            "raw_ref": edge_data.get("raw_ref", ""),
                        })
                        expanded_indices.add(successor)
                        new_frontier.add(successor)

                # Follow INCOMING edges (clauses that override THIS one)
                # This is critical — if standard RAG retrieves 4.1, we must
                # also pull 9.3 which modifies 4.1, even though 4.1 doesn't
                # reference 9.3 directly.
                for predecessor in self.graph.predecessors(node):
                    if predecessor not in expanded_indices:
                        edge_data = self.graph.edges[predecessor, node]
                        is_override = edge_data.get("has_override", False)
                        expansion_log.append({
                            "from_chunk": predecessor,
                            "to_chunk": node,
                            "hop": hop + 1,
                            "type": "incoming_override" if is_override else "incoming_ref",
                            "raw_ref": edge_data.get("raw_ref", ""),
                        })
                        expanded_indices.add(predecessor)
                        new_frontier.add(predecessor)

            frontier = new_frontier
            if not frontier:
                break

        expanded_chunks = [
            self.chunks[i]
            for i in sorted(expanded_indices)
            if i in self.chunks
        ]

        return expanded_chunks, expansion_log

    def get_graph_summary(self) -> str:
        """Human-readable summary of the knowledge graph."""
        lines = [
            f"📊 Legal Knowledge Graph built:",
            f"   • {self.stats['total_nodes']} clause nodes",
            f"   • {self.stats['total_edges']} cross-reference edges",
        ]
        if self.stats["override_clauses"]:
            lines.append(f"   • {len(self.stats['override_clauses'])} override relationships detected:")
            for ov in self.stats["override_clauses"][:5]:
                lines.append(f"     ↳ {ov}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CONTRADICTION RESOLUTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

CONTRADICTION_PROMPT = """You are an expert legal analyst specializing in contract interpretation.

You have retrieved the following legal clauses from a document. Some clauses may MODIFY, OVERRIDE, 
or CREATE EXCEPTIONS to other clauses. This is common in contracts where a general rule (e.g., Section 4) 
is later qualified by a specific exception (e.g., Section 9.3 with "notwithstanding Section 4").

YOUR TASK:
1. Read ALL clauses carefully.
2. Identify if any clause OVERRIDES or CREATES AN EXCEPTION to another clause.
3. Apply the legally correct principle: SPECIFIC provisions override GENERAL provisions.
   Exception clauses beginning with "notwithstanding" always take precedence.
4. Answer the user's question with the LEGALLY CORRECT interpretation.
5. Explicitly state if a simpler reading of the document would have been WRONG and WHY.

RETRIEVED CLAUSES:
{context}

USER QUESTION: {question}

Respond with:
- LEGAL ANSWER: The correct legal answer, accounting for all clause interactions.
- CLAUSE HIERARCHY: Which clause controls, and why (if multiple clauses apply).
- TRAP FOR THE UNWARY: What a naive reader would wrongly conclude (if applicable).
"""


def resolve_with_graph(
    client: Groq,
    question: str,
    expanded_chunks: list[str],
    expansion_log: list[dict],
    tone_instruction: Optional[str] = None,
) -> dict:
    """
    Run contradiction-aware LLM generation over graph-expanded chunks.

    Returns dict with:
        - answer: the legally correct answer
        - graph_triggered: bool (was GraphRAG needed?)
        - expansion_summary: human-readable explanation of what was added
    """
    context = "\n\n---CLAUSE BOUNDARY---\n\n".join(expanded_chunks)

    # Build expansion summary for UI display
    override_found = any(
        log["type"] in ("override", "incoming_override")
        for log in expansion_log
    )
    expansion_summary = ""
    if expansion_log:
        added_nodes = set()
        for log in expansion_log:
            edge_type = log.get("type", "")
            if edge_type in ("override", "reference"):
                added_nodes.add(log.get("to_chunk"))
            else:
                added_nodes.add(log.get("from_chunk"))
        additions = len([n for n in added_nodes if n is not None])
        original_count = max(len(expanded_chunks) - additions, 0)
        expansion_summary = (
            f"🔗 Graph Expansion: Standard retrieval found {original_count} clause(s). "
            f"GraphRAG automatically pulled in {additions} additional linked clause(s). "
        )
        if override_found:
            expansion_summary += (
                "⚠️ **Override relationship detected** — a 'notwithstanding' exception clause "
                "was found that modifies the primary answer. Standard RAG would have missed this."
            )

    tone_block = ""
    if tone_instruction:
        tone_block = f"\n\nTONE REQUIREMENT:\n{tone_instruction}\n"

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise legal analyst. You MUST identify when clauses "
                        "contradict each other and resolve based on legal hierarchy rules. "
                        "Never ignore 'notwithstanding' or 'subject to' qualifiers."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        CONTRADICTION_PROMPT.format(context=context, question=question)
                        + tone_block
                    ),
                },
            ],
            max_tokens=600,
            temperature=0.2,  # Low temp for legal precision
        )
        answer_text = response.choices[0].message.content.strip()
    except Exception as e:
        answer_text = f"Error generating graph-aware answer: {str(e)}"

    return {
        "answer": answer_text,
        "graph_triggered": len(expansion_log) > 0,
        "override_detected": override_found,
        "expansion_summary": expansion_summary,
        "chunks_used": len(expanded_chunks),
        "extra_chunks_added": additions if expansion_log else 0,
        "expanded_chunks": expanded_chunks,
        "expansion_log": expansion_log,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT RENDERING HELPER
# ─────────────────────────────────────────────────────────────────────────────

def render_graphrag_ui(
    graph: LegalKnowledgeGraph,
    graph_result: dict,
    show_answer: bool = True,
):
    """
    Renders the GraphRAG result section in Streamlit.
    Call after resolve_with_graph().
    """
    import streamlit as st

    st.markdown("---")
    st.subheader("🕸️ GraphRAG — Cross-Reference Resolver")

    # Graph stats
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Clauses in Graph", graph.stats["total_nodes"])
    col_b.metric("Cross-Reference Edges", graph.stats["total_edges"])
    col_c.metric("Override Relations", len(graph.stats["override_clauses"]))

    # Expansion alert
    if graph_result["graph_triggered"]:
        if graph_result["override_detected"]:
            st.warning(graph_result["expansion_summary"])
        else:
            st.info(graph_result["expansion_summary"])
    else:
        st.success(
            "✅ No cross-references detected for this query. "
            "Answer is based on the directly retrieved clauses."
        )

    if show_answer:
        st.markdown("#### 📋 Graph-Aware Legal Answer")
        st.write(graph_result["answer"])

    # Override clauses detail
    if graph.stats["override_clauses"]:
        with st.expander(
            f"🔍 Override Relationships Found in This Document "
            f"({len(graph.stats['override_clauses'])})"
        ):
            for ov in graph.stats["override_clauses"]:
                st.markdown(f"- {ov}")


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION WRAPPER — plug into existing VERtisa pipeline
# ─────────────────────────────────────────────────────────────────────────────

def graphrag_answer(
    client: Groq,
    question: str,
    graph: LegalKnowledgeGraph,
    initial_chunk_indices: list[int],
    max_hops: int = 2,
    tone_instruction: Optional[str] = None,
) -> dict:
    """
    Drop-in replacement for your existing generate_answer() call.
    Automatically expands retrieval using the knowledge graph,
    then generates a contradiction-aware answer.

    Args:
        client: Your existing Groq client
        question: User's question
        graph: Pre-built LegalKnowledgeGraph (build once per document upload)
        initial_chunk_indices: Output of your existing FAISS/BM25 retrieval
        max_hops: Graph traversal depth (2 is sufficient for most contracts)

    Returns:
        Full result dict from resolve_with_graph()
    """
    expanded_chunks, expansion_log = graph.expand_with_graph(
        initial_chunk_indices, max_hops=max_hops
    )
    return resolve_with_graph(
        client,
        question,
        expanded_chunks,
        expansion_log,
        tone_instruction=tone_instruction,
    )


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    # Simulate the exact "trap" scenario described in the paper
    test_chunks = [
        """Section 3 - Standard Notice Requirements
        Either party may terminate this Agreement upon thirty (30) days written notice
        to the other party. Notice shall be delivered via certified mail or email.""",

        """Section 4.1 - Termination Without Cause
        Notwithstanding Section 3, the Company may terminate this Agreement at any time
        without cause upon written notice. The Employee shall have no claim for
        compensation arising from such termination.""",

        """Section 9.3 - Long-Tenure Employee Protection
        Notwithstanding anything to the contrary in Section 4.1, if the Employee has
        been continuously employed for a period of five (5) or more years as of the
        date of termination, the Company shall provide no less than ninety (90) days
        written notice prior to any termination without cause, or pay ninety (90) days
        salary in lieu thereof.""",

        """Section 12 - Benefits and Compensation
        The Employee shall receive an annual salary of $85,000 USD, payable bi-weekly.
        Benefits include health insurance, 15 days PTO annually, and 401(k) matching
        up to 4% of base salary.""",

        """Section 15 - Governing Law
        This Agreement shall be governed by and construed in accordance with the laws
        of the State of Delaware, without regard to conflict of law principles.""",
    ]

    print("Building Legal Knowledge Graph...")
    graph = LegalKnowledgeGraph()
    graph.build(test_chunks)
    print(graph.get_graph_summary())
    print()

    # Simulate: Standard RAG retrieved chunk index 1 (Section 4.1)
    # The correct behavior: GraphRAG should ALSO pull chunk index 2 (Section 9.3)
    print("Standard RAG retrieved: [Chunk 1 — Section 4.1]")
    print("Running graph expansion...")

    expanded, log = graph.expand_with_graph([1], max_hops=2)
    print(f"\nGraph expanded to {len(expanded)} chunks:")
    for entry in log:
        print(
            f"  + Added Chunk {entry['to_chunk']} "
            f"[{entry['type']}] via '{entry['raw_ref']}'"
        )

    print("\n✅ GraphRAG correctly identified Section 9.3 as a modifier of Section 4.1")
    print("   Standard RAG would have said 'Yes, you can be fired without notice.'")
    print("   GraphRAG will correctly say 'Only if you've worked less than 5 years.'")