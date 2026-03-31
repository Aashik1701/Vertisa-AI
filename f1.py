"""
Vertisa AI — Feature 1: Multi-Agent Legal Debate System
======================================================
Fires 3 concurrent Groq API calls with different legal "personas":
  • Plaintiff Advocate  → Most aggressive pro-claimant reading
  • Defense Advocate    → Most protective/literal reading
  • Judge (Arbitrator)  → Balanced synthesis + likely ruling

No AutoGen, no CrewAI — just concurrent HTTP calls.
"""

import asyncio
import json
from groq import Groq


# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

PLAINTIFF_PROMPT = """You are an aggressive Plaintiff's attorney in a legal dispute.
Your job is to argue the MOST FAVORABLE interpretation of the contract for the claimant/plaintiff.
Stretch every ambiguous word to benefit your client. Find every obligation the other party has failed.

You MUST base your argument ONLY on the legal text provided below. Do not invent clauses.
If the text does not support your argument, say what IS there and why it helps the plaintiff.

CONTRACT CONTEXT:
{context}

LEGAL QUESTION: {question}

Provide your argument in 3-4 sentences. Start with "Plaintiff's Position:" """

DEFENSE_PROMPT = """You are a cautious Defense attorney protecting your client from liability.
Your job is to argue the MOST RESTRICTIVE and LITERAL interpretation of the contract.
Find every qualifier, exception, and limitation that protects your client.

You MUST base your argument ONLY on the legal text provided below. Do not invent clauses.
Focus on what the contract explicitly DOES NOT say, and use that absence to protect the defendant.

CONTRACT CONTEXT:
{context}

LEGAL QUESTION: {question}

Provide your argument in 3-4 sentences. Start with "Defense's Position:" """

JUDGE_PROMPT = """You are an experienced Judge who has just heard arguments from both sides.
Your role is to give a BALANCED, REALISTIC legal assessment of this dispute.
Identify which party has the stronger legal position and WHY, based purely on the contract text.
Note any ambiguities that would likely be decisive in court.

CONTRACT CONTEXT:
{context}

LEGAL QUESTION: {question}

Give your ruling in 4-5 sentences. Be direct. Start with "Court's Assessment:" """


# ─────────────────────────────────────────────────────────────────────────────
# ASYNC DEBATE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _call_groq_sync(client: Groq, prompt: str, persona_label: str) -> dict:
    """
    Single synchronous Groq call. Returns dict with label + response text.
    Called in a thread pool so it doesn't block the event loop.
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise legal professional. Answer based ONLY on the provided contract text.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=400,
            temperature=0.4,
        )
        return {
            "persona": persona_label,
            "text": response.choices[0].message.content.strip(),
            "error": None,
        }
    except Exception as e:
        return {
            "persona": persona_label,
            "text": f"[Error generating {persona_label} argument]",
            "error": str(e),
        }


async def _run_debate_async(client: Groq, question: str, context: str) -> dict:
    """
    Fires all 3 Groq calls concurrently using a thread pool.
    Returns all three arguments in ~same time as one call.
    """
    loop = asyncio.get_event_loop()

    prompts = [
        (PLAINTIFF_PROMPT.format(context=context, question=question), "Plaintiff"),
        (DEFENSE_PROMPT.format(context=context, question=question), "Defense"),
        (JUDGE_PROMPT.format(context=context, question=question), "Judge"),
    ]

    # Run all 3 in parallel using thread executor (Groq SDK is sync)
    tasks = [
        loop.run_in_executor(None, _call_groq_sync, client, prompt, label)
        for prompt, label in prompts
    ]

    results = await asyncio.gather(*tasks)

    return {r["persona"]: r["text"] for r in results}


def run_debate(client: Groq, question: str, context: str) -> dict:
    """
    Public entry point. Handles asyncio event loop safely for both
    Jupyter/Colab and Streamlit environments.

    Returns:
        {
            "Plaintiff": "Plaintiff's Position: ...",
            "Defense":   "Defense's Position: ...",
            "Judge":     "Court's Assessment: ...",
        }
    """
    try:
        # Colab/Jupyter already has a running loop — use nest_asyncio approach
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in Jupyter/Colab — use a thread to run the async code
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    _run_debate_async(client, question, context),
                )
                return future.result()
        else:
            return loop.run_until_complete(
                _run_debate_async(client, question, context)
            )
    except RuntimeError:
        # Fallback: just run synchronously if async fails
        results = {}
        for prompt_template, label in [
            (PLAINTIFF_PROMPT, "Plaintiff"),
            (DEFENSE_PROMPT, "Defense"),
            (JUDGE_PROMPT, "Judge"),
        ]:
            prompt = prompt_template.format(context=context, question=question)
            result = _call_groq_sync(client, prompt, label)
            results[label] = result["text"]
        return results


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT RENDERING HELPER
# ─────────────────────────────────────────────────────────────────────────────

def render_debate_ui(debate_results: dict):
    """
    Renders the three debate columns in Streamlit.
    Call this inside your Streamlit app after run_debate().
    """
    import streamlit as st

    st.markdown("---")
    st.subheader("⚖️ Multi-Agent Legal Debate")
    st.caption(
        "Three independent AI agents analyze the same clause from opposing perspectives. "
        "Law is not binary — see how the same text can be read differently."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """<div style='background:#1a1a2e;border-left:4px solid #e74c3c;
            padding:16px;border-radius:8px;height:100%'>
            <h4 style='color:#e74c3c;margin:0 0 8px'>🔴 Plaintiff</h4>""",
            unsafe_allow_html=True,
        )
        st.write(debate_results.get("Plaintiff", ""))
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            """<div style='background:#1a1a2e;border-left:4px solid #3498db;
            padding:16px;border-radius:8px;height:100%'>
            <h4 style='color:#3498db;margin:0 0 8px'>🔵 Defense</h4>""",
            unsafe_allow_html=True,
        )
        st.write(debate_results.get("Defense", ""))
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown(
            """<div style='background:#1a1a2e;border-left:4px solid #f39c12;
            padding:16px;border-radius:8px;height:100%'>
            <h4 style='color:#f39c12;margin:0 0 8px'>⚖️ Court</h4>""",
            unsafe_allow_html=True,
        )
        st.write(debate_results.get("Judge", ""))
        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST (run this file directly to verify it works)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    api_key = os.environ.get("GROQ_API_KEY", "your_groq_api_key_here")
    client = Groq(api_key=api_key)

    sample_context = """
    Section 4.1 - Termination Without Cause:
    The Company may terminate this Agreement at any time without cause upon written notice.

    Section 9.3 - Long-Tenure Exception:
    Notwithstanding anything to the contrary in Section 4.1, if the Employee has been
    continuously employed for a period of five (5) or more years, the Company shall
    provide no less than ninety (90) days written notice prior to any termination
    without cause, or pay in lieu thereof.
    """

    question = "Can the company fire me without giving notice?"

    print("Running Multi-Agent Debate... (3 concurrent calls)")
    result = run_debate(client, question, sample_context)

    print("\n" + "=" * 60)
    for persona, argument in result.items():
        print(f"\n[{persona.upper()}]\n{argument}")
    print("=" * 60)