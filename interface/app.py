"""
interface/app.py

Streamlit web interface for the Darwin AI Archaeologist.

Three modes:
  1. Ask Darwin — query with RAG grounding and citation display
  2. Failure Lab — run specific failure elicitation prompts
  3. Corpus Explorer — browse the embedded corpus and retrieved passages

Run:
    streamlit run interface/app.py
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    print("Install streamlit: pip install streamlit")
    sys.exit(1)

from config import CHROMA_DIR, RESULTS_DIR, LIFE_PERIODS
from pipeline.model import DarwinModel
from pipeline.retriever import DarwinRetriever


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Darwin AI Archaeologist",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .passage-card {
        background: #f8f6f0;
        border-left: 3px solid #8B7355;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 4px 4px 0;
        font-size: 0.9rem;
    }
    .citation {
        font-size: 0.75rem;
        color: #666;
        font-style: italic;
        margin-bottom: 0.3rem;
    }
    .score-badge {
        display: inline-block;
        background: #8B7355;
        color: white;
        padding: 1px 6px;
        border-radius: 10px;
        font-size: 0.7rem;
    }
    .failure-warning {
        background: #fff3cd;
        border: 1px solid #ffc107;
        padding: 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    .response-box {
        background: #fafafa;
        border: 1px solid #ddd;
        padding: 1.2rem;
        border-radius: 6px;
        font-family: Georgia, serif;
        line-height: 1.7;
    }
</style>
""", unsafe_allow_html=True)


# ── State ──────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_model():
    return DarwinModel()

@st.cache_resource
def get_retriever():
    return DarwinRetriever()

@st.cache_data
def load_failure_prompts():
    prompts_path = Path("data/failure_prompts.json")
    if prompts_path.exists():
        return json.loads(prompts_path.read_text(encoding="utf-8"))
    return {}


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔬 Darwin Archaeologist")
    st.caption("Study what the model gets wrong.")

    mode = st.radio(
        "Mode",
        ["Ask Darwin", "Failure Lab", "Corpus Explorer"],
        index=0,
    )

    st.divider()

    # Corpus stats
    try:
        retriever = get_retriever()
        count = retriever.count()
        st.metric("Corpus vectors", f"{count:,}")
    except Exception:
        st.warning("Vector store not ready. Run embedder.py first.")

    st.divider()
    st.caption("""
    **About this project**

    This is not a simulation of Darwin. It is a study of what breaks when you try.

    Each failure mode — temporal lock, register collapse, gap confabulation — is an argument about what makes a person irreducible to their textual trace.
    """)


# ── Mode: Ask Darwin ───────────────────────────────────────────────────────────

if mode == "Ask Darwin":
    st.header("Ask Darwin")
    st.caption("Every response is grounded in real Darwin passages. Citations shown below.")

    col1, col2 = st.columns([2, 1])

    with col2:
        date_context = st.text_input(
            "Date context (year)",
            value="1860",
            help="The year from which Darwin is responding. Affects what knowledge he should have."
        )
        filter_register = st.selectbox(
            "Filter corpus by register",
            ["any", "public", "private", "personal", "intimate"],
        )
        filter_doc_type = st.selectbox(
            "Filter corpus by document type",
            ["any", "letter", "notebook", "published", "diary", "autobiography", "manuscript"],
        )
        n_passages = st.slider("Passages to retrieve", 1, 8, 5)

    with col1:
        query = st.text_area(
            "Your question",
            placeholder="What do you think of the relationship between humans and other primates?",
            height=120,
        )

        if st.button("Ask Darwin", type="primary", disabled=not query):
            model = get_model()
            retriever = get_retriever()
            retriever.top_k = n_passages

            with st.spinner("Retrieving from Darwin's corpus..."):
                response = model.query(
                    prompt=query,
                    date_context=date_context,
                    filter_register=filter_register if filter_register != "any" else None,
                    filter_doc_type=filter_doc_type if filter_doc_type != "any" else None,
                )

            st.subheader("Darwin's response")
            st.markdown(
                f'<div class="response-box">{response.response_text}</div>',
                unsafe_allow_html=True
            )

            if response.retrieved_passages:
                st.subheader(f"Source passages ({len(response.retrieved_passages)} retrieved)")
                for i, p in enumerate(response.retrieved_passages, 1):
                    with st.expander(f"{i}. {p.citation_str()} — score: {p.score:.2f}"):
                        st.markdown(
                            f'<div class="passage-card">'
                            f'<div class="citation">{p.citation_str()} | {p.doc_type} | {p.register}</div>'
                            f'{p.text}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        if p.url:
                            st.markdown(f"[View source]({p.url})")
            else:
                st.warning("No relevant passages retrieved. The model is likely confabulating.")

            # Temporal check
            try:
                year = int(date_context)
                future_passages = [
                    p for p in response.retrieved_passages
                    if p.date_year and p.date_year > year
                ]
                if future_passages:
                    st.markdown(
                        f'<div class="failure-warning">⚠️ <strong>Temporal warning:</strong> '
                        f'{len(future_passages)} retrieved passage(s) post-date the context year {year}. '
                        f'The model may be drawing on knowledge Darwin could not have had.</div>',
                        unsafe_allow_html=True
                    )
            except ValueError:
                pass


# ── Mode: Failure Lab ──────────────────────────────────────────────────────────

elif mode == "Failure Lab":
    st.header("Failure Lab")
    st.caption("Run structured failure elicitation prompts. Study what the model gets wrong — and what that reveals.")

    prompts_data = load_failure_prompts()
    if not prompts_data:
        st.error("failure_prompts.json not found. Ensure data/failure_prompts.json exists.")
        st.stop()

    categories = prompts_data.get("categories", {})

    cat_choice = st.selectbox(
        "Failure category",
        list(categories.keys()),
        format_func=lambda x: {
            "temporal_lock": "Temporal Lock — uses knowledge Darwin couldn't have had",
            "register_collapse": "Register Collapse — emotional flatness across contexts",
            "gap_confabulation": "Gap Confabulation — fills silences with fiction",
            "embodiment_erasure": "Embodiment Erasure — no fatigue, pain, or constraint",
            "public_private_blur": "Public/Private Blur — over-indexes on published voice",
        }.get(x, x)
    )

    cat_data = categories.get(cat_choice, {})
    st.caption(cat_data.get("description", ""))

    prompts_in_cat = cat_data.get("prompts", [])
    if not prompts_in_cat:
        st.info("No prompts in this category.")
        st.stop()

    prompt_labels = [f"[{p['id']}] {p['prompt'][:60]}..." for p in prompts_in_cat]
    prompt_idx = st.selectbox("Select prompt", range(len(prompts_in_cat)), format_func=lambda i: prompt_labels[i])
    selected = prompts_in_cat[prompt_idx]

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Prompt:** {selected['prompt']}\n\n**Date context:** {selected['date_context']}")
    with col2:
        st.warning(f"**Expected failure:** {selected['expected_failure']}")
        st.success(f"**Ground truth:** {selected['ground_truth']}")

    n_runs = st.slider("Number of runs (to measure variance)", 1, 5, 3)

    if st.button("Run failure elicitation", type="primary"):
        model = get_model()
        responses = []

        progress = st.progress(0)
        for i in range(n_runs):
            with st.spinner(f"Run {i+1}/{n_runs}..."):
                resp = model.query(
                    prompt=selected["prompt"],
                    date_context=selected["date_context"],
                    filter_before_year=selected.get("filter_before_year"),
                    failure_category=cat_choice,
                )
                responses.append(resp)
                progress.progress((i + 1) / n_runs)
                time.sleep(0.3)

        st.subheader(f"Results ({n_runs} runs)")

        tabs = st.tabs([f"Run {i+1}" for i in range(len(responses))])
        for tab, resp in zip(tabs, responses):
            with tab:
                st.markdown(
                    f'<div class="response-box">{resp.response_text}</div>',
                    unsafe_allow_html=True
                )
                if resp.retrieved_passages:
                    st.caption(f"Based on {len(resp.retrieved_passages)} retrieved passages")
                    for p in resp.retrieved_passages[:2]:
                        st.caption(f"• {p.citation_str()} (score: {p.score:.2f})")
                else:
                    st.markdown(
                        '<div class="failure-warning">⚠️ No passages retrieved — high confabulation risk</div>',
                        unsafe_allow_html=True
                    )

        # Variance analysis
        if n_runs > 1:
            st.subheader("Variance analysis")
            texts = [r.response_text for r in responses]
            word_sets = [set(t.lower().split()) for t in texts]
            similarities = []
            for i in range(len(word_sets)):
                for j in range(i+1, len(word_sets)):
                    inter = len(word_sets[i] & word_sets[j])
                    union = len(word_sets[i] | word_sets[j])
                    if union:
                        similarities.append(inter / union)

            avg_sim = sum(similarities) / len(similarities) if similarities else 0

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Avg lexical similarity", f"{avg_sim:.2f}")
            col_b.metric("Variance level",
                         "Low" if avg_sim > 0.6 else "Medium" if avg_sim > 0.35 else "High")
            col_c.metric("Avg response length",
                         f"{sum(len(t) for t in texts) // len(texts)} chars")

            interpretation = (
                "**Low variance:** Responses are consistent — potentially consistently wrong. Most dangerous failure mode."
                if avg_sim > 0.6 else
                "**Medium variance:** Some uncertainty present."
                if avg_sim > 0.35 else
                "**High variance:** Model is uncertain here — at least it knows it doesn't know."
            )
            st.info(interpretation)


# ── Mode: Corpus Explorer ──────────────────────────────────────────────────────

elif mode == "Corpus Explorer":
    st.header("Corpus Explorer")
    st.caption("Browse what Darwin actually wrote. This is the ground truth the model should stay anchored to.")

    retriever = get_retriever()

    search_query = st.text_input("Search Darwin's corpus", placeholder="natural selection, coral reef, Emma...")

    col1, col2, col3 = st.columns(3)
    with col1:
        filter_period = st.selectbox(
            "Life period",
            ["any"] + [p["name"] for p in LIFE_PERIODS],
            format_func=lambda x: x if x == "any" else f"{x} ({next((p['description'] for p in LIFE_PERIODS if p['name'] == x), '')})"
        )
    with col2:
        filter_register = st.selectbox(
            "Register",
            ["any", "public", "private", "personal", "intimate"],
        )
    with col3:
        n_results = st.slider("Results", 3, 15, 5)

    if search_query:
        with st.spinner("Searching..."):
            passages = retriever.retrieve(
                query=search_query,
                top_k=n_results,
                filter_period=filter_period if filter_period != "any" else None,
                filter_register=filter_register if filter_register != "any" else None,
            )

        if passages:
            st.subheader(f"{len(passages)} passages found")
            for p in passages:
                with st.expander(f"📜 {p.citation_str()} — {p.doc_type} | {p.register} | score: {p.score:.2f}"):
                    st.markdown(
                        f'<div class="passage-card">'
                        f'<div class="citation">{p.citation_str()}</div>'
                        f'{p.text}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    if p.life_period:
                        st.caption(f"Life period: {p.life_period}")
                    if p.url:
                        st.markdown(f"[View original source]({p.url})")
        else:
            st.info("No passages found. Try a different query or remove filters.")
