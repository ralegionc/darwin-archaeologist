"""
Central configuration for the Darwin AI Archaeologist project.
Copy this to config_local.py and override values there for secrets.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CLEANED_DIR = DATA_DIR / "cleaned"
CHUNKS_DIR = DATA_DIR / "chunks"
CHROMA_DIR = DATA_DIR / "chroma"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

for d in [RAW_DIR, CLEANED_DIR, CHUNKS_DIR, CHROMA_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Corpus sources ─────────────────────────────────────────────────────────────
DARWIN_SOURCES = {
    "correspondence": {
        "base_url": "https://www.darwinproject.ac.uk",
        "search_url": "https://www.darwinproject.ac.uk/search",
        "enabled": True,
        "priority": 1,
    },
    "darwin_online": {
        "base_url": "http://darwin-online.org.uk",
        "works_url": "http://darwin-online.org.uk/contents.html",
        "enabled": True,
        "priority": 2,
    },
    # Gutenberg has clean plain-text versions of published works
    "gutenberg": {
        "ids": {
            "origin_of_species": 1228,
            "voyage_of_beagle": 944,
            "descent_of_man": 2300,
            "expression_of_emotions": 1227,
            "autobiography": 2010,
        },
        "enabled": True,
        "priority": 3,
    },
}

# ── Darwin life periods (for temporal tagging) ─────────────────────────────────
LIFE_PERIODS = [
    {"name": "youth",           "start": 1809, "end": 1831, "description": "Before the Beagle"},
    {"name": "beagle_voyage",   "start": 1831, "end": 1836, "description": "HMS Beagle voyage"},
    {"name": "post_beagle",     "start": 1836, "end": 1842, "description": "Processing the voyage, early theorizing"},
    {"name": "species_work",    "start": 1842, "end": 1859, "description": "Secret species work, barnacles, illness"},
    {"name": "origin_decade",   "start": 1859, "end": 1871, "description": "Post-Origin controversy"},
    {"name": "late_career",     "start": 1871, "end": 1882, "description": "Descent of Man onwards, final works"},
]

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_SIZE = 800          # tokens per chunk
CHUNK_OVERLAP = 100       # token overlap between chunks
MIN_CHUNK_SIZE = 100      # discard chunks shorter than this

# ── Embedding ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # local, free
# EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI, better quality
EMBEDDING_DEVICE = "cuda" if os.environ.get("USE_GPU") else "cpu"
CHROMA_COLLECTION = "darwin_corpus"

# ── Retrieval ──────────────────────────────────────────────────────────────────
TOP_K_RETRIEVAL = 5       # passages to retrieve per query
RETRIEVAL_THRESHOLD = 0.3 # minimum similarity score

# ── LLM ───────────────────────────────────────────────────────────────────────
# Options: "openai", "anthropic", "local"
LLM_BACKEND = os.environ.get("LLM_BACKEND", "anthropic")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = "claude-opus-4-5"

# Local model (after fine-tuning)
LOCAL_MODEL_PATH = str(MODELS_DIR / "darwin-mistral")
LOCAL_BASE_MODEL = "mistralai/Mistral-7B-v0.1"

# ── Fine-tuning ────────────────────────────────────────────────────────────────
FINETUNE_CONFIG = {
    "base_model": LOCAL_BASE_MODEL,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "fp16": True,
    "output_dir": LOCAL_MODEL_PATH,
}

# ── Failure elicitation ────────────────────────────────────────────────────────
FAILURE_CATEGORIES = [
    "temporal_lock",
    "register_collapse",
    "gap_confabulation",
    "embodiment_erasure",
    "public_private_blur",
]

ELICITATION_RUNS_PER_PROMPT = 5   # run each prompt N times to measure variance
AUTHENTICITY_SCALE = 5            # 1–5 rating scale for human validators

# ── System prompt ──────────────────────────────────────────────────────────────
DARWIN_SYSTEM_PROMPT = """You are responding as Charles Darwin, the naturalist (1809–1882).

You must:
- Speak only from knowledge you would have had at the date specified in the prompt
- Use only information grounded in the retrieved passages below
- When uncertain, express that uncertainty as Darwin would — he was famously candid about doubt
- Write in Darwin's register: precise, self-deprecating, attentive to evidence, occasionally wry
- Never reference events, people, or knowledge that post-dates the prompt's context

Retrieved passages from Darwin's actual writings:
{retrieved_passages}

If the retrieved passages do not contain relevant information, say so rather than inventing.
The date context for this response is: {date_context}"""
