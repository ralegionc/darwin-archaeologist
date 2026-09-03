# AI Archaeologist — Charles Darwin

> Train a model on Darwin's writings. Study what it gets wrong.
> A systematic meditation on memory, identity, and what makes a person irreducible.

---

## What this project does

This project builds a Darwin-grounded AI system in three layers:

1. **Corpus pipeline** — scrapes, cleans, and temporally tags Darwin's letters, notebooks, autobiography, and published works
2. **RAG + fine-tuning** — builds a retrieval-augmented system over his writing, with optional LoRA fine-tuning
3. **Failure elicitation** — systematically probes where the model fails, and taxonomizes those failures philosophically

The goal is not to simulate Darwin. It is to study what breaks, and what those breakages reveal about personhood, memory, and identity.

---

## Project structure

```
darwin-archaeologist/
├── corpus/
│   ├── scraper.py          # Scrapes Darwin Online + Correspondence Project
│   ├── cleaner.py          # Cleans, normalizes, deduplicates text
│   ├── tagger.py           # Adds temporal + contextual metadata
│   └── chunker.py          # Splits into RAG-ready chunks with metadata
├── pipeline/
│   ├── embedder.py         # Embeds corpus into vector store (Chroma)
│   ├── retriever.py        # RAG retrieval with citation
│   ├── finetune.py         # LoRA fine-tuning on Mistral/Llama
│   └── model.py            # Unified model interface (RAG + LLM)
├── interface/
│   ├── app.py              # Streamlit web interface
│   ├── elicitor.py         # Failure elicitation protocol runner
│   └── annotator.py        # Human validator annotation tool
├── scripts/
│   ├── run_scrape.sh
│   ├── run_embed.sh
│   └── run_elicit.sh
├── tests/
│   └── test_pipeline.py
├── data/
│   └── failure_prompts.json
├── requirements.txt
└── config.py
```

---

## Quickstart

```bash
pip install -r requirements.txt
python corpus/scraper.py --source all --output data/raw/
python corpus/cleaner.py --input data/raw/ --output data/cleaned/
python corpus/chunker.py --input data/cleaned/ --output data/chunks/
python pipeline/embedder.py --input data/chunks/ --store data/chroma/
streamlit run interface/app.py
python interface/elicitor.py --prompts data/failure_prompts.json --output results/
```

---

## The failure taxonomy

| Category | What fails | What it reveals |
|---|---|---|
| **Temporal lock** | Model uses knowledge Darwin didn't have | Consciousness is sequential; representations aren't |
| **Register collapse** | Emotional flatness across contexts | Identity is a distribution, not a mean |
| **Gap confabulation** | Fills silences with plausible fiction | The undocumented interior is structurally irretrievable |
| **Embodiment erasure** | No fatigue, pain, or physical constraint | Personhood includes the body that produces language |
| **Public/private blur** | Over-indexes on published voice | Archives are curated; no corpus is neutral |

---

## Darwin corpus overview

| Source | Documents | Period | Register |
|---|---|---|---|
| Correspondence Project | ~15,000 letters | 1821–1882 | Personal, varied |
| Darwin Online notebooks | ~30 notebooks | 1836–1860 | Private, speculative |
| Published works | 25 books | 1839–1881 | Public, polished |
| Autobiography | 1 document | 1876 | Intimate, retrospective |
| Beagle diary | 1 document | 1831–1836 | Embodied, pre-theory |
