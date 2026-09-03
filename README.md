# AI Archaeologist — Charles Darwin

> Train a model on Darwin's writings. Study what it gets wrong.

**A systematic study of where a language model grounded in one person's archive
breaks down, and what those breakages reveal about what a person is.**

There is a genre of project that trains a model on a dead writer and presents the
output as resurrection. This is the opposite of that. The interesting thing about
a Darwin-grounded model is not the passages where it sounds like Darwin. It is the
specific, repeatable ways it fails, because each failure mode marks something
about personhood that a text corpus cannot carry.

Darwin is unusually good material for this. He left roughly 15,000 letters, thirty
notebooks, twenty-five books, an autobiography written for his children, and the
Beagle diary of a twenty-two-year-old who had not yet had the idea. The same
person across sixty years, five registers, and the boundary between private
speculation and published caution. If any archive is dense enough to expose what
archives leave out, it is this one.

---

## The failure taxonomy

This is the core of the project. Each category is a way the model reliably fails,
paired with what that failure indicates.

| Category | What fails | What it reveals |
|---|---|---|
| **Temporal lock** | Model uses knowledge Darwin did not have | Consciousness is sequential; a representation trained on the whole corpus has no "not yet" |
| **Register collapse** | Emotional flatness across contexts | Identity is a distribution, not a mean. Averaging the letters and the books produces neither |
| **Gap confabulation** | Fills documented silences with plausible fiction | The undocumented interior is structurally irrecoverable, and the model cannot represent its own absence |
| **Embodiment erasure** | No fatigue, pain, or physical constraint | Personhood includes the body that produced the text. Darwin was chronically ill for forty years and it shaped everything |
| **Public/private blur** | Over-indexes on the published voice | Archives are curated. No corpus is neutral, and the published works survive better than the discarded ones |

`data/failure_prompts.json` holds the elicitation protocol: five categories of
prompt designed to trigger each failure mode deliberately rather than waiting to
encounter it.

The move that makes this a study rather than a demo is that failure is the
measurement. A model that never confabulated in a documented gap would be
evidence about the archive. One that always does is evidence about the method.

---

## The corpus

| Source | Documents | Period | Register |
|---|---|---|---|
| Darwin Correspondence Project | ~15,000 letters | 1821–1882 | Personal, varied by recipient |
| Darwin Online notebooks | ~30 notebooks | 1836–1860 | Private, speculative |
| Published works | 25 books | 1839–1881 | Public, polished, cautious |
| Autobiography | 1 document | 1876 | Intimate, retrospective |
| Beagle diary | 1 document | 1831–1836 | Embodied, pre-theory |

**These are targets, not a manifest.** The scrapers are written; nothing has been
scraped into this repository yet. `data/` currently contains only the failure
prompts, and `data/raw`, `cleaned`, `chunks` and `chroma` are created on first run.

The register column is doing real work in the design. A letter to Hooker, a
notebook entry, and a paragraph of *Origin* are three different voices from one
person, and collapsing them is exactly the failure that "register collapse" names.

---

## Architecture

```
   corpus/scraper.py      Darwin Online + Correspondence Project
        |                 -> data/raw/
        v
   corpus/cleaner.py      normalise, deduplicate, strip apparatus
        |                 -> data/cleaned/
        v
   corpus/chunker.py      RAG-ready chunks carrying source, date, register
        |                 -> data/chunks/
        v
   pipeline/embedder.py   embed into Chroma
        |                 -> data/chroma/
        v
   pipeline/retriever.py  retrieval with citation back to the source document
   pipeline/finetune.py   optional LoRA on Mistral or Llama
   pipeline/model.py      unified interface over RAG and the fine-tuned model
        |
        +--> interface/app.py         Streamlit, for exploration
        +--> interface/elicitor.py    runs the failure protocol over the taxonomy
        +--> interface/annotator.py   human validator marks each failure
```

Retrieval carries citations by design. A claim the model makes about Darwin should
be traceable to a letter with a date, which is also what makes temporal-lock
failures detectable rather than merely suspected.

## Running it

```bash
pip install -r requirements.txt

python corpus/scraper.py  --source all       --output data/raw/
python corpus/cleaner.py  --input data/raw/  --output data/cleaned/
python corpus/chunker.py  --input data/cleaned/ --output data/chunks/
python pipeline/embedder.py --input data/chunks/ --store data/chroma/

streamlit run interface/app.py
python interface/elicitor.py --prompts data/failure_prompts.json --output results/
```

Paths are centralised in `config.py`, which creates `data/raw`, `data/cleaned`,
`data/chunks`, `data/chroma`, `models/` and `results/` on import.

## Layout

```
corpus/     scraper.py, cleaner.py, chunker.py
pipeline/   embedder.py, retriever.py, finetune.py, model.py
interface/  app.py, elicitor.py, annotator.py
data/       failure_prompts.json  (the rest is generated)
config.py   paths and model settings
```

## State of the work

The pipeline is written end to end and has not been run against the real archive.
There are no scraped documents, no embeddings, no fine-tuned adapters and no
elicitation results in this repository. `corpus/tagger.py`, a `scripts/` directory
and `tests/` are referenced in the design and do not exist yet.

That is stated plainly because the failure taxonomy reads like findings and is not.
It is a set of predictions about what will break, written before the model was
built. Whether the five categories survive contact with an actual elicitation run
is the open question, and it is the whole point of running it.

## Limitations

**The archive is curated and the curation is invisible.** What survived to Darwin
Online survived because someone kept it. Letters he burned, conversations he had,
and the notebooks he did not keep are absent in a way that no amount of retrieval
fixes. Public/private blur names this and cannot correct it.

**Fine-tuning on 15,000 letters teaches style, not belief.** A LoRA adapter that
reproduces Darwin's cadence has not learned what he thought, and the difference is
easy to lose when the output is fluent.

**Failure categories are the author's, not derived.** Five categories chosen in
advance is a hypothesis. A grounded-theory pass over actual failures might produce
different or additional ones, and the annotator tool exists to support that.

**No control.** Running the identical pipeline on a second figure with a comparably
dense archive would separate what is true of Darwin from what is true of the
method. Nothing here does that yet.

## Roadmap

- Run the scrapers and report corpus statistics against the target table above
- Execute the elicitation protocol and publish the failure rates per category,
  which is the result this repository is structured to produce
- A second subject with a comparable archive, as a control on the taxonomy
- Register-conditioned retrieval, so a question can be asked of the letter-writing
  Darwin rather than the book-writing one
- Human annotation of a sample, using `interface/annotator.py`, to check whether
  the categories are separable in practice

## Sources

Darwin Correspondence Project, University of Cambridge. Darwin Online
(darwin-online.org.uk), John van Wyhe ed.

## License

MIT. See [LICENSE](LICENSE).
