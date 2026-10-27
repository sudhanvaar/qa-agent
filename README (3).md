# Knowledge-Base Q&A Agent (CLI RAG)

A command-line question-answering agent that answers **only** from a Markdown knowledge base. It was built for the Gutermann leak-detection product overview, but it works on any Markdown file or folder of `.md` files.

The agent retrieves the most relevant sections of the document, passes them to an LLM with strict grounding rules, and prints the answer along with the sections it used. If the document does not contain the answer, it says so instead of guessing.

## Features

- **Structure-aware chunking.** The Markdown is split by heading, so every chunk keeps its full heading path (e.g. `Real-Time Correlators For All Conditions > AQUASCAN 610`). Heading markers inside code fences are ignored.
- **Parent-section context.** Text that sits directly under a parent heading is attached to its children, so section-level facts (like "Permanent Leak Detection Monitoring" intro text) are not lost.
- **Hybrid retrieval.** Final score = `0.6 × dense (embedding similarity) + 0.4 × lexical (IDF-weighted keyword overlap)`. Dense search handles paraphrases; lexical search handles exact model names and part numbers.
- **Named-item forcing.** If the query names a product (e.g. "compare AQUASCOPE 3 and AQUASCOPE 550"), those sections are always included in the context, even if they did not rank in the top K.
- **Grounded answers.** The system prompt forbids outside knowledge, invented specs/prices/availability, and mixing up facts between neighbouring products. Questions the document cannot answer get an explicit "not enough information" reply.
- **Verified source citation.** The LLM is asked to name the sections it used. The agent then checks those names against the sections it actually retrieved and prints only the real ones under `From section(s):`. If none can be verified, it falls back to the closest retrieved sections and says so.
- **Flexible backends.** Any OpenAI-compatible LLM endpoint (OpenAI, Ollama, vLLM, LM Studio, etc.). Embeddings run locally or through the API.
- **Debug mode.** `--debug` prints each retrieved chunk with its score.

## Project structure

```
.
├── qa_agent.py           # the whole agent (chunking, retrieval, prompting, CLI)
├── product_overview.md   # knowledge base (default path)
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

- Python 3.10 or newer
- An OpenAI-compatible LLM endpoint

## Setup

```bash
git clone <your-repo-url>
cd <your-repo-name>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Configure the LLM

Set the environment variables for your provider.

**OpenAI**

```bash
export OPENAI_API_KEY="sk-..."
export LLM_MODEL="gpt-4o-mini"
```

**Local model via Ollama** (no API key needed, but the variable must be non-empty)

```bash
export OPENAI_BASE_URL="http://localhost:11434/v1"
export OPENAI_API_KEY="ollama"
export LLM_MODEL="llama3.2"
```

On Windows PowerShell use `$env:OPENAI_API_KEY="..."` instead of `export`.

## Usage

```bash
# Uses product_overview.md next to the script
python qa_agent.py

# Specify a file or a folder of .md files
python qa_agent.py --kb product_overview.md
python qa_agent.py --kb docs/

# Show retrieval scores
python qa_agent.py --debug

# Recommended for the Gutermann file: promote a plain-text title to a real section
python qa_agent.py --implicit-section "Acoustic Leak Detection Microphones"
```

In the Gutermann overview, the line `Acoustic Leak Detection Microphones` is plain text, not a Markdown heading, so without `--implicit-section` the AQUASCOPE products are filed under "Multi-Point Ofline Correlators".

Type a question at the `>` prompt. Type `exit` or `quit` (or press Ctrl+C / Ctrl+D) to leave.

### Example session

```
> Which correlator is designed for plastic and large-diameter pipes over long distances?
Agent: The AQUASCAN TM3 is described as a high-end correlator designed specifically to find
leaks on plastic and large diameter pipes over long distances. ...
From section(s):
  - product_overview > PRODUCT OVERVIEW EN int A4 Web v1.2 Oct 2025 > Real-Time Correlators For All Conditions > AQUASCAN TM3

> Can I order the ZONESCAN HYDRO today?
Agent: The document says ZONESCAN HYDRO is launching in Q2 2026 and that pre-series units
are available by contacting sales. It does not confirm it can be ordered now.
From section(s):
  - product_overview > PRODUCT OVERVIEW EN int A4 Web v1.2 Oct 2025 > Permanent Leak Detection Monitoring > ZONESCAN HYDRO

> Compare AQUASCOPE 3 and AQUASCOPE 550.
Agent: ... (point-by-point comparison, "not stated" where information is missing)
```

Outputs above are illustrative; exact wording depends on the LLM you use.

## Configuration

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | none | API key for the LLM endpoint (and API embeddings) |
| `OPENAI_BASE_URL` | OpenAI default | Point to any OpenAI-compatible server |
| `LLM_MODEL` | `llama3.2` | Chat model used to generate answers |
| `EMBED_BACKEND` | `auto` | `local` (sentence-transformers), `api` (provider embeddings), or `auto` (local if installed, else API) |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `EMBED_API_MODEL` | `text-embedding-3-small` | Embedding model when using the API backend |

### Command-line options

| Option | Default | Purpose |
|---|---|---|
| `--kb PATH` | `product_overview.md` beside the script | Markdown file or directory of `.md` files |
| `--max-chars N` | `1500` | Maximum characters per chunk |
| `--overlap N` | `150` | Overlap carried between split chunks |
| `--implicit-section "TITLE"` | none | Treat a plain-text line as a `##` heading (repeatable). Useful when the source has section titles that are not marked up as headings |
| `--debug` | off | Print retrieved chunks and scores |

## How it works

1. **Clean.** Remove HTML comments (such as the page and image markers in the converted PDF), flatten image links, normalise bullets and whitespace.
2. **Split by heading.** Build a section list that keeps each section's heading ancestry.
3. **Chunk by size.** Sections longer than `--max-chars` are split at paragraph or line boundaries, with a small overlap.
4. **Enrich each chunk.** Prefix the text with the document name, section path, and a short excerpt of the parent sections' own text. This is the text that gets embedded and sent to the LLM.
5. **Index.** Embed all chunks and build an IDF table for keyword scoring. Index a normalised name for every leaf heading so product names in a query can be detected.
6. **Retrieve.** Score every chunk with the hybrid score, take the top 5, and add any sections explicitly named in the query (up to 8 forced).
7. **Generate.** Send the chunks and the question to the LLM at `temperature=0` with the grounding system prompt.
8. **Print.** Show the answer, then verify the LLM's final `SOURCES:` line against the retrieved sections and print the confirmed ones.

## Design decisions

- **Heading-based chunking** fits product catalogues, where each product is a self-contained section; a fixed-size splitter would blur products together.
- **Hybrid scoring** because product names such as "AQUASCAN 620L" are poorly handled by embeddings alone, while keyword search alone misses paraphrased questions.
- **Forced inclusion of named items** makes comparison questions reliable, since both products are guaranteed to be in the context.
- **Strict prompt rules** on availability, pricing and cross-product attribution target the typical failure modes of RAG over spec sheets.
- **A single file with no vector database.** The knowledge base is small enough to hold in memory, so NumPy is all that is needed.

## Limitations

- Embeddings are recomputed at every start; there is no on-disk index cache. This is fine for small documents but slow for large corpora.
- Product names are matched on headings, so a document with no headings (or generic headings only) loses the forced-inclusion behaviour. Use `--implicit-section` to promote plain-text titles.
- Images in the source are dropped; only their alt text survives.
- Answer quality depends on the chosen LLM. Small local models may ignore the grounding rules more often.
- Single-turn only: each question is answered independently, with no conversation memory.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Knowledge base not found` | Check the `--kb` path, or place `product_overview.md` next to `qa_agent.py` |
| `warning: ... produced 1 chunk(s)` | The file has no Markdown headings. Add them or use `--implicit-section` |
| `LLM call failed` | Check `OPENAI_API_KEY`, `OPENAI_BASE_URL` and `LLM_MODEL`; confirm the server is running |
| `sentence-transformers` is slow to install | Skip it and run with `EMBED_BACKEND=api` |
