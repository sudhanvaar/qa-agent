#!/usr/bin/env python3
# --- Imports: standard library, typing helpers and numpy for vector math ---
import argparse
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

# --- Config: model names, retrieval size, hybrid-score weights, chunk sizes ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
TOP_K = 5
W_DENSE, W_LEX = 0.6, 0.4
MAX_CHARS = 1500
OVERLAP = 150
INTRO_MAX_CHARS = 600

# --- Word lists and regexes: headings too vague to name an item, noise words, markdown patterns ---
GENERIC_HEADINGS = {
    "overview", "introduction", "summary", "features", "details", "specifications",
    "specs", "description", "notes", "general", "contents", "faq", "contact",
}
STOPWORDS = set(
    "a an the and or of to in on for with is are was be do does did can i we you it "
    "this that what which who how me my our your at by from as about use uses using "
    "need want find recommend there any have has".split()
)
HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass
class Chunk:
    """One searchable piece of a document: its source file, heading path, body and parent context."""
    source: str
    headings: list[str]
    body: str
    context: str = ""
    text: str = field(init=False)

    def __post_init__(self):
        # Builds the full text that gets embedded and shown to the LLM (doc + section + parent context + body).
        parts = [f"Document: {self.source}"]
        if self.headings:
            parts.append("Section: " + " > ".join(self.headings))
        if self.context:
            parts.append(
                "Parent-section context (applies to everything under it):\n"
                + self.context
            )
        parts.append(self.body)
        self.text = "\n".join(parts)

    @property
    def label(self) -> str:
        # Human-readable section path, e.g. "Laptops > ProBook X".
        return " > ".join(self.headings) if self.headings else self.source


def default_clean(text: str) -> str:
    """Strip comments, image syntax, tabs, odd bullets and extra blank lines from Markdown."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.replace("\t", "    ")
    text = re.sub(r"^[ ]*[•▪◦*]\s+", "- ", text, flags=re.MULTILINE)
    text = re.sub(r"[ ]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def promote_implicit(text: str, titles: list[str]) -> str:
    """Turn chosen plain-text lines into '## ' headings so they become real sections."""
    if not titles:
        return text
    wanted = {title.strip() for title in titles}
    return "\n".join(
        f"## {line.strip()}" if line.strip() in wanted else line
        for line in text.splitlines()
    )


def split_sections(text: str) -> list[tuple[list[str], str]]:
    """Split Markdown into (heading-path, body) sections, ignoring '#' lines inside code fences."""
    sections: list[tuple[list[str], str]] = []
    headings: list[str] = []
    body: list[str] = []
    in_fence = False

    def flush():
        # Saves the text collected so far as one section under the current heading path.
        content = "\n".join(body).strip()
        if content:
            sections.append((list(headings), content))
        body.clear()

    for line in text.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
        match = None if in_fence else HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            del headings[level - 1:]
            headings.append(match.group(2))
        else:
            body.append(line)
    flush()
    return sections


def _units(body: str, max_chars: int) -> Iterator[str]:
    """Yield paragraphs, hard-splitting any paragraph longer than max_chars at a newline/space."""
    for paragraph in re.split(r"\n\s*\n", body):
        while len(paragraph) > max_chars:
            cut = max(
                paragraph.rfind("\n", 0, max_chars),
                paragraph.rfind(" ", 0, max_chars),
            )
            cut = cut if cut > 0 else max_chars
            yield paragraph[:cut].strip()
            paragraph = paragraph[cut:].strip()
        if paragraph:
            yield paragraph


def split_by_size(body: str, max_chars: int, overlap: int) -> list[str]:
    """Pack paragraphs into pieces up to max_chars, repeating a short last paragraph as overlap."""
    if len(body) <= max_chars:
        return [body]
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for unit in _units(body, max_chars):
        if current and size + len(unit) + 2 > max_chars:
            pieces.append("\n\n".join(current))
            tail = current[-1] if overlap and len(current[-1]) <= overlap else None
            current, size = ([tail], len(tail)) if tail else ([], 0)
        current.append(unit)
        size += len(unit) + 2
    if current:
        pieces.append("\n\n".join(current))
    return pieces


def chunk_markdown(
    path: str,
    max_chars: int = MAX_CHARS,
    overlap: int = OVERLAP,
    intro_max_chars: int = INTRO_MAX_CHARS,
    clean: Callable[[str], str] = default_clean,
    implicit_sections: list[str] | None = None,
) -> list[Chunk]:
    """Read one .md file, clean it, split by heading, attach parent-section context, and size-split into Chunks."""
    path_obj = Path(path)
    text = path_obj.read_text(encoding="utf-8")
    sections = split_sections(clean(promote_implicit(text, implicit_sections or [])))

    # Collect each section's own text so child sections can borrow it as parent context.
    own: dict[tuple[str, ...], str] = {}
    for headings, body in sections:
        key = tuple(headings)
        own[key] = (own[key] + "\n" + body) if key in own else body

    chunks: list[Chunk] = []
    for headings, body in sections:
        # Build a trimmed snippet of every ancestor section's text for this chunk.
        context_parts = []
        for depth in range(1, len(headings)):
            parent = own.get(tuple(headings[:depth]))
            if parent:
                snippet = parent[:intro_max_chars].rstrip()
                if len(parent) > intro_max_chars:
                    snippet += " ..."
                context_parts.append(f"[{headings[depth - 1]}] {snippet}")
        context = "\n".join(context_parts)
        for piece in split_by_size(body, max_chars, overlap):
            chunks.append(Chunk(path_obj.stem, headings, piece, context))
    return chunks


def load_chunks(path: str, **kwargs) -> list[Chunk]:
    """Load chunks from one file or every .md under a directory, warning about files that barely split."""
    source = Path(path)
    files = sorted(source.rglob("*.md")) if source.is_dir() else [source]
    chunks: list[Chunk] = []
    for file in files:
        found = chunk_markdown(str(file), **kwargs)
        if len(found) <= 1:
            print(
                f"  warning: {file.name} produced {len(found)} chunk(s); check its structure.",
                file=sys.stderr,
            )
        chunks.extend(found)
    return chunks


def tokenize(text: str) -> list[str]:
    """Lowercase, drop stopwords/short tokens and crudely singularize plurals for keyword matching."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    result = []
    for token in tokens:
        if token in STOPWORDS or len(token) < 2:
            continue
        if len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        result.append(token)
    return result


class Embedder:
    """Use local embeddings if available, otherwise an OpenAI-compatible API."""

    def __init__(self, client):
        # Picks local sentence-transformers or the API depending on EMBED_BACKEND and installed packages.
        self.client = client
        self.api_model = os.getenv("EMBED_API_MODEL", "text-embedding-3-small")
        self.local = None
        backend = os.getenv("EMBED_BACKEND", "auto")
        if backend != "api":
            try:
                from sentence_transformers import SentenceTransformer

                print(f"Loading local embedding model ({EMBED_MODEL})...", file=sys.stderr)
                self.local = SentenceTransformer(EMBED_MODEL)
            except ImportError:
                if backend == "local":
                    raise
        if self.local is None:
            print(f"Using API embeddings ({self.api_model}).", file=sys.stderr)

    def encode(self, texts: list[str]) -> np.ndarray:
        # Returns unit-length embedding vectors for a list of texts (local model or API).
        if self.local is not None:
            return self.local.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        response = self.client.embeddings.create(model=self.api_model, input=texts)
        embeddings = np.array([item.embedding for item in response.data], dtype=float)
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


class Retriever:
    """Hybrid (semantic + keyword) search over chunks, with forced inclusion of items named in the query."""
    MAX_FORCED = 8

    def __init__(self, chunks: list[Chunk], embedder: Embedder):
        # Precomputes chunk embeddings, token sets, IDF weights and a name -> chunk-index lookup.
        self.chunks = chunks
        self.model = embedder
        self.emb = self.model.encode([chunk.text for chunk in chunks])
        self.chunk_tokens = [set(tokenize(chunk.text)) for chunk in chunks]
        document_frequency: dict[str, int] = {}
        for tokens in self.chunk_tokens:
            for token in tokens:
                document_frequency[token] = document_frequency.get(token, 0) + 1
        self.idf = {
            token: math.log((len(chunks) + 1) / (count + 0.5)) + 1
            for token, count in document_frequency.items()
        }

        # Force inclusion of query-named items (especially in comparisons).
        self.names: dict[str, list[int]] = {}
        for index, chunk in enumerate(chunks):
            if chunk.headings:
                name = self._norm(chunk.headings[-1])
                if len(name) >= 4 and name not in GENERIC_HEADINGS:
                    self.names.setdefault(name, []).append(index)

    @staticmethod
    def _norm(text: str) -> str:
        # Lowercases and strips everything except letters/digits so names compare loosely.
        return re.sub(r"[^a-z0-9]", "", text.lower())

    def _named_items(self, query: str) -> list[int]:
        # Finds chunks whose section name literally appears in the query (capped at MAX_FORCED).
        normalized_query = self._norm(query)
        indices = []
        for name, matches in self.names.items():
            if name in normalized_query and len(matches) <= 3:
                indices.extend(matches)
        return indices[: self.MAX_FORCED]

    def search(self, query: str, k: int = TOP_K):
        # Scores every chunk (0.6 dense + 0.4 IDF-weighted keyword), then returns top-k plus any forced chunks.
        dense = self.emb @ self.model.encode([query])[0]
        query_tokens = set(tokenize(query))
        denominator = sum(self.idf.get(t, 1.0) for t in query_tokens) or 1.0
        lexical = np.array([
            sum(self.idf.get(t, 0.0) for t in query_tokens if t in tokens) / denominator
            for tokens in self.chunk_tokens
        ])
        scores = W_DENSE * dense + W_LEX * lexical

        ranked = list(np.argsort(-scores))
        forced = self._named_items(query)
        chosen = list(dict.fromkeys(forced + ranked))[:max(k, len(forced))]
        forced_set = set(forced)
        return [(self.chunks[i], float(scores[i]), i in forced_set) for i in chosen]

# Grounding rules: they directly affect generated answers.
SYSTEM_PROMPT = """You are a knowledge-base assistant. Answer ONLY from the \
CONTEXT provided. Rules:

1. Use no outside knowledge. If the context does not contain enough information \
to answer, say explicitly: "The document does not contain enough information to \
answer that," then state what the document DOES say that is relevant.
2. Never invent specs, prices, availability, order/shipping status or dates. If \
asked whether something can be ordered/bought and the context only mentions a \
launch date or "contact sales", say that: report what is stated and that the \
document does not confirm it can be ordered now.
3. Do not mix up items. Attribute each fact to the exact section whose text \
states it. A feature listed for one item is NOT a feature of another, even if \
they sit under the same parent section. Check technical claims carefully.
4. Text under "Parent-section context" applies to everything beneath that parent \
section; you may attribute it to those items, saying it is section-level.
5. For recommendations, pick the item(s) whose text explicitly matches ALL of the \
user's constraints, explain which words in the document justify it, and mention \
close alternatives only if the document supports them.
6. For comparisons, cover both items point by point, using only stated facts; \
say "not stated" where one item lacks information.
7. Be concise. End with a line exactly in the form:
SOURCES: <section name>, <section name>
listing only the entries you actually used."""


def build_context(hits) -> str:
    """Join the retrieved chunks into a numbered CONTEXT block separated by '---'."""
    return "\n\n---\n\n".join(f"[{i}] {chunk.text}" for i, (chunk, _, _) in enumerate(hits, 1))


def answer(client, query: str, hits) -> str:
    """Send the system prompt, context and question to the LLM (temperature 0) and return its reply."""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{build_context(hits)}\n\nQUESTION: {query}"},
        ],
    )
    return response.choices[0].message.content.strip()


def split_sources(text: str):
    """Separate the answer body from the trailing 'SOURCES:' line the LLM was told to write."""
    match = re.search(r"^\s*SOURCES:\s*(.+)$", text, flags=re.MULTILINE | re.IGNORECASE)
    if not match:
        return text, None
    return text[:match.start()].rstrip(), match.group(1).strip()


# ---------------------------------------------------------------------------
# NEW (bonus): show exactly which document/section(s) the answer came from.
# ---------------------------------------------------------------------------
def resolve_sections(sources: str | None, hits) -> list[str]:
    """Match the LLM's cited names against the retrieved chunks, so only real sections are shown."""
    if not sources:
        return []
    cited = Retriever._norm(sources)
    found: list[str] = []
    for chunk, _, _ in hits:
        if not chunk.headings:
            continue
        name = Retriever._norm(chunk.headings[-1])
        entry = f"{chunk.source} > {chunk.label}"
        if len(name) >= 3 and name in cited and entry not in found:
            found.append(entry)
    return found


def print_provenance(sources: str | None, hits) -> None:
    """Print verified source sections; fall back to the top retrieved sections if none could be matched."""
    verified = resolve_sections(sources, hits)
    if verified:
        print("From section(s):")
        for entry in verified:
            print(f"  - {entry}")
    else:
        print("Closest retrieved section(s) (LLM cited none that could be verified):")
        seen: list[str] = []
        for chunk, _, _ in hits[:3]:
            entry = f"{chunk.source} > {chunk.label}"
            if entry not in seen:
                seen.append(entry)
                print(f"  - {entry}")


def main():
    """CLI entry point: parse args, build chunks and retriever, then loop answering questions."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kb", default=str(Path(__file__).with_name("product_overview.md")),
        help="a Markdown file or directory of .md files",
    )
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS)
    parser.add_argument("--overlap", type=int, default=OVERLAP)
    parser.add_argument("--implicit-section", action="append", default=[])
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.kb):
        sys.exit(f"Knowledge base not found: {args.kb}")
    chunks = load_chunks(
        args.kb, max_chars=args.max_chars, overlap=args.overlap,
        implicit_sections=args.implicit_section,
    )
    if not chunks:
        sys.exit("No chunks were produced; is the file empty?")
    print(f"Parsed {len(chunks)} chunks from {len({c.source for c in chunks})} document(s).", file=sys.stderr)

    from openai import OpenAI
    client = OpenAI()
    retriever = Retriever(chunks, Embedder(client))
    print("Knowledge-base Q&A. Type 'exit' or 'quit' to leave.\n")
    while True:
        # Read one question; blank lines are skipped, exit/quit/Ctrl-D/Ctrl-C end the loop.
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        hits = retriever.search(query)
        if args.debug:
            for chunk, score, forced in hits:
                tag = " (forced: named in query)" if forced else ""
                print(f"  [debug] {score:.3f}  {chunk.source} / {chunk.label}{tag}")
        try:
            raw = answer(client, query, hits)
        except Exception as exc:
            print(f"Agent: LLM call failed: {exc}\n")
            continue
        body, sources = split_sources(raw)
        print(f"Agent: {body}")
        print_provenance(sources, hits)  # NEW: show which section(s) the answer came from
        print()


if __name__ == "__main__":
    main()
