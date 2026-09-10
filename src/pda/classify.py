"""LLM classification of filing paragraphs against the taxonomy.

Design notes worth defending in an interview:

*   Two-tier routing. Every chunk is labelled by a cheap model. Chunks where
    the cheap model and the keyword baseline disagree get re-labelled by a
    stronger model. Most paragraphs are obvious; spending Opus-tier tokens on
    all of them is how these projects end up costing $400 and never finishing.
*   Content-addressed cache. The key is a hash of the text, the taxonomy
    version and the model id. Re-running after a taxonomy edit correctly
    invalidates; re-running after a crash correctly resumes.
*   Batched prompts. Chunks go up several at a time with explicit indices,
    which cuts the per-call overhead of resending the taxonomy.
*   Forced abstention. The prompt says an empty list is the expected answer
    for most paragraphs. Without that instruction the model finds a theme in
    everything and precision collapses.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .baseline import classify_chunk as baseline_classify
from .config import Taxonomy, load_taxonomy

BULK_MODEL = os.environ.get("PDA_BULK_MODEL", "claude-haiku-4-5-20251001")
ADJUDICATOR_MODEL = os.environ.get("PDA_ADJUDICATOR_MODEL", "claude-sonnet-5")
BATCH_SIZE = int(os.environ.get("PDA_BATCH_SIZE", "6"))

_JSON_BLOCK = re.compile(r"\[.*\]", re.DOTALL)


@dataclass(frozen=True)
class Prediction:
    chunk_id: str
    labels: list[str]
    model: str
    taxonomy_version: str
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "labels": self.labels,
            "model": self.model,
            "taxonomy_version": self.taxonomy_version,
            "rationale": self.rationale,
        }


def build_system_prompt(taxonomy: Taxonomy) -> str:
    lines = [
        "You are labelling paragraphs from SEC 10-K filings by payments and "
        "merchant acquiring companies against a fixed theme taxonomy.",
        "",
        "THEMES:",
    ]
    for theme in taxonomy.themes:
        lines.append(f"\n[{theme.id}] {theme.name}")
        lines.append(f"  Definition: {theme.definition.strip()}")
        if theme.includes:
            lines.append("  Counts as this theme:")
            lines.extend(f"    - {item}" for item in theme.includes)
        if theme.excludes:
            lines.append("  Does NOT count:")
            lines.extend(f"    - {item}" for item in theme.excludes)
        if theme.examples.positive:
            lines.append(f"  Example (positive): {theme.examples.positive[0]}")
        if theme.examples.negative:
            lines.append(f"  Example (negative): {theme.examples.negative[0]}")

    lines += [
        "",
        "RULES:",
        "1. Most paragraphs match NO theme. An empty list is the correct and "
        "expected answer for boilerplate, financial tables, and generic "
        "language. Do not stretch to find a match.",
        "2. Assign a theme only if the paragraph makes a substantive claim "
        "that fits the definition, not because a keyword appears.",
        "3. A paragraph may carry more than one theme.",
        "4. Use only the exact theme ids listed above.",
        "",
        "OUTPUT: a JSON array with one object per input paragraph, in the same "
        'order, of the form {"index": <int>, "labels": [<theme ids>], '
        '"rationale": "<one short clause, or empty>"}. Return the JSON array '
        "and nothing else.",
    ]
    return "\n".join(lines)


def _cache_key(text: str, taxonomy_version: str, model: str) -> str:
    payload = f"{model}|{taxonomy_version}|{text}".encode()
    return hashlib.sha256(payload).hexdigest()[:32]


class Cache:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict | None:
        path = self.directory / f"{key}.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                return None
        return None

    def put(self, key: str, value: dict) -> None:
        (self.directory / f"{key}.json").write_text(json.dumps(value))


def _parse_response(raw: str, batch_size: int) -> list[dict]:
    match = _JSON_BLOCK.search(raw)
    if not match:
        raise ValueError(f"No JSON array in response: {raw[:200]}")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON array")
    by_index = {int(item.get("index", i)): item for i, item in enumerate(parsed)}
    return [by_index.get(i, {"index": i, "labels": []}) for i in range(batch_size)]


class Classifier:
    """Wraps the Anthropic client. Import is deferred so the rest of the
    package, and the whole test suite, runs without the SDK installed."""

    def __init__(
        self,
        taxonomy: Taxonomy | None = None,
        cache_dir: Path | None = None,
        api_key: str | None = None,
    ) -> None:
        self.taxonomy = taxonomy or load_taxonomy()
        self.system = build_system_prompt(self.taxonomy)
        self.cache = Cache(cache_dir) if cache_dir else None
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from anthropic import Anthropic  # deferred

            if not self._api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            self._client = Anthropic(api_key=self._api_key)
        return self._client

    def _call(self, texts: Sequence[str], model: str) -> list[dict]:
        numbered = "\n\n".join(
            f"--- PARAGRAPH {i} ---\n{text}" for i, text in enumerate(texts)
        )
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self.client.messages.create(
                    model=model,
                    max_tokens=1500,
                    system=self.system,
                    messages=[{"role": "user", "content": numbered}],
                )
                raw = "".join(
                    block.text for block in response.content if block.type == "text"
                )
                return _parse_response(raw, len(texts))
            except Exception as error:  # noqa: BLE001 - retried and re-raised
                last_error = error
                time.sleep(2**attempt)
        raise RuntimeError(f"Classification failed after retries: {last_error}")

    def classify(
        self,
        chunk_ids: Sequence[str],
        texts: Sequence[str],
        *,
        model: str = BULK_MODEL,
        batch_size: int = BATCH_SIZE,
    ) -> list[Prediction]:
        results: dict[str, Prediction] = {}
        pending_ids: list[str] = []
        pending_texts: list[str] = []

        for chunk_id, text in zip(chunk_ids, texts, strict=True):
            key = _cache_key(text, self.taxonomy.version, model)
            hit = self.cache.get(key) if self.cache else None
            if hit is not None:
                results[chunk_id] = Prediction(
                    chunk_id=chunk_id,
                    labels=self.taxonomy.validate_labels(hit.get("labels", [])),
                    model=model,
                    taxonomy_version=self.taxonomy.version,
                    rationale=hit.get("rationale", ""),
                )
            else:
                pending_ids.append(chunk_id)
                pending_texts.append(text)

        for start in range(0, len(pending_texts), batch_size):
            batch_ids = pending_ids[start : start + batch_size]
            batch_texts = pending_texts[start : start + batch_size]
            for chunk_id, text, item in zip(
                batch_ids, batch_texts, self._call(batch_texts, model), strict=True
            ):
                labels = self.taxonomy.validate_labels(item.get("labels", []))
                rationale = str(item.get("rationale", ""))[:300]
                if self.cache:
                    self.cache.put(
                        _cache_key(text, self.taxonomy.version, model),
                        {"labels": labels, "rationale": rationale},
                    )
                results[chunk_id] = Prediction(
                    chunk_id=chunk_id,
                    labels=labels,
                    model=model,
                    taxonomy_version=self.taxonomy.version,
                    rationale=rationale,
                )

        return [results[chunk_id] for chunk_id in chunk_ids]


def disagreements(
    texts: Sequence[str], predictions: Sequence[Prediction], taxonomy: Taxonomy
) -> list[int]:
    """Indices where the LLM and the keyword baseline disagree.

    These are the chunks worth spending the stronger model on: either the
    baseline caught a keyword the model dismissed, or the model found a theme
    with no lexical anchor. Both are where the errors live.
    """
    out: list[int] = []
    for index, (text, prediction) in enumerate(
        zip(texts, predictions, strict=True)
    ):
        if set(baseline_classify(text, taxonomy)) != set(prediction.labels):
            out.append(index)
    return out
