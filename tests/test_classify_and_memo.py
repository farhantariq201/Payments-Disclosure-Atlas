import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pda.chunks import Chunk
from pda.classify import (
    Classifier,
    Prediction,
    _parse_response,
    build_system_prompt,
    disagreements,
)
from pda.config import load_peers, load_taxonomy
from pda.memo import CompanyMemo, render_crosscut
from pda.timeline import build_panel, first_mentions

CONFIG = Path(__file__).resolve().parents[1] / "config"


def _taxonomy():
    return load_taxonomy(CONFIG / "taxonomy.yaml")


class FakeMessages:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        body = self.payloads.pop(0)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=body)]
        )


class FakeClient:
    def __init__(self, payloads):
        self.messages = FakeMessages(payloads)


def _classifier(payloads, tmp_path=None):
    classifier = Classifier(
        taxonomy=_taxonomy(), cache_dir=tmp_path, api_key="test-key"
    )
    classifier._client = FakeClient(payloads)
    return classifier


def test_prompt_contains_every_theme_and_the_abstention_rule():
    prompt = build_system_prompt(_taxonomy())
    for theme in _taxonomy().themes:
        assert f"[{theme.id}]" in prompt
    assert "empty list" in prompt.lower()


def test_parse_response_tolerates_prose_around_the_json():
    parsed = _parse_response(
        'Sure, here you go:\n[{"index": 0, "labels": ["a"]}]\nHope that helps.', 1
    )
    assert parsed[0]["labels"] == ["a"]


def test_parse_response_fills_gaps_for_skipped_indices():
    """A short response must not silently shift labels onto the wrong chunk."""
    parsed = _parse_response('[{"index": 2, "labels": ["a"]}]', 3)
    assert parsed[0]["labels"] == []
    assert parsed[2]["labels"] == ["a"]


def test_parse_response_rejects_non_json():
    with pytest.raises(ValueError):
        _parse_response("I could not complete this request.", 1)


def test_invented_theme_ids_are_dropped_from_predictions():
    payload = json.dumps(
        [{"index": 0, "labels": ["isv_channel_displacement", "hallucinated_theme"]}]
    )
    classifier = _classifier([payload])
    result = classifier.classify(["c1"], ["some text about software vendors"])
    assert result[0].labels == ["isv_channel_displacement"]


def test_predictions_come_back_in_input_order():
    payload = json.dumps(
        [
            {"index": 0, "labels": []},
            {"index": 1, "labels": ["take_rate_compression"]},
        ]
    )
    classifier = _classifier([payload])
    result = classifier.classify(["a", "b"], ["text one", "text two"])
    assert [r.chunk_id for r in result] == ["a", "b"]
    assert result[1].labels == ["take_rate_compression"]


def test_cache_prevents_a_second_api_call(tmp_path):
    payload = json.dumps([{"index": 0, "labels": ["ai_deployment"]}])
    classifier = _classifier([payload], tmp_path=tmp_path)
    classifier.classify(["c1"], ["machine learning in underwriting"])
    assert len(classifier._client.messages.calls) == 1

    again = _classifier([], tmp_path=tmp_path)
    result = again.classify(["c1"], ["machine learning in underwriting"])
    assert result[0].labels == ["ai_deployment"]
    assert again._client.messages.calls == []


def test_taxonomy_version_change_invalidates_the_cache(tmp_path):
    payload = json.dumps([{"index": 0, "labels": ["ai_deployment"]}])
    classifier = _classifier([payload], tmp_path=tmp_path)
    classifier.classify(["c1"], ["machine learning in underwriting"])

    bumped = _classifier([payload], tmp_path=tmp_path)
    bumped.taxonomy = bumped.taxonomy.model_copy(update={"version": "2.0.0"})
    bumped.classify(["c1"], ["machine learning in underwriting"])
    assert len(bumped._client.messages.calls) == 1


def test_disagreements_flag_the_chunks_worth_adjudicating():
    taxonomy = _taxonomy()
    texts = [
        "We face competition from integrated software vendors in our channel.",
        "Generic boilerplate with no thematic content whatsoever here.",
    ]
    predictions = [
        Prediction("a", [], "m", taxonomy.version),                       # model missed it
        Prediction("b", [], "m", taxonomy.version),                       # both agree
    ]
    assert disagreements(texts, predictions, taxonomy) == [0]


def _panel_fixture():
    peers, taxonomy = load_peers(CONFIG / "peers.yaml"), _taxonomy()
    chunks, labels = [], {}
    for year in (2022, 2023):
        for ordinal in range(3):
            chunk = Chunk(
                chunk_id=f"FI-{year}-item1a-{ordinal:04d}",
                ticker="FI",
                fiscal_year=year,
                item="item1a",
                ordinal=ordinal,
                text="Competition from integrated software vendors is intensifying.",
                text_hash=f"h{year}{ordinal}",
                filing_accession="a",
                filing_date=f"{year+1}-02-01",
                is_new_language=(ordinal == 0),
            )
            chunks.append(chunk)
            labels[chunk.chunk_id] = (
                ["isv_channel_displacement"] if ordinal == 0 else []
            )
    return chunks, labels, peers, taxonomy


def test_company_memo_renders_with_tables_and_quotes():
    chunks, labels, peers, taxonomy = _panel_fixture()
    panel = build_panel(chunks, labels, peers, taxonomy)
    document = CompanyMemo(
        ticker="FI",
        name="Fiserv, Inc.",
        cohort="incumbent",
        fiscal_years=[2022, 2023],
        financials={"Revenue": {2022: 1e9, 2023: 1.2e9}},
        theme_rows=[r for r in panel if r.ticker == "FI"],
        quotes=[("isv_channel_displacement", 2023, "Competition is intensifying.")],
    ).render(taxonomy)

    assert "Fiserv, Inc. (FI)" in document
    assert "ISV channel displacement" in document
    assert "1,200" in document           # revenue rendered in millions
    assert "as first filed" in document  # restatement caveat survives


def test_crosscut_reports_denominators_and_the_wrongness_section():
    chunks, labels, peers, taxonomy = _panel_fixture()
    panel = build_panel(chunks, labels, peers, taxonomy)
    document = render_crosscut(
        panel, first_mentions(panel, peers), peers, taxonomy,
        "isv_channel_displacement",
    )
    assert "Adoption curve" in document
    assert "Censored" in document
    assert "would have to be true for this to be wrong" in document
