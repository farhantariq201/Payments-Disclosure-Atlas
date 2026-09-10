"""Typed loaders for the peer set and theme taxonomy."""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(os.environ.get("PDA_CONFIG_DIR", REPO_ROOT / "config"))
DATA_DIR = Path(os.environ.get("PDA_DATA_DIR", REPO_ROOT / "data"))


class Peer(BaseModel):
    ticker: str
    name: str
    cohort: str = ""


class Cohort(BaseModel):
    id: str
    label: str
    thesis: str = ""
    members: list[Peer]


class Study(BaseModel):
    name: str
    start_fiscal_year: int
    end_fiscal_year: int
    forms: list[str] = Field(default_factory=lambda: ["10-K"])


class PeerSet(BaseModel):
    study: Study
    cohorts: list[Cohort]
    partial_history: dict[str, int] = Field(default_factory=dict)

    @property
    def peers(self) -> list[Peer]:
        out: list[Peer] = []
        for cohort in self.cohorts:
            for member in cohort.members:
                out.append(member.model_copy(update={"cohort": cohort.id}))
        return out

    @property
    def tickers(self) -> list[str]:
        return [p.ticker for p in self.peers]

    def cohort_of(self, ticker: str) -> str:
        for peer in self.peers:
            if peer.ticker == ticker:
                return peer.cohort
        raise KeyError(f"{ticker} is not in the peer set")

    def first_covered_year(self, ticker: str) -> int:
        """Earliest fiscal year for which absence of a theme is meaningful.

        For companies that IPO'd mid-window there is no filing before this
        year, so silence is not evidence of anything.
        """
        return self.partial_history.get(ticker, self.study.start_fiscal_year)


class ThemeExamples(BaseModel):
    positive: list[str] = Field(default_factory=list)
    negative: list[str] = Field(default_factory=list)


class Theme(BaseModel):
    id: str
    name: str
    definition: str
    includes: list[str] = Field(default_factory=list)
    excludes: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    examples: ThemeExamples = Field(default_factory=ThemeExamples)


class Taxonomy(BaseModel):
    version: str
    themes: list[Theme]
    sections_in_scope: list[str]

    @property
    def ids(self) -> list[str]:
        return [t.id for t in self.themes]

    def get(self, theme_id: str) -> Theme:
        for theme in self.themes:
            if theme.id == theme_id:
                return theme
        raise KeyError(theme_id)

    def validate_labels(self, labels: list[str]) -> list[str]:
        """Drop anything the model invented. Returns the clean, sorted list."""
        known = set(self.ids)
        return sorted({label for label in labels if label in known})


@cache
def load_peers(path: str | Path | None = None) -> PeerSet:
    target = Path(path) if path else CONFIG_DIR / "peers.yaml"
    return PeerSet.model_validate(yaml.safe_load(target.read_text()))


@cache
def load_taxonomy(path: str | Path | None = None) -> Taxonomy:
    target = Path(path) if path else CONFIG_DIR / "taxonomy.yaml"
    return Taxonomy.model_validate(yaml.safe_load(target.read_text()))
