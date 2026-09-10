"""Multi-label evaluation metrics with confidence intervals.

Every number the benchmark reports comes from here. Three deliberate choices:

*   Multi-label, not multi-class. A paragraph can carry two themes or none, so
    per-theme precision and recall are computed independently and the empty
    prediction is a legitimate, correct answer.
*   Bootstrap confidence intervals. On a 400-chunk gold set a rare theme might
    have twelve positives; a point estimate of F1 on twelve examples deserves
    an error bar, and quoting one without it is the most common way these
    projects overstate themselves.
*   Cohen's kappa for self-consistency. If you relabel 100 chunks a week later
    and agree with yourself only 70% of the time, your gold set has a ceiling
    and the model cannot be blamed for hitting it.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

Labels = Sequence[Sequence[str]]


@dataclass(frozen=True)
class ThemeScore:
    theme: str
    support: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    f1_low: float = 0.0
    f1_high: float = 0.0


@dataclass(frozen=True)
class BenchmarkResult:
    per_theme: list[ThemeScore]
    micro_f1: float
    macro_f1: float
    micro_f1_low: float
    micro_f1_high: float
    exact_match_rate: float
    abstain_accuracy: float
    n: int

    def as_rows(self) -> list[dict]:
        return [
            {
                "theme": s.theme,
                "support": s.support,
                "precision": round(s.precision, 3),
                "recall": round(s.recall, 3),
                "f1": round(s.f1, 3),
                "f1_ci": f"[{s.f1_low:.2f}, {s.f1_high:.2f}]",
            }
            for s in self.per_theme
        ]


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return precision, recall, f1


def _counts(gold: Labels, pred: Labels, theme: str) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for gold_row, pred_row in zip(gold, pred, strict=False):
        in_gold = theme in gold_row
        in_pred = theme in pred_row
        if in_gold and in_pred:
            tp += 1
        elif in_pred:
            fp += 1
        elif in_gold:
            fn += 1
    return tp, fp, fn


def _micro_f1(gold: Labels, pred: Labels, themes: Sequence[str]) -> float:
    tp = fp = fn = 0
    for theme in themes:
        t, f, n = _counts(gold, pred, theme)
        tp += t
        fp += f
        fn += n
    return _prf(tp, fp, fn)[2]


def _bootstrap_ci(
    gold: Labels,
    pred: Labels,
    themes: Sequence[str],
    *,
    theme: str | None = None,
    rounds: int = 1000,
    seed: int = 17,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not gold:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(gold)
    samples: list[float] = []
    for _ in range(rounds):
        idx = [rng.randrange(n) for _ in range(n)]
        g = [gold[i] for i in idx]
        p = [pred[i] for i in idx]
        if theme is None:
            samples.append(_micro_f1(g, p, themes))
        else:
            samples.append(_prf(*_counts(g, p, theme))[2])
    samples.sort()
    lo = samples[int(alpha / 2 * rounds)]
    hi = samples[min(rounds - 1, int((1 - alpha / 2) * rounds))]
    return lo, hi


def evaluate(
    gold: Labels,
    pred: Labels,
    themes: Sequence[str],
    *,
    bootstrap_rounds: int = 1000,
    seed: int = 17,
) -> BenchmarkResult:
    if len(gold) != len(pred):
        raise ValueError(f"gold has {len(gold)} rows, pred has {len(pred)}")

    gold = [sorted(set(row)) for row in gold]
    pred = [sorted(set(row)) for row in pred]

    per_theme: list[ThemeScore] = []
    for theme in themes:
        tp, fp, fn = _counts(gold, pred, theme)
        precision, recall, f1 = _prf(tp, fp, fn)
        low, high = (
            _bootstrap_ci(
                gold, pred, themes, theme=theme, rounds=bootstrap_rounds, seed=seed
            )
            if bootstrap_rounds
            else (0.0, 0.0)
        )
        per_theme.append(
            ThemeScore(
                theme=theme,
                support=tp + fn,
                tp=tp,
                fp=fp,
                fn=fn,
                precision=precision,
                recall=recall,
                f1=f1,
                f1_low=low,
                f1_high=high,
            )
        )

    micro = _micro_f1(gold, pred, themes)
    macro = sum(s.f1 for s in per_theme) / len(per_theme) if per_theme else 0.0
    micro_low, micro_high = (
        _bootstrap_ci(gold, pred, themes, rounds=bootstrap_rounds, seed=seed)
        if bootstrap_rounds
        else (0.0, 0.0)
    )

    exact = sum(1 for g, p in zip(gold, pred, strict=True) if g == p) / len(gold) if gold else 0.0

    empties = [(g, p) for g, p in zip(gold, pred, strict=True) if not g]
    abstain = (
        sum(1 for g, p in empties if not p) / len(empties) if empties else 0.0
    )

    return BenchmarkResult(
        per_theme=per_theme,
        micro_f1=micro,
        macro_f1=macro,
        micro_f1_low=micro_low,
        micro_f1_high=micro_high,
        exact_match_rate=exact,
        abstain_accuracy=abstain,
        n=len(gold),
    )


def cohens_kappa(first: Labels, second: Labels, theme: str) -> float:
    """Agreement between two labelling passes on one theme, chance-corrected.

    Used for self-consistency: relabel a slice of the gold set after a delay
    and measure whether you agree with yourself.
    """
    n = len(first)
    if n == 0:
        return 0.0
    if len(second) != n:
        raise ValueError(
            f"labelling passes differ in length: {n} vs {len(second)}"
        )
    pairs = list(zip(first, second, strict=True))
    both = sum(1 for a, b in pairs if theme in a and theme in b)
    neither = sum(1 for a, b in pairs if theme not in a and theme not in b)
    observed = (both + neither) / n

    p_first = sum(1 for a in first if theme in a) / n
    p_second = sum(1 for b in second if theme in b) / n
    expected = p_first * p_second + (1 - p_first) * (1 - p_second)

    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def confusion_pairs(
    gold: Labels, pred: Labels, themes: Sequence[str], top: int = 10
) -> list[tuple[str, str, int]]:
    """Which themes get mistaken for which, ranked.

    Drives the failure taxonomy in the writeup. A confusion between two themes
    usually means the boundary in the taxonomy is underspecified, which is a
    fixable problem rather than a model limitation.
    """
    tally: dict[tuple[str, str], int] = {}
    for gold_row, pred_row in zip(gold, pred, strict=True):
        missed = set(gold_row) - set(pred_row)
        spurious = set(pred_row) - set(gold_row)
        for m in missed:
            for s in spurious:
                tally[(m, s)] = tally.get((m, s), 0) + 1
    ranked = sorted(tally.items(), key=lambda kv: -kv[1])[:top]
    return [(a, b, count) for (a, b), count in ranked]
