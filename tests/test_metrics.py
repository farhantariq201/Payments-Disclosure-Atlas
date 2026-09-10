from pda.metrics import cohens_kappa, confusion_pairs, evaluate

THEMES = ["a", "b", "c"]


def test_precision_recall_computed_by_hand():
    gold = [["a"], ["a", "b"], [], ["b"]]
    pred = [["a"], ["a"], ["a"], ["b"]]
    result = evaluate(gold, pred, THEMES, bootstrap_rounds=0)

    a = next(s for s in result.per_theme if s.theme == "a")
    assert (a.tp, a.fp, a.fn) == (2, 1, 0)
    assert a.precision == 2 / 3
    assert a.recall == 1.0

    b = next(s for s in result.per_theme if s.theme == "b")
    assert (b.tp, b.fp, b.fn) == (1, 0, 1)


def test_theme_with_no_examples_scores_zero_not_nan():
    result = evaluate([["a"]], [["a"]], THEMES, bootstrap_rounds=0)
    c = next(s for s in result.per_theme if s.theme == "c")
    assert c.f1 == 0.0 and c.support == 0


def test_perfect_prediction():
    gold = [["a"], ["b", "c"], []]
    result = evaluate(gold, gold, THEMES, bootstrap_rounds=0)
    assert result.micro_f1 == 1.0
    assert result.exact_match_rate == 1.0
    assert result.abstain_accuracy == 1.0


def test_abstain_accuracy_penalises_over_labelling():
    """A classifier that finds a theme in every paragraph must score badly
    on the themeless rows even if its recall looks fine."""
    gold = [[], [], ["a"]]
    pred = [["a"], ["a"], ["a"]]
    result = evaluate(gold, pred, THEMES, bootstrap_rounds=0)
    assert result.abstain_accuracy == 0.0
    assert result.micro_f1 < 0.6


def test_duplicate_labels_do_not_inflate_counts():
    result = evaluate([["a", "a"]], [["a", "a", "a"]], THEMES, bootstrap_rounds=0)
    a = next(s for s in result.per_theme if s.theme == "a")
    assert (a.tp, a.fp) == (1, 0)


def test_length_mismatch_raises():
    try:
        evaluate([["a"]], [["a"], ["b"]], THEMES)
    except ValueError as error:
        assert "rows" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_bootstrap_interval_brackets_the_point_estimate():
    gold = [["a"], ["a"], [], ["b"], ["a", "b"], [], ["c"], ["a"]]
    pred = [["a"], [], [], ["b"], ["a"], ["a"], ["c"], ["a"]]
    result = evaluate(gold, pred, THEMES, bootstrap_rounds=300, seed=3)
    assert result.micro_f1_low <= result.micro_f1 <= result.micro_f1_high
    assert result.micro_f1_high > result.micro_f1_low


def test_bootstrap_is_reproducible_under_a_fixed_seed():
    gold = [["a"], [], ["b"], ["a"]]
    pred = [["a"], ["a"], ["b"], []]
    first = evaluate(gold, pred, THEMES, bootstrap_rounds=200, seed=11)
    second = evaluate(gold, pred, THEMES, bootstrap_rounds=200, seed=11)
    assert first.micro_f1_low == second.micro_f1_low


def test_kappa_corrects_for_chance_agreement():
    """Two passes that both almost never assign a theme agree ~always by
    luck. Raw agreement would read 0.9; kappa must not."""
    first = [[] for _ in range(9)] + [["a"]]
    second = [[] for _ in range(8)] + [["a"], []]
    assert cohens_kappa(first, second, "a") < 0.5


def test_kappa_perfect_agreement():
    rows = [["a"], [], ["a"], []]
    assert cohens_kappa(rows, rows, "a") == 1.0


def test_confusion_pairs_surface_taxonomy_boundary_problems():
    gold = [["a"], ["a"], ["a"]]
    pred = [["b"], ["b"], ["c"]]
    pairs = confusion_pairs(gold, pred, THEMES)
    assert pairs[0] == ("a", "b", 2)
