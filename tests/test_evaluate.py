from src.evaluate import evaluate_predictions, exact_match, f1_score


def test_exact_match_normalization():
    assert exact_match("The Nolan!", "nolan") == 1


def test_f1_partial_overlap():
    assert 0 < f1_score("christopher nolan directed it", "christopher nolan") < 1


def test_abstention_accuracy():
    recs = [
        {"pred": "x", "gold": ["x"], "abstained": False, "answerable": True},
        {"pred": "", "gold": [""], "abstained": True, "answerable": False},
        {"pred": "wrong", "gold": [""], "abstained": False, "answerable": False},
    ]
    m = evaluate_predictions(recs)
    assert m["em"] == 1.0
    assert m["abstention_acc"] == 0.5
