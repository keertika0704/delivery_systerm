"""Guards for the honesty of the narrative in notebooks/03_model_training.ipynb (no execution, text only)."""
from __future__ import annotations

import json

import pytest

from src import config

NB = config.ROOT / "notebooks" / "03_model_training.ipynb"


@pytest.fixture(scope="module")
def nb():
    if not NB.exists():
        pytest.skip("notebook not built")
    return json.loads(NB.read_text())


def _md(nb):
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def _code(nb):
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def test_top_calibration_bin_is_described_with_its_real_size(nb):
    cal = json.loads(config.MODEL_METRICS_PATH.read_text())["models"]["logistic_regression"]["calibration_test"]
    n_top = cal["bin_counts"][-1]
    text = _md(nb)
    assert f"n={n_top:,}" in text, "the small top score bin must be reported with its n"
    assert "highest-scored decile" not in text, "score bins are uneven (ties), they are not deciles"
    assert "Wilson" in text


def test_reliability_axes_do_not_claim_deciles_and_show_bin_sizes(nb):
    code = _code(nb)
    assert "(score decile)" not in code and "(decile)" not in code
    assert "n shown for bins under" in code


def test_flexible_model_claim_is_not_overstated(nb):
    text = _md(nb)
    assert "fit validation about as well as logistic regression" not in text
    assert "a sign that flexibility mostly learned" not in text


@pytest.mark.parametrize("phrase", [
    "depend on conditions that are not in the order record",
    "cannot see that short promises are risky because it never saw them",
    "what makes a delivery late",
    "promise policy and logistics conditions changed between windows",
])
def test_unhedged_causal_phrases_are_gone(nb, phrase):
    assert phrase not in _md(nb)


def test_explanations_are_marked_as_hypotheses(nb):
    text = _md(nb)
    assert "One plausible reason" in text
    assert "cannot be tested with this data" in text


# --- review round: protocol wording, importance flips, coefficient description, tie rule ---

def test_test_window_is_not_claimed_to_be_evaluated_once_or_decided_before_looking(nb):
    text = _md(nb) + "\n" + _code(nb)
    assert "evaluated once" not in text and "evaluated **once**" not in text
    assert "decided before looking at test" not in text
    assert "after every decision was frozen" not in text
    # the accurate protocol statement is present
    assert "No fitting, tuning, calibration, threshold, band or model-choice step uses the test window" in _md(nb)


def test_importance_sign_flips_separate_noise_from_clear_reversals(nb):
    text = _md(nb)
    assert "within about 2 SD of zero" in text and "only **`same_state`** is a large reversal" in text
    assert "np.sign" in _code(nb) and "2 * imp_df" in _code(nb), "the flip count must be split into noise vs clear flips in code"


def test_coefficient_text_does_not_name_a_selective_list_of_states(nb):
    text = _md(nb)
    assert "customer states PR, MG, DF and SP" not in text
    assert "AL, MA, SE, PA and CE" not in text
    assert "`same_state_1` (-1.29)" in text and "`seller_state_MA` (+1.35" in text
    assert "not** read as a ranking of places" in text


def test_top_decile_lift_states_the_tie_rule(nb):
    text = _md(nb)
    assert "tied scores at the 10% cut are split in expectation" in text


def test_lift_top10_values_match_the_tie_rule_in_the_saved_metrics():
    """The quoted lift is the expected value under random tie-breaking, not a plain argsort."""
    import numpy as np
    import pandas as pd
    from src import train_model as tm

    rng = np.random.default_rng(0)
    p = rng.integers(0, 5, 2000).astype(float)             # heavy ties straddling the 10% cut
    y = (rng.random(2000) < 0.05 + 0.03 * p).astype(float)
    got = tm.top_fraction_stats(y, p)["precision"]
    k = 200
    sims = []
    for _ in range(300):
        order = np.lexsort((rng.random(len(p)), -p))        # score descending, random order within ties
        sims.append(y[order[:k]].mean())
    assert abs(got - float(np.mean(sims))) < 0.01
