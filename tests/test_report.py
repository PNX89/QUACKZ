"""Tests for quackz.report.

A report is a document, so what is tested is what a reader would object to: numbers that
appear without the rule that graded them, a headline that does not match the verdicts
below it, a JSON file that only Python can read, and the tone rule, which is that the
library's name is a pun and its output is a risk memo.
"""

from __future__ import annotations

import json
import math
import re

import numpy as np
import pandas as pd
import pytest

from conftest import PPY, RESAMPLES
from quackz.checks import Verdict
from quackz.evaluate import DSR_BREAK_EVEN_CAP, DSR_TRIAL_GRID, evaluate
from quackz.report import (
    CHECK_TITLES,
    LIMITS,
    _json_safe,
    check_lines,
    headline,
    json_report,
    markdown_report,
    text_report,
)

RENDERERS = (text_report, markdown_report, json_report)


def flat(text: str) -> str:
    """One long line, so an assertion does not have to know where the wrapping fell."""
    return " ".join(text.split())


# The pun is allowed in the repository name, the tagline and the first line of the README.
# It is allowed nowhere in the output, and neither is the package name, which carries it.
FORBIDDEN = re.compile(
    r"\b(quackz?|quacks|duck|ducks|ducked|mallard|waddle|waddles|feather|feathers|pond|"
    r"beak|plumage|webbed|fowl|honk|bird|birds)\b",
    re.IGNORECASE,
)


@pytest.fixture(params=["honest_eval", "fallback_eval", "searched_eval", "reconciled_eval"])
def any_eval(request):
    return request.getfixturevalue(request.param)


# --------------------------------------------------------------------------------------
# Tone
# --------------------------------------------------------------------------------------


def test_no_output_carries_the_pun(any_eval):
    for render in RENDERERS:
        found = FORBIDDEN.search(render(any_eval))
        assert found is None, f"{render.__name__} contains {found.group(0)!r}"


def test_no_check_wording_carries_the_pun(any_eval):
    for line in check_lines(any_eval):
        assert FORBIDDEN.search(line.finding) is None
        assert FORBIDDEN.search(line.title) is None
        for detail in line.details:
            assert FORBIDDEN.search(detail) is None


def test_there_is_no_letter_grade_or_score(any_eval):
    for render in RENDERERS:
        text = render(any_eval).lower()
        assert "grade" not in text
        assert "score" not in text
        assert "out of 10" not in text


def test_no_dashes_are_used_as_punctuation(any_eval):
    """Written by codepoint so the file itself stays free of the characters it forbids."""
    em_dash, en_dash = chr(0x2014), chr(0x2013)
    for render in RENDERERS:
        text = render(any_eval)
        assert em_dash not in text
        assert en_dash not in text


# --------------------------------------------------------------------------------------
# The headline
# --------------------------------------------------------------------------------------


def test_headline_counts_match_the_verdicts(any_eval):
    counts = {level: 0 for level in Verdict}
    for verdict in any_eval.checks.verdicts().values():
        counts[verdict] += 1
    assert headline(any_eval) == (
        f"{counts[Verdict.FAIL]} FAIL, {counts[Verdict.WARN]} WARN, {counts[Verdict.PASS]} PASS"
    )


def test_headline_totals_every_check_that_ran(honest_eval, reconciled_eval):
    assert sum(int(part.split()[0]) for part in headline(honest_eval).split(", ")) == 7
    assert sum(int(part.split()[0]) for part in headline(reconciled_eval).split(", ")) == 8


def test_the_text_report_leads_with_the_verdict_and_the_headline(searched_eval):
    first = text_report(searched_eval).splitlines()[2]
    assert first == f"Verdict: {searched_eval.verdict.value}. {headline(searched_eval)}."


# --------------------------------------------------------------------------------------
# What must be in the report
# --------------------------------------------------------------------------------------


def test_the_report_states_the_annualization_it_used_and_where_it_came_from(
    honest_eval, honest_strategy
):
    assert "252.00 periods per year, supplied" in text_report(honest_eval)
    inferred = evaluate(*honest_strategy, bootstrap_resamples=RESAMPLES)
    assert "periods per year, inferred from the index" in text_report(inferred)


def test_the_report_states_the_seed_and_the_resample_count(honest_eval):
    assert f"{RESAMPLES} resamples, seed {honest_eval.meta.seed}" in text_report(honest_eval)


def test_every_check_appears_with_its_verdict(any_eval):
    text = text_report(any_eval)
    markdown = markdown_report(any_eval)
    for key, verdict in any_eval.checks.verdicts().items():
        title = CHECK_TITLES[key]
        assert f"[{verdict.value}] {title}" in text
        assert f"### {title}: {verdict.value}" in markdown


def test_a_check_that_did_not_run_is_not_reported(honest_eval):
    assert CHECK_TITLES["reconcile"] not in text_report(honest_eval)


def test_the_broker_comparison_points_at_the_gross_figure(honest_eval):
    """Two bps-per-turnover figures one cost apart, so the sentence must say which is which."""
    line = next(line for line in check_lines(honest_eval) if line.key == "concentration")
    detail = flat(" ".join(line.details))
    assert "measured on the net stream" in detail
    before_the_quote = detail.split("broker's quote")[0]
    assert before_the_quote.rindex("break-even cost above") > before_the_quote.rindex(
        "Edge per unit of turnover"
    )


def test_every_finding_carries_the_rule_that_graded_it(any_eval):
    # The exemption this once carried for noise_floor was unconditional, so it covered the
    # branch that could quote a threshold as well as the one that could not, and the zero
    # floor branch printed a FAIL naming no rule at all for as long as it stood.
    for line in check_lines(any_eval):
        assert "FAIL" in line.finding


def flat_position_market(*, drift: float, n: int = 400, seed: int = 11):
    """A constant long position on a drifting market, for the two zero-floor branches."""
    rng = np.random.default_rng(seed)
    bar_returns = 0.01 * rng.standard_normal(n) + drift
    index = pd.date_range("2019-01-02", periods=n, freq="B")
    prices = pd.Series(100.0 * np.cumprod(1.0 + bar_returns), index=index, name="close")
    return prices, pd.Series(np.ones(n), index=index, name="position")


def noise_floor_finding(evaluation) -> str:
    return next(line for line in check_lines(evaluation) if line.key == "noise_floor").finding


def test_a_zero_floor_that_fails_still_names_the_rule_that_failed_it():
    """The default n_trials of 1 puts the floor at zero, so a loser FAILs on the sign alone.

    That verdict used to print as a neutral remark with no cut-off in it, which is the one
    thing this module's docstring says a finding may never do.
    """
    prices, positions = flat_position_market(drift=-0.0005)
    evaluation = evaluate(prices, positions, periods_per_year=PPY, bootstrap_resamples=RESAMPLES)
    assert evaluation.checks.noise_floor.verdict is Verdict.FAIL
    finding = noise_floor_finding(evaluation)
    assert "one declared trial is no selection at all" in finding
    assert "FAIL at or below a Sharpe of zero" in finding


def test_a_zero_floor_from_undispersed_trials_names_the_count_it_actually_saw():
    """Trials that all scored alike put the floor at zero without there being only one.

    The sentence claimed a single trial in both cases, so this report said "one declared
    trial" while its own meta block said two hundred.
    """
    prices, positions = flat_position_market(drift=0.0004)
    evaluation = evaluate(
        prices,
        positions,
        periods_per_year=PPY,
        n_trials=200,
        var_trial_sharpes=0.0,
        bootstrap_resamples=RESAMPLES,
    )
    assert evaluation.checks.noise_floor.annualized == 0.0
    finding = noise_floor_finding(evaluation)
    assert "200 declared trials" in finding
    assert "one declared trial" not in finding
    assert "FAIL at or below a Sharpe of zero" in finding


def test_an_overridden_threshold_shows_up_in_the_wording(honest_strategy):
    evaluation = evaluate(
        *honest_strategy,
        periods_per_year=PPY,
        bootstrap_resamples=RESAMPLES,
        thresholds={"cost_break_even_bps_fail": 1.0, "cost_break_even_bps_warn": 2.0},
    )
    finding = next(line for line in check_lines(evaluation) if line.key == "cost_sweep").finding
    assert "FAIL below 1.0 bps, WARN below 2.0 bps" in finding


def test_the_trial_table_is_printed_with_the_published_grid(fallback_eval):
    text = text_report(fallback_eval)
    assert "DEFLATED SHARPE AGAINST THE NUMBER OF TRIALS" in text
    table = text.split("DEFLATED SHARPE AGAINST THE NUMBER OF TRIALS")[1]
    for count in DSR_TRIAL_GRID:
        assert f"{count:,}" in table
    for row in fallback_eval.checks.deflated_sharpe.by_trials:
        assert f"{row.dsr:,.4f}" in table


def test_the_break_even_trial_count_is_stated_in_words(honest_eval, honest_strategy, capped_eval):
    crosses = honest_eval.checks.deflated_sharpe.break_even_n_trials
    assert f"clears 0.95 up to {crosses:,} trials" in flat(text_report(honest_eval))

    prices, _ = honest_strategy
    churn = pd.Series(np.where(np.arange(len(prices)) % 2 == 0, 1.0, -1.0), index=prices.index)
    losing = evaluate(
        prices, churn, costs_bps=50.0, periods_per_year=PPY, bootstrap_resamples=RESAMPLES
    )
    assert losing.checks.deflated_sharpe.break_even_n_trials is None
    assert "does not reach 0.95 even at a single trial" in flat(text_report(losing))

    # The third arm. Without a fixture that reaches the cap, this sentence printed the
    # second arm's wording instead and named a break-even count of 999,999 that no longer
    # marked where the record gives out.
    assert capped_eval.checks.deflated_sharpe.break_even_capped is True
    assert f"still clears 0.95 at {DSR_BREAK_EVEN_CAP:,} trials, where the search stops" in flat(
        text_report(capped_eval)
    )


def test_the_cost_table_matches_the_check_it_came_from(honest_eval):
    table = text_report(honest_eval).split("COST SENSITIVITY")[1]
    for row in honest_eval.checks.cost_sweep.rows:
        assert f"{row.bps:,.1f}" in table
        assert f"{row.net_sharpe:,.2f}" in table


def test_the_limits_are_printed_in_full(any_eval):
    text = text_report(any_eval)
    for limit in LIMITS:
        assert limit.split(".")[0] in flat(text)


def test_the_fallback_warning_reaches_the_reader(fallback_eval):
    text = flat(text_report(fallback_eval))
    assert "iid-normal" in text
    assert "placeholder, not a bound" in text


# --------------------------------------------------------------------------------------
# Shape and determinism
# --------------------------------------------------------------------------------------


def test_rendering_is_deterministic(any_eval):
    for render in RENDERERS:
        assert render(any_eval) == render(any_eval)


def test_the_same_inputs_render_the_same_report(honest_strategy):
    first = evaluate(*honest_strategy, periods_per_year=PPY, bootstrap_resamples=RESAMPLES)
    second = evaluate(*honest_strategy, periods_per_year=PPY, bootstrap_resamples=RESAMPLES)
    assert text_report(first) == text_report(second)


def test_text_lines_stay_within_a_terminal(any_eval):
    for line in text_report(any_eval).splitlines():
        assert len(line) <= 90, line


def test_markdown_tables_are_well_formed(any_eval):
    markdown = markdown_report(any_eval)
    assert markdown.startswith("# Backtest audit")
    rows = [line for line in markdown.splitlines() if line.startswith("|")]
    assert len(rows) > 20
    for row in rows:
        assert row.endswith("|")
    for title in ("Deflated Sharpe", "Noise floor", "Cost sensitivity"):
        assert f"| {title} |" in markdown


# --------------------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------------------


def test_json_is_strict_and_complete(searched_eval):
    payload = json.loads(json_report(searched_eval))
    assert payload["verdict"] == searched_eval.verdict.value
    assert payload["headline"] == headline(searched_eval)
    assert payload["counts"]["FAIL"] + payload["counts"]["WARN"] + payload["counts"]["PASS"] == 7
    assert payload["check_verdicts"]["deflated_sharpe"] == (
        searched_eval.checks.deflated_sharpe.verdict.value
    )
    assert payload["metrics"]["sharpe"] == pytest.approx(searched_eval.metrics.sharpe)
    assert payload["meta"]["thresholds"]["dsr_warn"] == 0.95
    assert len(payload["checks"]["deflated_sharpe"]["by_trials"]) == len(
        searched_eval.checks.deflated_sharpe.by_trials
    )
    assert payload["limits"] == list(LIMITS)


def test_json_carries_no_token_only_python_can_read(honest_eval):
    # The noise floor ratio is infinite at one declared trial, which JSON cannot express.
    assert math.isinf(honest_eval.checks.noise_floor.ratio)
    text = json_report(honest_eval)
    assert "Infinity" not in text
    assert "NaN" not in text
    assert json.loads(text)["checks"]["noise_floor"]["ratio"] is None
    json.loads(text)  # strict by default: parses only because the value became null


def test_json_safe_flattens_the_types_json_does_not_have():
    assert _json_safe(math.inf) is None
    assert _json_safe(-math.inf) is None
    assert _json_safe(math.nan) is None
    assert _json_safe(np.float64(1.5)) == 1.5
    assert isinstance(_json_safe(np.int64(3)), int)
    assert _json_safe(np.True_) is True
    assert _json_safe(Verdict.WARN) == "WARN"
    assert _json_safe({"a": (1.0, np.float32(2.0))}) == {"a": [1.0, 2.0]}


def test_unbounded_values_are_spelled_out_in_the_text(honest_eval):
    text = flat(text_report(honest_eval))
    assert "a floor of zero: one declared trial is no selection at all" in text
    assert "inf" not in text.lower().replace("information", "").replace("inferred", "")
