"""Tests for quackz.evaluate.

The composition layer has one job beyond calling the checks: it must not let two of them
disagree. So most of what is asserted here is agreement. The deflated Sharpe and the noise
floor share one trial dispersion. The annualized numbers the caller sees and the per-period
numbers the statistics use are the same numbers, converted once. And the frequency
convention at this boundary, everything annualized, is pinned by comparing against
`quackz.metrics` called with the de-annualization done by hand.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from conftest import COSTS_BPS, PPY, RESAMPLES, build_honest
from quackz import metrics
from quackz import returns as rt
from quackz.checks import DEFAULT_THRESHOLDS, Thresholds, Verdict
from quackz.evaluate import DSR_TRIAL_GRID, MIN_OBSERVATIONS, Evaluation, evaluate
from quackz.returns import QuackzInputError, net_returns


def quick(prices, positions, **kwargs) -> Evaluation:
    kwargs.setdefault("periods_per_year", PPY)
    kwargs.setdefault("bootstrap_resamples", RESAMPLES)
    return evaluate(prices, positions, **kwargs)


# --------------------------------------------------------------------------------------
# The three fixtures, and what each of them is for
# --------------------------------------------------------------------------------------


def test_honest_strategy_passes_every_check(honest_eval):
    assert honest_eval.verdict is Verdict.PASS
    assert set(honest_eval.checks.verdicts().values()) == {Verdict.PASS}
    assert 1.0 < honest_eval.metrics.sharpe < 2.5


def test_peeking_strategy_is_caught_by_the_level_and_nothing_else(peeking_eval):
    """The whole argument for grading execution delay on the level, in one assertion.

    A position set from the return it is about to earn passes the deflated Sharpe, the
    noise floor, the bootstrap, the cost sweep, the concentration check and the stability
    check. Every one of those is a statistical statement about a return series, and this
    return series is statistically magnificent. Only the implausible LEVEL gives it away.
    """
    verdicts = peeking_eval.checks.verdicts()
    assert verdicts.pop("latency") is Verdict.FAIL
    assert set(verdicts.values()) == {Verdict.PASS}
    assert peeking_eval.metrics.sharpe > 10.0


def test_the_composed_report_validates_and_builds_its_streams_once(monkeypatch, honest_strategy):
    """Building the streams once is a claim about a mechanism, so pin the mechanism.

    Every check can stand alone, validating the inputs and building the streams it needs,
    and three of them used to do exactly that inside `evaluate`: four validation passes
    over the same two series, and four chances for two lines of one report to be describing
    different numbers.
    """
    validated: list[str] = []
    built = 0
    original_validate = rt._validate_series
    original_gross = rt._gross_from_checked

    def counting_validate(series, name):
        validated.append(name)
        return original_validate(series, name)

    def counting_gross(prices, positions):
        nonlocal built
        built += 1
        return original_gross(prices, positions)

    monkeypatch.setattr(rt, "_validate_series", counting_validate)
    monkeypatch.setattr(rt, "_gross_from_checked", counting_gross)
    quick(*honest_strategy, costs_bps=COSTS_BPS)

    assert validated == ["prices", "positions"]
    assert built == 1


def test_searched_strategy_sits_at_the_noise_floor_of_its_own_search(searched_eval):
    """The best of 200 random signals on a market with no edge in it."""
    floor = searched_eval.checks.noise_floor
    assert floor.variance_source == "trial_sharpes"
    assert floor.ratio < 1.1
    assert searched_eval.checks.deflated_sharpe.dsr < 0.6
    assert searched_eval.checks.deflated_sharpe.verdict is not Verdict.PASS
    assert searched_eval.verdict is not Verdict.PASS


def test_overall_verdict_is_the_worst_check(searched_eval, honest_eval, peeking_eval):
    for evaluation in (searched_eval, honest_eval, peeking_eval):
        severities = {Verdict.PASS: 0, Verdict.WARN: 1, Verdict.FAIL: 2}
        worst = max(evaluation.checks.verdicts().values(), key=severities.__getitem__)
        assert evaluation.verdict is worst


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------


def test_caller_series_are_not_mutated(honest_strategy):
    prices, positions = honest_strategy
    prices_before, positions_before = prices.copy(deep=True), positions.copy(deep=True)
    quick(prices, positions, costs_bps=7.5, n_trials=4, claimed_returns=prices.pct_change())
    pd.testing.assert_series_equal(prices, prices_before)
    pd.testing.assert_series_equal(positions, positions_before)


def test_nan_in_prices_is_rejected(honest_strategy):
    prices, positions = honest_strategy
    damaged = prices.copy()
    damaged.iloc[10] = np.nan
    with pytest.raises(QuackzInputError, match="NaN"):
        quick(damaged, positions)


def test_misaligned_index_is_rejected(honest_strategy):
    prices, positions = honest_strategy
    with pytest.raises(QuackzInputError, match="different indexes"):
        quick(prices, positions.iloc[1:])


def test_a_sample_too_short_for_the_narrowest_check_is_refused():
    prices, positions = build_honest(n=MIN_OBSERVATIONS)
    with pytest.raises(QuackzInputError, match=f"at least {MIN_OBSERVATIONS} return bars"):
        quick(prices, positions)


def test_the_shortest_acceptable_sample_still_evaluates():
    prices, positions = build_honest(n=MIN_OBSERVATIONS + 1)
    evaluation = quick(prices, positions)
    assert evaluation.meta.n_obs == MIN_OBSERVATIONS
    # Block lengths longer than the sample are dropped rather than raising.
    assert [block.block_length for block in evaluation.checks.bootstrap.blocks] == [5, 20]


def test_n_trials_must_be_a_positive_integer(honest_strategy):
    prices, positions = honest_strategy
    with pytest.raises(QuackzInputError, match="n_trials"):
        quick(prices, positions, n_trials=0)
    with pytest.raises(QuackzInputError, match="n_trials"):
        quick(prices, positions, n_trials=2.5)


# --------------------------------------------------------------------------------------
# Frequency conventions at the boundary
# --------------------------------------------------------------------------------------


def test_metrics_are_net_of_costs_and_the_gross_sharpe_is_reported_beside_them(honest_strategy):
    prices, positions = honest_strategy
    evaluation = quick(prices, positions, costs_bps=15.0)
    net = net_returns(prices, positions, costs_bps=15.0)
    assert evaluation.metrics.sharpe == pytest.approx(
        metrics.sharpe(net, periods_per_year=PPY), rel=1e-12
    )
    assert evaluation.metrics.sharpe_gross > evaluation.metrics.sharpe


def test_annualized_and_per_period_sharpe_are_one_number(honest_eval):
    metrics_block = honest_eval.metrics
    assert metrics_block.sharpe == pytest.approx(
        metrics_block.sharpe_per_period * math.sqrt(PPY), rel=1e-12
    )
    check = honest_eval.checks.deflated_sharpe
    assert check.observed_sharpe_per_period == pytest.approx(metrics_block.sharpe_per_period)
    assert check.expected_max_sharpe == pytest.approx(
        check.expected_max_sharpe_per_period * math.sqrt(PPY), rel=1e-12
    )


def test_trial_sharpes_are_taken_annualized_and_de_annualized_here(searched_strategy):
    """The one convention this module owns: what crosses the boundary is annualized."""
    prices, positions, annual_sharpes = searched_strategy
    evaluation = quick(
        prices, positions, n_trials=len(annual_sharpes), trial_sharpes=annual_sharpes
    )
    net = net_returns(prices, positions, costs_bps=0.0).to_numpy()
    skew, kurt = metrics.moments(net)
    expected = metrics.deflated_sharpe(
        sr=metrics.sharpe(net, periods_per_year=PPY) / math.sqrt(PPY),
        n_trials=len(annual_sharpes),
        n_obs=net.size,
        skew=skew,
        kurt_nonexcess=kurt,
        trial_sharpes=annual_sharpes / math.sqrt(PPY),
    )
    assert evaluation.checks.deflated_sharpe.dsr == pytest.approx(expected.dsr, rel=1e-12)
    assert evaluation.checks.deflated_sharpe.var_trial_sharpes == pytest.approx(
        float(np.var(annual_sharpes, ddof=1)), rel=1e-12
    )


def test_a_variance_and_the_sharpes_it_came_from_agree(searched_strategy):
    prices, positions, annual_sharpes = searched_strategy
    from_sharpes = quick(
        prices, positions, n_trials=len(annual_sharpes), trial_sharpes=annual_sharpes
    )
    from_variance = quick(
        prices,
        positions,
        n_trials=len(annual_sharpes),
        var_trial_sharpes=float(np.var(annual_sharpes, ddof=1)),
    )
    assert from_sharpes.checks.deflated_sharpe.dsr == pytest.approx(
        from_variance.checks.deflated_sharpe.dsr, rel=1e-12
    )
    assert from_sharpes.checks.noise_floor.annualized == pytest.approx(
        from_variance.checks.noise_floor.annualized, rel=1e-12
    )


def test_supplying_both_trial_inputs_is_an_error(honest_strategy):
    prices, positions = honest_strategy
    with pytest.raises(QuackzInputError, match="not both"):
        quick(prices, positions, n_trials=3, trial_sharpes=[1.0, 2.0], var_trial_sharpes=0.5)


def test_deflation_and_noise_floor_share_one_dispersion(searched_eval, fallback_eval):
    for evaluation in (searched_eval, fallback_eval):
        deflated = evaluation.checks.deflated_sharpe
        floor = evaluation.checks.noise_floor
        assert deflated.var_trial_sharpes == pytest.approx(floor.var_trial_sharpes, rel=1e-15)
        assert deflated.variance_source == floor.variance_source
        # The benchmark the DSR is measured against IS the floor the report prints.
        assert deflated.expected_max_sharpe == pytest.approx(floor.annualized, rel=1e-15)


def test_the_iid_fallback_warns_and_says_so_in_the_result(fallback_eval, honest_strategy):
    assert fallback_eval.checks.deflated_sharpe.variance_source == "iid_fallback"
    assert "iid-normal" in fallback_eval.checks.deflated_sharpe.warning
    assert "iid-normal" in fallback_eval.checks.noise_floor.note
    assert fallback_eval.checks.deflated_sharpe.var_trial_sharpes_per_period == pytest.approx(
        1.0 / fallback_eval.meta.n_obs
    )
    prices, positions = honest_strategy
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        quick(prices, positions, n_trials=5)
    assert any(issubclass(entry.category, UserWarning) for entry in caught)


def test_supplied_dispersion_does_not_warn(searched_eval):
    assert searched_eval.checks.deflated_sharpe.warning is None
    assert searched_eval.checks.noise_floor.note is None


def test_the_iid_fallback_is_a_placeholder_rather_than_a_bound(searched_strategy):
    """The fallback's error has no fixed sign, so nothing in the report may call it a bound.

    The two hundred configurations of this search are near-copies of one rule, so their
    Sharpes scatter by less than the iid-normal 1 / n_obs and the fallback deflates HARDER
    than the truth. A search over genuinely different rules scatters wider, and the same
    fallback then deflates too little. Both are real searches.
    """
    prices, positions, sharpes = searched_strategy
    declared = len(sharpes)
    guessed = quick(prices, positions, n_trials=declared).checks.deflated_sharpe
    narrow = quick(
        prices, positions, n_trials=declared, trial_sharpes=sharpes
    ).checks.deflated_sharpe
    wide = quick(
        prices, positions, n_trials=declared, trial_sharpes=np.linspace(-3.0, 3.0, declared)
    ).checks.deflated_sharpe

    assert guessed.variance_source == "iid_fallback"
    assert narrow.var_trial_sharpes < guessed.var_trial_sharpes < wide.var_trial_sharpes
    assert narrow.dsr > guessed.dsr > wide.dsr


# --------------------------------------------------------------------------------------
# The deflation table
# --------------------------------------------------------------------------------------


def test_the_trial_table_covers_the_published_grid_and_the_declared_count(fallback_eval):
    counts = [row.n_trials for row in fallback_eval.checks.deflated_sharpe.by_trials]
    assert set(DSR_TRIAL_GRID) <= set(counts)
    assert fallback_eval.meta.n_trials in counts
    assert counts == sorted(counts)


def test_deflation_falls_and_the_benchmark_rises_with_the_trial_count(fallback_eval):
    rows = fallback_eval.checks.deflated_sharpe.by_trials
    assert all(a.dsr >= b.dsr for a, b in pairwise(rows))
    assert all(a.expected_max_sharpe <= b.expected_max_sharpe for a, b in pairwise(rows))
    assert rows[0].n_trials == 1
    assert rows[0].expected_max_sharpe == 0.0


def test_at_one_trial_the_deflated_sharpe_is_the_probabilistic_sharpe(honest_eval):
    check = honest_eval.checks.deflated_sharpe
    assert check.n_trials == 1
    assert check.dsr == pytest.approx(check.psr_vs_zero, rel=1e-15)


def test_the_break_even_trial_count_brackets_the_target(honest_eval, honest_strategy):
    check = honest_eval.checks.deflated_sharpe
    assert check.break_even_n_trials is not None
    assert not check.break_even_capped
    prices, positions = honest_strategy
    at = quick(prices, positions, costs_bps=COSTS_BPS, n_trials=check.break_even_n_trials)
    beyond = quick(prices, positions, costs_bps=COSTS_BPS, n_trials=check.break_even_n_trials + 1)
    assert at.checks.deflated_sharpe.dsr >= DEFAULT_THRESHOLDS.dsr_warn
    assert beyond.checks.deflated_sharpe.dsr < DEFAULT_THRESHOLDS.dsr_warn


def test_a_record_that_never_clears_the_target_reports_no_break_even_count(honest_strategy):
    prices, _ = honest_strategy
    flat = pd.Series(np.where(np.arange(len(prices)) % 2 == 0, 1.0, -1.0), index=prices.index)
    evaluation = quick(prices, flat, costs_bps=50.0)
    assert evaluation.checks.deflated_sharpe.psr_vs_zero < DEFAULT_THRESHOLDS.dsr_warn
    assert evaluation.checks.deflated_sharpe.break_even_n_trials is None
    assert evaluation.checks.deflated_sharpe.min_track_record_length is None


# --------------------------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------------------------


def test_thresholds_override_reaches_the_checks(honest_strategy):
    prices, positions = honest_strategy
    strict = quick(prices, positions, thresholds={"latency_annual_sharpe_warn": 0.5})
    assert strict.checks.latency.verdict is Verdict.WARN
    assert strict.meta.thresholds.latency_annual_sharpe_warn == 0.5
    assert quick(prices, positions).checks.latency.verdict is Verdict.PASS


def test_thresholds_accepts_the_dataclass_as_well_as_a_mapping(honest_strategy):
    prices, positions = honest_strategy
    cuts = dataclasses.replace(DEFAULT_THRESHOLDS, cost_break_even_bps_warn=1e6)
    assert quick(prices, positions, thresholds=cuts).checks.cost_sweep.verdict is Verdict.WARN


def test_an_unknown_threshold_name_is_rejected(honest_strategy):
    prices, positions = honest_strategy
    with pytest.raises(QuackzInputError, match="unknown threshold name"):
        quick(prices, positions, thresholds={"dsr_warning": 0.9})
    with pytest.raises(QuackzInputError, match="thresholds must be"):
        quick(prices, positions, thresholds=0.95)


def test_a_swapped_threshold_ladder_is_rejected():
    with pytest.raises(QuackzInputError, match="must be lower"):
        Thresholds(dsr_warn=0.5, dsr_fail=0.9)
    with pytest.raises(QuackzInputError, match="must be higher"):
        Thresholds(bootstrap_p_value_warn=0.2, bootstrap_p_value_fail=0.1)
    with pytest.raises(QuackzInputError, match="finite"):
        Thresholds(dsr_warn=math.nan)


def test_equal_warn_and_fail_cut_offs_collapse_the_warn_band(honest_strategy):
    prices, positions = honest_strategy
    evaluation = quick(
        prices,
        positions,
        thresholds={"latency_annual_sharpe_warn": 0.5, "latency_annual_sharpe_fail": 0.5},
    )
    assert evaluation.checks.latency.verdict is Verdict.FAIL


# --------------------------------------------------------------------------------------
# Reconciliation of a claimed stream
# --------------------------------------------------------------------------------------


def test_a_claimed_stream_equal_to_the_recomputed_one_reconciles_exactly(honest_strategy):
    prices, positions = honest_strategy
    claimed = net_returns(prices, positions, costs_bps=COSTS_BPS)
    evaluation = quick(prices, positions, costs_bps=COSTS_BPS, claimed_returns=claimed)
    check = evaluation.checks.reconcile
    assert check.correlation == pytest.approx(1.0)
    assert check.cumulative_gap == pytest.approx(0.0, abs=1e-12)
    assert check.verdict is Verdict.PASS
    assert check.candidate_causes == ()


def test_a_claimed_stream_on_the_price_bars_drops_its_leading_value(honest_strategy):
    prices, positions = honest_strategy
    claimed = net_returns(prices, positions, costs_bps=COSTS_BPS)
    on_price_bars = np.concatenate(([0.0], claimed.to_numpy()))
    evaluation = quick(prices, positions, costs_bps=COSTS_BPS, claimed_returns=on_price_bars)
    assert evaluation.checks.reconcile.correlation == pytest.approx(1.0)


def test_a_claimed_stream_of_the_wrong_length_is_an_error(honest_strategy):
    prices, positions = honest_strategy
    with pytest.raises(QuackzInputError, match="claimed_returns has 5 values"):
        quick(prices, positions, claimed_returns=np.zeros(5))


def test_a_claimed_stream_missing_bars_names_them(honest_strategy):
    prices, positions = honest_strategy
    claimed = net_returns(prices, positions, costs_bps=COSTS_BPS).iloc[5:]
    with pytest.raises(QuackzInputError, match="missing 5 of the"):
        quick(prices, positions, costs_bps=COSTS_BPS, claimed_returns=claimed)


def test_reconcile_is_absent_when_nothing_was_claimed(honest_eval, reconciled_eval):
    assert honest_eval.checks.reconcile is None
    assert "reconcile" not in honest_eval.checks.verdicts()
    assert reconciled_eval.checks.reconcile is not None
    assert reconciled_eval.checks.reconcile.gap_direction == "claimed_higher"


# --------------------------------------------------------------------------------------
# The Evaluation object itself
# --------------------------------------------------------------------------------------


def test_meta_records_what_the_run_used(honest_strategy):
    prices, positions = honest_strategy
    evaluation = quick(prices, positions, costs_bps=3.5, n_trials=9, bootstrap_seed=42)
    meta = evaluation.meta
    assert (meta.periods_per_year, meta.periods_per_year_inferred) == (PPY, False)
    assert (meta.costs_bps, meta.seed, meta.n_trials) == (3.5, 42, 9)
    assert (meta.n_obs, meta.n_resamples) == (len(prices) - 1, RESAMPLES)
    assert meta.first_bar == prices.index[1].date().isoformat()
    assert meta.last_bar == prices.index[-1].date().isoformat()
    assert meta.library_version.count(".") == 2
    assert meta.thresholds == DEFAULT_THRESHOLDS
    assert evaluation.checks.bootstrap.seed == 42


def test_periods_per_year_is_inferred_when_it_is_not_given(honest_strategy):
    prices, positions = honest_strategy
    evaluation = evaluate(prices, positions, bootstrap_resamples=RESAMPLES)
    assert evaluation.meta.periods_per_year_inferred
    # Business days: about five sevenths of the calendar, and nothing like 252.
    assert 255.0 < evaluation.meta.periods_per_year < 266.0


def test_the_whole_evaluation_is_frozen(honest_eval):
    for target in (honest_eval, honest_eval.meta, honest_eval.metrics, honest_eval.checks):
        with pytest.raises(dataclasses.FrozenInstanceError):
            target.verdict = Verdict.PASS


def test_bulky_rows_stay_out_of_the_repr(honest_eval):
    text = repr(honest_eval.checks.deflated_sharpe)
    assert "by_trials" not in text
    assert "dsr=" in text


def test_evaluate_is_deterministic(honest_strategy):
    prices, positions = honest_strategy
    assert quick(prices, positions, n_trials=3) == quick(prices, positions, n_trials=3)


def test_every_leaf_value_is_a_plain_python_type(searched_eval):
    def walk(value):
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        else:
            assert isinstance(value, (bool, int, float, str, type(None))), repr(value)
            assert not isinstance(value, np.generic), repr(value)

    walk(dataclasses.asdict(searched_eval))
