"""Tests for quackz.checks.

Two ideas run through this file. First, a check that fires on everything is worthless, so
every check that can fire has a fixture that must NOT fire alongside the one that must.
Second, the quantities the report quotes are cross-checked against each other: the
break-even cost against the edge per unit turnover, the bootstrap standard error against
the variance term inside the probabilistic Sharpe, the vectorised drawdown against the
scalar one.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from quackz import checks, metrics
from quackz import returns as rt
from quackz.checks import Verdict
from quackz.returns import QuackzInputError

PPY = 252.0


def index_for(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2018-01-02", periods=n, freq="B")


def prices_from_returns(bar_returns: np.ndarray) -> pd.Series:
    return pd.Series(
        100.0 * np.cumprod(1.0 + bar_returns), index=index_for(len(bar_returns)), name="close"
    )


def drifting_market(*, n: int = 1000, drift: float = 0.0005, vol: float = 0.01, seed: int = 5):
    """A price series with a small positive drift, and a flat long position on it."""
    rng = np.random.default_rng(seed)
    bar_returns = drift + vol * rng.standard_normal(n)
    prices = prices_from_returns(bar_returns)
    positions = pd.Series(np.ones(n), index=prices.index, name="signal")
    return prices, positions


def churning_strategy(*, n: int = 1000, seed: int = 5):
    """The same market traded with a position that flips every single bar."""
    prices, _ = drifting_market(n=n, seed=seed)
    flipping = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    return prices, pd.Series(flipping, index=prices.index, name="signal")


def honest_fast_decay(*, n: int = 1200, edge: float = 0.20, vol: float = 0.01, seed: int = 7):
    """A signal that genuinely predicts the next bar and nothing beyond it.

    `r[t] = vol * (edge * signal[t-1] + noise[t])`, traded as `position[t] = sign(signal[t])`.
    The whole edge lives in one bar, so a delay of one bar destroys all of it. That is what
    real short-horizon alpha looks like, and the level it reaches is unremarkable.
    """
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal(n)
    noise = rng.standard_normal(n)
    bar_returns = np.empty(n)
    bar_returns[0] = 0.0
    bar_returns[1:] = vol * (edge * signal[:-1] + noise[1:])
    prices = prices_from_returns(bar_returns)
    return prices, pd.Series(np.sign(signal), index=prices.index, name="signal")


def implausible_level(*, n: int = 1200, vol: float = 0.01, seed: int = 7):
    """A position set from the return it is about to earn.

    Its decay profile is the same as `honest_fast_decay`, gone after one bar. Only the
    LEVEL separates the two, which is exactly why the verdict fires on the level.
    """
    rng = np.random.default_rng(seed)
    bar_returns = vol * rng.standard_normal(n)
    prices = prices_from_returns(bar_returns)
    peeking = np.zeros(n)
    peeking[:-1] = np.sign(bar_returns[1:])
    return prices, pd.Series(peeking, index=prices.index, name="signal")


def slow_rebalance_leak(*, n: int = 1200, vol: float = 0.01, rebalance: int = 5, seed: int = 11):
    """A position rebalanced every `rebalance` bars from the return it is about to earn.

    The off-by-one a rebalance loop actually makes: the new position is chosen with the
    close that follows the decision rather than the one at it. Between rebalances the
    position is only held, so it turns over slowly and its autocorrelation is high, while
    the whole edge sits in the single bar after each decision.

    The level it reaches is ordinary, which is the point. This is the leak the Sharpe rule
    was never going to see.
    """
    rng = np.random.default_rng(seed)
    bar_returns = vol * rng.standard_normal(n)
    prices = prices_from_returns(bar_returns)
    positions = np.zeros(n)
    for start in range(0, n - 1, rebalance):
        positions[start : start + rebalance] = np.sign(bar_returns[start + 1])
    return prices, pd.Series(positions, index=prices.index, name="signal")


# --------------------------------------------------------------------------------------
# reconcile
# --------------------------------------------------------------------------------------


def test_reconcile_of_a_series_against_itself_is_perfect():
    prices, positions = drifting_market()
    gross = rt.gross_returns(prices, positions)
    result = checks.reconcile(claimed=gross, recomputed=gross)
    assert result.correlation == pytest.approx(1.0)
    assert result.tracking_error == 0.0
    assert result.cumulative_gap == pytest.approx(0.0, abs=1e-15)
    assert result.gap_direction == "identical"
    assert result.candidate_causes == ()
    assert result.verdict is Verdict.PASS


def test_reconcile_reports_the_direction_of_the_gap():
    prices, positions = drifting_market()
    gross = rt.gross_returns(prices, positions)
    net = rt.net_returns(prices, positions, costs_bps=100.0)
    assert checks.reconcile(claimed=gross, recomputed=net).gap_direction == "claimed_higher"
    assert checks.reconcile(claimed=net, recomputed=gross).gap_direction == "claimed_lower"


def test_reconcile_flags_a_material_gap():
    prices, positions = drifting_market()
    gross = rt.gross_returns(prices, positions)
    result = checks.reconcile(claimed=gross + 0.001, recomputed=gross)
    assert result.cumulative_gap > checks.RECONCILE_CUMULATIVE_GAP_FAIL
    assert result.verdict is Verdict.FAIL


def test_reconcile_does_not_conclude_lookahead():
    """Candidate causes are listed, never ranked into an accusation.

    The conventional explanations come first and the one about information timing is
    hedged in its own text, because this check cannot tell the two apart.
    """
    prices, positions = drifting_market()
    gross = rt.gross_returns(prices, positions)
    causes = checks.reconcile(claimed=gross + 0.001, recomputed=gross).candidate_causes
    assert len(causes) >= 4
    assert "cost" in causes[0]
    assert "cannot separate" in causes[-1]
    assert not any("proves" in cause or "must be" in cause for cause in causes)


def test_reconcile_refuses_to_join_two_different_indexes():
    prices, positions = drifting_market(n=50)
    gross = rt.gross_returns(prices, positions)
    with pytest.raises(QuackzInputError, match="will not inner-join"):
        checks.reconcile(claimed=gross, recomputed=gross.iloc[:-1])


def test_reconcile_rejects_a_length_mismatch():
    with pytest.raises(QuackzInputError, match="same bars"):
        checks.reconcile(claimed=np.array([0.1, 0.2, 0.3]), recomputed=np.array([0.1, 0.2]))


def test_reconcile_rejects_a_constant_stream():
    with pytest.raises(QuackzInputError, match="undefined"):
        checks.reconcile(claimed=np.zeros(10), recomputed=np.linspace(0.0, 0.01, 10))


# --------------------------------------------------------------------------------------
# latency_sensitivity
# --------------------------------------------------------------------------------------


def test_honest_fast_decay_does_not_fire():
    """A one-bar signal loses everything to a one-bar delay, and that is not a finding."""
    prices, positions = honest_fast_decay()
    result = checks.latency_sensitivity(prices, positions, max_lag=3, periods_per_year=PPY)
    assert result.sharpe_by_lag[0] == pytest.approx(2.31, abs=0.05)
    assert abs(result.sharpe_by_lag[1]) < 0.5
    assert result.verdict is Verdict.PASS


def test_an_implausible_level_fires():
    prices, positions = implausible_level()
    result = checks.latency_sensitivity(prices, positions, max_lag=3, periods_per_year=PPY)
    assert result.sharpe_by_lag[0] > checks.LATENCY_ANNUAL_SHARPE_FAIL
    assert result.verdict is Verdict.FAIL


def test_the_two_fixtures_decay_alike_and_differ_only_in_level():
    """The point of the level rule, pinned: raw decay cannot separate these two, level can."""
    honest = checks.latency_sensitivity(*honest_fast_decay(), max_lag=3, periods_per_year=PPY)
    peeking = checks.latency_sensitivity(*implausible_level(), max_lag=3, periods_per_year=PPY)
    for result in (honest, peeking):
        assert abs(result.sharpe_by_lag[1]) < 0.5 * abs(result.sharpe_by_lag[0])
        # Both turn over every other bar, so neither is a case the decay rule speaks about.
        assert result.mean_holding_period < checks.LATENCY_DECAY_MIN_HOLDING_BARS
        assert result.retention_ratio is None
    assert peeking.sharpe_by_lag[0] > 8.0 * honest.sharpe_by_lag[0]
    assert honest.verdict is Verdict.PASS
    assert peeking.verdict is Verdict.FAIL


def test_a_slow_position_that_dies_one_bar_late_fires_where_the_level_cannot():
    """The leak the level rule was never going to catch: an ordinary Sharpe, an instant death.

    The position is rebalanced every five bars from the return it is about to earn, so it
    is held for ten bars on average and its edge is gone after one. A Sharpe of 2.8 is not
    a number anybody blinks at.
    """
    prices, positions = slow_rebalance_leak()
    result = checks.latency_sensitivity(prices, positions, max_lag=3, periods_per_year=PPY)

    assert result.sharpe_by_lag[0] < checks.LATENCY_ANNUAL_SHARPE_WARN
    assert result.mean_holding_period > checks.LATENCY_DECAY_MIN_HOLDING_BARS
    assert result.position_autocorr > 0.5
    assert result.retention_ratio < checks.LATENCY_RETENTION_RATIO_FAIL
    assert result.verdict is Verdict.FAIL


def test_a_slow_honest_strategy_keeps_what_its_holding_period_implies(honest_strategy):
    """The other side of the same rule: a persistent signal survives the delay it should."""
    result = checks.latency_sensitivity(*honest_strategy, max_lag=3, periods_per_year=PPY)
    assert result.mean_holding_period > checks.LATENCY_DECAY_MIN_HOLDING_BARS
    assert result.retention_ratio > checks.LATENCY_RETENTION_RATIO_WARN
    assert result.verdict is Verdict.PASS


def test_the_retention_figures_are_the_arithmetic_they_claim_to_be():
    prices, positions = slow_rebalance_leak(rebalance=10)
    result = checks.latency_sensitivity(prices, positions, periods_per_year=PPY)
    assert result.expected_retention_1bar == pytest.approx(1.0 - 1.0 / result.mean_holding_period)
    assert result.retention_1bar == pytest.approx(result.sharpe_by_lag[1] / result.sharpe_by_lag[0])
    assert result.retention_ratio == pytest.approx(
        result.retention_1bar / result.expected_retention_1bar
    )


def test_the_decay_rule_stays_quiet_when_there_is_no_edge_to_lose():
    """A non-positive Sharpe at lag 0 has no retention to measure, so the rule declines."""
    prices, positions = drifting_market(n=300, drift=-0.001)
    result = checks.latency_sensitivity(prices, positions, periods_per_year=PPY)
    assert result.sharpe_by_lag[0] < 0.0
    assert result.retention_1bar is None
    assert result.retention_ratio is None
    assert result.verdict is Verdict.PASS


def test_every_lag_is_measured_on_the_same_bars():
    prices, positions = honest_fast_decay(n=300)
    result = checks.latency_sensitivity(prices, positions, max_lag=4, periods_per_year=PPY)
    assert result.n_obs == len(prices) - 1 - 4
    assert result.lags == (0, 1, 2, 3, 4)
    assert len(result.sharpe_by_lag) == 5


def test_a_wider_lag_range_shortens_the_common_window():
    """If the lag-0 Sharpe did not move, the lags were not sharing a window."""
    prices, positions = honest_fast_decay(n=300)
    narrow = checks.latency_sensitivity(prices, positions, max_lag=1, periods_per_year=PPY)
    wide = checks.latency_sensitivity(prices, positions, max_lag=6, periods_per_year=PPY)
    assert wide.n_obs == narrow.n_obs - 5
    assert narrow.sharpe_by_lag[0] != wide.sharpe_by_lag[0]


def test_mean_holding_period_of_a_position_held_throughout():
    prices, positions = drifting_market(n=200)
    result = checks.latency_sensitivity(prices, positions, periods_per_year=PPY)
    # One entry and one exit over 199 return bars, so the position is held for all of them.
    assert result.mean_holding_period == pytest.approx(199.0)
    assert result.position_autocorr == 1.0


def test_mean_holding_period_of_a_position_that_flips_every_bar():
    prices, positions = churning_strategy(n=200)
    result = checks.latency_sensitivity(prices, positions, periods_per_year=PPY)
    assert result.mean_holding_period == pytest.approx(1.0, rel=0.01)
    assert result.position_autocorr < -0.9


def test_latency_sensitivity_rejects_a_lag_longer_than_the_sample():
    prices, positions = drifting_market(n=6)
    with pytest.raises(QuackzInputError, match="common to every lag"):
        checks.latency_sensitivity(prices, positions, max_lag=5, periods_per_year=PPY)


def test_latency_sensitivity_rejects_a_zero_lag_range():
    prices, positions = drifting_market(n=50)
    with pytest.raises(QuackzInputError, match="max_lag"):
        checks.latency_sensitivity(prices, positions, max_lag=0, periods_per_year=PPY)


# --------------------------------------------------------------------------------------
# cost_sweep
# --------------------------------------------------------------------------------------


def test_break_even_is_closed_form_and_exact():
    """Charging exactly the break-even cost leaves a mean net return of zero.

    Nothing is interpolated: the grid here brackets the crossing so coarsely that any
    interpolated answer would be wrong by orders of magnitude.
    """
    prices, positions = churning_strategy()
    coarse = checks.cost_sweep(prices, positions, bps_grid=[0.0, 500.0], periods_per_year=PPY)
    at_break_even = checks.cost_sweep(
        prices, positions, bps_grid=[coarse.break_even_bps], periods_per_year=PPY
    )
    # Machine precision against the size of the terms being cancelled, not against the
    # near-zero mean they cancel to.
    bar_scale = float(np.abs(rt.gross_returns(prices, positions)).mean())
    assert abs(at_break_even.rows[0].mean_net_return) < 1e-15 * bar_scale


def test_break_even_matches_its_definition_to_machine_precision():
    prices, positions = churning_strategy()
    result = checks.cost_sweep(prices, positions, bps_grid=[0.0], periods_per_year=PPY)
    expected = 1e4 * result.mean_gross_return / result.mean_turnover
    assert result.break_even_bps == expected


def test_mean_net_return_is_monotone_non_increasing_in_cost():
    """The Sharpe is not guaranteed monotone, so only the mean net return is asserted."""
    prices, positions = churning_strategy()
    result = checks.cost_sweep(
        prices, positions, bps_grid=[0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0], periods_per_year=PPY
    )
    means = [row.mean_net_return for row in result.rows]
    assert all(later <= earlier for earlier, later in pairwise(means))


def test_the_grid_is_reported_in_ascending_order():
    prices, positions = drifting_market()
    result = checks.cost_sweep(prices, positions, bps_grid=[10.0, 0.0, 5.0], periods_per_year=PPY)
    assert [row.bps for row in result.rows] == [0.0, 5.0, 10.0]


def test_a_buy_and_hold_edge_survives_realistic_costs():
    prices, positions = drifting_market()
    result = checks.cost_sweep(prices, positions, bps_grid=[0.0, 5.0], periods_per_year=PPY)
    # Two trades over a thousand bars, so the cost per bar is negligible.
    assert result.break_even_bps > 1000.0
    assert result.verdict is Verdict.PASS


def test_a_bar_by_bar_flip_does_not_survive_realistic_costs():
    prices, positions = churning_strategy()
    result = checks.cost_sweep(prices, positions, bps_grid=[0.0, 5.0], periods_per_year=PPY)
    assert result.break_even_bps < checks.COST_BREAK_EVEN_BPS_FAIL
    assert result.verdict is Verdict.FAIL


def test_zero_cost_row_matches_the_gross_stream():
    prices, positions = churning_strategy(n=200)
    result = checks.cost_sweep(prices, positions, bps_grid=[0.0], periods_per_year=PPY)
    gross = rt.gross_returns(prices, positions)
    assert result.rows[0].mean_net_return == pytest.approx(float(gross.mean()), rel=1e-15)
    assert result.rows[0].net_sharpe == pytest.approx(
        metrics.sharpe(gross, periods_per_year=PPY), rel=1e-15
    )


@pytest.mark.parametrize("grid", [[], [-1.0], [float("nan")], [1.0, float("inf")]])
def test_cost_sweep_rejects_a_bad_grid(grid):
    prices, positions = drifting_market(n=50)
    with pytest.raises(QuackzInputError, match="bps_grid"):
        checks.cost_sweep(prices, positions, bps_grid=grid, periods_per_year=PPY)


def test_a_strategy_that_never_trades_has_no_break_even():
    prices, _ = drifting_market(n=100)
    flat = pd.Series(np.zeros(100), index=prices.index)
    result = checks.cost_sweep(prices, flat, bps_grid=[0.0, 10.0], periods_per_year=PPY)
    assert result.mean_turnover == 0.0
    assert result.break_even_bps == 0.0


# --------------------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------------------


def bootstrap_returns(*, n: int = 500, drift: float = 0.0006, seed: int = 21) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return drift + 0.01 * rng.standard_normal(n)


def test_the_same_seed_reproduces_the_whole_result():
    data = bootstrap_returns()
    first = checks.bootstrap(data, periods_per_year=PPY, n_resamples=200, seed=3)
    second = checks.bootstrap(data, periods_per_year=PPY, n_resamples=200, seed=3)
    assert first == second


def test_a_different_seed_gives_a_different_draw():
    data = bootstrap_returns()
    first = checks.bootstrap(data, periods_per_year=PPY, n_resamples=200, seed=3)
    second = checks.bootstrap(data, periods_per_year=PPY, n_resamples=200, seed=4)
    assert first.blocks[0].sharpe_percentiles != second.blocks[0].sharpe_percentiles
    assert first.observed_sharpe == second.observed_sharpe


def test_the_null_imposed_p_value_is_near_one_half_when_there_is_no_edge():
    """Recentring a series that already has a zero mean leaves the null equal to the data.

    The observed Sharpe is then exactly zero, so about half the resamples must reach it.
    Anything far from one half would mean the null is not being imposed properly.
    """
    raw = bootstrap_returns(drift=0.0)
    data = raw - raw.mean()
    result = checks.bootstrap(data, periods_per_year=PPY, n_resamples=500, seed=11)
    assert result.observed_sharpe == pytest.approx(0.0, abs=1e-13)
    assert 0.35 < result.max_p_value < 0.65
    assert result.verdict is Verdict.FAIL


def test_the_null_imposed_p_value_is_tiny_when_the_edge_is_large():
    rng = np.random.default_rng(2)
    data = 0.004 + 0.001 * rng.standard_normal(400)
    result = checks.bootstrap(data, periods_per_year=PPY, n_resamples=500, seed=11)
    assert result.max_p_value == 0.0
    assert result.verdict is Verdict.PASS


def test_the_p_value_reported_is_the_least_favourable_block_length():
    data = bootstrap_returns()
    result = checks.bootstrap(data, periods_per_year=PPY, n_resamples=200, seed=3)
    assert result.max_p_value == max(block.p_value for block in result.blocks)
    assert [block.block_length for block in result.blocks] == [5, 20, 60]


def test_the_studentized_interval_brackets_the_observed_sharpe():
    data = bootstrap_returns()
    result = checks.bootstrap(data, periods_per_year=PPY, n_resamples=500, seed=3)
    for block in result.blocks:
        assert block.ci_low < result.observed_sharpe < block.ci_high


def test_reported_percentiles_are_ordered():
    data = bootstrap_returns()
    result = checks.bootstrap(data, periods_per_year=PPY, n_resamples=300, seed=3)
    assert result.percentiles == checks.BOOTSTRAP_PERCENTILES
    for block in result.blocks:
        assert list(block.sharpe_percentiles) == sorted(block.sharpe_percentiles)
        assert list(block.max_drawdown_percentiles) == sorted(block.max_drawdown_percentiles)
        assert all(value <= 0.0 for value in block.max_drawdown_percentiles)


def test_reported_sharpes_are_annualized():
    data = bootstrap_returns()
    per_period = float(data.mean() / data.std(ddof=1))
    result = checks.bootstrap(data, periods_per_year=PPY, n_resamples=200, seed=3)
    assert result.observed_sharpe == pytest.approx(per_period * math.sqrt(PPY), rel=1e-15)
    assert result.observed_max_drawdown == pytest.approx(metrics.max_drawdown(data), rel=1e-15)


def test_a_short_block_bootstrap_understates_the_drawdown_tail():
    """The limitation the README states, demonstrated rather than asserted.

    These returns contain one long losing regime. A five-bar block cannot reproduce a
    hundred-bar stretch, so its 5th percentile drawdown comes out shallower than the
    drawdown that actually happened, which would tell a reader that the realised drawdown
    was a freak event. Longer blocks recover more of the tail, which is the whole reason
    the check sweeps a grid of block lengths instead of picking one.
    """
    rng = np.random.default_rng(9)
    n = 600
    data = 0.004 + 0.01 * rng.standard_normal(n)
    data[200:340] -= 0.012
    observed = metrics.max_drawdown(data)
    result = checks.bootstrap(data, periods_per_year=PPY, n_resamples=800, seed=1)

    tails = [block.max_drawdown_percentiles[0] for block in result.blocks]
    assert tails[0] > observed
    assert all(later < earlier for earlier, later in pairwise(tails))
    assert tails[-1] < observed


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"block_lengths": [0]}, "at least 1"),
        ({"block_lengths": [10_000]}, "exceeds"),
        ({"block_lengths": []}, "at least one block length"),
        ({"n_resamples": 1}, "n_resamples"),
        ({"seed": -1}, "seed"),
    ],
)
def test_bootstrap_rejects_bad_arguments(kwargs, message):
    data = bootstrap_returns(n=100)
    call = {"periods_per_year": PPY, "n_resamples": 50, "seed": 0, **kwargs}
    with pytest.raises(QuackzInputError, match=message):
        checks.bootstrap(data, **call)


def test_bootstrap_rejects_a_constant_series():
    with pytest.raises(QuackzInputError, match="constant"):
        checks.bootstrap(np.zeros(100), periods_per_year=PPY, n_resamples=50, seed=0)


def test_a_degenerate_resample_does_not_poison_the_batch():
    """A strategy that trades rarely can resample into a row of identical bars.

    That row has no dispersion, so its Sharpe and its standard error are zero rather than
    a NaN, and the arithmetic must not warn on the way there. Test configuration turns a
    RuntimeWarning into an error, so this passing is the assertion.
    """
    rng = np.random.default_rng(0)
    paths = np.zeros((3, 50))
    paths[0] = 0.01
    paths[2] = 0.01 * rng.standard_normal(50)
    errors = checks._sharpe_hac_standard_error_rows(paths, lags=2)
    sharpes = checks._sharpe_rows(paths)
    assert errors[0] == 0.0
    assert errors[1] == 0.0
    assert errors[2] > 0.0
    assert list(sharpes[:2]) == [0.0, 0.0]


def test_stationary_bootstrap_draws_stay_inside_the_sample():
    rng = np.random.default_rng(0)
    indices = checks._stationary_bootstrap_indices(40, block_length=5, n_resamples=100, rng=rng)
    assert indices.shape == (100, 40)
    assert indices.min() >= 0
    assert indices.max() <= 39


def test_a_long_block_length_produces_long_runs_of_consecutive_bars():
    """The defining property of the stationary bootstrap: geometric blocks of the original."""
    rng = np.random.default_rng(0)
    short = checks._stationary_bootstrap_indices(200, block_length=2, n_resamples=50, rng=rng)
    wide = checks._stationary_bootstrap_indices(200, block_length=50, n_resamples=50, rng=rng)

    def consecutive_fraction(indices: np.ndarray) -> float:
        return float(np.mean(np.diff(indices, axis=1) % 200 == 1))

    assert consecutive_fraction(short) == pytest.approx(0.5, abs=0.05)
    assert consecutive_fraction(wide) == pytest.approx(0.98, abs=0.02)


def test_vectorised_drawdown_matches_the_scalar_one():
    rng = np.random.default_rng(4)
    paths = 0.01 * rng.standard_normal((25, 80))
    rows = checks._max_drawdown_rows(paths)
    expected = [metrics.max_drawdown(path) for path in paths]
    assert list(rows) == pytest.approx(expected, rel=1e-15)


def test_the_bootstrap_standard_error_agrees_with_the_probabilistic_sharpe():
    """Ledoit and Wolf's delta method and Bailey's PSR share one variance term.

    With no HAC lags, `se**2 * n` is exactly 1 - skew*SR + ((kurt - 1) / 4) * SR**2, the
    quantity inside `probabilistic_sharpe`. If these ever diverged, the bootstrap interval
    and the PSR printed beside it in the same report would be describing different
    distributions. The Sharpe here uses the population deviation, because the delta method
    is expressed in population moments.
    """
    rng = np.random.default_rng(3)
    data = 0.0004 + 0.01 * rng.standard_normal(500)
    standard_error = float(checks._sharpe_hac_standard_error_rows(data[None, :], lags=0)[0])
    sr = float(data.mean() / data.std(ddof=0))
    skew, kurt_nonexcess = metrics.moments(data)
    variance_term = 1.0 - skew * sr + ((kurt_nonexcess - 1.0) / 4.0) * sr**2
    assert standard_error**2 * data.size == pytest.approx(variance_term, rel=1e-12)


def test_the_standard_error_reduces_to_the_textbook_form_for_a_normal_sample():
    rng = np.random.default_rng(8)
    data = 0.0005 + 0.01 * rng.standard_normal(20_000)
    standard_error = float(checks._sharpe_hac_standard_error_rows(data[None, :], lags=0)[0])
    sr = float(data.mean() / data.std(ddof=0))
    assert standard_error == pytest.approx(math.sqrt((1.0 + sr**2 / 2.0) / data.size), rel=0.05)


# --------------------------------------------------------------------------------------
# noise_floor
# --------------------------------------------------------------------------------------


def test_noise_floor_golden_value():
    """Bailey and Lopez de Prado Eq. 1, not the crude sqrt(2 ln N) asymptotic.

    The asymptotic gives 0.102531 per bar here, 17.7 percent higher, and shipping it
    beside a deflated Sharpe built on Eq. 1 would put two contradictory numbers in one
    report. `test_bailey_floor_sits_well_below_the_crude_asymptotic_bound` in
    test_metrics.py pins that gap.
    """
    result = checks.noise_floor(n_trials=200, n_obs=1008, periods_per_year=252.0)
    assert result.per_period == pytest.approx(0.087106, abs=5e-7)
    assert result.annualized == pytest.approx(1.382762, abs=5e-7)
    assert result.var_trial_sharpes == pytest.approx(1.0 / 1008)
    assert result.variance_source == "iid_fallback"


def test_the_annualized_floor_is_the_per_period_floor_scaled_by_root_time():
    result = checks.noise_floor(n_trials=50, n_obs=500, periods_per_year=52.0)
    assert result.annualized == pytest.approx(result.per_period * math.sqrt(52.0), rel=1e-15)


def test_the_iid_fallback_says_that_it_is_a_fallback():
    result = checks.noise_floor(n_trials=10, n_obs=250, periods_per_year=PPY)
    assert result.note is not None
    assert "1/n_obs" in result.note


def test_a_supplied_trial_variance_is_used_and_carries_no_note():
    result = checks.noise_floor(
        n_trials=10, n_obs=250, periods_per_year=PPY, var_trial_sharpes=0.01
    )
    assert result.var_trial_sharpes == 0.01
    assert result.variance_source == "var_trial_sharpes"
    assert result.note is None
    assert result.per_period == metrics.expected_max_sharpe(n_trials=10, var_trial_sharpes=0.01)


def test_a_realistic_search_variance_raises_the_floor_above_the_iid_fallback():
    lenient = checks.noise_floor(n_trials=200, n_obs=1000, periods_per_year=PPY)
    honest = checks.noise_floor(
        n_trials=200, n_obs=1000, periods_per_year=PPY, var_trial_sharpes=4.0 / 1000
    )
    assert honest.annualized > lenient.annualized


def test_a_single_trial_has_no_floor():
    result = checks.noise_floor(n_trials=1, n_obs=1000, periods_per_year=PPY)
    assert result.per_period == 0.0
    assert result.annualized == 0.0


def test_the_floor_rises_with_the_number_of_trials():
    floors = [
        checks.noise_floor(n_trials=n, n_obs=1000, periods_per_year=PPY).annualized
        for n in (2, 10, 100, 1000)
    ]
    assert all(earlier < later for earlier, later in pairwise(floors))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_trials": 0}, "n_trials"),
        ({"n_obs": 1}, "n_obs"),
        ({"periods_per_year": 0.0}, "periods_per_year"),
        ({"var_trial_sharpes": -1.0}, "var_trial_sharpes"),
    ],
)
def test_noise_floor_rejects_bad_arguments(kwargs, message):
    call = {"n_trials": 10, "n_obs": 100, "periods_per_year": PPY, **kwargs}
    with pytest.raises(QuackzInputError, match=message):
        checks.noise_floor(**call)


# --------------------------------------------------------------------------------------
# subperiod_stability
# --------------------------------------------------------------------------------------


def test_subperiod_stability_does_not_fire_on_pure_noise():
    """Thirty noise samples, none of which should FAIL and few of which should WARN.

    This is the calibration test for the whole check. A naive dispersion threshold fails
    most of these, because five window Sharpes of a strategy with no edge scatter by
    about 1.1 annualized purely from sampling noise.
    """
    verdicts = []
    ratios = []
    for seed in range(30):
        prices, positions = drifting_market(n=1000, drift=0.0, seed=1000 + seed)
        result = checks.subperiod_stability(prices, positions, n_splits=5, periods_per_year=PPY)
        verdicts.append(result.verdict)
        ratios.append(result.dispersion_ratio)
    assert Verdict.FAIL not in verdicts
    assert verdicts.count(Verdict.PASS) >= 27
    assert float(np.mean(ratios)) == pytest.approx(1.0, abs=0.2)


def test_subperiod_stability_fires_when_one_window_holds_the_whole_result():
    rng = np.random.default_rng(17)
    n = 1000
    bar_returns = 0.01 * rng.standard_normal(n)
    bar_returns[600:800] += 0.006
    prices = prices_from_returns(bar_returns)
    positions = pd.Series(np.ones(n), index=prices.index)
    result = checks.subperiod_stability(prices, positions, n_splits=5, periods_per_year=PPY)
    assert result.dispersion_ratio > checks.STABILITY_DISPERSION_RATIO_FAIL
    assert result.verdict is Verdict.FAIL


def test_the_expected_standard_error_follows_the_lo_formula():
    prices, positions = drifting_market(n=1000)
    result = checks.subperiod_stability(prices, positions, n_splits=5, periods_per_year=PPY)
    sr_per_period = result.full_sample_sharpe / math.sqrt(PPY)
    # array_split leaves the windows differing by at most one bar, so the predicted
    # standard error is averaged over the window lengths actually used.
    expected = math.sqrt(
        float(np.mean([(1.0 + sr_per_period**2 / 2.0) / w for w in result.window_n_obs])) * PPY
    )
    assert result.expected_sharpe_se == pytest.approx(expected, rel=1e-12)
    assert result.dispersion_ratio == pytest.approx(
        result.sharpe_dispersion / result.expected_sharpe_se, rel=1e-15
    )


def test_windows_are_contiguous_and_cover_the_whole_sample():
    prices, positions = drifting_market(n=1003)
    result = checks.subperiod_stability(prices, positions, n_splits=4, periods_per_year=PPY)
    assert sum(result.window_n_obs) == len(prices) - 1
    assert max(result.window_n_obs) - min(result.window_n_obs) <= 1
    assert result.worst_window_sharpe == min(result.window_sharpes)
    assert len(result.window_sharpes) == 4


def test_subperiod_stability_charges_costs_when_asked_to():
    prices, positions = churning_strategy()
    free = checks.subperiod_stability(prices, positions, n_splits=4, periods_per_year=PPY)
    charged = checks.subperiod_stability(
        prices, positions, n_splits=4, periods_per_year=PPY, costs_bps=10.0
    )
    assert charged.full_sample_sharpe < free.full_sample_sharpe


def test_subperiod_stability_needs_enough_bars_per_window():
    prices, positions = drifting_market(n=9)
    with pytest.raises(QuackzInputError, match="return bars"):
        checks.subperiod_stability(prices, positions, n_splits=5, periods_per_year=PPY)


def test_subperiod_stability_does_not_borrow_validation_vocabulary():
    """The check measures one fixed signal over time; it is not a validation procedure.

    Naming it after one would claim a guarantee it cannot give, so the words that carry
    that claim are kept out of the API surface entirely.
    """
    documentation = checks.subperiod_stability.__doc__ or ""
    surface = documentation + " ".join(checks.SubperiodStabilityResult.__annotations__)
    assert "OOS" not in surface
    assert "walk-forward" not in surface.lower()
    assert "walk forward" not in surface.lower()
    assert "out of sample" not in surface.lower()
    assert "Nothing is refitted" in documentation


# --------------------------------------------------------------------------------------
# concentration
# --------------------------------------------------------------------------------------


def concentrated_returns(*, n: int = 500, seed: int = 13) -> np.ndarray:
    """Small losses most days and five days that carry the whole result."""
    rng = np.random.default_rng(seed)
    data = -0.0002 + 0.002 * rng.standard_normal(n)
    data[(np.array([0.08, 0.24, 0.52, 0.66, 0.94]) * n).astype(int)] += 0.08
    return data


def test_dropping_the_best_bars_lowers_the_sharpe():
    data = concentrated_returns()
    result = checks.concentration(data, drop_top=(1, 5, 10), periods_per_year=PPY)
    assert result.baseline_sharpe > result.rows[0].sharpe
    sharpes = [row.sharpe for row in result.rows]
    assert all(later <= earlier for earlier, later in pairwise(sharpes))


def test_the_profit_share_rises_with_the_number_of_bars_dropped():
    data = concentrated_returns()
    result = checks.concentration(data, drop_top=(1, 5, 10), periods_per_year=PPY)
    shares = [row.share_of_gross_profit for row in result.rows]
    assert all(earlier < later for earlier, later in pairwise(shares))
    assert all(0.0 <= share <= 1.0 for share in shares)


def test_concentration_fires_when_a_handful_of_bars_carry_the_result():
    data = concentrated_returns()
    result = checks.concentration(data, drop_top=(1, 5, 10), periods_per_year=PPY)
    assert result.rows[-1].share_of_gross_profit > checks.CONCENTRATION_PROFIT_SHARE_FAIL
    assert result.verdict is Verdict.FAIL


def test_concentration_does_not_fire_on_an_evenly_earned_result():
    rng = np.random.default_rng(31)
    data = 0.0005 + 0.005 * rng.standard_normal(1000)
    result = checks.concentration(data, drop_top=(1, 5, 10), periods_per_year=PPY)
    assert result.rows[-1].share_of_gross_profit < 0.1
    assert result.verdict is Verdict.PASS


def test_the_share_is_computed_by_hand_on_a_tiny_series():
    # Positive bars sum to 1.0; the best two are 0.5 and 0.3.
    data = np.array([0.5, -0.2, 0.3, 0.1, -0.4, 0.1])
    result = checks.concentration(data, drop_top=(1, 2), periods_per_year=1.0)
    assert result.rows[0].share_of_gross_profit == pytest.approx(0.5)
    assert result.rows[1].share_of_gross_profit == pytest.approx(0.8)


def test_edge_per_turnover_is_the_break_even_cost_in_disguise():
    """The most quotable line in the report is also the cost sweep's crossing point."""
    prices, positions = churning_strategy()
    streams = rt.build_returns(prices, positions, periods_per_year=PPY)
    sweep = checks.cost_sweep(prices, positions, bps_grid=[0.0], periods_per_year=PPY)
    result = checks.concentration(
        streams.gross, periods_per_year=PPY, turnover_series=streams.turnover
    )
    assert result.edge_per_turnover_bps == pytest.approx(sweep.break_even_bps, rel=1e-15)
    assert result.total_turnover == pytest.approx(float(streams.turnover.sum()), rel=1e-15)


def test_turnover_fields_stay_empty_when_no_turnover_is_supplied():
    result = checks.concentration(concentrated_returns(), periods_per_year=PPY)
    assert result.total_turnover is None
    assert result.edge_per_turnover_bps is None


def test_concentration_rejects_a_turnover_series_of_the_wrong_length():
    data = concentrated_returns(n=100)
    with pytest.raises(QuackzInputError, match="same bars"):
        checks.concentration(data, periods_per_year=PPY, turnover_series=np.ones(99))


def test_concentration_refuses_to_drop_the_whole_sample():
    with pytest.raises(QuackzInputError, match="not enough"):
        checks.concentration(np.arange(10) / 100.0, drop_top=(9,), periods_per_year=PPY)


def test_concentration_reports_counts_in_ascending_order():
    result = checks.concentration(concentrated_returns(), drop_top=(10, 1, 5), periods_per_year=PPY)
    assert [row.drop_top for row in result.rows] == [1, 5, 10]


# --------------------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------------------


def test_verdicts_serialise_as_their_own_names():
    import json

    assert json.dumps({"verdict": Verdict.WARN}) == '{"verdict": "WARN"}'


def test_the_worst_verdict_wins():
    assert checks.worst_verdict(Verdict.PASS, Verdict.WARN, Verdict.FAIL) is Verdict.FAIL
    assert checks.worst_verdict(Verdict.PASS, Verdict.WARN) is Verdict.WARN
    assert checks.worst_verdict(Verdict.PASS) is Verdict.PASS
    assert checks.worst_verdict() is Verdict.PASS


def float_leaves(value: object) -> list[float]:
    if isinstance(value, float):
        return [value]
    if isinstance(value, tuple):
        return [leaf for item in value for leaf in float_leaves(item)]
    if hasattr(value, "__dataclass_fields__"):
        return [leaf for item in vars(value).values() for leaf in float_leaves(item)]
    return []


def test_no_check_result_leaks_a_nan():
    """A NaN in a report is a silent wrong answer; every degenerate path returns a value."""
    prices, positions = drifting_market(n=300)
    streams = rt.build_returns(prices, positions, periods_per_year=PPY)
    results = [
        checks.reconcile(claimed=streams.gross, recomputed=streams.net),
        checks.latency_sensitivity(prices, positions, periods_per_year=PPY),
        checks.cost_sweep(prices, positions, bps_grid=[0.0, 5.0], periods_per_year=PPY),
        checks.noise_floor(n_trials=20, n_obs=299, periods_per_year=PPY),
        checks.subperiod_stability(prices, positions, n_splits=3, periods_per_year=PPY),
        checks.concentration(streams.net, periods_per_year=PPY, turnover_series=streams.turnover),
        checks.bootstrap(streams.net, periods_per_year=PPY, n_resamples=100, seed=0),
    ]
    for result in results:
        leaves = float_leaves(result)
        assert leaves
        assert not any(math.isnan(leaf) for leaf in leaves), result
