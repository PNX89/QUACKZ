"""Tests for quackz.returns.

The cost alignment is the part of this library that is easiest to get subtly wrong and
hardest to notice, so it is pinned bar by bar rather than in aggregate.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from quackz import returns as rt
from quackz.returns import QuackzInputError

# Five prices whose returns are exact enough to check in the head: +10 percent,
# +10 percent, flat, then 110 / 121 - 1.
PRICES = [100.0, 110.0, 121.0, 121.0, 110.0]
BAR_RETURNS = [0.10, 0.10, 0.0, 110.0 / 121.0 - 1.0]


def make_index(n: int, start: str = "2020-01-06") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="B")


def make_series(values: list[float], name: str) -> pd.Series:
    return pd.Series(values, index=make_index(len(values)), name=name)


def prices_series() -> pd.Series:
    return make_series(PRICES, "close")


def positions_series(values: list[float]) -> pd.Series:
    return make_series(values, "signal")


# --------------------------------------------------------------------------------------
# The one-bar shift
# --------------------------------------------------------------------------------------


def test_gross_returns_use_the_previous_bar_position():
    positions = [1.0, 1.0, 0.0, 1.0, 1.0]
    gross = rt.gross_returns(prices_series(), positions_series(positions))
    expected = [positions[t - 1] * BAR_RETURNS[t - 1] for t in range(1, len(PRICES))]
    assert list(gross) == pytest.approx(expected, rel=1e-15)


def test_gross_returns_drop_exactly_one_leading_bar():
    gross = rt.gross_returns(prices_series(), positions_series([1.0] * 5))
    assert len(gross) == len(PRICES) - 1
    assert gross.index[0] == prices_series().index[1]
    assert gross.index[-1] == prices_series().index[-1]


def test_the_final_position_never_earns_a_return():
    """positions[-1] is set at the last close, so no return exists for it to earn."""
    base = rt.gross_returns(prices_series(), positions_series([1.0, 1.0, 1.0, 1.0, 0.0]))
    changed = rt.gross_returns(prices_series(), positions_series([1.0, 1.0, 1.0, 1.0, 99.0]))
    pd.testing.assert_series_equal(base, changed)


def test_a_position_taken_one_bar_late_earns_a_different_return():
    early = rt.gross_returns(prices_series(), positions_series([1.0, 0.0, 0.0, 0.0, 0.0]))
    late = rt.gross_returns(prices_series(), positions_series([0.0, 1.0, 0.0, 0.0, 0.0]))
    assert early.iloc[0] == pytest.approx(0.10)
    assert late.iloc[0] == 0.0
    assert late.iloc[1] == pytest.approx(0.10)


# --------------------------------------------------------------------------------------
# Cost alignment
# --------------------------------------------------------------------------------------


def test_turnover_charges_the_entry_trade():
    """positions[-1] is flat, so the trade into positions[0] is charged on the first bar.

    A fillna(0) on a differenced position series drops this charge, which flatters any
    strategy that takes one position and holds it.
    """
    traded = rt.turnover(positions_series([1.0, 1.0, 1.0, 1.0, 1.0]))
    assert traded.iloc[0] == pytest.approx(1.0)


def test_entry_cost_reaches_the_net_return_of_the_first_bar():
    prices = prices_series()
    positions = positions_series([1.0, 1.0, 1.0, 1.0, 1.0])
    gross = rt.gross_returns(prices, positions)
    net = rt.net_returns(prices, positions, costs_bps=50.0)
    assert net.iloc[0] == pytest.approx(gross.iloc[0] - 0.005, rel=1e-15)
    assert net.iloc[0] < gross.iloc[0]


def test_turnover_is_charged_on_the_bar_whose_return_the_new_position_earns():
    # Position path 0, 0, 1, 1, 1: the trade happens at bar 2's close, so it is charged on
    # bar 3, which is the first bar the new position earns.
    traded = rt.turnover(positions_series([0.0, 0.0, 1.0, 1.0, 1.0]), liquidate_final=False)
    assert list(traded) == pytest.approx([0.0, 0.0, 1.0, 0.0])


def test_turnover_charges_the_final_liquidation():
    # Position in force at the end of the sample is positions[-2] = 1, and closing it
    # costs one unit, added to the last bar.
    positions = positions_series([0.0, 0.0, 1.0, 1.0, 1.0])
    with_exit = rt.turnover(positions, liquidate_final=True)
    without_exit = rt.turnover(positions, liquidate_final=False)
    assert list(with_exit) == pytest.approx([0.0, 0.0, 1.0, 1.0])
    assert with_exit.iloc[-1] - without_exit.iloc[-1] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "positions",
    [
        [1.0, 1.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0, 1.0],
        [0.5, -0.5, 1.0, -1.0, 0.25],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ],
)
def test_total_turnover_is_the_closed_round_trip_path(positions):
    """Flat to the traded path and back to flat, with positions[-1] never participating."""
    traded_path = [0.0, *positions[:-1], 0.0]
    expected = float(np.abs(np.diff(traded_path)).sum())
    assert float(rt.turnover(positions_series(positions)).sum()) == pytest.approx(
        expected, rel=1e-14
    )


def test_turnover_hand_computed():
    # Path 0 -> 1 -> 1 -> 0 -> 1, then liquidate: 1, 0, 1, (1 + 1).
    traded = rt.turnover(positions_series([1.0, 1.0, 0.0, 1.0, 1.0]))
    assert list(traded) == pytest.approx([1.0, 0.0, 1.0, 2.0])


def test_turnover_is_indexed_like_gross_returns():
    prices = prices_series()
    positions = positions_series([1.0, 0.0, 1.0, 0.0, 1.0])
    assert rt.turnover(positions).index.equals(rt.gross_returns(prices, positions).index)


def test_net_returns_are_gross_minus_costs_times_turnover():
    prices = prices_series()
    positions = positions_series([0.5, -0.5, 1.0, -1.0, 0.25])
    gross = rt.gross_returns(prices, positions)
    traded = rt.turnover(positions)
    net = rt.net_returns(prices, positions, costs_bps=12.5)
    pd.testing.assert_series_equal(net, (gross - 0.00125 * traded).rename("net_return"))


def test_break_even_identity_holds_to_machine_precision():
    """The closed-form break-even in cost_sweep depends on exactly this identity."""
    prices = prices_series()
    positions = positions_series([1.0, 0.0, 1.0, 1.0, 0.0])
    streams = rt.build_returns(prices, positions, costs_bps=7.0)
    expected = float(streams.gross.mean()) - (7.0 / 1e4) * float(streams.turnover.mean())
    assert float(streams.net.mean()) == pytest.approx(expected, rel=1e-15)


def test_zero_cost_net_returns_equal_gross_returns():
    prices = prices_series()
    positions = positions_series([1.0, -1.0, 1.0, -1.0, 1.0])
    streams = rt.build_returns(prices, positions, costs_bps=0.0)
    pd.testing.assert_series_equal(
        streams.net, streams.gross.rename("net_return"), check_names=True
    )


def test_higher_costs_never_raise_the_mean_net_return():
    prices = prices_series()
    positions = positions_series([1.0, -1.0, 1.0, -1.0, 1.0])
    means = [
        float(rt.build_returns(prices, positions, costs_bps=bps).net.mean())
        for bps in (0.0, 1.0, 5.0, 25.0, 100.0)
    ]
    assert all(later <= earlier for earlier, later in pairwise(means))


def test_negative_costs_are_rejected():
    with pytest.raises(QuackzInputError, match="non-negative"):
        rt.net_returns(prices_series(), positions_series([1.0] * 5), costs_bps=-1.0)


# --------------------------------------------------------------------------------------
# Annualization
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [3, 50, 366, 4000])
def test_a_calendar_daily_index_infers_the_days_in_a_year_exactly(n):
    """One bar a day is 365.25 bars a year at any sample length, or the count is wrong."""
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    assert rt.infer_periods_per_year(index) == pytest.approx(365.25)


def test_the_inferred_factor_does_not_move_with_the_length_of_a_regular_index():
    """Counting timestamps instead of intervals shows up as a short-sample upward bias.

    n stamps bound n - 1 intervals, so counting the stamps multiplies the factor by
    n / (n - 1): 33 percent too high on four daily bars and 0.4 percent too high on 251.
    Every annualized Sharpe in the report is scaled by this number.
    """
    short = rt.infer_periods_per_year(pd.date_range("2020-01-01", periods=4, freq="D"))
    long = rt.infer_periods_per_year(pd.date_range("2020-01-01", periods=4000, freq="D"))
    assert short == pytest.approx(long)


def test_a_business_day_index_infers_261_rather_than_the_252_of_a_trading_calendar():
    """The gap is holidays, which pandas' B frequency does not model and the data cannot show.

    This is the reason `periods_per_year` is worth supplying explicitly on daily equity
    bars: inference answers what the index contains, not what the exchange was open for.
    """
    index = pd.date_range("2015-01-01", "2024-12-31", freq="B")
    assert rt.infer_periods_per_year(index) == pytest.approx(260.84, abs=0.05)


def test_inferred_monthly_frequency_lands_on_twelve():
    index = pd.date_range("2010-01-31", "2024-12-31", freq="ME")
    assert rt.infer_periods_per_year(index) == pytest.approx(12.0, abs=0.01)


def test_an_explicit_annualization_factor_is_never_overridden():
    ppy, inferred = rt.resolve_periods_per_year(make_index(100), 365.0)
    assert ppy == 365.0
    assert inferred is False


def test_an_inferred_annualization_factor_is_flagged_as_inferred():
    ppy, inferred = rt.resolve_periods_per_year(make_index(100), None)
    assert inferred is True
    assert ppy > 0


def test_intraday_data_refuses_to_guess_an_annualization_factor():
    index = pd.DatetimeIndex(pd.date_range("2020-01-01 09:30", periods=60, freq="min"))
    with pytest.raises(QuackzInputError, match="less than one whole day"):
        rt.infer_periods_per_year(index)


@pytest.mark.parametrize("bad_ppy", [0.0, -1.0, float("nan"), float("inf")])
def test_an_impossible_annualization_factor_is_rejected(bad_ppy):
    with pytest.raises(QuackzInputError, match="positive finite"):
        rt.resolve_periods_per_year(make_index(10), bad_ppy)


# --------------------------------------------------------------------------------------
# Input validation and the NaN policy
# --------------------------------------------------------------------------------------


def test_caller_series_are_never_mutated():
    prices = prices_series()
    positions = positions_series([1.0, 0.0, 1.0, 0.0, 1.0])
    prices_before = prices.copy(deep=True)
    positions_before = positions.copy(deep=True)

    rt.build_returns(prices, positions, costs_bps=10.0)
    rt.gross_returns(prices, positions)
    rt.net_returns(prices, positions, costs_bps=10.0)
    rt.turnover(positions)

    pd.testing.assert_series_equal(prices, prices_before)
    pd.testing.assert_series_equal(positions, positions_before)
    assert prices.name == "close"
    assert positions.name == "signal"


def test_integer_positions_are_accepted_without_changing_the_caller_dtype():
    prices = prices_series()
    positions = pd.Series([1, 1, 0, 0, 1], index=make_index(5), name="signal")
    streams = rt.build_returns(prices, positions)
    assert positions.dtype == np.int64
    assert streams.gross.dtype == np.float64


def test_a_nan_in_prices_is_rejected_with_the_offending_timestamp():
    prices = prices_series()
    prices.iloc[2] = np.nan
    with pytest.raises(QuackzInputError, match="NaN"):
        rt.gross_returns(prices, positions_series([1.0] * 5))


def test_a_nan_in_positions_is_rejected():
    positions = positions_series([1.0, np.nan, 1.0, 1.0, 1.0])
    with pytest.raises(QuackzInputError, match="NaN"):
        rt.gross_returns(prices_series(), positions)


def test_misaligned_indexes_are_reported_rather_than_inner_joined():
    prices = prices_series()
    positions = positions_series([1.0] * 5)
    positions.index = make_index(5, start="2020-02-03")
    with pytest.raises(QuackzInputError, match="different indexes"):
        rt.gross_returns(prices, positions)


def test_a_missing_position_row_is_not_silently_dropped():
    prices = prices_series()
    positions = positions_series([1.0] * 5).drop(index=prices.index[2])
    with pytest.raises(QuackzInputError, match="only in prices"):
        rt.gross_returns(prices, positions)


def test_a_non_datetime_index_is_rejected():
    prices = pd.Series(PRICES, name="close")
    positions = pd.Series([1.0] * 5, name="signal")
    with pytest.raises(QuackzInputError, match="DatetimeIndex"):
        rt.gross_returns(prices, positions)


def test_an_unsorted_index_is_rejected():
    prices = prices_series().iloc[::-1]
    positions = positions_series([1.0] * 5).iloc[::-1]
    with pytest.raises(QuackzInputError, match="sorted ascending"):
        rt.gross_returns(prices, positions)


def test_duplicate_timestamps_are_rejected():
    index = make_index(4).append(pd.DatetimeIndex([make_index(4)[-1]]))
    prices = pd.Series(PRICES, index=index, name="close")
    positions = pd.Series([1.0] * 5, index=index, name="signal")
    with pytest.raises(QuackzInputError, match="duplicate timestamps"):
        rt.gross_returns(prices, positions)


@pytest.mark.parametrize("bad_price", [0.0, -10.0])
def test_non_positive_prices_are_rejected(bad_price):
    prices = prices_series()
    prices.iloc[3] = bad_price
    with pytest.raises(QuackzInputError, match="strictly positive"):
        rt.gross_returns(prices, positions_series([1.0] * 5))


def test_too_few_observations_are_rejected():
    prices = make_series([100.0, 101.0], "close")
    positions = make_series([1.0, 1.0], "signal")
    with pytest.raises(QuackzInputError, match="at least 3 observations"):
        rt.gross_returns(prices, positions)


def test_a_dataframe_is_rejected_with_a_clear_message():
    with pytest.raises(QuackzInputError, match="pandas Series"):
        rt.gross_returns(prices_series().to_frame(), positions_series([1.0] * 5))


def test_a_non_numeric_series_is_rejected():
    positions = pd.Series(["long"] * 5, index=make_index(5), name="signal")
    with pytest.raises(QuackzInputError, match="numeric"):
        rt.gross_returns(prices_series(), positions)


# --------------------------------------------------------------------------------------
# build_returns
# --------------------------------------------------------------------------------------


def test_build_returns_agrees_with_the_individual_helpers():
    prices = prices_series()
    positions = positions_series([1.0, 0.0, 1.0, 1.0, 0.0])
    streams = rt.build_returns(prices, positions, costs_bps=8.0, periods_per_year=252.0)
    pd.testing.assert_series_equal(streams.gross, rt.gross_returns(prices, positions))
    pd.testing.assert_series_equal(streams.turnover, rt.turnover(positions))
    pd.testing.assert_series_equal(streams.net, rt.net_returns(prices, positions, costs_bps=8.0))


def test_build_returns_carries_the_annualization_provenance():
    prices = prices_series()
    positions = positions_series([1.0] * 5)

    explicit = rt.build_returns(prices, positions, periods_per_year=252.0)
    assert explicit.periods_per_year == 252.0
    assert explicit.periods_per_year_inferred is False

    inferred = rt.build_returns(prices, positions)
    assert inferred.periods_per_year_inferred is True
    assert inferred.periods_per_year == pytest.approx(rt.infer_periods_per_year(prices.index))


def test_build_returns_reports_the_observation_count_after_the_shift():
    streams = rt.build_returns(prices_series(), positions_series([1.0] * 5))
    assert streams.n_obs == len(PRICES) - 1
    assert streams.n_obs == len(streams.net)


def test_return_streams_are_frozen():
    streams = rt.build_returns(prices_series(), positions_series([1.0] * 5))
    with pytest.raises(AttributeError):
        streams.periods_per_year = 1.0


def test_return_streams_repr_stays_readable():
    """Bulky series are excluded from the repr so a report never dumps a whole backtest."""
    text = repr(rt.build_returns(prices_series(), positions_series([1.0] * 5)))
    assert "periods_per_year" in text
    assert "2020-01" not in text
