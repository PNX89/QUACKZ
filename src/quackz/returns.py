"""Turning prices and positions into returns, in exactly one place.

Every check in the library consumes the series built here, so the conventions are
defined once and tested in isolation.

Position convention
    `positions[t]` is the target exposure DECIDED AT bar t's close, so it earns bar
    t+1's return:

        gross_t = positions[t-1] * prices.pct_change()[t]

    The returned series therefore starts one bar after the price series.

Cost convention
    net_t = positions[t-1] * r_t - (costs_bps / 1e4) * |positions[t-1] - positions[t-2]|

    Two edges of that formula are where cost models usually leak profit.

    Entry. At the first return bar the term is |positions[0] - positions[-1]| with
    positions[-1] taken as flat, so the trade INTO the first position is charged. A
    `fillna(0)` on a differenced position series silently drops that charge, which
    flatters any strategy that takes one large position and holds it.

    Exit. The position in force at the end of the sample is `positions[-2]`, the last
    one that actually earned a return, and closing it is charged on the final bar. That
    makes the cost path a closed round trip from flat back to flat, so total turnover
    equals the true traded path. `positions[-1]` never participates: it is set at the
    last close and no further return exists for it to earn. Pass
    `liquidate_final=False` to leave the exit uncharged and mark to market instead.

Causality
    The shift proves that THIS library's arithmetic is causal. It says nothing about the
    supplied positions: lookahead baked into the caller's signal is invisible here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "MIN_OBSERVATIONS",
    "QuackzInputError",
    "ReturnStreams",
    "build_returns",
    "gross_returns",
    "infer_periods_per_year",
    "net_returns",
    "resolve_periods_per_year",
    "turnover",
    "validate_inputs",
]

# A shift of one plus a position difference of one needs three price bars before any
# cost-bearing return exists.
MIN_OBSERVATIONS = 3


class QuackzInputError(ValueError):
    """Raised when prices or positions violate a documented input convention."""


def _validate_series(series: object, name: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise QuackzInputError(f"{name} must be a pandas Series, got {type(series).__name__}")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise QuackzInputError(
            f"{name} must have a DatetimeIndex, got {type(series.index).__name__}. Parse the "
            "date column first, for example pd.read_csv(..., parse_dates=['date'])."
        )
    if len(series) < MIN_OBSERVATIONS:
        raise QuackzInputError(
            f"{name} needs at least {MIN_OBSERVATIONS} observations, got {len(series)}"
        )
    if not series.index.is_monotonic_increasing:
        raise QuackzInputError(f"{name} index must be sorted ascending; call .sort_index() first")
    if series.index.has_duplicates:
        duplicates = series.index[series.index.duplicated()].unique()[:3].tolist()
        raise QuackzInputError(
            f"{name} index has duplicate timestamps (first few: {duplicates}); "
            "collapse or drop them before evaluating"
        )
    if not pd.api.types.is_numeric_dtype(series):
        raise QuackzInputError(f"{name} must be numeric, got dtype {series.dtype}")

    # Defensive copy at the boundary: caller Series are never mutated, and copy-on-write
    # semantics differ across pandas 2.x and 3.x so this cannot be left to the library.
    out = series.astype(float).copy()
    out.name = name

    if out.isna().any():
        first_bad = out.index[out.isna()][0]
        raise QuackzInputError(
            f"{name} has {int(out.isna().sum())} NaN value(s), first at {first_bad}. QUACKZ drops "
            "only the single leading NaN created by its own shift; decide explicitly how to "
            "handle the rest (fill, drop, or fix the source data)."
        )
    if not np.isfinite(out.to_numpy()).all():
        raise QuackzInputError(f"{name} contains infinite values")
    return out


def _check_alignment(prices: pd.Series, positions: pd.Series) -> None:
    if prices.index.equals(positions.index):
        return
    only_prices = prices.index.difference(positions.index)
    only_positions = positions.index.difference(prices.index)
    raise QuackzInputError(
        "prices and positions have different indexes: "
        f"{len(only_prices)} timestamp(s) only in prices (first few: {only_prices[:3].tolist()}), "
        f"{len(only_positions)} only in positions (first few: {only_positions[:3].tolist()}). "
        "QUACKZ will not inner-join them, because a silent join turns missing positions into "
        "an implicitly different strategy. Reindex deliberately and re-run."
    )


def validate_inputs(prices: object, positions: object) -> tuple[pd.Series, pd.Series]:
    """Validate, defensively copy and align prices and positions.

    Returns fresh float Series; the caller's objects are never touched.
    """
    checked_prices = _validate_series(prices, "prices")
    checked_positions = _validate_series(positions, "positions")
    if (checked_prices <= 0).any():
        first_bad = checked_prices.index[checked_prices <= 0][0]
        raise QuackzInputError(
            f"prices must be strictly positive (first violation at {first_bad}); percentage "
            "returns are undefined at or below zero"
        )
    _check_alignment(checked_prices, checked_positions)
    return checked_prices, checked_positions


def infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    """Observations per year implied by the index itself.

    `(len(index) - 1) / span_in_days * 365.25`. The numerator counts INTERVALS, not
    timestamps: n stamps bound n - 1 of them, and counting the stamps instead inflates the
    factor by n / (n - 1), which inflates every annualized Sharpe built on it. The bias is
    small on a long sample and large on a short one, and it is always upward, which is the
    wrong direction for a library that exists to catch inflated Sharpes.

    Inferring beats mapping a frequency string to a constant, because the constant is a
    guess about holidays and half-days that the data already answers. Note what that means
    for daily equity bars: a holiday-free business-day index infers about 261, not the 252
    a trading calendar gives, so pass `periods_per_year` explicitly when the difference
    matters. The caller must report the value it used.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise QuackzInputError(f"index must be a DatetimeIndex, got {type(index).__name__}")
    if len(index) < 2:
        raise QuackzInputError("cannot infer periods_per_year from fewer than 2 timestamps")
    span_days = (index[-1] - index[0]).days
    if span_days <= 0:
        raise QuackzInputError(
            "cannot infer periods_per_year: the index spans less than one whole day. Pass "
            "periods_per_year explicitly for intraday data."
        )
    return (len(index) - 1) / span_days * 365.25


def resolve_periods_per_year(
    index: pd.DatetimeIndex, periods_per_year: float | None
) -> tuple[float, bool]:
    """Return (value, was_inferred). An explicit value is used as given, never overridden."""
    if periods_per_year is None:
        return infer_periods_per_year(index), True
    ppy = float(periods_per_year)
    if not np.isfinite(ppy) or ppy <= 0:
        raise QuackzInputError(
            f"periods_per_year must be a positive finite number, got {periods_per_year!r}"
        )
    return ppy, False


def _check_costs_bps(costs_bps: float) -> float:
    bps = float(costs_bps)
    if not np.isfinite(bps) or bps < 0:
        raise QuackzInputError(f"costs_bps must be a non-negative finite number, got {costs_bps!r}")
    return bps


def _gross_from_checked(prices: pd.Series, positions: pd.Series) -> pd.Series:
    # fill_method=None is explicit rather than default-dependent; NaNs are already rejected.
    asset_returns = prices.pct_change(fill_method=None)
    gross = (positions.shift(1) * asset_returns).iloc[1:]
    if not np.isfinite(gross.to_numpy()).all():
        raise QuackzInputError("gross returns contain non-finite values; check the price series")
    return gross.rename("gross_return")


def _turnover_from_checked(positions: pd.Series, *, liquidate_final: bool) -> pd.Series:
    values = positions.to_numpy()
    previous = np.concatenate(([0.0], values[:-1]))
    traded = np.abs(values - previous)[:-1]
    if liquidate_final:
        traded = traded.copy()
        traded[-1] += abs(float(values[-2]))
    return pd.Series(traded, index=positions.index[1:], name="turnover")


def gross_returns(prices: pd.Series, positions: pd.Series) -> pd.Series:
    """Costless strategy returns: `positions[t-1] * prices.pct_change()[t]`.

    The single leading NaN produced by the shift is dropped, so the result is one
    observation shorter than the inputs.
    """
    return _gross_from_checked(*validate_inputs(prices, positions))


def turnover(positions: pd.Series, *, liquidate_final: bool = True) -> pd.Series:
    """Traded notional attributed to each return bar: `|positions[t-1] - positions[t-2]|`.

    Indexed like `gross_returns`, so `net = gross - (bps / 1e4) * turnover` needs no
    further alignment. `positions[-1]` is treated as flat, which charges the entry trade,
    and with `liquidate_final=True` the closing trade out of `positions[-2]` is added to
    the final bar, so the series sums to the full round-trip path from flat to flat.
    """
    checked = _validate_series(positions, "positions")
    return _turnover_from_checked(checked, liquidate_final=liquidate_final)


def net_returns(
    prices: pd.Series,
    positions: pd.Series,
    *,
    costs_bps: float = 0.0,
    liquidate_final: bool = True,
) -> pd.Series:
    """Gross returns less linear transaction costs at `costs_bps` basis points of turnover."""
    bps = _check_costs_bps(costs_bps)
    checked_prices, checked_positions = validate_inputs(prices, positions)
    gross = _gross_from_checked(checked_prices, checked_positions)
    traded = _turnover_from_checked(checked_positions, liquidate_final=liquidate_final)
    return (gross - (bps / 1e4) * traded).rename("net_return")


@dataclass(frozen=True)
class ReturnStreams:
    """Everything downstream checks need, built once so they cannot disagree."""

    gross: pd.Series = field(repr=False)
    net: pd.Series = field(repr=False)
    turnover: pd.Series = field(repr=False)
    periods_per_year: float
    periods_per_year_inferred: bool
    costs_bps: float
    n_obs: int


def build_returns(
    prices: pd.Series,
    positions: pd.Series,
    *,
    costs_bps: float = 0.0,
    periods_per_year: float | None = None,
    liquidate_final: bool = True,
) -> ReturnStreams:
    """Validate inputs once and build the gross, net and turnover series together.

    `periods_per_year` is inferred from the PRICE index when None, and the flag saying so
    travels with the result because a report that hides its annualization factor is not
    reproducible.
    """
    bps = _check_costs_bps(costs_bps)
    checked_prices, checked_positions = validate_inputs(prices, positions)
    ppy, inferred = resolve_periods_per_year(checked_prices.index, periods_per_year)

    gross = _gross_from_checked(checked_prices, checked_positions)
    traded = _turnover_from_checked(checked_positions, liquidate_final=liquidate_final)
    net = (gross - (bps / 1e4) * traded).rename("net_return")
    return ReturnStreams(
        gross=gross,
        net=net,
        turnover=traded,
        periods_per_year=ppy,
        periods_per_year_inferred=inferred,
        costs_bps=bps,
        n_obs=len(gross),
    )
