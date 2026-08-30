"""Per-check diagnostics: what a backtest's own numbers imply once you push on them.

Every check returns a frozen dataclass carrying the quantities it computed and, where a
verdict is possible, a PASS / WARN / FAIL. Thresholds are module-level constants with the
reasoning for each value written beside it, so a reader can disagree with a number without
reading the code that consumes it. They are gathered into `Thresholds`, which every check
accepts and `quackz.evaluate.evaluate` threads through a whole report, so a caller who
disagrees with a cut-off overrides it in one place rather than reimplementing the check.

Verdicts are statements about evidence, never about intent. A FAIL says the backtest's own
arithmetic does not support the claim being made. None of these checks can see how the
supplied positions were constructed, so none of them can establish that information from
the future entered the signal, and none of them says so.

Frequency
    Anything named `sharpe` in a returned value is ANNUALIZED by sqrt(periods_per_year).
    The statistical machinery underneath (the PSR family in `quackz.metrics`, the
    studentized bootstrap here) works at the observation frequency and converts at the
    boundary, which is where the conversion is visible.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, fields
from enum import StrEnum

import numpy as np
import numpy.typing as npt
import pandas as pd

from quackz import metrics
from quackz.metrics import _newey_west_lags
from quackz.returns import (
    QuackzInputError,
    ReturnStreams,
    _turnover_from_checked,
    build_returns,
)

__all__ = [
    "BOOTSTRAP_CI_CONFIDENCE",
    "BOOTSTRAP_PERCENTILES",
    "BOOTSTRAP_P_VALUE_FAIL",
    "BOOTSTRAP_P_VALUE_WARN",
    "CONCENTRATION_PROFIT_SHARE_FAIL",
    "CONCENTRATION_PROFIT_SHARE_WARN",
    "COST_BREAK_EVEN_BPS_FAIL",
    "COST_BREAK_EVEN_BPS_WARN",
    "DEFAULT_THRESHOLDS",
    "DSR_FAIL",
    "DSR_WARN",
    "LATENCY_ANNUAL_SHARPE_FAIL",
    "LATENCY_ANNUAL_SHARPE_WARN",
    "LATENCY_DECAY_MIN_HOLDING_BARS",
    "LATENCY_RETENTION_RATIO_FAIL",
    "LATENCY_RETENTION_RATIO_WARN",
    "NOISE_FLOOR_RATIO_FAIL",
    "NOISE_FLOOR_RATIO_WARN",
    "RECONCILE_CORRELATION_FAIL",
    "RECONCILE_CORRELATION_WARN",
    "RECONCILE_CUMULATIVE_GAP_FAIL",
    "RECONCILE_CUMULATIVE_GAP_WARN",
    "STABILITY_DISPERSION_RATIO_FAIL",
    "STABILITY_DISPERSION_RATIO_WARN",
    "BootstrapBlockResult",
    "BootstrapResult",
    "ConcentrationResult",
    "ConcentrationRow",
    "CostSweepResult",
    "CostSweepRow",
    "LatencyResult",
    "NoiseFloorResult",
    "ReconcileResult",
    "SubperiodStabilityResult",
    "Thresholds",
    "Verdict",
    "bootstrap",
    "concentration",
    "cost_sweep",
    "latency_sensitivity",
    "noise_floor",
    "reconcile",
    "subperiod_stability",
    "worst_verdict",
]


class Verdict(StrEnum):
    """Outcome of one check. A str subclass, so it serialises to JSON as its own name."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


_SEVERITY = {Verdict.PASS: 0, Verdict.WARN: 1, Verdict.FAIL: 2}


def worst_verdict(*verdicts: Verdict) -> Verdict:
    """Least favourable of the verdicts given. PASS when none are given."""
    return max(verdicts, key=_SEVERITY.__getitem__, default=Verdict.PASS)


# --------------------------------------------------------------------------------------
# Thresholds. Every one of these is a judgement call, so every one carries its reasoning.
# --------------------------------------------------------------------------------------

# reconcile: below this correlation the two series are not describing the same trading
# path, so no convention difference explains the gap on its own.
RECONCILE_CORRELATION_FAIL = 0.99
# Two implementations of one rule that differ only in cost, financing or rebalance
# conventions still track each other above this.
RECONCILE_CORRELATION_WARN = 0.999
# Terminal wealth gap, as a fraction of starting capital: 0.10 is ten cents on the euro.
RECONCILE_CUMULATIVE_GAP_FAIL = 0.10
RECONCILE_CUMULATIVE_GAP_WARN = 0.01

# latency_sensitivity: an annualized Sharpe this high sits far outside the range of
# documented liquid-market results at any meaningful capacity, so the first hypothesis is
# a timing or data problem rather than an edge. Fires on the LEVEL at lag 0.
LATENCY_ANNUAL_SHARPE_FAIL = 10.0
LATENCY_ANNUAL_SHARPE_WARN = 5.0

# latency_sensitivity, decay: a one-bar delay changes the position only on the bars where
# it moved, so a position with a mean holding period of h bars is misplaced on about 1/h of
# them, and an edge spread across the holding period keeps roughly (h - 1)/h of its Sharpe.
# What is graded is the fraction of that expectation actually kept. At half of it the edge
# is concentrated near the decision; at a quarter it is not spread across the holding period
# at all but sits in the single bar after the position changed, which is the bar a rebalance
# loop reads when it reads one bar too far ahead.
LATENCY_RETENTION_RATIO_WARN = 0.50
LATENCY_RETENTION_RATIO_FAIL = 0.25

# The decay rule has no power below this mean holding period, so it is not applied there. A
# position that turns over every other bar is EXPECTED to lose everything to a one-bar
# delay: that is what genuine short-horizon alpha looks like, and grading it would fail the
# honest case. Five bars is a strategy that still holds most of its exposure a day later.
LATENCY_DECAY_MIN_HOLDING_BARS = 5.0

# The rule divides one Sharpe by another, so it also needs a numerator that is not itself
# noise: the ratio of two numbers that are both noise is noise, and on a strategy with no
# edge at all it lands anywhere. Below this many standard errors above zero there is no edge
# to lose, and the rule stays quiet. Measured against sqrt(periods_per_year / n_obs), one
# standard error of an annualized Sharpe near zero under iid returns (Lo 2002). Sweeping
# forty edgeless slow-moving strategies, dropping this guard fails six of them.
LATENCY_DECAY_MIN_SHARPE_SE = 2.0

# cost_sweep: a round trip in liquid cash equities costs roughly 5 to 10 bps all in
# (spread, commission, impact). An edge whose break-even sits below that does not survive
# contact with a broker; one that clears 20 bps has room for a worse fill than assumed.
COST_BREAK_EVEN_BPS_FAIL = 5.0
COST_BREAK_EVEN_BPS_WARN = 20.0

# bootstrap: conventional cut-offs on the null-imposed p-value, applied to the LEAST
# favourable block length so that a lucky choice of block cannot rescue the verdict.
BOOTSTRAP_P_VALUE_WARN = 0.05
BOOTSTRAP_P_VALUE_FAIL = 0.10
BOOTSTRAP_CI_CONFIDENCE = 0.95
BOOTSTRAP_PERCENTILES = (5.0, 25.0, 50.0, 75.0, 95.0)

# subperiod_stability: ratio of the observed dispersion of window Sharpes to the
# dispersion that sampling noise alone predicts. Under a stable signal the ratio is chi
# distributed around 1; with five windows it clears 1.5 about one time in sixteen and 2.5
# about one time in twenty thousand.
STABILITY_DISPERSION_RATIO_WARN = 1.5
STABILITY_DISPERSION_RATIO_FAIL = 2.5

# concentration: share of gross profit contributed by the largest requested number of
# bars. For a normal sample of about a thousand bars the top ten carry roughly 7 percent,
# so 30 percent is several times the noise baseline and 50 percent means the track record
# is a handful of events rather than a process.
CONCENTRATION_PROFIT_SHARE_WARN = 0.30
CONCENTRATION_PROFIT_SHARE_FAIL = 0.50

# quackz.evaluate, deflated Sharpe: the probability that the result survives the selection
# the declared trial count implies. 0.95 is the confidence a track record is conventionally
# asked to clear; below 0.5 the search is a likelier explanation of the number than an edge.
DSR_WARN = 0.95
DSR_FAIL = 0.50

# quackz.evaluate, noise floor: observed annualized Sharpe divided by the Sharpe a search of
# the declared size expects to find in pure noise. At or below 1 the result is what the
# search alone produces; below 1.5 the gap is inside the estimation error of both numbers.
NOISE_FLOOR_RATIO_WARN = 1.5
NOISE_FLOOR_RATIO_FAIL = 1.0

# Two series that differ by less than this in terminal wealth are the same series.
_GAP_TOLERANCE = 1e-12


# Fields that are themselves probabilities or profit shares, so a cut-off outside [0, 1] is
# not a strict-but-valid preference, it is a value the quantity being compared can never take.
# `dsr_warn` is stricter still: `evaluate` passes it straight through to
# `metrics.min_track_record_length` as `confidence`, which requires the open interval, so 0.0
# and 1.0 crash there with a message that never mentions `dsr_warn`.
_PROBABILITY_FIELDS = frozenset(
    {
        "dsr_fail",
        "bootstrap_p_value_warn",
        "bootstrap_p_value_fail",
        "concentration_profit_share_warn",
        "concentration_profit_share_fail",
    }
)
# Pearson correlation coefficients, so [-1, 1] rather than [0, 1].
_CORRELATION_FIELDS = frozenset({"reconcile_correlation_warn", "reconcile_correlation_fail"})

# Milder cut-off, harsher cut-off, and the direction the harsher one must lie in. A swapped
# pair is silent otherwise: every verdict comes back PASS and the report looks clean.
_THRESHOLD_LADDERS = (
    ("reconcile_correlation_warn", "reconcile_correlation_fail", "lower"),
    ("reconcile_cumulative_gap_warn", "reconcile_cumulative_gap_fail", "higher"),
    ("latency_annual_sharpe_warn", "latency_annual_sharpe_fail", "higher"),
    ("latency_retention_ratio_warn", "latency_retention_ratio_fail", "lower"),
    ("cost_break_even_bps_warn", "cost_break_even_bps_fail", "lower"),
    ("bootstrap_p_value_warn", "bootstrap_p_value_fail", "higher"),
    ("stability_dispersion_ratio_warn", "stability_dispersion_ratio_fail", "higher"),
    ("concentration_profit_share_warn", "concentration_profit_share_fail", "higher"),
    ("dsr_warn", "dsr_fail", "lower"),
    ("noise_floor_ratio_warn", "noise_floor_ratio_fail", "lower"),
)


@dataclass(frozen=True)
class Thresholds:
    """Every cut-off a verdict in this library is formed from, gathered into one object.

    The defaults are the module constants above, each of which carries the reasoning for its
    value beside it. Override with `dataclasses.replace(DEFAULT_THRESHOLDS, dsr_warn=0.99)`,
    or pass a mapping of names to values to `quackz.evaluate.evaluate`. Each check also
    takes a `thresholds` keyword, so one set of cut-offs grades a whole report.

    An equal WARN and FAIL cut-off is allowed and collapses the WARN band to nothing.
    """

    reconcile_correlation_warn: float = RECONCILE_CORRELATION_WARN
    reconcile_correlation_fail: float = RECONCILE_CORRELATION_FAIL
    reconcile_cumulative_gap_warn: float = RECONCILE_CUMULATIVE_GAP_WARN
    reconcile_cumulative_gap_fail: float = RECONCILE_CUMULATIVE_GAP_FAIL
    latency_annual_sharpe_warn: float = LATENCY_ANNUAL_SHARPE_WARN
    latency_annual_sharpe_fail: float = LATENCY_ANNUAL_SHARPE_FAIL
    latency_retention_ratio_warn: float = LATENCY_RETENTION_RATIO_WARN
    latency_retention_ratio_fail: float = LATENCY_RETENTION_RATIO_FAIL
    cost_break_even_bps_warn: float = COST_BREAK_EVEN_BPS_WARN
    cost_break_even_bps_fail: float = COST_BREAK_EVEN_BPS_FAIL
    bootstrap_p_value_warn: float = BOOTSTRAP_P_VALUE_WARN
    bootstrap_p_value_fail: float = BOOTSTRAP_P_VALUE_FAIL
    stability_dispersion_ratio_warn: float = STABILITY_DISPERSION_RATIO_WARN
    stability_dispersion_ratio_fail: float = STABILITY_DISPERSION_RATIO_FAIL
    concentration_profit_share_warn: float = CONCENTRATION_PROFIT_SHARE_WARN
    concentration_profit_share_fail: float = CONCENTRATION_PROFIT_SHARE_FAIL
    dsr_warn: float = DSR_WARN
    dsr_fail: float = DSR_FAIL
    noise_floor_ratio_warn: float = NOISE_FLOOR_RATIO_WARN
    noise_floor_ratio_fail: float = NOISE_FLOOR_RATIO_FAIL

    def __post_init__(self) -> None:
        for spec in fields(self):
            value = getattr(self, spec.name)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise QuackzInputError(
                    f"threshold {spec.name} must be a finite number, got {value!r}"
                )
            value = float(value)
            if spec.name == "dsr_warn" and not 0.0 < value < 1.0:
                raise QuackzInputError(
                    f"dsr_warn must be strictly between 0 and 1, got {value!r}; it is used "
                    "as a confidence level as well as a comparison, and 0 or 1 is not a "
                    "confidence anything can be evaluated at"
                )
            if spec.name in _PROBABILITY_FIELDS and not 0.0 <= value <= 1.0:
                raise QuackzInputError(
                    f"threshold {spec.name} is a probability or a share of profit and must "
                    f"lie in [0, 1], got {value!r}"
                )
            if spec.name in _CORRELATION_FIELDS and not -1.0 <= value <= 1.0:
                raise QuackzInputError(
                    f"threshold {spec.name} is a correlation and must lie in [-1, 1], got {value!r}"
                )
            object.__setattr__(self, spec.name, value)
        for warn_name, fail_name, direction in _THRESHOLD_LADDERS:
            warn_value = getattr(self, warn_name)
            fail_value = getattr(self, fail_name)
            ordered = fail_value <= warn_value if direction == "lower" else fail_value >= warn_value
            if not ordered:
                raise QuackzInputError(
                    f"{fail_name}={fail_value} must be {direction} than or equal to "
                    f"{warn_name}={warn_value}; as given, a result can never reach FAIL "
                    "without passing through WARN in the wrong direction"
                )


DEFAULT_THRESHOLDS = Thresholds()


def _finite_array(values: npt.ArrayLike, *, name: str, min_len: int = 2) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise QuackzInputError(f"{name} must be one dimensional, got shape {arr.shape}")
    if arr.size < min_len:
        raise QuackzInputError(f"{name} needs at least {min_len} observations, got {arr.size}")
    if not np.isfinite(arr).all():
        bad = int(np.count_nonzero(~np.isfinite(arr)))
        raise QuackzInputError(
            f"{name} contains {bad} non-finite value(s); clean or drop them before evaluating"
        )
    return arr


def _check_int(value: object, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise QuackzInputError(f"{name} must be an integer, got {value!r}")
    out = int(value)
    if out < minimum:
        raise QuackzInputError(f"{name} must be at least {minimum}, got {out}")
    return out


def _check_periods_per_year(periods_per_year: float) -> float:
    """For checks with no price index to infer from, so `None` is not an option here."""
    ppy = float(periods_per_year)
    if not math.isfinite(ppy) or ppy <= 0.0:
        raise QuackzInputError(
            f"periods_per_year must be a positive finite number, got {periods_per_year!r}"
        )
    return ppy


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Elementwise division that yields 0.0 where the denominator is not positive.

    A degenerate bootstrap resample (every observation identical) has no dispersion and no
    edge, so a Sharpe of zero is the honest value. Dividing and cleaning up afterwards
    would raise a RuntimeWarning, which the test suite turns into an error.
    """
    out = np.zeros_like(numerator, dtype=float)
    return np.divide(numerator, denominator, out=out, where=denominator > 0.0)


def _compounded_growth(arr: np.ndarray, *, name: str) -> float:
    """Terminal wealth minus one, computed in log space so a long series cannot overflow."""
    if (arr <= -1.0).any():
        raise QuackzInputError(
            f"{name} contains a value at or below -1 (total loss); the compounded curve is "
            "undefined beyond that point"
        )
    return float(np.expm1(np.log1p(arr).sum()))


# --------------------------------------------------------------------------------------
# reconcile
# --------------------------------------------------------------------------------------

_RECONCILE_CAUSES_COMMON = (
    "a different transaction cost or slippage model, or costs charged on a different "
    "definition of turnover",
    "financing, borrow or margin interest applied to one series and not the other",
    "a different rebalance convention: execution at the open rather than the close, or a "
    "one-bar difference in when the position is deemed live",
    "dividends, coupons or other cash flows included in one series only",
    "a different sizing or leverage convention, including compounding against a growing "
    "equity base in one series and a fixed notional in the other",
)

_RECONCILE_CAUSE_CLAIMED_HIGHER = (
    "information that was not available at the decision time entering the claimed series; "
    "this check cannot separate that from the conventions above, and the conventions are "
    "the more common explanation"
)


@dataclass(frozen=True)
class ReconcileResult:
    """How far a claimed return stream sits from the one QUACKZ recomputes."""

    correlation: float
    tracking_error: float
    cumulative_gap: float
    gap_direction: str
    n_obs: int
    candidate_causes: tuple[str, ...]
    verdict: Verdict


def reconcile(
    *,
    claimed: npt.ArrayLike,
    recomputed: npt.ArrayLike,
    thresholds: Thresholds | None = None,
) -> ReconcileResult:
    """Compare a claimed per-period return stream with the one implied by prices and positions.

    Reports the correlation, the per-period tracking error (standard deviation of the
    difference), the gap in terminal wealth and the direction of that gap, together with a
    list of candidate causes.

    A gap is not evidence of lookahead and this function never says that it is. In practice
    almost every divergence is a convention: a different cost model, financing, or a
    rebalance timed differently. Only after those are ruled out by hand does anything else
    become worth considering, which is why the candidate causes are listed rather than
    ranked into a conclusion.

    Both series must be per-period simple returns covering the same bars. Two pandas Series
    with different indexes are rejected rather than joined, because a silent join compares
    two different strategies.
    """
    cuts = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    # The isinstance calls are inline rather than held in a name: both parameters are
    # ArrayLike, and only a reader that sees the narrowing here can tell that the index
    # comparison below is reached on Series alone.
    if (
        isinstance(claimed, pd.Series)
        and isinstance(recomputed, pd.Series)
        and not claimed.index.equals(recomputed.index)
    ):
        only_claimed = claimed.index.difference(recomputed.index)
        only_recomputed = recomputed.index.difference(claimed.index)
        raise QuackzInputError(
            "claimed and recomputed cover different bars: "
            f"{len(only_claimed)} timestamp(s) only in claimed "
            f"(first few: {only_claimed[:3].tolist()}), "
            f"{len(only_recomputed)} only in recomputed "
            f"(first few: {only_recomputed[:3].tolist()}). Align them deliberately; "
            "QUACKZ will not inner-join two return streams."
        )

    a = _finite_array(claimed, name="claimed")
    b = _finite_array(recomputed, name="recomputed")
    if a.size != b.size:
        raise QuackzInputError(
            f"claimed has {a.size} observations and recomputed has {b.size}; the two must "
            "cover the same bars. The recomputed stream is one bar shorter than the price "
            "series because of the position shift, so trim the claimed stream to match."
        )

    sd_a = float(a.std(ddof=1))
    sd_b = float(b.std(ddof=1))
    if sd_a == 0.0 or sd_b == 0.0:
        raise QuackzInputError(
            "claimed or recomputed is constant, so the correlation between them is "
            "undefined; a flat return stream cannot be reconciled against anything"
        )
    correlation = float(np.corrcoef(a, b)[0, 1])
    tracking_error = float((a - b).std(ddof=1))
    gap = _compounded_growth(a, name="claimed") - _compounded_growth(b, name="recomputed")

    causes: tuple[str, ...]
    if gap > _GAP_TOLERANCE:
        direction = "claimed_higher"
        causes = (*_RECONCILE_CAUSES_COMMON, _RECONCILE_CAUSE_CLAIMED_HIGHER)
    elif gap < -_GAP_TOLERANCE:
        direction = "claimed_lower"
        causes = _RECONCILE_CAUSES_COMMON
    else:
        direction = "identical"
        causes = ()

    if (
        correlation < cuts.reconcile_correlation_fail
        or abs(gap) > cuts.reconcile_cumulative_gap_fail
    ):
        verdict = Verdict.FAIL
    elif (
        correlation < cuts.reconcile_correlation_warn
        or abs(gap) > cuts.reconcile_cumulative_gap_warn
    ):
        verdict = Verdict.WARN
    else:
        verdict = Verdict.PASS

    return ReconcileResult(
        correlation=correlation,
        tracking_error=tracking_error,
        cumulative_gap=gap,
        gap_direction=direction,
        n_obs=int(a.size),
        candidate_causes=causes,
        verdict=verdict,
    )


# --------------------------------------------------------------------------------------
# latency_sensitivity
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LatencyResult:
    """Sharpe as the position is delivered progressively later, plus the scale of the signal.

    `retention_1bar` is the fraction of the undelayed Sharpe that survives a one-bar delay,
    and `expected_retention_1bar` is what the mean holding period implies it should be.
    `retention_ratio` is the first over the second and is the quantity the decay verdict is
    formed from; it is None when the rule was not applied, either because the undelayed
    Sharpe is not positive or because the position turns over too fast for the rule to say
    anything (see `LATENCY_DECAY_MIN_HOLDING_BARS`).
    """

    lags: tuple[int, ...]
    sharpe_by_lag: tuple[float, ...]
    position_autocorr: float
    mean_holding_period: float
    sharpe_standard_error: float
    retention_1bar: float | None
    expected_retention_1bar: float
    retention_ratio: float | None
    n_obs: int
    verdict: Verdict


def _is_constant(arr: np.ndarray) -> bool:
    spread = float(np.max(arr) - np.min(arr))
    return spread <= 1e-14 * max(float(np.max(np.abs(arr))), 1.0)


def latency_sensitivity(
    prices: pd.Series,
    positions: pd.Series,
    *,
    max_lag: int = 3,
    periods_per_year: float,
    thresholds: Thresholds | None = None,
) -> LatencyResult:
    """Sharpe recomputed with the position delivered 0 to `max_lag` bars later than intended.

    All lags are evaluated on the SAME bars, the last `n - 1 - max_lag` of them, so the
    numbers differ only in timing and not in the sample they were measured on.

    Two rules form the verdict, and the report names whichever one fired.

    The LEVEL at lag 0. An annualized Sharpe far above what liquid markets have historically
    paid is a claim about the data or the timing before it is a claim about an edge.

    The DECAY from lag 0 to lag 1, measured against the holding period rather than in the
    raw. Raw decay says nothing on its own: a two-bar signal has no reason to survive a
    three-bar delay, and losing everything by lag 1 is what real short-horizon alpha looks
    like. But a one-bar delay only misplaces the position on the bars where it moved, so a
    position held h bars keeps roughly `(h - 1)/h` of an edge that is spread across its
    holding period. `retention_ratio` is what it actually kept over that expectation, and a
    slow-moving strategy that keeps almost none of its Sharpe one bar later is one whose
    edge lives entirely in the bar after the decision, which is the bar a rebalance loop
    reads when it reads one bar too far ahead. The rule stays quiet on two kinds of run it
    cannot speak about: a mean holding period below `LATENCY_DECAY_MIN_HOLDING_BARS`, where
    the expectation is too small to test against, and a Sharpe at zero delay within
    `LATENCY_DECAY_MIN_SHARPE_SE` standard errors of zero, where there is no edge to lose
    and the ratio of two noisy numbers is noise.

    What this still cannot see: a leak whose horizon matches the holding period. A position
    built from a twenty-bar forward return and held twenty bars loses as little to a one-bar
    delay as an honest one does, and passes both rules.

    `mean_holding_period` is `2 * sum(|position|) / total turnover`, in bars: a position
    held once from flat back to flat gives the number of bars it was on. `position_autocorr`
    is the lag-1 autocorrelation of the position series itself, defined as 1.0 for a
    constant exposure.

    Costs are deliberately excluded. Delaying a position moves the turnover around without
    changing its total, so charging costs would add a constant to every lag and obscure the
    timing effect this check exists to isolate.
    """
    streams = build_returns(prices, positions, costs_bps=0.0, periods_per_year=periods_per_year)
    return _latency_from_streams(streams, max_lag=max_lag, thresholds=thresholds)


def _latency_from_streams(
    streams: ReturnStreams,
    *,
    max_lag: int = 3,
    thresholds: Thresholds | None = None,
) -> LatencyResult:
    """`latency_sensitivity` over streams already built. See there for what it measures."""
    cuts = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    lag_count = _check_int(max_lag, name="max_lag", minimum=1)
    checked_prices, checked_positions = streams.prices, streams.positions
    ppy = streams.periods_per_year

    asset_returns = checked_prices.pct_change(fill_method=None).to_numpy()[1:]
    pos = checked_positions.to_numpy()
    window = asset_returns.size - lag_count
    if window < 2:
        raise QuackzInputError(
            f"max_lag={lag_count} leaves {window} bar(s) common to every lag; a Sharpe needs "
            f"at least 2. Supply at least {lag_count + 3} observations or lower max_lag."
        )

    sharpe_by_lag = []
    for lag in range(lag_count + 1):
        lagged = pos[lag_count - lag : asset_returns.size - lag] * asset_returns[lag_count:]
        sharpe_by_lag.append(metrics.sharpe(lagged, periods_per_year=ppy))

    position_autocorr = 1.0 if _is_constant(pos) else metrics.autocorr_1(pos)

    # The full round trip from flat back to flat, whatever the streams were built with: the
    # holding period is a property of the position path, not of a cost convention.
    traded = float(_turnover_from_checked(checked_positions, liquidate_final=True).sum())
    exposure = float(np.abs(pos[:-1]).sum())
    holding_period = 2.0 * exposure / traded if traded > 0.0 else 0.0

    level = sharpe_by_lag[0]
    if level >= cuts.latency_annual_sharpe_fail:
        level_verdict = Verdict.FAIL
    elif level >= cuts.latency_annual_sharpe_warn:
        level_verdict = Verdict.WARN
    else:
        level_verdict = Verdict.PASS

    # One standard error of the annualized Sharpe at zero delay, under iid returns.
    standard_error = math.sqrt(ppy / window)
    expected_retention = 1.0 - 1.0 / holding_period if holding_period > 1.0 else 0.0
    retention: float | None = None
    ratio: float | None = None
    decay_verdict = Verdict.PASS
    # A degenerate lag-0 Sharpe is infinite rather than large (a return stream with no
    # dispersion at all), and dividing one infinity by another manufactures a NaN. The level
    # rule has already failed that run; the decay rule has nothing to add to it.
    if math.isfinite(level) and level > LATENCY_DECAY_MIN_SHARPE_SE * standard_error:
        retention = sharpe_by_lag[1] / level
        # A position that turns over faster than the rule can speak about is graded on the
        # level alone; so is one whose Sharpe is not far enough from zero to have anything
        # to lose, which is the branch above.
        if holding_period >= LATENCY_DECAY_MIN_HOLDING_BARS and expected_retention > 0.0:
            ratio = retention / expected_retention
            if ratio < cuts.latency_retention_ratio_fail:
                decay_verdict = Verdict.FAIL
            elif ratio < cuts.latency_retention_ratio_warn:
                decay_verdict = Verdict.WARN

    return LatencyResult(
        lags=tuple(range(lag_count + 1)),
        sharpe_by_lag=tuple(sharpe_by_lag),
        position_autocorr=position_autocorr,
        mean_holding_period=holding_period,
        sharpe_standard_error=standard_error,
        retention_1bar=retention,
        expected_retention_1bar=expected_retention,
        retention_ratio=ratio,
        n_obs=window,
        verdict=worst_verdict(level_verdict, decay_verdict),
    )


# --------------------------------------------------------------------------------------
# cost_sweep
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CostSweepRow:
    """One point on the cost grid."""

    bps: float
    mean_net_return: float
    net_sharpe: float


@dataclass(frozen=True)
class CostSweepResult:
    """Where the edge goes as costs rise, and the cost level at which it is gone."""

    rows: tuple[CostSweepRow, ...]
    break_even_bps: float
    mean_gross_return: float
    mean_turnover: float
    total_turnover: float
    n_obs: int
    verdict: Verdict


def cost_sweep(
    prices: pd.Series,
    positions: pd.Series,
    *,
    bps_grid: Sequence[float],
    periods_per_year: float,
    thresholds: Thresholds | None = None,
) -> CostSweepResult:
    """Mean net return and net Sharpe across a grid of per-unit-turnover costs.

    The break-even cost is closed form, not interpolated off the grid:

        break_even_bps = 1e4 * mean(gross_return) / mean(turnover)

    which follows directly from `net = gross - (bps / 1e4) * turnover`, so the mean net
    return is exactly zero at that cost whatever the grid happens to contain. The grid
    exists to show the shape of the decay, not to locate the crossing.

    Mean net return is monotone non-increasing in bps by construction. The net SHARPE is
    not: costs shift the mean down while also changing the dispersion, so a Sharpe that
    ticks up at one grid point is arithmetic, not an anomaly.

    `bps_grid` is sorted ascending before evaluation.
    """
    streams = build_returns(prices, positions, costs_bps=0.0, periods_per_year=periods_per_year)
    return _cost_sweep_from_streams(streams, bps_grid=bps_grid, thresholds=thresholds)


def _cost_sweep_from_streams(
    streams: ReturnStreams,
    *,
    bps_grid: Sequence[float],
    thresholds: Thresholds | None = None,
) -> CostSweepResult:
    """`cost_sweep` over streams already built.

    Only the gross and turnover series are read, and neither depends on the cost the
    streams were built with, so a composed report can pass the streams it already has.
    """
    cuts = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    grid = [float(b) for b in bps_grid]
    if not grid:
        raise QuackzInputError("bps_grid must contain at least one cost level")
    for value in grid:
        if not math.isfinite(value) or value < 0.0:
            raise QuackzInputError(
                f"every entry of bps_grid must be a non-negative finite number, got {value!r}"
            )
    grid.sort()

    gross = streams.gross.to_numpy()
    traded = streams.turnover.to_numpy()
    ppy = streams.periods_per_year

    mean_gross = float(gross.mean())
    mean_turnover = float(traded.mean())
    if mean_turnover > 0.0:
        break_even = 1e4 * mean_gross / mean_turnover
    else:
        # Nothing was ever traded, so no cost per unit of turnover can consume the edge.
        break_even = math.inf if mean_gross > 0.0 else 0.0

    rows = []
    for bps in grid:
        net = gross - (bps / 1e4) * traded
        rows.append(
            CostSweepRow(
                bps=bps,
                mean_net_return=float(net.mean()),
                net_sharpe=metrics.sharpe(net, periods_per_year=ppy),
            )
        )

    if break_even < cuts.cost_break_even_bps_fail:
        verdict = Verdict.FAIL
    elif break_even < cuts.cost_break_even_bps_warn:
        verdict = Verdict.WARN
    else:
        verdict = Verdict.PASS

    return CostSweepResult(
        rows=tuple(rows),
        break_even_bps=break_even,
        mean_gross_return=mean_gross,
        mean_turnover=mean_turnover,
        total_turnover=float(traded.sum()),
        n_obs=streams.n_obs,
        verdict=verdict,
    )


# --------------------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapBlockResult:
    """Resampling results at one mean block length."""

    block_length: int
    sharpe_percentiles: tuple[float, ...]
    max_drawdown_percentiles: tuple[float, ...]
    p_value: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class BootstrapResult:
    """Stationary bootstrap of the realised return stream, across a grid of block lengths."""

    blocks: tuple[BootstrapBlockResult, ...]
    percentiles: tuple[float, ...]
    observed_sharpe: float
    observed_max_drawdown: float
    hac_standard_error: float
    max_p_value: float
    confidence: float
    n_resamples: int
    n_obs: int
    seed: int
    periods_per_year: float
    verdict: Verdict


def _stationary_bootstrap_indices(
    n_obs: int, *, block_length: int, n_resamples: int, rng: np.random.Generator
) -> np.ndarray:
    """Politis and Romano (1994) stationary bootstrap index matrix, shape (n_resamples, n_obs).

    Each step continues the current block with probability 1 - 1/L and starts a new block
    at a uniformly drawn position otherwise, so block lengths are geometric with mean L.
    Continuation wraps around the end of the sample, which is what makes the resampled
    series stationary.
    """
    probability_new_block = 1.0 / block_length
    indices = np.empty((n_resamples, n_obs), dtype=np.int64)
    indices[:, 0] = rng.integers(0, n_obs, size=n_resamples)
    starts_new = rng.random((n_resamples, n_obs - 1)) < probability_new_block
    fresh_starts = rng.integers(0, n_obs, size=(n_resamples, n_obs - 1))
    for step in range(1, n_obs):
        continued = (indices[:, step - 1] + 1) % n_obs
        indices[:, step] = np.where(starts_new[:, step - 1], fresh_starts[:, step - 1], continued)
    return indices


def _negligible_dispersion(paths: np.ndarray, spread: np.ndarray) -> np.ndarray:
    """True where a row's spread is rounding dust relative to the level of its own data.

    A resample of a strategy that trades rarely can come out with every bar identical.
    Testing that against an exact zero is not enough: subtracting the mean from identical
    values leaves dust of order 1e-18, and dividing by it manufactures a Sharpe of 1e16
    out of nothing. This is the same relative test `quackz.metrics` applies to a constant
    series, and it must stay the same one so a resample and the original sample are judged
    degenerate under one rule.
    """
    level = np.abs(paths).mean(axis=1)
    return spread <= 1e-14 * np.maximum(level, 1e-300)


def _sharpe_rows(paths: np.ndarray) -> np.ndarray:
    """Per-period Sharpe of every row, sample standard deviation (ddof=1)."""
    deviation = paths.std(axis=1, ddof=1)
    deviation = np.where(_negligible_dispersion(paths, deviation), 0.0, deviation)
    return _safe_divide(paths.mean(axis=1), deviation)


def _max_drawdown_rows(paths: np.ndarray) -> np.ndarray:
    """Maximum drawdown of every row, as a negative fraction.

    The leading 1.0 is kept in the equity curve for the same reason as in
    `quackz.metrics.max_drawdown`: without it a fall on the first bar has no prior peak and
    reports a drawdown of zero.
    """
    if (paths <= -1.0).any():
        raise QuackzInputError(
            "a resampled path contains a return at or below -1; the compounded curve is "
            "undefined beyond that point"
        )
    equity = np.cumprod(1.0 + paths, axis=1)
    equity = np.concatenate([np.ones((paths.shape[0], 1)), equity], axis=1)
    peak = np.maximum.accumulate(equity, axis=1)
    return (equity / peak - 1.0).min(axis=1)


def _sharpe_hac_standard_error_rows(paths: np.ndarray, *, lags: int) -> np.ndarray:
    """HAC standard error of the per-period Sharpe of every row (Ledoit and Wolf 2008).

    Delta method on the pair (mean, second raw moment) with a Bartlett-kernel long-run
    covariance, which is what makes the interval valid under serial correlation and
    non-normal tails. With `lags=0` this reduces exactly to the Bailey and Lopez de Prado
    variance term 1 - skew*SR + ((kurt - 1) / 4) * SR**2, divided by the sample size; the
    test suite pins that identity, and it is the reason the bootstrap interval and the PSR
    in the same report cannot contradict each other.
    """
    n_obs = paths.shape[1]
    mean = paths.mean(axis=1)
    second_moment = (paths**2).mean(axis=1)

    dev_mean = paths - mean[:, None]
    dev_second = paths**2 - second_moment[:, None]
    s11 = (dev_mean * dev_mean).sum(axis=1) / n_obs
    s12 = (dev_mean * dev_second).sum(axis=1) / n_obs
    s22 = (dev_second * dev_second).sum(axis=1) / n_obs

    # The population variance, taken from the deviations rather than from
    # second_moment - mean**2, which loses every significant digit on a flat row. Read
    # before the lag loop starts adding autocovariances to s11.
    variance = np.where(_negligible_dispersion(paths, np.sqrt(s11)), 0.0, s11)
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        s11 += 2.0 * weight * (dev_mean[:, lag:] * dev_mean[:, :-lag]).sum(axis=1) / n_obs
        s22 += 2.0 * weight * (dev_second[:, lag:] * dev_second[:, :-lag]).sum(axis=1) / n_obs
        cross = (dev_mean[:, lag:] * dev_second[:, :-lag]).sum(axis=1) + (
            dev_second[:, lag:] * dev_mean[:, :-lag]
        ).sum(axis=1)
        s12 += weight * cross / n_obs

    scale = _safe_divide(np.ones_like(variance), variance**1.5)
    grad_mean = second_moment * scale
    grad_second = -0.5 * mean * scale
    long_run = grad_mean**2 * s11 + 2.0 * grad_mean * grad_second * s12 + grad_second**2 * s22
    # The Bartlett kernel keeps the long-run covariance positive semidefinite, so a
    # negative value here is floating point noise around zero.
    return np.sqrt(np.maximum(long_run, 0.0) / n_obs)


def bootstrap(
    returns: npt.ArrayLike,
    *,
    periods_per_year: float,
    n_resamples: int = 1000,
    block_lengths: Sequence[int] = (5, 20, 60),
    seed: int,
    thresholds: Thresholds | None = None,
) -> BootstrapResult:
    """Stationary bootstrap of a realised return stream across a grid of mean block lengths.

    For each block length it reports percentiles of the resampled Sharpe and of the
    resampled maximum drawdown, plus two inferential quantities.

    The p-value is NULL IMPOSED: the returns are recentred to a mean of exactly zero, the
    recentred series is resampled, and the reported value is the fraction of resamples
    whose Sharpe reaches the observed one. That is a test of the hypothesis that the true
    mean is zero. It is not "the probability that the Sharpe is negative", which is not a
    quantity a frequentist bootstrap produces, and a p-value of 0.0 means "below
    1 / n_resamples", not zero.

    The interval is STUDENTIZED (bootstrap-t) rather than a percentile interval, following
    Ledoit and Wolf (2008): each resample contributes (SR* - SR_hat) / SE*, with SE* a HAC
    standard error computed on that resample, and the quantiles of that pivot are applied
    to the HAC standard error of the original sample. A percentile interval on a Sharpe is
    badly calibrated under serial correlation; this one is not.

    The verdict uses the LEAST favourable p-value across the block lengths, so that picking
    a flattering block length cannot change the answer.

    Two limits worth stating plainly. Resampling probes the sampling noise in the realised
    P&L; it says nothing about how the strategy would have behaved on a different price
    path, because every resample is built from bars this strategy actually traded. And a
    block bootstrap destroys dependence beyond the block length, so the drawdown
    percentiles UNDERSTATE the tail: the worst drawdowns in real markets come from runs
    longer than any block.

    All reported Sharpe figures are annualized by sqrt(periods_per_year).
    """
    cuts = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    arr = _finite_array(returns, name="returns", min_len=10)
    # Checked here, against the caller's own series, before any resample is built from it.
    # Every resample is bars of `arr` rearranged, never a new value, so a total loss found
    # only after resampling would blame "a resampled path" for a bar the caller supplied,
    # naming neither the input nor where in it the bad bar sits.
    bad = np.flatnonzero(arr <= -1.0)
    if bad.size:
        raise QuackzInputError(
            f"returns contains a value at or below -1 (total loss) at index {int(bad[0])}; "
            "the compounded curve is undefined beyond that point"
        )
    ppy = _check_periods_per_year(periods_per_year)
    n_obs = int(arr.size)
    resamples = _check_int(n_resamples, name="n_resamples", minimum=2)
    seed_value = _check_int(seed, name="seed", minimum=0)

    blocks = [_check_int(b, name="block_lengths entry", minimum=1) for b in block_lengths]
    if not blocks:
        raise QuackzInputError("block_lengths must contain at least one block length")
    for block in blocks:
        if block > n_obs:
            raise QuackzInputError(
                f"block length {block} exceeds the {n_obs} observations available; a block "
                "longer than the sample makes every resample a circular shift of it"
            )

    if float(arr.std(ddof=1)) == 0.0:
        raise QuackzInputError(
            "returns is constant, so every resample of it is identical and the bootstrap "
            "carries no information"
        )

    lags = min(_newey_west_lags(n_obs), n_obs - 1)
    annualize = math.sqrt(ppy)
    observed_sharpe = float(arr.mean() / arr.std(ddof=1))
    standard_error = float(_sharpe_hac_standard_error_rows(arr[None, :], lags=lags)[0])
    tail = (1.0 - BOOTSTRAP_CI_CONFIDENCE) / 2.0

    centred = arr - arr.mean()
    rng = np.random.default_rng(seed_value)
    block_results = []
    for block in blocks:
        indices = _stationary_bootstrap_indices(
            n_obs, block_length=block, n_resamples=resamples, rng=rng
        )
        paths = arr[indices]
        sharpes = _sharpe_rows(paths)
        drawdowns = _max_drawdown_rows(paths)
        errors = _sharpe_hac_standard_error_rows(paths, lags=lags)
        pivot = _safe_divide(sharpes - observed_sharpe, errors)
        low_quantile, high_quantile = np.percentile(pivot, [100.0 * tail, 100.0 * (1.0 - tail)])

        null_indices = _stationary_bootstrap_indices(
            n_obs, block_length=block, n_resamples=resamples, rng=rng
        )
        null_sharpes = _sharpe_rows(centred[null_indices])

        block_results.append(
            BootstrapBlockResult(
                block_length=block,
                sharpe_percentiles=tuple(
                    float(x) * annualize for x in np.percentile(sharpes, BOOTSTRAP_PERCENTILES)
                ),
                max_drawdown_percentiles=tuple(
                    float(x) for x in np.percentile(drawdowns, BOOTSTRAP_PERCENTILES)
                ),
                p_value=float(np.mean(null_sharpes >= observed_sharpe)),
                ci_low=(observed_sharpe - float(high_quantile) * standard_error) * annualize,
                ci_high=(observed_sharpe - float(low_quantile) * standard_error) * annualize,
            )
        )

    max_p_value = max(block.p_value for block in block_results)
    if max_p_value > cuts.bootstrap_p_value_fail:
        verdict = Verdict.FAIL
    elif max_p_value > cuts.bootstrap_p_value_warn:
        verdict = Verdict.WARN
    else:
        verdict = Verdict.PASS

    return BootstrapResult(
        blocks=tuple(block_results),
        percentiles=BOOTSTRAP_PERCENTILES,
        observed_sharpe=observed_sharpe * annualize,
        observed_max_drawdown=float(_max_drawdown_rows(arr[None, :])[0]),
        hac_standard_error=standard_error * annualize,
        max_p_value=max_p_value,
        confidence=BOOTSTRAP_CI_CONFIDENCE,
        n_resamples=resamples,
        n_obs=n_obs,
        seed=seed_value,
        periods_per_year=ppy,
        verdict=verdict,
    )


# --------------------------------------------------------------------------------------
# noise_floor
# --------------------------------------------------------------------------------------

_NOISE_FLOOR_IID_NOTE = (
    "V[{SR_n}] was not supplied, so the iid-normal null variance 1/n_obs was used. Where "
    "the configurations differed genuinely their Sharpes scatter wider than that and the "
    "true floor is higher; where the search was a fine grid over one strategy they scatter "
    "narrower and it is lower. Pass the Sharpe of every configuration the search touched "
    "to replace the guess with the measurement."
)


@dataclass(frozen=True)
class NoiseFloorResult:
    """The Sharpe a search of this size expects to find in pure noise."""

    per_period: float
    annualized: float
    n_trials: int
    n_obs: int
    var_trial_sharpes: float
    variance_source: str
    periods_per_year: float
    note: str | None


def noise_floor(
    *,
    n_trials: int,
    n_obs: int,
    periods_per_year: float,
    var_trial_sharpes: float | None = None,
) -> NoiseFloorResult:
    """Expected maximum Sharpe across `n_trials` trials of a strategy with no edge.

    This is Bailey and Lopez de Prado (2014) Eq. 1, the same extreme-value expectation the
    deflated Sharpe uses as its benchmark, so the floor printed in a report and the
    deflation applied in the same report cannot disagree. A crude sqrt(2 ln N) asymptotic
    would be easier to write and would overstate this number by 15 to 40 percent at
    realistic N, which is exactly the size of the effect being measured.

    Returned at both frequencies: `per_period` is the raw expectation and `annualized` is
    `per_period * sqrt(periods_per_year)`.

    This result carries no verdict, because the floor is a property of the SEARCH, not of
    the strategy. The comparison that matters, the observed Sharpe against this floor,
    needs both and is made by the caller.
    """
    trials = _check_int(n_trials, name="n_trials", minimum=1)
    observations = _check_int(n_obs, name="n_obs", minimum=2)
    ppy = _check_periods_per_year(periods_per_year)

    if var_trial_sharpes is None:
        variance = 1.0 / observations
        source = "iid_fallback"
        note: str | None = _NOISE_FLOOR_IID_NOTE
    else:
        variance = float(var_trial_sharpes)
        if not math.isfinite(variance) or variance < 0.0:
            raise QuackzInputError(
                f"var_trial_sharpes must be a non-negative finite number, got {var_trial_sharpes!r}"
            )
        source = "var_trial_sharpes"
        note = None

    per_period = metrics.expected_max_sharpe(n_trials=trials, var_trial_sharpes=variance)
    return NoiseFloorResult(
        per_period=per_period,
        annualized=per_period * math.sqrt(ppy),
        n_trials=trials,
        n_obs=observations,
        var_trial_sharpes=variance,
        variance_source=source,
        periods_per_year=ppy,
        note=note,
    )


# --------------------------------------------------------------------------------------
# subperiod_stability
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SubperiodStabilityResult:
    """Sharpe window by window, judged against the dispersion sampling noise alone explains."""

    window_sharpes: tuple[float, ...]
    window_n_obs: tuple[int, ...]
    worst_window_sharpe: float
    sharpe_dispersion: float
    expected_sharpe_se: float
    dispersion_ratio: float
    full_sample_sharpe: float
    n_splits: int
    verdict: Verdict


def subperiod_stability(
    prices: pd.Series,
    positions: pd.Series,
    *,
    n_splits: int,
    periods_per_year: float,
    costs_bps: float = 0.0,
    thresholds: Thresholds | None = None,
) -> SubperiodStabilityResult:
    """Sharpe of one FIXED signal measured over contiguous, equal-sized windows.

    This measures temporal stability: whether the same rule, unchanged, paid at a similar
    rate in the first part of the sample as in the last. It is not a validation procedure.
    Nothing is refitted, no parameter is chosen on one window and tested on another, and
    passing this check says nothing about how the rule was arrived at. For splitters that
    keep training data strictly in the past, see `quackz.splits`.

    The verdict is noise aware, which matters more here than anywhere else in the library.
    A Sharpe measured on n bars has a standard error of roughly sqrt((1 + SR**2 / 2) / n),
    so with five windows of two hundred bars the window Sharpes of a perfectly stable
    strategy still scatter by around 1.1 annualized. Flagging that scatter as instability
    would fail almost every honest strategy. What is reported is `dispersion_ratio`, the
    observed dispersion divided by the dispersion predicted by that standard error, and
    only a ratio meaningfully above 1 is evidence of anything.

    The standard error is the iid Lo (2002) form. A Newey-West estimator sits one module
    away in `quackz.metrics`, and under positive autocorrelation it would predict a wider
    scatter than this one does, so the ratio reported here is biased towards WARN for a
    strategy whose returns are serially correlated. Read it beside `autocorr_1` and the
    Newey-West t-statistic in the same report.
    """
    streams = build_returns(
        prices,
        positions,
        costs_bps=costs_bps,
        periods_per_year=periods_per_year,
    )
    return _subperiod_stability_from_streams(streams, n_splits=n_splits, thresholds=thresholds)


def _subperiod_stability_from_streams(
    streams: ReturnStreams,
    *,
    n_splits: int,
    thresholds: Thresholds | None = None,
) -> SubperiodStabilityResult:
    """`subperiod_stability` over streams already built, windowing their net series."""
    cuts = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    splits = _check_int(n_splits, name="n_splits", minimum=2)
    arr = streams.net.to_numpy()
    ppy = streams.periods_per_year
    if arr.size < 2 * splits:
        raise QuackzInputError(
            f"n_splits={splits} needs at least {2 * splits} return bars for a Sharpe in every "
            f"window, got {arr.size}"
        )

    windows = np.array_split(arr, splits)
    window_sharpes = tuple(metrics.sharpe(w, periods_per_year=ppy) for w in windows)
    if not all(math.isfinite(s) for s in window_sharpes):
        raise QuackzInputError(
            "at least one window has no return dispersion at all, so its Sharpe is infinite; "
            "the strategy is flat over part of the sample and cannot be judged this way"
        )

    full_sample_sharpe = metrics.sharpe(arr, periods_per_year=ppy)
    sr_per_period = full_sample_sharpe / math.sqrt(ppy)
    # Lo (2002): the sampling variance of a Sharpe on n iid bars. Averaged over the
    # windows because array_split leaves them differing in length by at most one bar.
    expected_variance = float(np.mean([(1.0 + sr_per_period**2 / 2.0) / w.size for w in windows]))
    expected_se = math.sqrt(expected_variance * ppy)
    dispersion = float(np.std(window_sharpes, ddof=1))
    ratio = dispersion / expected_se

    if ratio >= cuts.stability_dispersion_ratio_fail:
        verdict = Verdict.FAIL
    elif ratio >= cuts.stability_dispersion_ratio_warn:
        verdict = Verdict.WARN
    else:
        verdict = Verdict.PASS

    return SubperiodStabilityResult(
        window_sharpes=window_sharpes,
        window_n_obs=tuple(int(w.size) for w in windows),
        worst_window_sharpe=min(window_sharpes),
        sharpe_dispersion=dispersion,
        expected_sharpe_se=expected_se,
        dispersion_ratio=ratio,
        full_sample_sharpe=full_sample_sharpe,
        n_splits=splits,
        verdict=verdict,
    )


# --------------------------------------------------------------------------------------
# concentration
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ConcentrationRow:
    """The track record with its k best bars removed."""

    drop_top: int
    sharpe: float
    share_of_gross_profit: float


@dataclass(frozen=True)
class ConcentrationResult:
    """How much of the result rests on a handful of bars, and what it cost to earn."""

    baseline_sharpe: float
    rows: tuple[ConcentrationRow, ...]
    total_turnover: float | None
    edge_per_turnover_bps: float | None
    n_obs: int
    verdict: Verdict


def concentration(
    returns: npt.ArrayLike,
    *,
    drop_top: Sequence[int] = (1, 5, 10),
    periods_per_year: float,
    turnover_series: npt.ArrayLike | None = None,
    thresholds: Thresholds | None = None,
) -> ConcentrationResult:
    """Sharpe recomputed with the k best bars removed, plus the edge earned per unit traded.

    A Sharpe that survives losing its ten best bars out of a thousand describes a process.
    One that collapses describes a few events, and the annualized number attached to it is
    not an estimate of anything repeatable.

    `share_of_gross_profit` is the fraction of the sum of PROFITABLE bars contributed by
    the k best. Measuring against gross profit rather than net profit keeps the quantity
    inside [0, 1] and well defined for a strategy that lost money overall, where a share of
    the net total would be meaningless or negative.

    Pass `turnover_series` (the series from `quackz.returns.turnover`, aligned to these
    returns) to also get the total traded notional and `edge_per_turnover_bps`, which is
    `1e4 * mean(return) / mean(turnover)`: the basis points earned per unit of notional
    turned over.

    Which stream you hand it decides what that number means, and the difference is exactly
    `costs_bps`. Handed GROSS returns it is identical to the break-even cost from
    `cost_sweep`, the figure to compare against a broker's quote. Handed the NET stream, as
    `quackz.evaluate` does so that the rows above describe the track record actually being
    claimed, it is that same figure with the cost already charged taken out, and comparing
    it against a broker's quote would charge the cost twice.
    """
    cuts = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    arr = _finite_array(returns, name="returns")
    ppy = _check_periods_per_year(periods_per_year)
    n_obs = int(arr.size)

    counts = [_check_int(k, name="drop_top entry", minimum=1) for k in drop_top]
    if not counts:
        raise QuackzInputError("drop_top must contain at least one count")
    for k in counts:
        if k > n_obs - 2:
            raise QuackzInputError(
                f"dropping the top {k} of {n_obs} bars leaves fewer than 2, which is not "
                "enough for a Sharpe"
            )
    counts.sort()

    ranked = np.argsort(arr, kind="stable")[::-1]
    gross_profit = float(arr[arr > 0.0].sum())
    rows = []
    for k in counts:
        keep = np.ones(n_obs, dtype=bool)
        keep[ranked[:k]] = False
        top_sum = float(arr[ranked[:k]].sum())
        rows.append(
            ConcentrationRow(
                drop_top=k,
                sharpe=metrics.sharpe(arr[keep], periods_per_year=ppy),
                share_of_gross_profit=top_sum / gross_profit if gross_profit > 0.0 else 0.0,
            )
        )

    total_turnover: float | None = None
    edge_per_turnover_bps: float | None = None
    if turnover_series is not None:
        traded = _finite_array(turnover_series, name="turnover_series")
        if traded.size != n_obs:
            raise QuackzInputError(
                f"turnover_series has {traded.size} observations and returns has {n_obs}; "
                "they must cover the same bars"
            )
        if (traded < 0.0).any():
            raise QuackzInputError("turnover_series must be non-negative")
        total_turnover = float(traded.sum())
        mean_turnover = float(traded.mean())
        if mean_turnover > 0.0:
            edge_per_turnover_bps = 1e4 * float(arr.mean()) / mean_turnover
        else:
            edge_per_turnover_bps = math.inf if float(arr.mean()) > 0.0 else 0.0

    worst_share = rows[-1].share_of_gross_profit
    if worst_share > cuts.concentration_profit_share_fail:
        verdict = Verdict.FAIL
    elif worst_share > cuts.concentration_profit_share_warn:
        verdict = Verdict.WARN
    else:
        verdict = Verdict.PASS

    return ConcentrationResult(
        baseline_sharpe=metrics.sharpe(arr, periods_per_year=ppy),
        rows=tuple(rows),
        total_turnover=total_turnover,
        edge_per_turnover_bps=edge_per_turnover_bps,
        n_obs=n_obs,
        verdict=verdict,
    )
