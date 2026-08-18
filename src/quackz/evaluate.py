"""One call that runs every check and returns a single frozen result.

`evaluate` is the composition layer. It validates the inputs and builds the return streams
exactly once and hands that one object to every check, so no two checks can disagree about
what the strategy earned, and it forms the two verdicts that need quantities from more than
one place: the deflated Sharpe, and the observed Sharpe against the noise floor of the
declared search.

Each check in `quackz.checks` also stands alone, taking prices and positions and building
what it needs for itself, because a check that can only run inside a report is a check
nobody reaches for. The composed path takes the `_from_streams` form of the three that
would otherwise rebuild: same arithmetic, one validation pass rather than four.

Frequency, stated once because it is the easiest thing to get wrong. Everything crossing
this module's boundary is ANNUALIZED: `sharpe`, `trial_sharpes`, `var_trial_sharpes`. The
statistical machinery in `quackz.metrics` works at the observation frequency, and the
conversion happens here, in the open: a Sharpe is divided by sqrt(periods_per_year) and a
variance of Sharpes by periods_per_year. This mirrors the way the Bailey and Lopez de Prado
worked examples are presented, and it means a caller who computed trial Sharpes with
`metrics.sharpe(..., periods_per_year=252)` can pass them straight in.

What an Evaluation is not. It is a report, not a container for the data: the return series
is deliberately absent, because it is one `quackz.returns.build_returns` call away and
carrying a few thousand floats through every serialisation would make the JSON output
useless. Every field here is a plain float, int, str or tuple of them.

Causality. The one-bar shift proves that THIS library's arithmetic is causal. It proves
nothing about the positions handed in: a signal built from information that was not
available at the decision time will pass every check below, because none of them can see
how the position was constructed. That limit is not fixable from inside a function that
receives only prices and positions, and it is stated in the report as well as here.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, replace

import numpy as np
import numpy.typing as npt
import pandas as pd

from quackz import checks, metrics
from quackz._version import __version__
from quackz.checks import (
    DEFAULT_THRESHOLDS,
    BootstrapResult,
    ConcentrationResult,
    CostSweepResult,
    LatencyResult,
    ReconcileResult,
    SubperiodStabilityResult,
    Thresholds,
    Verdict,
    worst_verdict,
)
from quackz.returns import QuackzInputError, build_returns

__all__ = [
    "BOOTSTRAP_BLOCK_LENGTHS",
    "CONCENTRATION_DROP_TOP",
    "COST_BPS_GRID",
    "DSR_BREAK_EVEN_CAP",
    "DSR_TRIAL_GRID",
    "LATENCY_MAX_LAG",
    "MIN_OBSERVATIONS",
    "STABILITY_N_SPLITS",
    "DeflatedSharpeCheck",
    "DsrTrialRow",
    "Evaluation",
    "EvaluationChecks",
    "Meta",
    "NoiseFloorCheck",
    "Performance",
    "evaluate",
]

# The composed audit is only as long as its narrowest check allows: the concentration check
# drops the ten best bars and still needs two, the stability check cuts five windows, and
# the bootstrap needs ten observations. Below this the report would be arithmetic about
# nothing, so it is refused rather than produced with a caveat nobody reads.
MIN_OBSERVATIONS = 20

# A three bar delay covers the plausible operational failures (a stale price, a missed
# close, a next-open fill) without leaving so little common window that the Sharpes are noise.
LATENCY_MAX_LAG = 3

# Five windows of a normal sample leave enough bars each for a Sharpe to mean something
# while still resolving a strategy that only worked in one part of the sample.
STABILITY_N_SPLITS = 5

# One bar shows single-event dependence, ten shows whether the record is a process. Both are
# small against the minimum sample, so the dropped bars cannot be a material part of it.
CONCENTRATION_DROP_TOP = (1, 5, 10)

# Geometric mean block lengths spanning a week, a month and a quarter of daily bars. Blocks
# longer than the sample are dropped, since every resample of one is a circular shift.
BOOTSTRAP_BLOCK_LENGTHS = (5, 20, 60)

# Costs from free to punitive, in basis points of turnover. The caller's own cost level is
# merged in, so the table always contains the number the strategy was actually charged.
COST_BPS_GRID = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)

# Trial counts for the deflation table. The declared count is merged in.
DSR_TRIAL_GRID = (1, 10, 50, 100, 500, 1000)

# The break-even trial search stops here. A strategy that still clears the confidence level
# after a million trials is not going to be undone by the millionth and first.
DSR_BREAK_EVEN_CAP = 1_000_000


@dataclass(frozen=True)
class Meta:
    """What the report was computed from, in enough detail to reproduce it."""

    periods_per_year: float
    periods_per_year_inferred: bool
    n_obs: int
    costs_bps: float
    seed: int
    n_trials: int
    n_resamples: int
    first_bar: str
    last_bar: str
    library_version: str
    thresholds: Thresholds


@dataclass(frozen=True)
class Performance:
    """Point estimates, net of the costs supplied, with the gross Sharpe for comparison."""

    sharpe: float
    sharpe_per_period: float
    sharpe_gross: float
    annualized_return: float
    annualized_vol: float
    total_return: float
    sortino: float
    calmar: float
    max_drawdown: float
    ulcer_index: float
    cvar_5pct: float
    skew: float
    kurt_nonexcess: float
    autocorr_1: float
    naive_tstat: float
    hac_tstat: float
    n_obs: int


@dataclass(frozen=True)
class DsrTrialRow:
    """The deflated Sharpe the same track record would carry after a search of `n_trials`."""

    n_trials: int
    expected_max_sharpe: float
    dsr: float


@dataclass(frozen=True)
class DeflatedSharpeCheck:
    """The observed Sharpe deflated by the selection the declared trial count implies."""

    dsr: float
    psr_vs_zero: float
    observed_sharpe: float
    observed_sharpe_per_period: float
    expected_max_sharpe: float
    expected_max_sharpe_per_period: float
    var_trial_sharpes: float
    var_trial_sharpes_per_period: float
    variance_source: str
    n_trials: int
    n_obs: int
    min_track_record_length: float | None
    confidence: float
    break_even_n_trials: int | None
    break_even_capped: bool
    by_trials: tuple[DsrTrialRow, ...] = field(repr=False)
    warning: str | None
    verdict: Verdict


@dataclass(frozen=True)
class NoiseFloorCheck:
    """The observed Sharpe against the best a search of the same size finds in pure noise."""

    observed_sharpe: float
    annualized: float
    per_period: float
    ratio: float
    n_trials: int
    n_obs: int
    var_trial_sharpes: float
    variance_source: str
    note: str | None
    verdict: Verdict


@dataclass(frozen=True)
class EvaluationChecks:
    """Every check, in the order the report prints them. `reconcile` is None when unused."""

    deflated_sharpe: DeflatedSharpeCheck
    noise_floor: NoiseFloorCheck
    cost_sweep: CostSweepResult
    concentration: ConcentrationResult
    bootstrap: BootstrapResult
    subperiod_stability: SubperiodStabilityResult
    latency: LatencyResult
    reconcile: ReconcileResult | None

    def verdicts(self) -> dict[str, Verdict]:
        """Verdict per check name, skipping checks that did not run."""
        out = {}
        for spec in fields(self):
            result = getattr(self, spec.name)
            if result is not None:
                out[spec.name] = result.verdict
        return out


@dataclass(frozen=True)
class Evaluation:
    """The whole audit: point estimates, per-check verdicts, the worst of them, and provenance."""

    metrics: Performance
    checks: EvaluationChecks
    verdict: Verdict
    meta: Meta


def _resolve_thresholds(thresholds: Thresholds | Mapping[str, float] | None) -> Thresholds:
    if thresholds is None:
        return DEFAULT_THRESHOLDS
    if isinstance(thresholds, Thresholds):
        return thresholds
    if isinstance(thresholds, Mapping):
        known = {spec.name for spec in fields(Thresholds)}
        unknown = sorted(set(thresholds) - known)
        if unknown:
            raise QuackzInputError(
                f"unknown threshold name(s): {', '.join(unknown)}. Known names are: "
                f"{', '.join(sorted(known))}"
            )
        return replace(DEFAULT_THRESHOLDS, **dict(thresholds))
    raise QuackzInputError(
        "thresholds must be a Thresholds, a mapping of overrides, or None, got "
        f"{type(thresholds).__name__}"
    )


def _check_n_trials(n_trials: object) -> int:
    if isinstance(n_trials, bool) or not isinstance(n_trials, (int, np.integer)):
        raise QuackzInputError(f"n_trials must be an integer, got {n_trials!r}")
    if int(n_trials) < 1:
        raise QuackzInputError(f"n_trials must be at least 1, got {n_trials}")
    return int(n_trials)


def _align_claimed(
    claimed: npt.ArrayLike, *, index: pd.DatetimeIndex, n_price_bars: int
) -> pd.Series:
    """Put a claimed return stream on the return index, or say precisely why it does not fit."""
    if isinstance(claimed, pd.Series) and isinstance(claimed.index, pd.DatetimeIndex):
        missing = index.difference(claimed.index)
        if len(missing):
            raise QuackzInputError(
                f"claimed_returns is missing {len(missing)} of the {len(index)} return bars "
                f"(first few: {missing[:3].tolist()}). Align it deliberately; the recomputed "
                "stream starts one bar after the price series because of the position shift."
            )
        return claimed.reindex(index).astype(float)

    arr = np.asarray(claimed, dtype=float)
    if arr.ndim != 1:
        raise QuackzInputError(f"claimed_returns must be one dimensional, got shape {arr.shape}")
    if arr.size == len(index):
        return pd.Series(arr, index=index)
    if arr.size == n_price_bars:
        # One value per PRICE bar: the first has no return to correspond to, so it is dropped,
        # exactly as the position shift drops the first bar of the recomputed stream.
        return pd.Series(arr[1:], index=index)
    raise QuackzInputError(
        f"claimed_returns has {arr.size} values, but the sample has {len(index)} return bars "
        f"(from {n_price_bars} price bars). Pass one value per return bar, or one per price "
        "bar and the leading value will be dropped."
    )


def _trial_variance(
    *,
    trial_sharpes: npt.ArrayLike | None,
    var_trial_sharpes: float | None,
    periods_per_year: float,
) -> tuple[np.ndarray | None, float | None]:
    """De-annualize the caller's trial dispersion. Returns (per-period sharpes, per-period var)."""
    if trial_sharpes is not None and var_trial_sharpes is not None:
        raise QuackzInputError(
            "pass either trial_sharpes or var_trial_sharpes, not both; they are two ways to "
            "specify the same quantity and supplying both hides which one was used"
        )
    if trial_sharpes is not None:
        annual = np.asarray(trial_sharpes, dtype=float)
        if annual.ndim != 1:
            raise QuackzInputError(f"trial_sharpes must be one dimensional, got {annual.shape}")
        if annual.size < 2:
            raise QuackzInputError(
                f"trial_sharpes needs at least 2 trials to have a variance, got {annual.size}"
            )
        if not np.isfinite(annual).all():
            raise QuackzInputError("trial_sharpes contains non-finite value(s)")
        return annual / math.sqrt(periods_per_year), None
    if var_trial_sharpes is not None:
        annual_var = float(var_trial_sharpes)
        if not math.isfinite(annual_var) or annual_var < 0.0:
            raise QuackzInputError(
                f"var_trial_sharpes must be a non-negative finite number, got {var_trial_sharpes!r}"
            )
        return None, annual_var / periods_per_year
    return None, None


def _dsr_at(
    n_trials: int, *, sr: float, n_obs: int, skew: float, kurt: float, variance: float
) -> tuple[float, float]:
    """(expected max Sharpe, DSR) at `n_trials`, holding the trial dispersion fixed."""
    sr_0 = metrics.expected_max_sharpe(n_trials=n_trials, var_trial_sharpes=variance)
    dsr = metrics.probabilistic_sharpe(
        sr=sr, sr_benchmark=sr_0, n_obs=n_obs, skew=skew, kurt_nonexcess=kurt
    )
    return sr_0, dsr


def _break_even_trials(
    dsr_for: Callable[[int], float], *, target: float
) -> tuple[int | None, bool]:
    """Largest trial count whose DSR still reaches `target`, by bisection.

    The DSR falls monotonically in the number of trials, since the benchmark it is measured
    against rises with it, so bisection is exact rather than approximate. Returns (count,
    capped); None means the record misses the target even with no selection at all.
    """
    if dsr_for(1) < target:
        return None, False
    if dsr_for(DSR_BREAK_EVEN_CAP) >= target:
        return DSR_BREAK_EVEN_CAP, True
    low, high = 1, DSR_BREAK_EVEN_CAP
    while high - low > 1:
        middle = (low + high) // 2
        if dsr_for(middle) >= target:
            low = middle
        else:
            high = middle
    return low, False


def _dsr_verdict(dsr: float, *, warn: float, fail: float) -> Verdict:
    if dsr < fail:
        return Verdict.FAIL
    if dsr < warn:
        return Verdict.WARN
    return Verdict.PASS


def _noise_floor_verdict(ratio: float, *, warn: float, fail: float) -> Verdict:
    """A Sharpe exactly AT the floor is what the search alone produces, so it fails."""
    if ratio <= fail:
        return Verdict.FAIL
    if ratio < warn:
        return Verdict.WARN
    return Verdict.PASS


def _floor_ratio(observed: float, floor: float) -> float:
    """Observed Sharpe as a multiple of the noise floor, with a floor of zero handled."""
    if floor > 0.0:
        return observed / floor
    # n_trials = 1 declares no selection, so the floor is exactly zero and any positive
    # Sharpe clears it by an unbounded multiple.
    if observed > 0.0:
        return math.inf
    return 0.0 if observed == 0.0 else -math.inf


def evaluate(
    prices: pd.Series,
    positions: pd.Series,
    *,
    costs_bps: float = 0.0,
    periods_per_year: float | None = None,
    n_trials: int = 1,
    trial_sharpes: npt.ArrayLike | None = None,
    var_trial_sharpes: float | None = None,
    claimed_returns: npt.ArrayLike | None = None,
    bootstrap_seed: int = 0,
    thresholds: Thresholds | Mapping[str, float] | None = None,
    bootstrap_resamples: int = 1000,
) -> Evaluation:
    """Run every check against one price and position series and return a single Evaluation.

    `prices` is a close series on a DatetimeIndex; `positions` is the target exposure decided
    AT bar t's close, aligned to it, so it earns bar t+1's return. Neither is mutated.

    `costs_bps` is charged on turnover, `|pos[t-1] - pos[t-2]|`, including the trade into the
    first position and the trade out of the last. Every number in `Evaluation.metrics` is net
    of it except `sharpe_gross`.

    `periods_per_year` is inferred from the index when None, and the value used is recorded
    in `meta` and printed by the report. `n_trials` is the number of configurations the
    search touched; it is a DECLARATION, and a search that is not disclosed cannot be
    deflated. `trial_sharpes` (ANNUALIZED, one per configuration) or `var_trial_sharpes`
    (ANNUALIZED variance of them) supplies the dispersion the deflation needs. With neither,
    the iid-normal fallback is used and the result carries a warning saying it understates
    the deflation.

    `claimed_returns` is an optional per-period return stream from the caller's own backtest,
    reconciled against the NET stream recomputed here. If the claimed stream is gross of
    costs, evaluate with `costs_bps=0.0` or the reconciliation compares two different things.
    It may be given per return bar, or per price bar, in which case the leading value is
    dropped.

    `thresholds` overrides any grading cut-off, as a `quackz.checks.Thresholds` or a mapping
    of field names to values. `bootstrap_resamples` is the only tuning knob the CLI exposes;
    the remaining shapes (cost grid, block lengths, window count) are module constants with
    their reasoning beside them.

    What passing does not establish. Every check reads the positions as given, so a signal
    built with information that was not available at the decision time will pass all of them.
    Survivorship and point-in-time universe construction are properties of the data and are
    out of scope entirely.
    """
    cuts = _resolve_thresholds(thresholds)
    trials = _check_n_trials(n_trials)

    streams = build_returns(
        prices, positions, costs_bps=costs_bps, periods_per_year=periods_per_year
    )
    n_obs = streams.n_obs
    if n_obs < MIN_OBSERVATIONS:
        raise QuackzInputError(
            f"the composed audit needs at least {MIN_OBSERVATIONS} return bars, got {n_obs}. "
            "The concentration check drops the ten best bars, the stability check cuts five "
            "windows and the bootstrap needs ten observations; below that the report would "
            "be arithmetic about nothing. Run the individual checks in quackz.checks instead."
        )

    net = streams.net.to_numpy()
    gross = streams.gross.to_numpy()
    ppy = streams.periods_per_year
    root_ppy = math.sqrt(ppy)

    # Said here, once, in the library's own error type. Otherwise the first estimator to
    # divide by a dispersion of zero raises on behalf of all of them, and what reaches the
    # caller is a sentence about skewness rather than about their position column.
    if metrics._has_no_dispersion(net):
        raise QuackzInputError(
            "the net return stream is constant, so there is nothing here to measure: a "
            "Sharpe, a skewness and a bootstrap all need dispersion. A position series "
            "that never leaves zero earns exactly this, and so does a price series that "
            "never moves. Individual checks in quackz.checks still accept a flat window."
        )

    skew, kurt = metrics.moments(net)
    sharpe_annual = metrics.sharpe(net, periods_per_year=ppy)
    sharpe_per_period = sharpe_annual / root_ppy
    annual_return = metrics.annualized_return(net, periods_per_year=ppy)
    performance = Performance(
        sharpe=sharpe_annual,
        sharpe_per_period=sharpe_per_period,
        sharpe_gross=metrics.sharpe(gross, periods_per_year=ppy),
        annualized_return=annual_return,
        annualized_vol=metrics.annualized_vol(net, periods_per_year=ppy),
        total_return=float(np.expm1(np.log1p(net).sum())),
        sortino=metrics.sortino(net, periods_per_year=ppy),
        calmar=metrics.calmar(net, periods_per_year=ppy),
        max_drawdown=metrics.max_drawdown(net),
        ulcer_index=metrics.ulcer_index(net),
        cvar_5pct=metrics.cvar(net, alpha=0.05),
        skew=skew,
        kurt_nonexcess=kurt,
        autocorr_1=metrics.autocorr_1(net),
        naive_tstat=sharpe_per_period * math.sqrt(n_obs),
        hac_tstat=metrics.hac_tstat(net),
        n_obs=n_obs,
    )

    trials_per_period, variance_per_period = _trial_variance(
        trial_sharpes=trial_sharpes,
        var_trial_sharpes=var_trial_sharpes,
        periods_per_year=ppy,
    )
    dsr_result = metrics.deflated_sharpe(
        sr=sharpe_per_period,
        n_trials=trials,
        n_obs=n_obs,
        skew=skew,
        kurt_nonexcess=kurt,
        trial_sharpes=trials_per_period,
        var_trial_sharpes=variance_per_period,
    )
    variance_used = dsr_result.var_trial_sharpes

    def deflation_at(count: int) -> tuple[float, float]:
        return _dsr_at(
            count,
            sr=sharpe_per_period,
            n_obs=n_obs,
            skew=skew,
            kurt=kurt,
            variance=variance_used,
        )

    by_trials = []
    for count in sorted({*DSR_TRIAL_GRID, trials}):
        benchmark, deflated_value = deflation_at(count)
        by_trials.append(
            DsrTrialRow(
                n_trials=count, expected_max_sharpe=benchmark * root_ppy, dsr=deflated_value
            )
        )
    break_even, capped = _break_even_trials(
        lambda count: deflation_at(count)[1], target=cuts.dsr_warn
    )

    min_trl: float | None = None
    if sharpe_per_period > 0.0:
        min_trl = metrics.min_track_record_length(
            sr=sharpe_per_period,
            sr_benchmark=0.0,
            skew=skew,
            kurt_nonexcess=kurt,
            confidence=cuts.dsr_warn,
        )
    deflated = DeflatedSharpeCheck(
        dsr=dsr_result.dsr,
        psr_vs_zero=metrics.probabilistic_sharpe(
            sr=sharpe_per_period, sr_benchmark=0.0, n_obs=n_obs, skew=skew, kurt_nonexcess=kurt
        ),
        observed_sharpe=sharpe_annual,
        observed_sharpe_per_period=sharpe_per_period,
        expected_max_sharpe=dsr_result.expected_max_sharpe * root_ppy,
        expected_max_sharpe_per_period=dsr_result.expected_max_sharpe,
        var_trial_sharpes=variance_used * ppy,
        var_trial_sharpes_per_period=variance_used,
        variance_source=dsr_result.variance_source,
        n_trials=trials,
        n_obs=n_obs,
        min_track_record_length=min_trl,
        confidence=cuts.dsr_warn,
        break_even_n_trials=break_even,
        break_even_capped=capped,
        by_trials=tuple(by_trials),
        warning=dsr_result.warning,
        verdict=_dsr_verdict(dsr_result.dsr, warn=cuts.dsr_warn, fail=cuts.dsr_fail),
    )

    # The floor and the deflation must be built from the same dispersion or the report
    # contradicts itself, so the fallback is passed through as a fallback rather than as a
    # number: both then land on 1 / n_obs and the floor keeps its note about it.
    floor = checks.noise_floor(
        n_trials=trials,
        n_obs=n_obs,
        periods_per_year=ppy,
        var_trial_sharpes=None if dsr_result.variance_source == "iid_fallback" else variance_used,
    )
    ratio = _floor_ratio(sharpe_annual, floor.annualized)
    noise_floor = NoiseFloorCheck(
        observed_sharpe=sharpe_annual,
        annualized=floor.annualized,
        per_period=floor.per_period,
        ratio=ratio,
        n_trials=trials,
        n_obs=n_obs,
        var_trial_sharpes=floor.var_trial_sharpes * ppy,
        # The dispersion was decided once, where the deflation was formed. Taking the label
        # from the floor instead would print "var_trial_sharpes" beside a number the caller
        # supplied as trial Sharpes, and the two lines of the report would disagree about
        # where the same quantity came from.
        variance_source=dsr_result.variance_source,
        note=floor.note,
        verdict=_noise_floor_verdict(
            ratio, warn=cuts.noise_floor_ratio_warn, fail=cuts.noise_floor_ratio_fail
        ),
    )

    blocks = tuple(b for b in BOOTSTRAP_BLOCK_LENGTHS if b <= n_obs)
    reconciled = None
    if claimed_returns is not None:
        reconciled = checks.reconcile(
            claimed=_align_claimed(
                claimed_returns, index=streams.net.index, n_price_bars=n_obs + 1
            ),
            recomputed=streams.net,
            thresholds=cuts,
        )

    evaluation_checks = EvaluationChecks(
        deflated_sharpe=deflated,
        noise_floor=noise_floor,
        cost_sweep=checks._cost_sweep_from_streams(
            streams,
            bps_grid=sorted({*COST_BPS_GRID, streams.costs_bps}),
            thresholds=cuts,
        ),
        concentration=checks.concentration(
            net,
            drop_top=CONCENTRATION_DROP_TOP,
            periods_per_year=ppy,
            turnover_series=streams.turnover,
            thresholds=cuts,
        ),
        bootstrap=checks.bootstrap(
            net,
            periods_per_year=ppy,
            n_resamples=bootstrap_resamples,
            block_lengths=blocks,
            seed=bootstrap_seed,
            thresholds=cuts,
        ),
        subperiod_stability=checks._subperiod_stability_from_streams(
            streams,
            n_splits=STABILITY_N_SPLITS,
            thresholds=cuts,
        ),
        latency=checks._latency_from_streams(
            streams,
            max_lag=LATENCY_MAX_LAG,
            thresholds=cuts,
        ),
        reconcile=reconciled,
    )

    index = streams.net.index
    meta = Meta(
        periods_per_year=ppy,
        periods_per_year_inferred=streams.periods_per_year_inferred,
        n_obs=n_obs,
        costs_bps=streams.costs_bps,
        seed=int(bootstrap_seed),
        n_trials=trials,
        n_resamples=int(bootstrap_resamples),
        first_bar=index[0].date().isoformat(),
        last_bar=index[-1].date().isoformat(),
        library_version=__version__,
        thresholds=cuts,
    )
    return Evaluation(
        metrics=performance,
        checks=evaluation_checks,
        verdict=worst_verdict(*evaluation_checks.verdicts().values()),
        meta=meta,
    )
