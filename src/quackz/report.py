"""Rendering an Evaluation as text, markdown or JSON.

Three rules shape this module.

One wording, three formats. Every verdict sentence is built once, in `check_lines`, and the
text and markdown renderers lay the same strings out differently. A report whose markdown
says something its text does not is a report nobody can quote.

Every number arrives with the rule that graded it. A line reads "break-even 2.1 bps; FAIL
below 5.0, WARN below 20.0", so a reader can disagree with the cut-off without opening the
source, and an overridden cut-off is visible in the output rather than silent. The cut-offs
themselves are not defined here: they live in `quackz.checks` as named module-level
constants, each with its reasoning beside it, gathered into `Thresholds` and overridable
through `quackz.evaluate.evaluate`.

No summary score. There is no letter grade and no weighted total, because both invite the
reader to skip the checks, and the checks are the report. The headline counts verdicts and
stops there.
"""

from __future__ import annotations

import json
import math
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from quackz.checks import (
    LATENCY_DECAY_MIN_HOLDING_BARS,
    LATENCY_DECAY_MIN_SHARPE_SE,
    BootstrapResult,
    ConcentrationResult,
    CostSweepResult,
    LatencyResult,
    ReconcileResult,
    SubperiodStabilityResult,
    Thresholds,
    Verdict,
)
from quackz.evaluate import DeflatedSharpeCheck, Evaluation, NoiseFloorCheck

__all__ = [
    "CHECK_TITLES",
    "LIMITS",
    "CheckLine",
    "check_lines",
    "headline",
    "json_report",
    "markdown_report",
    "text_report",
]

CHECK_TITLES = {
    "deflated_sharpe": "Deflated Sharpe",
    "noise_floor": "Noise floor",
    "cost_sweep": "Cost sensitivity",
    "concentration": "Profit concentration",
    "bootstrap": "Resampling",
    "subperiod_stability": "Subperiod stability",
    "latency": "Execution delay",
    "reconcile": "Reconciliation",
}

LIMITS = (
    "Positions are read as given. A signal built from information that was not available "
    "at the decision time will pass every check above, because none of them can see how "
    "the position was constructed.",
    "The trial count is a declaration. A search that is not disclosed cannot be deflated, "
    "and the deflated Sharpe is only as honest as the number supplied.",
    "Resampling probes the sampling noise in the realised profit and loss on this price "
    "path. It says nothing about a different path, and a block bootstrap destroys "
    "dependence beyond the block length, so the drawdown percentiles understate the tail.",
    "Survivorship and point-in-time universe construction are properties of the data, not "
    "of the arithmetic, and are out of scope.",
    "One instrument, one position series. There is no portfolio, correlation or capacity "
    "treatment here.",
)

_WIDTH = 78
_LABEL = 26
_VERDICT_ORDER = (Verdict.FAIL, Verdict.WARN, Verdict.PASS)


@dataclass(frozen=True)
class CheckLine:
    """One check as the report states it: the verdict, the finding, and the evidence."""

    key: str
    title: str
    verdict: Verdict
    finding: str
    details: tuple[str, ...]


def _num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "not supplied"
    number = float(value)
    if math.isnan(number):
        return "undefined"
    if math.isinf(number):
        return "unbounded" if number > 0 else "-unbounded"
    return f"{number:,.{digits}f}"


def _pct(value: float | None, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return _num(value, digits)
    return f"{float(value) * 100.0:,.{digits}f}%"


def _bps(value: float | None, digits: int = 1) -> str:
    return f"{_num(value, digits)} bps"


def _counts(evaluation: Evaluation) -> dict[Verdict, int]:
    verdicts = evaluation.checks.verdicts().values()
    return {level: sum(1 for v in verdicts if v is level) for level in _VERDICT_ORDER}


def headline(evaluation: Evaluation) -> str:
    """The one-line summary: "2 FAIL, 1 WARN, 5 PASS"."""
    counts = _counts(evaluation)
    return ", ".join(f"{counts[level]} {level.value}" for level in _VERDICT_ORDER)


def _deflated_sharpe(check: DeflatedSharpeCheck, cuts: Thresholds) -> tuple[str, list[str]]:
    plural = "trial" if check.n_trials == 1 else "trials"
    finding = (
        f"Deflated Sharpe {_num(check.dsr, 3)} against {check.n_trials:,} declared "
        f"{plural}; FAIL below {_num(cuts.dsr_fail)}, WARN below {_num(cuts.dsr_warn)}."
    )
    details = [
        f"Observed Sharpe {_num(check.observed_sharpe)} annualized against a benchmark of "
        f"{_num(check.expected_max_sharpe)}, the best a search of this size expects from "
        "pure noise.",
        f"Trial Sharpe dispersion V = {_num(check.var_trial_sharpes, 4)} per year "
        f"({_num(check.var_trial_sharpes_per_period, 6)} per bar), source "
        f"{check.variance_source}.",
        f"Probabilistic Sharpe against a zero benchmark: {_num(check.psr_vs_zero, 3)}.",
    ]
    if check.min_track_record_length is None:
        details.append(
            "Minimum track record length is undefined at a Sharpe of zero or below: no "
            "sample size makes it significant."
        )
    else:
        details.append(
            f"Bars needed for {_pct(check.confidence, 0)} confidence the true Sharpe beats "
            f"zero: {_num(check.min_track_record_length, 0)}, against {check.n_obs:,} observed."
        )
    if check.warning:
        details.append(check.warning)
    return finding, details


def _noise_floor(check: NoiseFloorCheck, cuts: Thresholds) -> tuple[str, list[str]]:
    if check.annualized <= 0.0:
        # A floor of zero arrives from two places that do not mean the same thing: one
        # declared trial, or a search whose trials all scored alike. This line named the
        # first in both cases, so a report could say "one declared trial" under a meta
        # block reporting two hundred. There is also nothing to divide by here, so the
        # ratio cut-offs cannot be quoted; the rule that actually grades the line is the
        # sign of the observed Sharpe, and that is what is quoted in its place.
        if check.n_trials == 1:
            reason = "one declared trial is no selection at all"
        else:
            reason = (
                f"{check.n_trials:,} declared trials with no dispersion between them, so the "
                "search had nothing to select from"
            )
        finding = (
            f"Observed Sharpe {_num(check.observed_sharpe)} against a floor of zero: "
            f"{reason}; with no floor to divide by, FAIL at or below a Sharpe of zero and "
            "no WARN band."
        )
    else:
        finding = (
            f"Observed Sharpe {_num(check.observed_sharpe)} is {_num(check.ratio)} times the "
            f"{_num(check.annualized)} a search of {check.n_trials} trials expects from pure "
            f"noise; FAIL at or below {_num(cuts.noise_floor_ratio_fail)}, WARN below "
            f"{_num(cuts.noise_floor_ratio_warn)}."
        )
    details = [
        f"Floor per bar {_num(check.per_period, 4)}, annualized {_num(check.annualized)}, "
        f"from {check.n_obs:,} observations and V = {_num(check.var_trial_sharpes, 4)} "
        f"per year ({check.variance_source}).",
        "The floor is the expected maximum of the search, Bailey and Lopez de Prado (2014) "
        "equation 1, the same benchmark the deflated Sharpe is measured against.",
    ]
    if check.note:
        details.append(check.note)
    return finding, details


def _cost_sweep(check: CostSweepResult, cuts: Thresholds) -> tuple[str, list[str]]:
    finding = (
        f"Break-even cost {_bps(check.break_even_bps)} per unit of turnover; FAIL below "
        f"{_bps(cuts.cost_break_even_bps_fail)}, WARN below "
        f"{_bps(cuts.cost_break_even_bps_warn)}."
    )
    return finding, [
        f"Gross edge {_num(check.mean_gross_return * 1e4, 2)} bps per bar against mean "
        f"turnover {_num(check.mean_turnover, 3)}, total turnover "
        f"{_num(check.total_turnover, 1)}. The full grid is tabulated below.",
        "Break-even is closed form, 1e4 * mean(gross) / mean(turnover), not read off the "
        "grid. Mean net return falls monotonically in cost; the net Sharpe need not.",
    ]


def _concentration(check: ConcentrationResult, cuts: Thresholds) -> tuple[str, list[str]]:
    worst = check.rows[-1]
    finding = (
        f"The best {worst.drop_top} of {check.n_obs:,} bars carry "
        f"{_pct(worst.share_of_gross_profit, 1)} of gross profit; FAIL above "
        f"{_pct(cuts.concentration_profit_share_fail, 0)}, WARN above "
        f"{_pct(cuts.concentration_profit_share_warn, 0)}."
    )
    details = [
        f"Sharpe {_num(check.baseline_sharpe)} on the full sample; "
        + ", ".join(f"{_num(row.sharpe)} without the best {row.drop_top}" for row in check.rows)
        + "."
    ]
    if check.edge_per_turnover_bps is not None:
        details.append(
            f"Edge per unit of turnover: {_bps(check.edge_per_turnover_bps, 2)}, on total "
            f"turnover {_num(check.total_turnover, 1)}. This is measured on the net stream, "
            "so the cost already charged has been taken out of it. The break-even cost "
            "above is the same quantity gross of costs, and that is the one to hold against "
            "a broker's quote."
        )
    return finding, details


def _bootstrap(check: BootstrapResult, cuts: Thresholds) -> tuple[str, list[str]]:
    finding = (
        f"Null-imposed p-value {_num(check.max_p_value, 3)} at the least favourable block "
        f"length; FAIL above {_num(cuts.bootstrap_p_value_fail)}, WARN above "
        f"{_num(cuts.bootstrap_p_value_warn)}."
    )
    low, high = check.percentiles[0], check.percentiles[-1]
    details = [
        f"{check.n_resamples:,} stationary bootstrap resamples per block length, seed "
        f"{check.seed}. Observed Sharpe {_num(check.observed_sharpe)}, maximum drawdown "
        f"{_pct(check.observed_max_drawdown, 1)}.",
    ]
    for block in check.blocks:
        details.append(
            f"Block {block.block_length}: p-value {_num(block.p_value, 3)}, Sharpe "
            f"{_num(low, 0)}th to {_num(high, 0)}th percentile {_num(block.sharpe_percentiles[0])} "
            f"to {_num(block.sharpe_percentiles[-1])}, studentized "
            f"{_pct(check.confidence, 0)} interval {_num(block.ci_low)} to "
            f"{_num(block.ci_high)}, drawdown {_num(low, 0)}th percentile "
            f"{_pct(block.max_drawdown_percentiles[0], 1)}."
        )
    details.append(
        "The p-value is the fraction of resamples of a mean-zero version of these returns "
        "that reach the observed Sharpe. It is not the probability that the Sharpe is "
        "negative, and 0.000 means below one in the resample count."
    )
    return finding, details


def _subperiod_stability(
    check: SubperiodStabilityResult, cuts: Thresholds
) -> tuple[str, list[str]]:
    finding = (
        f"Window Sharpe dispersion {_num(check.sharpe_dispersion)} against "
        f"{_num(check.expected_sharpe_se)} expected from sampling noise alone, ratio "
        f"{_num(check.dispersion_ratio)}; FAIL at or above "
        f"{_num(cuts.stability_dispersion_ratio_fail)}, WARN at or above "
        f"{_num(cuts.stability_dispersion_ratio_warn)}."
    )
    windows = ", ".join(_num(value) for value in check.window_sharpes)
    return finding, [
        f"Sharpe by window: {windows}. Worst {_num(check.worst_window_sharpe)}, full sample "
        f"{_num(check.full_sample_sharpe)}.",
        "The windows are contiguous slices of one fixed signal, so this measures temporal "
        "stability, not validation: nothing is refitted between them.",
    ]


def _latency(check: LatencyResult, cuts: Thresholds) -> tuple[str, list[str]]:
    finding = (
        f"Sharpe {_num(check.sharpe_by_lag[0])} with no delay; FAIL at or above "
        f"{_num(cuts.latency_annual_sharpe_fail)}, WARN at or above "
        f"{_num(cuts.latency_annual_sharpe_warn)}."
    )
    if check.retention_ratio is None:
        if check.retention_1bar is None:
            reason = (
                f"a Sharpe of {_num(check.sharpe_by_lag[0])} at zero delay is inside "
                f"{_num(LATENCY_DECAY_MIN_SHARPE_SE, 0)} standard errors of zero "
                f"({_num(check.sharpe_standard_error)} each), so there is no edge here whose "
                "decay could mean anything"
            )
        else:
            reason = (
                f"a mean holding period of {_num(check.mean_holding_period, 1)} bars is below "
                f"the {_num(LATENCY_DECAY_MIN_HOLDING_BARS, 1)} it needs, so losing the "
                "Sharpe to a one-bar delay is the expected behaviour rather than a finding"
            )
        finding += f" The decay rule does not apply here: {reason}."
    else:
        finding += (
            f" One bar late it keeps {_pct(check.retention_1bar, 1)} of that Sharpe against "
            f"the {_pct(check.expected_retention_1bar, 1)} a "
            f"{_num(check.mean_holding_period, 1)} bar holding period implies, ratio "
            f"{_num(check.retention_ratio)}; FAIL below "
            f"{_num(cuts.latency_retention_ratio_fail)}, WARN below "
            f"{_num(cuts.latency_retention_ratio_warn)}."
        )
    decay = ", ".join(
        f"{lag} bar{'' if lag == 1 else 's'} {_num(value)}"
        for lag, value in zip(check.lags, check.sharpe_by_lag, strict=True)
    )
    return finding, [
        f"Sharpe by delay: {decay}, all measured on the same {check.n_obs:,} bars and gross "
        "of costs, so only the timing changes between them.",
        f"Mean holding period {_num(check.mean_holding_period, 1)} bars, position "
        f"autocorrelation {_num(check.position_autocorr, 3)}.",
        "A one-bar delay only misplaces the position on the bars where it moved, so a "
        "position held h bars keeps about (h - 1) / h of an edge spread across its holding "
        "period. Raw decay carries no verdict of its own, and never can: a two-bar signal "
        "has no reason to survive a three-bar delay.",
        "Neither rule sees a leak whose horizon matches the holding period. A position "
        "built from a twenty-bar forward return and held twenty bars loses as little to a "
        "one-bar delay as an honest one does.",
    ]


def _reconcile(check: ReconcileResult, cuts: Thresholds) -> tuple[str, list[str]]:
    finding = (
        f"Correlation {_num(check.correlation, 4)} with the recomputed stream and a terminal "
        f"wealth gap of {_pct(check.cumulative_gap, 2)}; FAIL below "
        f"{_num(cuts.reconcile_correlation_fail, 3)} correlation or "
        f"{_pct(cuts.reconcile_cumulative_gap_fail, 0)} gap."
    )
    details = [
        f"Tracking error {_pct(check.tracking_error, 3)} per bar over {check.n_obs:,} bars, "
        f"direction {check.gap_direction}.",
    ]
    if check.candidate_causes:
        details.append("Candidate causes, most common first:")
        details.extend(check.candidate_causes)
    return finding, details


# The value type is deliberately loose in its first argument: each builder takes the one
# check result it renders, and the dispatch key is what pairs them. Narrowing it would mean
# a union of eight result types that no caller ever needs.
_BUILDERS: dict[str, Callable[[Any, Thresholds], tuple[str, list[str]]]] = {
    "deflated_sharpe": _deflated_sharpe,
    "noise_floor": _noise_floor,
    "cost_sweep": _cost_sweep,
    "concentration": _concentration,
    "bootstrap": _bootstrap,
    "subperiod_stability": _subperiod_stability,
    "latency": _latency,
    "reconcile": _reconcile,
}


def check_lines(evaluation: Evaluation) -> tuple[CheckLine, ...]:
    """Every check as one finding plus its supporting lines, in report order."""
    cuts = evaluation.meta.thresholds
    lines = []
    for key, verdict in evaluation.checks.verdicts().items():
        result = getattr(evaluation.checks, key)
        finding, details = _BUILDERS[key](result, cuts)
        lines.append(
            CheckLine(
                key=key,
                title=CHECK_TITLES[key],
                verdict=verdict,
                finding=finding,
                details=tuple(details),
            )
        )
    return tuple(lines)


def _meta_rows(evaluation: Evaluation) -> list[tuple[str, str]]:
    meta = evaluation.meta
    check = evaluation.checks.deflated_sharpe
    source = "inferred from the index" if meta.periods_per_year_inferred else "supplied"
    return [
        ("Sample", f"{meta.first_bar} to {meta.last_bar}, {meta.n_obs:,} return bars"),
        ("Annualization", f"{_num(meta.periods_per_year)} periods per year, {source}"),
        ("Costs charged", f"{_bps(meta.costs_bps, 2)} per unit of turnover"),
        (
            "Trials declared",
            f"{meta.n_trials:,}, dispersion V = {_num(check.var_trial_sharpes, 4)} per year "
            f"({check.variance_source})",
        ),
        ("Resampling", f"{meta.n_resamples:,} resamples, seed {meta.seed}"),
        ("Report version", meta.library_version),
    ]


def _metric_rows(evaluation: Evaluation) -> list[tuple[str, str]]:
    m = evaluation.metrics
    return [
        ("Sharpe, annualized", _num(m.sharpe)),
        ("Sharpe, per bar", _num(m.sharpe_per_period, 4)),
        ("Sharpe, gross of costs", _num(m.sharpe_gross)),
        ("Annualized return", _pct(m.annualized_return)),
        ("Annualized volatility", _pct(m.annualized_vol)),
        ("Total return", _pct(m.total_return)),
        ("Sortino", _num(m.sortino)),
        ("Calmar", _num(m.calmar)),
        ("Maximum drawdown", _pct(m.max_drawdown)),
        ("Ulcer index", _num(m.ulcer_index, 4)),
        ("CVaR, worst 5 percent", _pct(m.cvar_5pct)),
        ("Skewness", _num(m.skew, 3)),
        ("Kurtosis, non-excess", _num(m.kurt_nonexcess, 3)),
        ("Lag-1 autocorrelation", _num(m.autocorr_1, 4)),
        ("t-statistic, naive", _num(m.naive_tstat)),
        ("t-statistic, Newey-West", _num(m.hac_tstat)),
    ]


def _trial_rows(evaluation: Evaluation) -> list[tuple[str, str, str]]:
    check = evaluation.checks.deflated_sharpe
    rows = []
    for row in check.by_trials:
        marker = " *" if row.n_trials == check.n_trials else ""
        rows.append((f"{row.n_trials:,}{marker}", _num(row.expected_max_sharpe), _num(row.dsr, 4)))
    return rows


def _break_even_sentence(evaluation: Evaluation) -> str:
    check = evaluation.checks.deflated_sharpe
    target = _num(check.confidence)
    if check.break_even_n_trials is None:
        return (
            f"The deflated Sharpe does not reach {target} even at a single trial, so no "
            "search size clears it: the record is too short or too noisy for that "
            "confidence against a zero benchmark."
        )
    if check.break_even_capped:
        return (
            f"The deflated Sharpe still clears {target} at {check.break_even_n_trials:,} "
            "trials, where the search stops."
        )
    return (
        f"The deflated Sharpe clears {target} up to {check.break_even_n_trials:,} trials and "
        f"falls below it at {check.break_even_n_trials + 1:,}: from there on, a record this "
        "size is not distinguishable from the best of the search."
    )


def _cost_rows(evaluation: Evaluation) -> list[tuple[str, str, str]]:
    return [
        (_num(row.bps, 1), _pct(row.mean_net_return, 4), _num(row.net_sharpe))
        for row in evaluation.checks.cost_sweep.rows
    ]


def _wrap(text: str, *, indent: str, hanging: str) -> list[str]:
    return textwrap.wrap(text, width=_WIDTH, initial_indent=indent, subsequent_indent=hanging) or [
        indent.rstrip()
    ]


def text_report(evaluation: Evaluation) -> str:
    """The report as fixed-width text, which is what the CLI prints."""
    out = ["BACKTEST AUDIT", "=" * _WIDTH]
    out.append(f"Verdict: {evaluation.verdict.value}. {headline(evaluation)}.")
    out.append("")
    for label, value in _meta_rows(evaluation):
        out.append(f"{label:<{_LABEL}}{value}")

    out += ["", "PERFORMANCE, NET OF COSTS", "-" * _WIDTH]
    for label, value in _metric_rows(evaluation):
        out.append(f"  {label:<{_LABEL}}{value:>14}")

    out += ["", "CHECKS", "-" * _WIDTH]
    for line in check_lines(evaluation):
        out.append(f"[{line.verdict.value}] {line.title}")
        out += _wrap(line.finding, indent=" " * 7, hanging=" " * 7)
        for detail in line.details:
            out += _wrap(detail, indent=" " * 7 + "- ", hanging=" " * 9)
        out.append("")

    out += ["DEFLATED SHARPE AGAINST THE NUMBER OF TRIALS", "-" * _WIDTH]
    out.append(f"  {'Trials':>10}  {'Benchmark Sharpe':>18}  {'DSR':>8}")
    for trials, benchmark, dsr in _trial_rows(evaluation):
        out.append(f"  {trials:>10}  {benchmark:>18}  {dsr:>8}")
    out.append("  * the declared trial count.")
    out += _wrap(_break_even_sentence(evaluation), indent="  ", hanging="  ")
    out += _wrap(
        "The dispersion of trial Sharpes is held fixed while the count varies, so the "
        "column answers what this record would be worth had the search been larger.",
        indent="  ",
        hanging="  ",
    )

    out += ["", "COST SENSITIVITY", "-" * _WIDTH]
    out.append(f"  {'Cost (bps)':>10}  {'Mean net return':>18}  {'Net Sharpe':>12}")
    for bps, mean_net, sharpe in _cost_rows(evaluation):
        out.append(f"  {bps:>10}  {mean_net:>18}  {sharpe:>12}")

    out += ["", "LIMITS OF THIS REPORT", "-" * _WIDTH]
    for limit in LIMITS:
        out += _wrap(limit, indent="  - ", hanging="    ")
    out.append("")
    return "\n".join(out)


def _md_table(headers: tuple[str, ...], rows: Sequence[tuple[str, ...]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    out.append("")
    return out


def markdown_report(evaluation: Evaluation) -> str:
    """The same report as markdown, for pasting into a pull request or a memo."""
    out = ["# Backtest audit", ""]
    out.append(f"> **{evaluation.verdict.value}.** {headline(evaluation)}.")
    out.append("")
    out += _md_table(("Field", "Value"), _meta_rows(evaluation))

    out += ["## Performance, net of costs", ""]
    out += _md_table(("Metric", "Value"), _metric_rows(evaluation))

    lines = check_lines(evaluation)
    out += ["## Checks", ""]
    out += _md_table(
        ("Check", "Verdict", "Finding"),
        [(line.title, line.verdict.value, line.finding) for line in lines],
    )
    for line in lines:
        out += [f"### {line.title}: {line.verdict.value}", "", line.finding, ""]
        out += [f"- {detail.strip()}" for detail in line.details]
        out.append("")

    out += ["## Deflated Sharpe against the number of trials", ""]
    out += _md_table(("Trials", "Benchmark Sharpe", "DSR"), _trial_rows(evaluation))
    out += [
        "`*` marks the declared trial count.",
        "",
        _break_even_sentence(evaluation),
        "",
        "The dispersion of trial Sharpes is held fixed while the count varies.",
        "",
        "## Cost sensitivity",
        "",
    ]
    out += _md_table(("Cost (bps)", "Mean net return", "Net Sharpe"), _cost_rows(evaluation))

    out += ["## Limits of this report", ""]
    out += [f"- {limit}" for limit in LIMITS]
    out.append("")
    return "\n".join(out)


def _json_safe(value: Any) -> Any:
    """Plain JSON types, with non-finite floats as null.

    JSON has no literal for infinity or NaN. Emitting one anyway produces a document that
    Python reads back and nothing else does, so an unbounded break-even cost is serialised
    as null; the text report spells it out in words.
    """
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Verdict):
        return value.value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def json_report(evaluation: Evaluation, *, indent: int = 2) -> str:
    """The whole Evaluation as strict JSON, plus the headline the other formats print."""
    counts = _counts(evaluation)
    payload = {
        "verdict": evaluation.verdict.value,
        "headline": headline(evaluation),
        "counts": {level.value: counts[level] for level in _VERDICT_ORDER},
        "check_verdicts": {
            key: verdict.value for key, verdict in evaluation.checks.verdicts().items()
        },
        "findings": {line.key: line.finding for line in check_lines(evaluation)},
        "meta": asdict(evaluation.meta),
        "metrics": asdict(evaluation.metrics),
        "checks": asdict(evaluation.checks),
        "limits": list(LIMITS),
    }
    return json.dumps(_json_safe(payload), indent=indent, allow_nan=False)
