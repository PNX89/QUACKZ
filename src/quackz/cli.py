"""Command line entry point: a CSV of prices and positions in, an audit out.

    quackz report data.csv --date-col date --price-col close --pos-col position

Exit codes, because this is meant to run in continuous integration.

    0   every check passed, or warned while --fail-on is `fail` (the default)
    1   at least one check FAILed, or WARNed while --fail-on is `warn`
    2   the command could not be run at all: a bad flag, a missing file, a
        column that is not in the CSV, or data that breaks a documented input
        convention

The text report always goes to standard output, so it can be piped. `--json` and `--md`
are additive: they write those formats to files in addition to the text on stdout, and
the note saying where they went goes to standard error to keep stdout clean.

Input errors are reported as one sentence naming the flag and the fix. A user who
misspells a column name gets the list of columns the file actually has, never a pandas
traceback.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from quackz._version import __version__
from quackz.checks import Verdict
from quackz.evaluate import Evaluation, evaluate
from quackz.report import json_report, markdown_report, text_report

__all__ = ["build_parser", "main"]

EXIT_OK = 0
EXIT_VERDICT = 1
EXIT_USAGE = 2


class CliError(Exception):
    """An input problem the user can fix, reported without a traceback."""


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, exposed so the flags can be tested without a subprocess."""
    parser = argparse.ArgumentParser(
        prog="quackz",
        description="Audit a backtest for the ways backtests overstate themselves.",
    )
    parser.add_argument("--version", action="version", version=f"quackz {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    report = subparsers.add_parser(
        "report",
        help="run every check against a CSV of prices and positions",
        description=(
            "Run every check against a CSV and print a report. Exit code 0 means pass or "
            "warn, 1 means a FAIL verdict, 2 means the command could not run."
        ),
    )
    report.add_argument("csv", type=Path, help="CSV file with a date, price and position column")
    report.add_argument("--date-col", default="date", help="timestamp column (default: date)")
    report.add_argument("--price-col", default="close", help="close price column (default: close)")
    report.add_argument(
        "--pos-col",
        default="position",
        help="target exposure decided at that bar's close (default: position)",
    )
    report.add_argument(
        "--claimed-col",
        default=None,
        help="optional column of per-period returns claimed by your own backtest, "
        "reconciled against the stream recomputed here",
    )
    report.add_argument(
        "--costs-bps",
        type=float,
        default=0.0,
        help="transaction cost in basis points of turnover (default: 0)",
    )
    report.add_argument(
        "--n-trials",
        type=int,
        default=1,
        help="configurations the search touched, used to deflate the Sharpe (default: 1)",
    )
    report.add_argument(
        "--bootstrap-n",
        type=int,
        default=1000,
        help="resamples per block length (default: 1000)",
    )
    report.add_argument(
        "--periods-per-year",
        type=float,
        default=None,
        help="annualization factor; inferred from the index when omitted",
    )
    report.add_argument("--seed", type=int, default=0, help="bootstrap seed (default: 0)")
    report.add_argument("--json", type=Path, default=None, metavar="PATH", help="also write JSON")
    report.add_argument("--md", type=Path, default=None, metavar="PATH", help="also write markdown")
    report.add_argument(
        "--fail-on",
        choices=("fail", "warn"),
        default="fail",
        help="verdict that makes the command exit 1 (default: fail)",
    )
    return parser


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise CliError(f"no such file: {path}")
    if path.is_dir():
        raise CliError(f"{path} is a directory, not a CSV file")
    try:
        frame = pd.read_csv(path)
    except UnicodeDecodeError as exc:
        raise CliError(f"{path} is not text this reader can decode: {exc}") from exc
    except pd.errors.EmptyDataError as exc:
        raise CliError(f"{path} is empty") from exc
    except pd.errors.ParserError as exc:
        raise CliError(f"{path} is not a well formed CSV: {exc}") from exc
    if frame.empty:
        raise CliError(f"{path} has a header but no rows")
    return frame


def _column(frame: pd.DataFrame, name: str, *, flag: str) -> pd.Series:
    if name not in frame.columns:
        available = ", ".join(str(column) for column in frame.columns)
        raise CliError(f"{flag} {name!r} is not a column in the file. Columns present: {available}")
    return frame[name]


def _index_from(frame: pd.DataFrame, name: str) -> pd.DatetimeIndex:
    column = _column(frame, name, flag="--date-col")
    try:
        stamps = pd.to_datetime(column)
    except (ValueError, TypeError) as exc:
        raise CliError(
            f"--date-col {name!r} could not be read as timestamps: {exc}. Give the column "
            "that holds the bar date."
        ) from exc
    if stamps.isna().any():
        first = int(stamps.isna().to_numpy().nonzero()[0][0])
        raise CliError(
            f"--date-col {name!r} has an unparseable timestamp on row {first + 2} of the file"
        )
    index = pd.DatetimeIndex(stamps)
    if not index.is_monotonic_increasing:
        breaks = (index[1:] < index[:-1]).nonzero()[0]
        raise CliError(
            f"--date-col {name!r} is not sorted ascending (first step backwards at row "
            f"{int(breaks[0]) + 3} of the file). Sort the file by date and run again."
        )
    return index


def load_evaluation(args: argparse.Namespace) -> Evaluation:
    """Read the CSV named by `args` and run the audit. Raises CliError on bad input."""
    frame = _read_frame(args.csv)
    index = _index_from(frame, args.date_col)
    prices = pd.Series(_column(frame, args.price_col, flag="--price-col").to_numpy(), index=index)
    positions = pd.Series(_column(frame, args.pos_col, flag="--pos-col").to_numpy(), index=index)
    claimed = None
    if args.claimed_col is not None:
        claimed = pd.Series(
            _column(frame, args.claimed_col, flag="--claimed-col").to_numpy(), index=index
        )
    return evaluate(
        prices,
        positions,
        costs_bps=args.costs_bps,
        periods_per_year=args.periods_per_year,
        n_trials=args.n_trials,
        claimed_returns=claimed,
        bootstrap_seed=args.seed,
        bootstrap_resamples=args.bootstrap_n,
    )


def exit_code(verdict: Verdict, *, fail_on: str) -> int:
    """0 unless the verdict is at or beyond what `--fail-on` treats as a failure."""
    if verdict is Verdict.FAIL:
        return EXIT_VERDICT
    if verdict is Verdict.WARN and fail_on == "warn":
        return EXIT_VERDICT
    return EXIT_OK


def _write(path: Path, text: str, *, label: str) -> None:
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise CliError(f"could not write the {label} report to {path}: {exc}") from exc
    print(f"wrote {label} to {path}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line. Returns the process exit code rather than raising SystemExit."""
    args = build_parser().parse_args(argv)
    try:
        # The library warns when it has to fall back to an assumption. Nothing is hidden,
        # but Python's default warning format prints a source line, which reads as a
        # traceback to someone who is running a command rather than writing Python.
        with warnings.catch_warnings(record=True) as raised:
            warnings.simplefilter("always")
            evaluation = load_evaluation(args)
        for entry in raised:
            print(f"note: {entry.message}", file=sys.stderr)
        print(text_report(evaluation))
        if args.json is not None:
            _write(args.json, json_report(evaluation) + "\n", label="JSON")
        if args.md is not None:
            _write(args.md, markdown_report(evaluation), label="markdown")
    # QuackzInputError and the ValueErrors raised by quackz.metrics are both statements
    # about the data, so they are reported as usage errors rather than as a traceback.
    except (CliError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    return exit_code(evaluation.verdict, fail_on=args.fail_on)


if __name__ == "__main__":
    raise SystemExit(main())
