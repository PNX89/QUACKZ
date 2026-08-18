"""Tests for quackz.cli.

The exit codes are the contract, because they are what a continuous integration job reads,
so they are tested through a real subprocess rather than by calling `main` in process. The
other thing tested here is the failure mode a user actually hits: a misspelled column name
must produce one sentence naming the column and listing the ones that exist, never a
traceback out of pandas.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

import pandas as pd
import pytest

from conftest import PPY, RESAMPLES, write_csv
from quackz.checks import Verdict
from quackz.cli import build_parser, exit_code
from quackz.evaluate import evaluate
from quackz.returns import net_returns

FAST = ("--bootstrap-n", "150", "--periods-per-year", str(PPY))


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "quackz.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture(scope="session")
def honest_csv(tmp_path_factory, honest_strategy) -> str:
    prices, positions = honest_strategy
    path = tmp_path_factory.mktemp("data") / "honest.csv"
    write_csv(path, prices, positions, claimed=net_returns(prices, positions, costs_bps=2.0))
    return str(path)


@pytest.fixture(scope="session")
def peeking_csv(tmp_path_factory, peeking_strategy) -> str:
    prices, positions = peeking_strategy
    path = tmp_path_factory.mktemp("data") / "peeking.csv"
    write_csv(path, prices, positions)
    return str(path)


# --------------------------------------------------------------------------------------
# Exit codes
# --------------------------------------------------------------------------------------


def test_a_clean_run_exits_zero_and_prints_the_report(honest_csv):
    result = run("report", honest_csv, *FAST)
    assert result.returncode == 0
    assert result.stdout.startswith("BACKTEST AUDIT")
    assert "Verdict: PASS." in result.stdout


def test_a_failing_verdict_exits_one(peeking_csv):
    result = run("report", peeking_csv, *FAST)
    assert result.returncode == 1
    assert "[FAIL] Execution delay" in result.stdout


def test_fail_on_warn_turns_a_warning_into_a_failure(honest_csv, honest_strategy):
    """The same command, one flag apart, on a run whose worst verdict is a WARN."""
    warned = evaluate(
        *honest_strategy, periods_per_year=PPY, n_trials=5, bootstrap_resamples=150, costs_bps=0.0
    )
    assert warned.verdict is Verdict.WARN

    args = ("report", honest_csv, *FAST, "--n-trials", "5")
    assert run(*args).returncode == 0
    assert run(*args, "--fail-on", "warn").returncode == 1


def test_a_missing_file_exits_two(tmp_path):
    result = run("report", str(tmp_path / "nowhere.csv"))
    assert result.returncode == 2
    assert result.stderr.strip().startswith("error: no such file")
    assert "Traceback" not in result.stderr


def test_an_unknown_flag_exits_two(honest_csv):
    result = run("report", honest_csv, "--annualize", "252")
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr


def test_no_subcommand_exits_two():
    assert run().returncode == 2


def test_version_prints_and_exits_zero():
    result = run("--version")
    assert result.returncode == 0
    assert result.stdout.split()[0] == "quackz"


# --------------------------------------------------------------------------------------
# Input errors, in the words a user can act on
# --------------------------------------------------------------------------------------


def test_a_misspelled_column_names_itself_and_lists_the_alternatives(honest_csv):
    result = run("report", honest_csv, "--price-col", "adj_close")
    assert result.returncode == 2
    assert "--price-col 'adj_close' is not a column in the file" in result.stderr
    assert "Columns present: date, close, position, pnl" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("flag", ["--date-col", "--pos-col", "--claimed-col"])
def test_every_column_flag_reports_its_own_name(honest_csv, flag):
    result = run("report", honest_csv, flag, "missing")
    assert result.returncode == 2
    assert f"{flag} 'missing'" in result.stderr


def test_an_unsorted_file_is_refused_with_the_fix(tmp_path, honest_strategy):
    prices, positions = honest_strategy
    path = tmp_path / "unsorted.csv"
    write_csv(path, prices, positions)
    frame = pd.read_csv(path)
    frame = pd.concat([frame.iloc[5:], frame.iloc[:5]], ignore_index=True)
    frame.to_csv(path, index=False)
    result = run("report", str(path))
    assert result.returncode == 2
    assert "not sorted ascending" in result.stderr
    assert "Sort the file by date" in result.stderr


def test_a_hole_in_the_price_column_is_refused_cleanly(tmp_path, honest_strategy):
    prices, positions = honest_strategy
    path = tmp_path / "holed.csv"
    write_csv(path, prices, positions)
    frame = pd.read_csv(path)
    frame.loc[7, "close"] = None
    frame.to_csv(path, index=False)
    result = run("report", str(path))
    assert result.returncode == 2
    assert result.stderr.startswith("error: prices has 1 NaN value")
    assert "Traceback" not in result.stderr


def test_a_sample_too_short_to_audit_says_so(tmp_path, honest_strategy):
    prices, positions = honest_strategy
    path = tmp_path / "short.csv"
    write_csv(path, prices.iloc[:12], positions.iloc[:12])
    result = run("report", str(path))
    assert result.returncode == 2
    assert "at least 20 return bars" in result.stderr


def test_an_empty_file_is_refused(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    result = run("report", str(path))
    assert result.returncode == 2
    assert "is empty" in result.stderr


# --------------------------------------------------------------------------------------
# Additional output
# --------------------------------------------------------------------------------------


def test_json_and_markdown_are_additive(tmp_path, honest_csv):
    json_path = tmp_path / "out.json"
    md_path = tmp_path / "out.md"
    result = run("report", honest_csv, *FAST, "--json", str(json_path), "--md", str(md_path))
    assert result.returncode == 0
    assert result.stdout.startswith("BACKTEST AUDIT")
    assert json_path.read_text().startswith("{")
    assert md_path.read_text().startswith("# Backtest audit")
    assert f"wrote JSON to {json_path}" in result.stderr
    assert f"wrote markdown to {md_path}" in result.stderr


def test_the_claimed_column_adds_the_reconciliation(honest_csv):
    """The bundled pnl column is the net stream at 2 bps, so it must reconcile exactly."""
    without = run("report", honest_csv, *FAST, "--costs-bps", "2")
    with_claim = run("report", honest_csv, *FAST, "--costs-bps", "2", "--claimed-col", "pnl")
    assert "Reconciliation" not in without.stdout
    assert "[PASS] Reconciliation" in with_claim.stdout


def test_the_report_is_reproducible_across_processes(honest_csv):
    first = run("report", honest_csv, *FAST, "--seed", "7")
    second = run("report", honest_csv, *FAST, "--seed", "7")
    assert first.stdout == second.stdout
    assert "seed 7" in first.stdout


def test_the_installed_console_script_runs(honest_csv):
    executable = shutil.which("quackz")
    if executable is None:
        pytest.skip("the console script is not on PATH in this environment")
    result = subprocess.run(
        [executable, "report", honest_csv, *FAST], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0
    assert result.stdout.startswith("BACKTEST AUDIT")


# --------------------------------------------------------------------------------------
# The pieces, without a subprocess
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("verdict", "fail_on", "expected"),
    [
        (Verdict.PASS, "fail", 0),
        (Verdict.WARN, "fail", 0),
        (Verdict.FAIL, "fail", 1),
        (Verdict.PASS, "warn", 0),
        (Verdict.WARN, "warn", 1),
        (Verdict.FAIL, "warn", 1),
    ],
)
def test_the_exit_code_table(verdict, fail_on, expected):
    assert exit_code(verdict, fail_on=fail_on) == expected


def test_the_defaults_are_the_documented_ones(honest_csv):
    args = build_parser().parse_args(["report", honest_csv])
    assert (args.date_col, args.price_col, args.pos_col) == ("date", "close", "position")
    assert (args.costs_bps, args.n_trials, args.seed) == (0.0, 1, 0)
    assert (args.bootstrap_n, args.periods_per_year, args.fail_on) == (1000, None, "fail")
    assert (args.claimed_col, args.json, args.md) == (None, None, None)


def test_the_evaluation_run_by_the_cli_matches_the_library(honest_csv, honest_strategy):
    args = build_parser().parse_args(
        ["report", honest_csv, "--costs-bps", "2", *FAST, "--bootstrap-n", str(RESAMPLES)]
    )
    from quackz.cli import load_evaluation

    through_cli = load_evaluation(args)
    direct = evaluate(
        *honest_strategy,
        costs_bps=2.0,
        periods_per_year=PPY,
        bootstrap_resamples=RESAMPLES,
    )
    assert through_cli.verdict is direct.verdict
    assert through_cli.checks.verdicts() == direct.checks.verdicts()
    # Not exact equality: a CSV holds about sixteen significant digits, so the prices come
    # back a few units in the last place away from the ones the fixture generated.
    assert through_cli.metrics.sharpe == pytest.approx(direct.metrics.sharpe, rel=1e-9)
    assert through_cli.checks.cost_sweep.break_even_bps == pytest.approx(
        direct.checks.cost_sweep.break_even_bps, rel=1e-9
    )


def test_a_library_warning_is_reported_as_a_note_not_a_traceback(honest_csv):
    result = run("report", honest_csv, *FAST, "--n-trials", "5")
    assert result.returncode == 0
    assert result.stderr.startswith("note: V[{SR_n}] was not supplied")
    assert "UserWarning" not in result.stderr
    assert "warnings.warn" not in result.stderr
