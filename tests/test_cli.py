"""Tests for quackz.cli.

The exit codes are the contract, because they are what a continuous integration job reads,
so they are tested through a real subprocess rather than by calling `main` in process. The
other thing tested here is the failure mode a user actually hits: a misspelled column name
must produce one sentence naming the column and listing the ones that exist, never a
traceback out of pandas.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import PPY, RESAMPLES, write_csv
from quackz import cli
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


@pytest.fixture(scope="session")
def searched_csv(tmp_path_factory, searched_strategy) -> str:
    prices, positions, _ = searched_strategy
    path = tmp_path_factory.mktemp("data") / "searched.csv"
    write_csv(path, prices, positions)
    return str(path)


@pytest.fixture(scope="session")
def searched_trials(tmp_path_factory, searched_strategy) -> str:
    """The Sharpe of all two hundred configurations, in the shape a search would write it."""
    *_, sharpes = searched_strategy
    path = tmp_path_factory.mktemp("data") / "trials.txt"
    body = "\n".join(f"{value:.12f}" for value in sharpes)
    path.write_text(f"# annualized Sharpe, one per configuration\n{body}\n", encoding="utf-8")
    return str(path)


def deflated_sharpe_of(stdout: str) -> float:
    match = re.search(r"Deflated Sharpe (\d+\.\d+) against", stdout)
    assert match is not None, stdout
    return float(match.group(1))


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


def test_a_directory_where_the_csv_should_be_exits_two(tmp_path):
    result = run("report", str(tmp_path))
    assert result.returncode == 2
    assert "is a directory, not a CSV file" in result.stderr
    assert "Traceback" not in result.stderr


def test_a_file_that_is_not_decodable_text_exits_two(tmp_path):
    """UnicodeDecodeError is a ValueError, so an OSError handler alone would miss it."""
    path = tmp_path / "utf16.csv"
    path.write_bytes("date,close,position\n2020-01-01,100,1\n".encode("utf-16"))
    result = run("report", str(path))
    assert result.returncode == 2
    assert "not text this reader can decode" in result.stderr
    assert "Traceback" not in result.stderr


def test_a_file_the_reader_cannot_open_exits_two_rather_than_reporting_a_verdict(
    monkeypatch, honest_csv, capsys
):
    """Exit 1 means a check FAILed. A file that could not be opened must never claim it."""

    def refuse(*args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(cli.pd, "read_csv", refuse)
    assert cli.main(["report", honest_csv]) == 2
    assert "could not read" in capsys.readouterr().err


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="root opens a mode 000 file whatever its permissions say",
)
def test_a_file_with_no_read_permission_exits_two(tmp_path, honest_strategy):
    prices, positions = honest_strategy
    path = tmp_path / "locked.csv"
    write_csv(path, prices, positions)
    path.chmod(0o000)
    try:
        result = run("report", str(path))
    finally:
        path.chmod(0o600)
    assert result.returncode == 2
    assert "could not read" in result.stderr
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
# The dispersion of the search, which is what makes the deflation honest
# --------------------------------------------------------------------------------------


def test_the_trial_sharpes_file_reaches_the_deflation(
    searched_csv, searched_trials, searched_strategy
):
    """The headline capability, from the command line rather than only from Python.

    Without the file the report deflates against the iid-normal placeholder. With it, the
    command line lands on exactly the number the library produces from the same Sharpes.
    """
    prices, positions, sharpes = searched_strategy
    direct = evaluate(
        prices,
        positions,
        periods_per_year=PPY,
        n_trials=len(sharpes),
        trial_sharpes=sharpes,
        bootstrap_resamples=RESAMPLES,
    )
    declared = ("--n-trials", str(len(sharpes)))
    fallback = run("report", searched_csv, *FAST, *declared)
    supplied = run("report", searched_csv, *FAST, *declared, "--trial-sharpes", searched_trials)

    assert "source iid_fallback" in fallback.stdout
    assert "source trial_sharpes" in supplied.stdout
    assert deflated_sharpe_of(supplied.stdout) == pytest.approx(
        direct.checks.deflated_sharpe.dsr, abs=1e-3
    )
    assert deflated_sharpe_of(supplied.stdout) != deflated_sharpe_of(fallback.stdout)


def test_the_trial_count_defaults_to_the_number_of_sharpes_supplied(searched_csv, searched_trials):
    """Declaring one trial beside a file of two hundred can only be an omission."""
    result = run("report", searched_csv, *FAST, "--trial-sharpes", searched_trials)
    assert result.returncode in (0, 1)
    assert "note: --n-trials was not given" in result.stderr
    assert "against 200 declared trials" in result.stdout


def test_the_variance_can_be_given_directly(searched_csv):
    result = run("report", searched_csv, *FAST, "--n-trials", "200", "--var-trial-sharpes", "1.5")
    assert "source var_trial_sharpes" in result.stdout
    assert "V = 1.5000 per year" in result.stdout


def test_the_two_ways_to_state_the_dispersion_are_mutually_exclusive(searched_csv, searched_trials):
    result = run(
        "report",
        searched_csv,
        *FAST,
        "--trial-sharpes",
        searched_trials,
        "--var-trial-sharpes",
        "1.5",
    )
    assert result.returncode == 2
    assert "not allowed with argument" in result.stderr


def test_a_negative_variance_is_refused(searched_csv):
    result = run("report", searched_csv, *FAST, "--var-trial-sharpes", "-1")
    assert result.returncode == 2
    assert "non-negative" in result.stderr


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("0.4\nnot-a-number\n0.9\n", "line 2: 'not-a-number' is not a number"),
        ("0.4\n", "holds 1 value(s)"),
        ("# every line a comment\n\n", "holds 0 value(s)"),
    ],
)
def test_an_unusable_trial_sharpes_file_names_the_problem(tmp_path, searched_csv, body, expected):
    path = tmp_path / "trials.txt"
    path.write_text(body, encoding="utf-8")
    result = run("report", searched_csv, *FAST, "--trial-sharpes", str(path))
    assert result.returncode == 2
    assert expected in result.stderr
    assert "Traceback" not in result.stderr


def test_a_missing_trial_sharpes_file_exits_two(tmp_path, searched_csv):
    result = run("report", searched_csv, *FAST, "--trial-sharpes", str(tmp_path / "gone.txt"))
    assert result.returncode == 2
    assert "no such file" in result.stderr


def test_a_trial_sharpes_file_that_is_not_decodable_text_exits_two(tmp_path, searched_csv):
    path = tmp_path / "trials.txt"
    path.write_bytes("0.4\n0.9\n".encode("utf-16"))
    result = run("report", searched_csv, *FAST, "--trial-sharpes", str(path))
    assert result.returncode == 2
    assert "is not UTF-8 text" in result.stderr
    assert "Traceback" not in result.stderr


def test_the_trial_sharpes_file_accepts_a_row_as_well_as_a_column(tmp_path, searched_strategy):
    """A search writes a column; a person types a row. Both are the same file to the reader."""
    from quackz.cli import _read_trial_sharpes

    column = tmp_path / "column.txt"
    column.write_text("1.0\n# a comment\n\n-0.5\n2.25\n", encoding="utf-8")
    row = tmp_path / "row.txt"
    row.write_text("1.0, -0.5, 2.25  # the same three\n", encoding="utf-8")

    expected = np.array([1.0, -0.5, 2.25])
    np.testing.assert_allclose(_read_trial_sharpes(column), expected)
    np.testing.assert_allclose(_read_trial_sharpes(row), expected)


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
    assert (args.trial_sharpes, args.var_trial_sharpes) == (None, None)


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
