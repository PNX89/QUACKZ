"""Tests for the two things a reader meets first: the README and the examples.

Documentation rots silently. A renamed flag or a moved threshold leaves the code correct
and the README wrong, and the README is the part a reader trusts. So the quickstart block
is executed rather than read, every command line in the README is parsed by the real
argument parser, the thresholds table is compared against the constants it claims to
document, and both examples are run.

The examples are loaded from their files rather than imported, because `examples/` is not
a package and must not become one just to be testable.
"""

from __future__ import annotations

import importlib.util
import re
import shlex
import sys
from pathlib import Path
from types import ModuleType

import pytest

from quackz.checks import DEFAULT_THRESHOLDS
from quackz.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")

# Row label in the README thresholds table, then the fields of Thresholds it documents.
DOCUMENTED_THRESHOLDS = {
    "Deflated Sharpe": ("dsr_warn", "dsr_fail"),
    "Noise floor ratio": ("noise_floor_ratio_warn", "noise_floor_ratio_fail"),
    "Cost break-even": ("cost_break_even_bps_warn", "cost_break_even_bps_fail"),
    "Bootstrap p-value": ("bootstrap_p_value_warn", "bootstrap_p_value_fail"),
    "Subperiod dispersion ratio": (
        "stability_dispersion_ratio_warn",
        "stability_dispersion_ratio_fail",
    ),
    "Profit concentration": (
        "concentration_profit_share_warn",
        "concentration_profit_share_fail",
    ),
    "Execution delay level": ("latency_annual_sharpe_warn", "latency_annual_sharpe_fail"),
    "Reconciliation correlation": ("reconcile_correlation_warn", "reconcile_correlation_fail"),
    "Reconciliation wealth gap": (
        "reconcile_cumulative_gap_warn",
        "reconcile_cumulative_gap_fail",
    ),
}


def section(title: str) -> str:
    """The README text under one heading, up to the next heading of the same level."""
    start = README.index(f"\n## {title}\n")
    rest = README[start + 1 :]
    end = rest.find("\n## ")
    return rest if end < 0 else rest[:end]


def code_blocks(text: str, language: str) -> list[str]:
    return re.findall(rf"```{language}\n(.*?)```", text, re.S)


def load_example(name: str) -> ModuleType:
    path = ROOT / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before execution so a dataclass or a pickle defined in the example can
    # find its own module, and removed afterwards so the test leaves no import behind.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[name]
    return module


def quackz_commands() -> list[list[str]]:
    """Every `quackz ...` invocation anywhere in the README, as argument lists."""
    out = []
    for block in code_blocks(README, "bash"):
        for line in block.splitlines():
            words = shlex.split(line.split("#", 1)[0].strip())
            if words[:2] == ["uv", "run"]:
                words = words[2:]
            if words and words[0] == "quackz":
                out.append(words[1:])
    return out


# --------------------------------------------------------------------------------------
# The quickstart
# --------------------------------------------------------------------------------------


def test_the_quickstart_python_block_runs(capsys):
    blocks = code_blocks(section("Quickstart"), "python")
    assert len(blocks) == 1
    exec(compile(blocks[0], "README.md", "exec"), {"__name__": "__readme__"})

    printed = capsys.readouterr().out.splitlines()
    assert re.fullmatch(r"\d+ FAIL, \d+ WARN, \d+ PASS", printed[0])
    assert float(printed[1]) > 0.0
    assert "BACKTEST AUDIT" in printed[2]


def test_every_documented_command_line_parses():
    commands = quackz_commands()
    assert commands, "the README no longer shows a quackz command line"
    for argv in commands:
        parsed = build_parser().parse_args(argv)
        assert parsed.command == "report"


def test_the_documented_flags_are_the_real_ones():
    """A flag renamed in the CLI must not survive in the README."""
    documented = set(re.findall(r"(?<![\w-])--[a-z][a-z-]+", section("What it prints")))
    known = {action for action in build_parser().parse_args(["report", "x"]).__dict__}
    assert documented, "the README no longer shows any flags"
    for flag in documented:
        assert flag[2:].replace("-", "_") in known, flag


# --------------------------------------------------------------------------------------
# The thresholds table
# --------------------------------------------------------------------------------------


def _documented_value(cell: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)(%?)", cell)
    assert match is not None, cell
    value = float(match.group(1))
    return value / 100.0 if match.group(2) else value


@pytest.mark.parametrize(("label", "fields"), sorted(DOCUMENTED_THRESHOLDS.items()))
def test_the_thresholds_table_matches_the_constants(label, fields):
    rows = [line for line in section("Default thresholds").splitlines() if line.startswith("| ")]
    matching = [row for row in rows if row.split("|")[1].strip() == label]
    assert len(matching) == 1, f"{label} is not a row of the thresholds table"

    cells = matching[0].split("|")
    for cell, name in zip(cells[2:4], fields, strict=True):
        assert _documented_value(cell) == pytest.approx(getattr(DEFAULT_THRESHOLDS, name)), name


def test_every_threshold_is_documented():
    rows = [line for line in section("Default thresholds").splitlines() if line.startswith("| ")]
    documented = {name for fields in DOCUMENTED_THRESHOLDS.values() for name in fields}
    assert documented == set(DEFAULT_THRESHOLDS.__dataclass_fields__)
    # Two header rows plus one row per documented pair.
    assert len(rows) == len(DOCUMENTED_THRESHOLDS) + 2


def test_the_override_example_uses_real_threshold_names():
    for block in code_blocks(section("Default thresholds"), "python"):
        for name in re.findall(r'"([a-z_]+)":', block):
            assert name in DEFAULT_THRESHOLDS.__dataclass_fields__, name


# --------------------------------------------------------------------------------------
# The examples
# --------------------------------------------------------------------------------------


def test_the_overfit_demo_fails_on_selection_and_nothing_else(capsys):
    load_example("overfit_demo").main()
    printed = capsys.readouterr().out

    assert printed.startswith("Search: 200 random signals")
    assert "Verdict: FAIL." in printed
    assert "[FAIL] Deflated Sharpe" in printed
    assert "[FAIL] Noise floor" in printed
    # The whole point of the demo: the returns of the winning strategy are well behaved,
    # and selection is visible only against the search that produced them.
    assert printed.count("[FAIL]") == 2
    assert "[PASS] Resampling" in printed


def test_the_momentum_demo_passes_every_check(capsys):
    load_example("momentum_demo").main()
    printed = capsys.readouterr().out

    assert printed.startswith("Trend rule on")
    assert "Verdict: PASS." in printed
    assert "[FAIL]" not in printed
    assert "[WARN]" not in printed


@pytest.mark.parametrize("name", ["overfit_demo", "momentum_demo"])
def test_the_examples_are_deterministic(name, capsys):
    module = load_example(name)
    module.main()
    first = capsys.readouterr().out
    module.main()
    assert capsys.readouterr().out == first
