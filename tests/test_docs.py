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

import html
import importlib.util
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from quackz import __version__
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
    "Execution delay retention": (
        "latency_retention_ratio_warn",
        "latency_retention_ratio_fail",
    ),
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


def readme_output_block() -> list[str]:
    """The demo block from the README, with the shell prompt and the marked cuts removed.

    A cut is one line whose text begins with `...`, which is why the README keeps them to
    one line each. Everything else in that block claims to be output and is checked as such.
    """
    lines = []
    for line in code_blocks(section("What it prints"), "text")[0].splitlines():
        if line.lstrip().startswith("...") or line.startswith("$ ") or not line.strip():
            continue
        lines.append(line)
    return lines


def test_every_line_of_the_readme_block_is_real_output(capsys):
    """The block says it is a real run, so every line of it has to come out of one.

    This is the claim the rest of the README rests on. A number nudged by hand, a row
    dropped without a marker, or a verdict count left behind by a threshold change all fail
    here rather than in front of a reader.
    """
    load_example("overfit_demo").main()
    printed = capsys.readouterr().out.splitlines()

    quoted = readme_output_block()
    assert len(quoted) > 30, "the README block no longer quotes the demo"
    for line in quoted:
        assert line in printed, line


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


def _escaped(text: str) -> str:
    """The card is HTML, so the captured output appears in it escaped, not raw."""
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def test_the_committed_demo_output_still_matches_a_live_run() -> None:
    """The Pages card publishes this output, so a stale copy is a lie on a public page.

    The card is generated outside this repository and committed, because the generator is
    deliberately not a repository and no job here could check it out. That puts the freshness
    burden on the only test suite that can run the demo and compare, which is this one.
    """
    committed = (ROOT / "docs" / "evidence" / "demo.txt").read_text(encoding="utf-8")
    live = subprocess.run(
        [sys.executable, "examples/overfit_demo.py"],
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
        cwd=ROOT,
    ).stdout
    assert committed == live, (
        "docs/evidence/demo.txt no longer matches a live run. "
        "Run: uv run python scripts/capture_evidence.py, then regenerate the card."
    )


def test_the_published_card_carries_the_output_it_claims_to() -> None:
    card = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
    demo = (ROOT / "docs" / "evidence" / "demo.txt").read_text(encoding="utf-8")
    assert _escaped(demo.rstrip()) in card, "the card's terminal block is not the captured output"
    # The card tells a reader that a test fails when the output stops matching. This is that
    # test, and this assertion is what stops that sentence becoming false by deletion.
    assert "a test fails when it" in card


def test_the_card_states_numbers_that_are_true_today() -> None:
    facts = json.loads((ROOT / "docs" / "evidence" / "facts.json").read_text(encoding="utf-8"))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-o", "addopts=", "--collect-only", "-q"],
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
        cwd=ROOT,
    )
    match = re.search(r"^(\d+) tests? collected", result.stdout, re.MULTILINE)
    assert match is not None, f"no collection total in:\n{result.stdout[-400:]}"
    assert facts["tests"] == int(match.group(1)), "facts.json's test total is stale"
    # Against the package version, never `git describe`: actions/checkout clones without tags,
    # so a git-based assertion tests the shape of the checkout rather than the release.
    assert facts["release"] == f"v{__version__}"
    card = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
    assert f"<dd>{facts['tests']}</dd>" in card
    assert f"<dd>{facts['release']}</dd>" in card


def test_the_readme_frame_is_built_from_the_captured_output() -> None:
    """The animated frame in the first screenful has to be the real run, not a picture of one.

    Every text line the SVG draws, minus the prompt line it adds and the truncation note it
    ends with, must appear in the captured output in the same order. Written this way rather
    than by re-deriving the generator's truncation arithmetic, because a test that reimplements
    the thing it checks passes for the wrong reason.
    """
    svg = (ROOT / "docs" / "demo.svg").read_text(encoding="utf-8")
    demo = (ROOT / "docs" / "evidence" / "demo.txt").read_text(encoding="utf-8")

    drawn = [html.unescape(m) for m in re.findall(r"<text[^>]*>(.*?)</text>", svg, re.DOTALL)]
    assert drawn, "the frame draws no text at all"
    assert drawn[0].startswith("$ "), "the frame does not open on the command it ran"
    assert drawn[-1].startswith("... ") and "more lines" in drawn[-1]

    body = [line for line in drawn[1:-2] if line.strip()]
    haystack = demo.splitlines()
    position = 0
    for line in body:
        stem = line[:-3] if line.endswith("...") else line
        while position < len(haystack) and not haystack[position].startswith(stem):
            position += 1
        assert position < len(haystack), f"the frame draws a line the run never printed: {line!r}"
        position += 1

    # ASCII only, checked here as well as by the tree scan in test_bytes.py, and for a different
    # reason: the frame is GENERATED, so a non ASCII glyph would arrive from a code change rather
    # than from anyone typing one, and it is served through a proxy that renders it as an image
    # where nobody can select the character and look at it. The comment that used to sit here
    # named a test in a sibling repository as covering this tree. It did not exist here.
    assert svg.isascii()
    assert "<script" not in svg, "a README image is served through a proxy that strips script"
