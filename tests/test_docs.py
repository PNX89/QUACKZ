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
import tomllib
from functools import cache
from pathlib import Path
from types import ModuleType

import pytest

from quackz import __version__
from quackz.checks import DEFAULT_THRESHOLDS
from quackz.cli import build_parser
from test_doc_contract import IMPLEMENTATIONS

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


# A cut in the README block is one line whose text begins with this, which is why the README
# keeps every cut to a single line. A cut in the middle of the block declares how many rows
# it stands for; the one at the end declares how many checks follow it and what they say.
CUT_MARKER = "..."
ROWS_CUT = re.compile(r"^\.\.\. (\d+) rows? cut here\b")
TRAILING_CUT = re.compile(r"^\.\.\. (\d+) more checks cut here, all PASS$")


def check_trailing_cut(marker: str, remaining: list[str]) -> None:
    """The last cut stands for whole checks, so what it claims about them is checkable."""
    declared = TRAILING_CUT.match(marker)
    assert declared is not None, (
        f"this test knows two shapes of cut marker and this is neither: {marker!r}. Teach it "
        "the new shape rather than leaving the lines behind it unchecked."
    )
    verdicts = [line for line in remaining if line.startswith("[")]
    assert len(verdicts) == int(declared.group(1)), (
        f"the marker says {declared.group(1)} checks follow it; the run prints {len(verdicts)}"
    )
    failing = [line for line in verdicts if not line.startswith("[PASS]")]
    assert failing == [], (
        f"the marker says the checks it hides all pass, and these do not: {failing}"
    )


def readme_output_block() -> list[str]:
    """The demo block from the README, with the shell prompt and blank lines removed.

    The cut markers are KEPT. They are what licenses a gap between the block and the run,
    so a reader of this list that could not see them would have no way to tell a cut the
    README declares from a line that quietly went missing. Everything else in that block
    claims to be output and is checked as such.
    """
    lines = []
    for line in code_blocks(section("What it prints"), "text")[0].splitlines():
        if line.startswith("$ ") or not line.strip():
            continue
        lines.append(line)
    return lines


def test_every_line_of_the_readme_block_is_real_output(capsys):
    """The block says it is a real run, so every line of it has to come out of one, in order.

    This is the claim the rest of the README rests on, and it used to be checked with `line
    in printed`. That is a membership test: it caught a number nudged by hand and was blind
    to a row deleted with no marker and to two rows swapped, so an entire FAIL check could
    be lifted out of the block and the suite stayed green. What walks the two together now
    is a cursor, and the only thing allowed to advance it past a printed line is a cut
    marker, which is precisely what the README promises about its three cuts.
    """
    load_example("overfit_demo").main()
    printed = [line for line in capsys.readouterr().out.splitlines() if line.strip()]

    quoted = readme_output_block()
    assert len(quoted) > 30, "the README block no longer quotes the demo"
    cuts = [line for line in quoted if line.lstrip().startswith(CUT_MARKER)]
    assert len(cuts) == 3, (
        "the README says the three marked lines are the only places anything was taken "
        f"out of this block, and it now has {len(cuts)}. A new cut is a new gap this test "
        "stops watching, so declare it in the prose as well or do not make it."
    )

    cursor = 0
    for position, entry in enumerate(quoted):
        rows = ROWS_CUT.match(entry.lstrip())
        if rows is not None:
            # Each marker says how many rows it stands for, so the gap it opens is exactly
            # that wide. Letting a marker mean "some lines" instead would put the one row
            # sitting between two of them back out of sight, which is where it was.
            cursor += int(rows.group(1))
            continue
        if entry.lstrip().startswith(CUT_MARKER):
            assert position == len(quoted) - 1, (
                f"a cut that declares no row count only makes sense at the end: {entry!r}"
            )
            check_trailing_cut(entry.lstrip(), printed[cursor:])
            cursor = len(printed)
            continue
        assert cursor < len(printed), f"the block quotes a line the run never printed: {entry!r}"
        assert printed[cursor] == entry, (
            "the block and the run diverge with no cut marked between them.\n"
            f"  block: {entry!r}\n"
            f"  run:   {printed[cursor]!r}"
        )
        cursor += 1

    # The "### Deflated Sharpe" section quotes a second table, in the same "this is what it
    # prints" register, with none of its eleven numbers checked anywhere: it carries no cut
    # marker, so it was never reached by `readme_output_block`, which only ever looks at the
    # first text block of "What it prints". Checked here against the SAME live run, on the
    # full, untruncated output, since the table is printed further down than this block cuts.
    trial_blocks = [b for b in code_blocks(README, "text") if "DEFLATED SHARPE AGAINST" in b]
    assert len(trial_blocks) == 1, "the deflation table block moved, or there is more than one"
    trial_lines = [line for line in trial_blocks[0].splitlines() if line.strip()]
    assert len(trial_lines) > 10, "the deflation table block no longer quotes the trial grid"
    trial_position = 0
    for line in trial_lines:
        while trial_position < len(printed) and printed[trial_position] != line:
            trial_position += 1
        assert trial_position < len(printed), (
            f"the deflation table quotes a line the run never printed: {line!r}"
        )
        trial_position += 1


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
    """The card shows the run in two parts, and both of them still have to be the run.

    The first screenful sits in the open block and the rest is behind a disclosure, so
    looking for the whole capture as one string would fail on a card that is perfectly
    honest. What has to hold is that the blocks put back together are the captured output
    byte for byte, which is the claim, rather than how the card chooses to fold it.
    """
    card = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
    demo = (ROOT / "docs" / "evidence" / "demo.txt").read_text(encoding="utf-8")

    blocks = re.findall(r"<pre[^>]*>(.*?)</pre>", card, re.DOTALL)
    assert len(blocks) == 2, (
        f"the card lays the captured run out in {len(blocks)} blocks rather than two. If the "
        "card legitimately changed shape, follow it here deliberately; the point of the count "
        "is that the shape cannot change without someone reading this test again."
    )
    assert "\n".join(blocks) == _escaped(demo.rstrip()), (
        "the card's terminal blocks are not the captured output"
    )

    # The figure is read out of the sentence that carries it rather than looked for in the
    # page: a card this long contains almost any small number somewhere.
    summary = re.search(r"<summary>(.*?)</summary>", card, re.DOTALL)
    assert summary is not None, "the rest of the run is no longer behind a labelled disclosure"
    declared = re.search(r"([\d,]+)", summary.group(1))
    assert declared is not None, (
        f"the disclosure does not say how much it holds: {summary.group(1)!r}"
    )
    assert int(declared.group(1).replace(",", "")) == len(blocks[1].splitlines()), (
        f"the disclosure says {summary.group(1)!r} and hides {len(blocks[1].splitlines())} lines"
    )

    # The card tells a reader that a test fails when the output stops matching. This is that
    # test, and this assertion is what stops that sentence becoming false by deletion.
    assert "a test fails when it" in card


@cache
def collected() -> str:
    """What pytest says it would run, asked once and shared by the two tests that need it."""
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-o", "addopts=", "--collect-only", "-q"],
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
        cwd=ROOT,
    ).stdout


def test_the_card_states_numbers_that_are_true_today() -> None:
    facts = json.loads((ROOT / "docs" / "evidence" / "facts.json").read_text(encoding="utf-8"))
    match = re.search(r"^(\d+) tests? collected", collected(), re.MULTILINE)
    assert match is not None, f"no collection total in:\n{collected()[-400:]}"
    assert facts["tests"] == int(match.group(1)), "facts.json's test total is stale"
    # Against the package version, never `git describe`: actions/checkout clones without tags,
    # so a git-based assertion tests the shape of the checkout rather than the release.
    assert facts["release"] == f"v{__version__}"
    card = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
    assert f"<dd>{facts['tests']}</dd>" in card
    assert f"<dd>{facts['release']}</dd>" in card
    # The card's own claim is that it "cannot quietly drift from the code it describes"
    # because a test fails when it stops matching a live run. Before this, the sentence
    # covered only half the card: these two facts are written by the same capture script
    # and read from the same facts.json, and neither was checked against the card at all.
    assert f"<dd>{facts['python']}</dd>" in card, (
        "the card's Python range no longer matches facts.json"
    )
    assert f"captured on {facts['captured']}" in card, (
        "the card's capture date no longer matches facts.json"
    )


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


def test_shipping_py_typed_means_a_type_checker_actually_runs() -> None:
    """`py.typed` tells a downstream user that these annotations were checked by something.

    Nothing checked them. The shared CI workflow's type step is opt in and defaults to off,
    so the marker file promised a checked package while no checker ran on any of the four
    interpreters, and CONTRIBUTING told contributors that typing was a gate here. All three
    parts are asserted together, because it was the wiring rather than the claim that went
    missing and any one of them alone leaves the promise unkept.
    """
    assert (ROOT / "src" / "quackz" / "py.typed").exists()

    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dev = config["dependency-groups"]["dev"]
    assert any(requirement.startswith("mypy") for requirement in dev), (
        f"mypy is not in the dev group, so `uv run mypy` cannot run: {dev}"
    )
    # Scoped to src on purpose: src is the whole of what the marker file promises about.
    assert config["tool"]["mypy"]["files"] == ["src"]

    # Read out of the `with:` block of the job that calls the shared workflow, not searched
    # for anywhere in the file. The input defaults to false, so an absent key and a key set
    # to false are the same skipped step, and only the block that carries it can say which.
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    inputs = ci.split("\n    with:\n", 1)
    assert len(inputs) == 2, "the checks job passes no inputs to the shared workflow at all"
    given = dict(
        re.findall(r"^      ([a-z-]+): *(.+?)\s*$", inputs[1], re.MULTILINE),
    )
    assert given.get("run-mypy") == "true", (
        f"CI does not enable the shared workflow's type step, so nothing runs mypy: {given}"
    )


# The contract kinds this repository is on the hook for, pinned by name and by count. They
# are read out of the generated contract file, so a kind quietly disappearing from the shared
# manifest would otherwise leave the test below checking one thing fewer and still passing.
CONTRACT_KINDS = {"NUMBER", "COMMAND", "OUTPUT", "REFERENCE"}


def test_every_contract_kind_names_a_test_that_pytest_actually_collects() -> None:
    """The generated contract resolves its names against the raw text of the suite.

    `f"def {name}(" in suite` is a substring search, so a comment, a docstring or any string
    literal carrying the name satisfies it: a kind can lose its implementation entirely while
    the contract still reports it implemented. Renaming the OUTPUT test and leaving its old
    name in a comment above it was enough, and the whole suite stayed green.

    That file is generated from a shared manifest and is rewritten on every publish, so the
    fix cannot live in it. The resolution it should be doing is done here instead, against
    the node ids pytest collects, where a comment cannot appear.
    """
    assert set(IMPLEMENTATIONS) == CONTRACT_KINDS, (
        "the contract kinds this repository implements have changed. Update CONTRACT_KINDS "
        f"deliberately rather than letting this test cover fewer of them: {set(IMPLEMENTATIONS)}"
    )
    names = set()
    for line in collected().splitlines():
        if "::" in line:
            names.add(line.split("::", 1)[1].strip().split("[", 1)[0])
    assert names, "pytest collected nothing, so this test would pass for the wrong reason"

    missing = {kind: name for kind, name in IMPLEMENTATIONS.items() if name not in names}
    assert missing == {}, (
        f"these contract kinds name a test that pytest does not collect: {missing}. The name "
        "may still appear in the tree as a comment or a string, which is what the generated "
        "contract check would accept and what this one will not."
    )
