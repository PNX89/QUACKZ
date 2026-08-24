"""Capture the demo's real output and the numbers the Pages card states.

WHY THIS EXISTS. The card at pnx89.github.io/QUACKZ shows the output of a real run and four
numbers about this repository. Both are committed, which means both can go stale. This writes
them from the source of truth in one step, and `tests/test_docs.py` fails when what is
committed stops matching a live run, so staleness is a red build rather than a quiet lie on a
public page.

WHAT IT WRITES.
    docs/evidence/demo.txt    stdout of the demo command, byte for byte
    docs/evidence/facts.json  test total, supported Python range, release tag, capture date

Every number comes from a command, never from a memory. The test total is collected by pytest
in a subprocess, the Python range is read out of the CI matrix, and the release is read from
git. Run it with:

    uv run python scripts/capture_evidence.py
"""

from __future__ import annotations

import datetime
import json
import os
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
EVIDENCE = REPO / "docs" / "evidence"
DEMO = ["uv", "run", "python", "examples/overfit_demo.py"]


def run(*args: str, cwd: pathlib.Path = REPO) -> str:
    result = subprocess.run(args, capture_output=True, text=True, cwd=cwd, timeout=600)
    if result.returncode:
        raise SystemExit(f"{args[0]} failed: {result.stderr.strip()[:400]}")
    return result.stdout


def test_total() -> int:
    """Collected, not counted by hand. -o addopts= neutralises this repository's own `-q`,
    which otherwise prints per-file totals instead of the line this parses."""
    out = run(sys.executable, "-m", "pytest", "-o", "addopts=", "--collect-only", "-q")
    match = re.search(r"^(\d+) tests? collected", out, re.MULTILINE)
    if not match:
        raise SystemExit(f"could not read a collection total from:\n{out[-400:]}")
    return int(match.group(1))


def python_range() -> str:
    """Read from the CI matrix, so the card cannot claim support CI does not test."""
    workflow = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    versions = sorted({v for v in re.findall(r'"(3\.\d+)"', workflow)}, key=lambda v: int(v[2:]))
    if not versions:
        raise SystemExit("no Python versions found in the CI matrix")
    return f"{versions[0]} to {versions[-1]}"


def release() -> str:
    """From the package version, cross-checked against the tag when one is reachable.

    `git describe` alone would be wrong here: a shallow checkout has no tags, and a version
    bumped without tagging would still report the old tag. The version is the claim; the tag
    is checked against it locally, where tags exist, so a mismatch is caught at capture time
    rather than published.
    """
    from quackz import __version__

    tag = f"v{__version__}"
    result = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    described = result.stdout.strip()
    if result.returncode == 0 and described != tag:
        raise SystemExit(
            f"the newest tag is {described} but the package version is {__version__}. "
            "Tag the release or fix the version before publishing a card that names one."
        )
    return tag


def main() -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=True)

    output = run(*DEMO)
    if not output.strip():
        raise SystemExit("the demo produced no output, refusing to write empty evidence")
    (EVIDENCE / "demo.txt").write_text(output, encoding="utf-8")

    # GITHUB_RUN_ID is set only inside Actions. Locally the card says "captured on <date>"
    # with no link rather than inventing one.
    run_id = os.environ.get("GITHUB_RUN_ID")
    repo_slug = os.environ.get("GITHUB_REPOSITORY", "PNX89/QUACKZ")
    facts = {
        "tests": test_total(),
        "python": python_range(),
        "release": release(),
        "captured": datetime.date.today().isoformat(),
        "runUrl": f"https://github.com/{repo_slug}/actions/runs/{run_id}" if run_id else None,
    }
    (EVIDENCE / "facts.json").write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {EVIDENCE / 'demo.txt'} ({len(output.splitlines())} lines)")
    print(f"wrote {EVIDENCE / 'facts.json'} {facts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
