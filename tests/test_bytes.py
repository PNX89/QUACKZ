"""No character above ASCII, in the tree or in a rendered report, and the reason is arithmetic.

A comment in `test_docs.py` said `test_every_text_file_in_the_repository_is_pure_ascii` covered
the tree. That test lives in a sibling repository. Here there was none, under any name.

WHY IT MATTERS IN THIS ONE SPECIFICALLY. Everything this package produces is numbers in
fixed-width columns: `f"{trials:>10}  {benchmark:>18}  {dsr:>8}"` and a dozen lines like it. Two
things break that, and both are invisible to the person reading the diff.

    U+2212 MINUS SIGN     draws exactly like the hyphen-minus in a Sharpe ratio of -0.31 and is
                          rejected by float(), json.loads() and every spreadsheet import
    U+2010 HYPHEN         the same trick in the other direction, and it survives copy and paste

A report is not decoration here. It is the artefact somebody pastes into a spreadsheet to argue
about, so a number in it that will not parse is worse than one that is wrong, because a wrong
number gets argued with and an unparseable one gets retyped.

The tree rule and the output rule are both asserted, because neither implies the other: the
renderers build strings from format specs, so an ASCII source can still emit a non-ASCII glyph
through a numeric formatter, and a clean report says nothing about the fixtures beside it.
"""

from __future__ import annotations

import pathlib
import subprocess
import unicodedata

import pytest

from quackz.report import json_report, markdown_report, text_report

REPO = pathlib.Path(__file__).resolve().parents[1]

#: EVERY evaluation shape the suite builds, not one of them. The renderers branch on the
#: verdict, so checking a single fixture would leave the columns that only a failing
#: evaluation prints unscanned, and those are the ones carrying negative numbers.
EVALUATIONS = ("honest_eval", "peeking_eval", "searched_eval", "reconciled_eval", "fallback_eval")

#: Extensions whose bytes are not meant to be read, so a byte above 0x7F carries no meaning.
OPAQUE = frozenset({".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".woff", ".woff2", ".zip"})


def committed() -> list[pathlib.Path]:
    """What git tracks, split on NUL because a path may legally contain a newline."""
    named = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "-z"], capture_output=True, check=True
    )
    return [
        candidate
        for entry in named.stdout.decode().split("\0")
        if entry
        for candidate in [REPO / entry]
        if candidate.is_file() and candidate.suffix.lower() not in OPAQUE
    ]


def describe(text: str) -> list[str]:
    """Every non-ASCII character in a string, named, because 'non-ASCII' is not actionable.

    U+2212 and U+002D are the same three pixels on screen. A failure that does not print the
    code point sends the reader looking for something they cannot see.
    """
    return [
        f"U+{ord(character):04X} {unicodedata.name(character, 'unnamed')}"
        for character in dict.fromkeys(character for character in text if not character.isascii())
    ]


def test_no_committed_file_carries_a_byte_above_ascii() -> None:
    """Bytes, not decoded text.

    Decoding first means a file that is not valid UTF-8 raises and gets skipped, so the files
    most likely to be carrying something odd would be the exempt ones.
    """
    offences, scanned = [], 0
    for path in committed():
        raw = path.read_bytes()
        scanned += 1
        if any(byte > 0x7F for byte in raw):
            offences.append(f"{path.relative_to(REPO)}: {describe(raw.decode('utf-8', 'replace'))}")
    assert scanned > 20, f"git listed {scanned} files, which is too few to be the whole tree"
    assert offences == [], offences


@pytest.mark.parametrize("fixture", EVALUATIONS)
def test_no_renderer_emits_a_character_that_will_not_parse_as_a_number(
    request: pytest.FixtureRequest, fixture: str
) -> None:
    """The half the tree scan cannot reach.

    Source files hold format specs, not output. A locale, a numpy repr or a future thousands
    separator could put U+2212 or a narrow no-break space into a column while every file in the
    repository stayed pure ASCII, and the report would look right in a terminal and fail on
    import.
    """
    evaluation = request.getfixturevalue(fixture)
    for render in (text_report, markdown_report, json_report):
        rendered = render(evaluation)
        assert isinstance(rendered, str)
        assert describe(rendered) == [], f"{render.__name__} emitted {describe(rendered)}"


def test_the_minus_sign_this_forbids_is_the_one_that_looks_right(searched_eval) -> None:
    """The check, measured against the thing it exists to catch, rather than assumed.

    Without this, a scan that had quietly stopped detecting anything would still pass on a clean
    tree for ever. It also records the fact the docstring above rests on: the two characters are
    indistinguishable to a reader and completely different to a parser.
    """
    # WRITTEN AS AN ESCAPE, and the first version of this line was not. It held the character
    # itself, which meant this file broke the rule it exists to enforce the moment it was
    # committed, and ruff's RUF001 caught it before the test could. A test carrying a literal
    # copy of what it forbids is the smallest possible version of the defect this repository is
    # about, so it is written the way the rule requires: escaped, and named in the assertion.
    unicode_minus, ascii_minus = "\u2212", "-"
    assert describe(f"{unicode_minus}0.31") == ["U+2212 MINUS SIGN"]
    assert describe(f"{ascii_minus}0.31") == []
    assert float(f"{ascii_minus}0.31") == -0.31
    try:
        float(f"{unicode_minus}0.31")
    except ValueError:
        pass
    else:  # pragma: no cover - reached only if CPython starts accepting it
        raise AssertionError("float() now accepts U+2212, so this rule needs a new reason")
    assert text_report(searched_eval).count(ascii_minus) > 0, (
        "no report in this fixture contains a minus at all, so the column this test is about is "
        "not being exercised by it"
    )
