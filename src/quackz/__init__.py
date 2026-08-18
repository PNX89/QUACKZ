"""QUACKZ: audits a trading backtest for the ways backtests lie.

Input is a price series and a position series; output is a per-check verdict on the
statistical claims a backtest makes about itself.

Read `quackz.returns` first. It defines the position and cost conventions that every
other module depends on, and getting those wrong is the difference between an audit and
a second backtest.

The individual checks stay behind `quackz.checks` rather than being flattened into this
namespace: names like `bootstrap` and `reconcile` mean too many things at the top level of
a package to be worth the keystrokes saved. `evaluate` runs all of them at once and is the
entry point most callers want.
"""

from __future__ import annotations

from quackz import checks, metrics, report, returns, splits
from quackz._version import __version__
from quackz.checks import Thresholds, Verdict

# This rebinds the name `quackz.evaluate` from the module to the function inside it, which
# is the call almost everyone wants. Reach the module itself as `from quackz.evaluate
# import ...`, which resolves through sys.modules and is unaffected.
from quackz.evaluate import Evaluation, evaluate
from quackz.metrics import (
    DSRResult,
    annualized_return,
    annualized_vol,
    autocorr_1,
    calmar,
    cvar,
    deflated_sharpe,
    expected_max_sharpe,
    hac_tstat,
    max_drawdown,
    min_track_record_length,
    moments,
    probabilistic_sharpe,
    sharpe,
    sortino,
    ulcer_index,
)
from quackz.returns import (
    QuackzInputError,
    ReturnStreams,
    build_returns,
    gross_returns,
    infer_periods_per_year,
    net_returns,
    turnover,
)
from quackz.splits import EmbargoedKFold, WalkForward

__all__ = [
    "DSRResult",
    "EmbargoedKFold",
    "Evaluation",
    "QuackzInputError",
    "ReturnStreams",
    "Thresholds",
    "Verdict",
    "WalkForward",
    "__version__",
    "annualized_return",
    "annualized_vol",
    "autocorr_1",
    "build_returns",
    "calmar",
    "checks",
    "cvar",
    "deflated_sharpe",
    "evaluate",
    "expected_max_sharpe",
    "gross_returns",
    "hac_tstat",
    "infer_periods_per_year",
    "max_drawdown",
    "metrics",
    "min_track_record_length",
    "moments",
    "net_returns",
    "probabilistic_sharpe",
    "report",
    "returns",
    "sharpe",
    "sortino",
    "splits",
    "turnover",
    "ulcer_index",
]
