"""Fixtures shared by the end-to-end tests.

Two strategies, both synthetic and seeded, because the point of a fixture here is that its
verdicts are known in advance.

`honest_strategy` has a small genuine edge on a persistent state variable, uses only
information available at the decision time, and trades rarely enough to survive costs. It
is what an unexciting real strategy looks like, and nothing in the report should FAIL on it.

`searched_strategy` is the overfit case built the way overfitting actually happens: two
hundred random signals are scored on the same sample and the best one is kept. Its Sharpe
is a selection artefact, and because the search reports the Sharpe of every configuration
it touched, it also exercises the honest way to populate V[{SR_n}].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quackz import metrics
from quackz.evaluate import Evaluation, evaluate
from quackz.returns import gross_returns, net_returns

PPY = 252.0
N_BARS = 600
COSTS_BPS = 2.0

# Enough resamples for the percentiles to be stable to two decimals, few enough that the
# whole suite stays under a couple of seconds.
RESAMPLES = 200


def _prices(bar_returns: np.ndarray, *, start: str = "2019-01-02") -> pd.Series:
    index = pd.date_range(start, periods=len(bar_returns), freq="B")
    return pd.Series(100.0 * np.cumprod(1.0 + bar_returns), index=index, name="close")


def build_honest(*, n: int = N_BARS, seed: int = 17) -> tuple[pd.Series, pd.Series]:
    """A persistent state that pays a little, traded on its sign with a one-bar lag."""
    rng = np.random.default_rng(seed)
    state = np.zeros(n)
    shocks = rng.standard_normal(n)
    for t in range(1, n):
        state[t] = 0.97 * state[t - 1] + shocks[t]
    bar_returns = 0.01 * (0.05 * state + rng.standard_normal(n))
    prices = _prices(bar_returns)
    return prices, pd.Series(np.sign(state), index=prices.index, name="position")


def build_searched(
    *, n: int = N_BARS, n_trials: int = 200, seed: int = 23
) -> tuple[pd.Series, pd.Series, np.ndarray]:
    """The best of `n_trials` random signals on a market with no edge in it at all.

    Returns the prices, the winning position series, and the ANNUALIZED Sharpe of every
    trial, which is what `evaluate(trial_sharpes=...)` wants.
    """
    rng = np.random.default_rng(seed)
    prices = _prices(0.01 * rng.standard_normal(n))
    sharpes = np.empty(n_trials)
    candidates = np.empty((n_trials, n))
    for trial in range(n_trials):
        # A random walk thresholded at zero: persistent enough to hold a position for a
        # while, which is what a plausible-looking overfit signal does.
        candidates[trial] = np.sign(np.cumsum(rng.standard_normal(n)))
        positions = pd.Series(candidates[trial], index=prices.index)
        sharpes[trial] = metrics.sharpe(gross_returns(prices, positions), periods_per_year=PPY)
    winner = int(np.argmax(sharpes))
    positions = pd.Series(candidates[winner], index=prices.index, name="position")
    return prices, positions, sharpes


def build_peeking(*, n: int = N_BARS, seed: int = 31) -> tuple[pd.Series, pd.Series]:
    """A position set from the return it is about to earn: the level is the tell."""
    rng = np.random.default_rng(seed)
    bar_returns = 0.01 * rng.standard_normal(n)
    prices = _prices(bar_returns)
    peeking = np.zeros(n)
    peeking[:-1] = np.sign(bar_returns[1:])
    return prices, pd.Series(peeking, index=prices.index, name="position")


@pytest.fixture(scope="session")
def honest_strategy() -> tuple[pd.Series, pd.Series]:
    return build_honest()


@pytest.fixture(scope="session")
def peeking_strategy() -> tuple[pd.Series, pd.Series]:
    return build_peeking()


@pytest.fixture(scope="session")
def searched_strategy() -> tuple[pd.Series, pd.Series, np.ndarray]:
    return build_searched()


@pytest.fixture(scope="session")
def honest_eval(honest_strategy) -> Evaluation:
    """The honest strategy with no search declared: every check should pass."""
    prices, positions = honest_strategy
    return evaluate(
        prices,
        positions,
        costs_bps=COSTS_BPS,
        periods_per_year=PPY,
        bootstrap_resamples=RESAMPLES,
    )


@pytest.fixture(scope="session")
def fallback_eval(honest_strategy) -> Evaluation:
    """Trials declared without their Sharpes, which is the iid fallback path."""
    prices, positions = honest_strategy
    return evaluate(
        prices,
        positions,
        costs_bps=COSTS_BPS,
        periods_per_year=PPY,
        n_trials=5,
        bootstrap_resamples=RESAMPLES,
    )


@pytest.fixture(scope="session")
def searched_eval(searched_strategy) -> Evaluation:
    prices, positions, trial_sharpes = searched_strategy
    return evaluate(
        prices,
        positions,
        costs_bps=COSTS_BPS,
        periods_per_year=PPY,
        n_trials=len(trial_sharpes),
        trial_sharpes=trial_sharpes,
        bootstrap_resamples=RESAMPLES,
    )


@pytest.fixture(scope="session")
def peeking_eval(peeking_strategy) -> Evaluation:
    prices, positions = peeking_strategy
    return evaluate(
        prices,
        positions,
        costs_bps=COSTS_BPS,
        periods_per_year=PPY,
        bootstrap_resamples=RESAMPLES,
    )


@pytest.fixture(scope="session")
def reconciled_eval(honest_strategy) -> Evaluation:
    """The honest strategy against a claimed stream that is 2 percent too generous."""
    prices, positions = honest_strategy
    claimed = net_returns(prices, positions, costs_bps=COSTS_BPS) * 1.02
    return evaluate(
        prices,
        positions,
        costs_bps=COSTS_BPS,
        periods_per_year=PPY,
        claimed_returns=claimed,
        bootstrap_resamples=RESAMPLES,
    )


# The dispersion declared alongside the capped record. Named because two fixtures and the
# test that brackets the cap all have to quote the same one; a search re-run at a different
# dispersion is a different search, and the break-even count would not be comparable.
CAPPED_VAR_TRIAL_SHARPES = 0.02


def build_capped(*, n: int = N_BARS, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    """A record so strong the break-even trial search runs into its own cap.

    A Sharpe near ten held for 600 bars keeps the deflated Sharpe above the target even at
    a million trials. Nothing about that is realistic, which is the point of having a cap
    at all, and it is the third arm of the break-even sentence that no other fixture reaches.
    """
    rng = np.random.default_rng(seed)
    bar_returns = 0.0025 + 0.004 * rng.standard_normal(n)
    prices = _prices(bar_returns)
    return prices, pd.Series(np.ones(n), index=prices.index, name="position")


@pytest.fixture(scope="session")
def capped_strategy() -> tuple[pd.Series, pd.Series]:
    return build_capped()


@pytest.fixture(scope="session")
def capped_eval(capped_strategy) -> Evaluation:
    prices, positions = capped_strategy
    return evaluate(
        prices,
        positions,
        periods_per_year=PPY,
        n_trials=10,
        var_trial_sharpes=CAPPED_VAR_TRIAL_SHARPES,
        bootstrap_resamples=RESAMPLES,
    )


def write_csv(
    path,
    prices: pd.Series,
    positions: pd.Series,
    *,
    claimed: pd.Series | None = None,
) -> None:
    """Write the fixture out in the shape the CLI reads."""
    frame = pd.DataFrame(
        {
            "date": prices.index.strftime("%Y-%m-%d"),
            "close": prices.to_numpy(),
            "position": positions.to_numpy(),
        }
    )
    if claimed is not None:
        values = np.asarray(claimed, dtype=float)
        if values.size == len(prices) - 1:
            # A return stream is one bar shorter than the prices it came from, so the first
            # row gets a zero. `evaluate` drops that leading value rather than aligning it.
            values = np.concatenate(([0.0], values))
        frame["pnl"] = values
    frame.to_csv(path, index=False)
