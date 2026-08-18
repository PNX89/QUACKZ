"""What a search of two hundred random signals produces, and what QUACKZ says about it.

The market here has no edge in it at all: driftless synthetic prices with a fixed seed. Two
hundred random signals are scored on that one sample and the best is kept, which is how
overfitting actually happens. The winner carries an annualized Sharpe near 3 on a one-year
track record.

The point of the demo is what the report does with it. The Sharpe of every configuration
the search touched is passed in as `trial_sharpes`, which is the honest way to populate
V[{SR_n}], and the deflated Sharpe and the noise floor then have the dispersion of the
search itself to measure the winner against. Everything else in the report passes: the
resampling, the cost sweep, the concentration, the stability and the latency checks all see
a well behaved track record, because it is one. Selection is not visible in the returns of
the strategy that won; it is visible only against the search that produced it.

Deterministic, offline and about a second to run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quackz import evaluate, metrics
from quackz.report import text_report
from quackz.returns import gross_returns

SEED = 7
N_BARS = 252
N_TRIALS = 200
SIGNAL_SMOOTHING = 10
ANNUAL_VOL = 0.20
PERIODS_PER_YEAR = 252.0
COSTS_BPS = 2.0
START = "2023-01-02"


def synthetic_prices(rng: np.random.Generator, *, n_bars: int = N_BARS) -> pd.Series:
    """A driftless random walk in price, so nothing in it can be predicted."""
    bar_returns = ANNUAL_VOL / np.sqrt(PERIODS_PER_YEAR) * rng.standard_normal(n_bars)
    index = pd.date_range(START, periods=n_bars, freq="B")
    return pd.Series(100.0 * np.cumprod(1.0 + bar_returns), index=index, name="close")


def random_signal(rng: np.random.Generator, *, n_bars: int, window: int) -> np.ndarray:
    """A smoothed random walk thresholded at zero, which holds a position for a while.

    The moving average is causal (only values up to bar t enter bar t's signal), so the
    demo cannot be dismissed as lookahead dressed up as overfitting.
    """
    weights = np.ones(window) / window
    smoothed = np.convolve(rng.standard_normal(n_bars), weights)[:n_bars]
    return np.sign(smoothed)


def search(
    prices: pd.Series, rng: np.random.Generator, *, n_trials: int = N_TRIALS
) -> tuple[pd.Series, np.ndarray]:
    """Score `n_trials` random signals on one sample and keep the best.

    Returns the winning position series and the ANNUALIZED Sharpe of every trial, which is
    what `evaluate(trial_sharpes=...)` expects.
    """
    n_bars = len(prices)
    candidates = np.empty((n_trials, n_bars))
    sharpes = np.empty(n_trials)
    for trial in range(n_trials):
        candidates[trial] = random_signal(rng, n_bars=n_bars, window=SIGNAL_SMOOTHING)
        positions = pd.Series(candidates[trial], index=prices.index)
        sharpes[trial] = metrics.sharpe(
            gross_returns(prices, positions), periods_per_year=PERIODS_PER_YEAR
        )
    winner = int(np.argmax(sharpes))
    return pd.Series(candidates[winner], index=prices.index, name="position"), sharpes


def main() -> None:
    rng = np.random.default_rng(SEED)
    prices = synthetic_prices(rng)
    positions, trial_sharpes = search(prices, rng)

    print(
        f"Search: {N_TRIALS} random signals on {N_BARS} bars of driftless synthetic prices, "
        f"seed {SEED}."
    )
    print(
        f"Trial Sharpes: best {trial_sharpes.max():.2f}, median "
        f"{np.median(trial_sharpes):.2f}, worst {trial_sharpes.min():.2f}, standard "
        f"deviation {trial_sharpes.std(ddof=1):.2f}."
    )
    print()

    evaluation = evaluate(
        prices,
        positions,
        costs_bps=COSTS_BPS,
        periods_per_year=PERIODS_PER_YEAR,
        n_trials=N_TRIALS,
        trial_sharpes=trial_sharpes,
        bootstrap_seed=0,
    )
    print(text_report(evaluation))


if __name__ == "__main__":
    main()
