"""What an honest, unexciting strategy looks like in the same report.

The prices here carry a slow, persistent drift, so a trend rule has something real to
find. The rule is the simplest one there is: hold the sign of the trailing k-bar return,
decided at the close and earning the next bar.

The lookback is chosen from a grid of five, on the same sample the report then judges, and
that small search is declared honestly: `n_trials=5` with the Sharpe of all five lookbacks
as `trial_sharpes`. Declaring it costs a little deflation and is what the deflated Sharpe
is for. A search of five is not a search of two hundred, and the report says so.

Deterministic, offline and about a second to run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quackz import evaluate, metrics
from quackz.report import text_report
from quackz.returns import net_returns

SEED = 11
N_BARS = 1512
LOOKBACKS = (20, 40, 60, 90, 120)
DRIFT_PERSISTENCE = 0.98
DRIFT_VOL = 0.05
ANNUAL_VOL = 0.18
PERIODS_PER_YEAR = 252.0
COSTS_BPS = 2.0
START = "2019-01-02"


def synthetic_prices(rng: np.random.Generator, *, n_bars: int = N_BARS) -> pd.Series:
    """Prices whose expected return is a slow AR(1) state, so trends are real but weak.

    The state is never handed to the strategy: it has to be inferred from past prices, and
    its standard deviation is about a quarter of the bar-to-bar noise, which is the honest
    end of what trend following has to work with.
    """
    sigma = ANNUAL_VOL / np.sqrt(PERIODS_PER_YEAR)
    shocks = rng.standard_normal(n_bars)
    drift = np.zeros(n_bars)
    for bar in range(1, n_bars):
        drift[bar] = DRIFT_PERSISTENCE * drift[bar - 1] + DRIFT_VOL * sigma * shocks[bar]
    bar_returns = drift + sigma * rng.standard_normal(n_bars)
    index = pd.date_range(START, periods=n_bars, freq="B")
    return pd.Series(100.0 * np.cumprod(1.0 + bar_returns), index=index, name="close")


def momentum_positions(prices: pd.Series, *, lookback: int) -> pd.Series:
    """Sign of the trailing `lookback` bar return, known at the close of the bar it indexes.

    The first `lookback` bars have no history to measure, so the position is flat there.
    """
    trailing = prices / prices.shift(lookback) - 1.0
    return pd.Series(np.sign(trailing.fillna(0.0)), index=prices.index, name="position")


def choose_lookback(prices: pd.Series) -> tuple[int, np.ndarray]:
    """Pick the lookback with the best net Sharpe, and keep every trial's Sharpe."""
    sharpes = np.array(
        [
            metrics.sharpe(
                net_returns(prices, momentum_positions(prices, lookback=k), costs_bps=COSTS_BPS),
                periods_per_year=PERIODS_PER_YEAR,
            )
            for k in LOOKBACKS
        ]
    )
    return LOOKBACKS[int(np.argmax(sharpes))], sharpes


def main() -> None:
    rng = np.random.default_rng(SEED)
    prices = synthetic_prices(rng)
    lookback, trial_sharpes = choose_lookback(prices)
    positions = momentum_positions(prices, lookback=lookback)

    print(f"Trend rule on {N_BARS} bars of synthetic prices with a persistent drift, seed {SEED}.")
    print(
        f"Lookback grid {LOOKBACKS}, Sharpes "
        + ", ".join(f"{s:.2f}" for s in trial_sharpes)
        + f"; kept {lookback} bars."
    )
    print()

    evaluation = evaluate(
        prices,
        positions,
        costs_bps=COSTS_BPS,
        periods_per_year=PERIODS_PER_YEAR,
        n_trials=len(LOOKBACKS),
        trial_sharpes=trial_sharpes,
        bootstrap_seed=0,
    )
    print(text_report(evaluation))


if __name__ == "__main__":
    main()
