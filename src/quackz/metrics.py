"""Point estimates and the sampling behaviour of a Sharpe ratio.

Conventions that hold for every function in this module.

Frequency
    `returns` is a per-period simple return series (a fraction, not a percentage).
    Functions that take `periods_per_year` return an ANNUALIZED number. Functions in
    the statistical-inference group (`probabilistic_sharpe`, `deflated_sharpe`,
    `min_track_record_length`) take `sr` and `sr_benchmark` at the OBSERVATION
    frequency, not annualized. Converting is the caller's job:
    `sr_per_period = sr_annual / sqrt(periods_per_year)`. Feeding an annualized Sharpe
    into those three is the most common way to get a wrong answer that still looks
    plausible, so the parameters are keyword-only and the docstrings say it again.

Kurtosis
    Kurtosis is NON-EXCESS everywhere: a normal sample gives roughly 3, not roughly 0.
    The parameter is named `kurt_nonexcess` so a call site cannot get it wrong quietly,
    and values below 1 are rejected because no distribution has kurtosis below 1.

Normal distribution
    `statistics.NormalDist` supplies both the cdf and the inverse cdf (Wichura AS241,
    accurate to about 1e-16), which keeps scipy out of the runtime dependencies. The
    test suite pins the agreement with `scipy.stats.norm` at 1e-12.

Degenerate inputs
    Zero volatility and zero drawdown return a signed infinity or 0.0 rather than
    raising, because a report window can legitimately hold a flat position. Quantities
    that have no limiting value at all (skewness, kurtosis and autocorrelation of a
    constant series) raise instead.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import numpy.typing as npt

__all__ = [
    "DSRResult",
    "annualized_return",
    "annualized_vol",
    "autocorr_1",
    "calmar",
    "cvar",
    "deflated_sharpe",
    "expected_max_sharpe",
    "hac_tstat",
    "max_drawdown",
    "min_track_record_length",
    "moments",
    "probabilistic_sharpe",
    "sharpe",
    "sortino",
    "ulcer_index",
]

_NORM = NormalDist()

# Euler-Mascheroni constant, as used in Bailey and Lopez de Prado (2014) Eq. 1.
_EULER_MASCHERONI = 0.5772156649

# The PSR variance term is a variance and must be positive. Anything at or below this
# means the supplied moments are mutually inconsistent (or an annualized Sharpe was
# passed where a per-period one belongs), which is an error, not a NaN to propagate.
_VARIANCE_TERM_FLOOR = 1e-12


def _as_array(returns: npt.ArrayLike, *, name: str = "returns", min_len: int = 2) -> np.ndarray:
    arr = np.asarray(returns, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one dimensional, got shape {arr.shape}")
    if arr.size < min_len:
        raise ValueError(f"{name} needs at least {min_len} observations, got {arr.size}")
    if not np.isfinite(arr).all():
        bad = int(np.count_nonzero(~np.isfinite(arr)))
        raise ValueError(f"{name} contains {bad} non-finite value(s); clean or drop them first")
    return arr


def _check_periods_per_year(periods_per_year: float) -> float:
    ppy = float(periods_per_year)
    if not math.isfinite(ppy) or ppy <= 0:
        raise ValueError(
            f"periods_per_year must be a positive finite number, got {periods_per_year!r}"
        )
    return ppy


def _check_kurt(kurt_nonexcess: float) -> float:
    kurt = float(kurt_nonexcess)
    if not math.isfinite(kurt) or kurt < 1.0:
        raise ValueError(
            f"kurt_nonexcess must be at least 1, got {kurt_nonexcess!r}. This parameter is "
            "NON-EXCESS kurtosis (a normal sample gives about 3); a value near 0 means excess "
            "kurtosis was passed by mistake."
        )
    return kurt


def _deviations(arr: np.ndarray, quantity: str) -> tuple[np.ndarray, float]:
    """Deviations from the mean and the second central moment, rejecting a constant series.

    Exact equality against zero is not enough: `arr - arr.mean()` on a constant series
    leaves rounding dust of order 1e-19, which a division by m2**1.5 would amplify into a
    fabricated shape statistic. The test is therefore relative to the level of the data.
    """
    dev = arr - arr.mean()
    m2 = float(np.mean(dev**2))
    level = float(np.mean(np.abs(arr)))
    if m2 <= 0.0 or math.sqrt(m2) <= 1e-14 * max(level, 1e-300):
        raise ValueError(
            f"returns is constant to within floating point precision, so {quantity} is undefined"
        )
    return dev, m2


def _degenerate_ratio(numerator: float) -> float:
    """Signed limit of numerator / 0."""
    if numerator > 0:
        return math.inf
    if numerator < 0:
        return -math.inf
    return 0.0


def _equity_curve(arr: np.ndarray) -> np.ndarray:
    """Compounded equity starting at 1.0, with the starting point kept as index 0.

    Keeping the 1.0 matters: without it a fall on the very first bar has no prior peak
    to be measured against and silently reports a zero drawdown.
    """
    if (arr <= -1.0).any():
        raise ValueError(
            "returns contains a value at or below -1 (total loss); the compounded equity "
            "curve is undefined beyond that point"
        )
    return np.concatenate(([1.0], np.cumprod(1.0 + arr)))


def _drawdown_curve(arr: np.ndarray) -> np.ndarray:
    """Drawdown as a non-positive fraction of the running peak, one value per return bar."""
    equity = _equity_curve(arr)
    peak = np.maximum.accumulate(equity)
    return (equity / peak - 1.0)[1:]


def sharpe(returns: npt.ArrayLike, *, periods_per_year: float) -> float:
    """Annualized Sharpe ratio, zero risk-free rate, sample standard deviation (ddof=1).

    Annualizing by sqrt(periods_per_year) assumes serially independent returns. Use
    `autocorr_1` and `hac_tstat` to see how badly that assumption is violated (Lo 2002).
    """
    arr = _as_array(returns)
    ppy = _check_periods_per_year(periods_per_year)
    mu = float(arr.mean())
    sd = float(arr.std(ddof=1))
    if sd == 0.0:
        return _degenerate_ratio(mu)
    return mu / sd * math.sqrt(ppy)


def sortino(returns: npt.ArrayLike, *, periods_per_year: float, target: float = 0.0) -> float:
    """Annualized Sortino ratio.

    Downside deviation uses the FULL sample in the denominator (upside bars enter as
    zeros), which is the original Sortino definition. Dividing only by the count of
    downside bars is a different and more flattering statistic.
    """
    arr = _as_array(returns)
    ppy = _check_periods_per_year(periods_per_year)
    excess = arr - float(target)
    downside = np.minimum(excess, 0.0)
    dd = math.sqrt(float(np.mean(downside**2)))
    mu = float(excess.mean())
    if dd == 0.0:
        return _degenerate_ratio(mu)
    return mu / dd * math.sqrt(ppy)


def calmar(returns: npt.ArrayLike, *, periods_per_year: float) -> float:
    """Annualized geometric return divided by the absolute maximum drawdown."""
    arr = _as_array(returns)
    ppy = _check_periods_per_year(periods_per_year)
    ann = annualized_return(arr, periods_per_year=ppy)
    mdd = abs(max_drawdown(arr))
    if mdd == 0.0:
        return _degenerate_ratio(ann)
    return ann / mdd


def max_drawdown(returns: npt.ArrayLike) -> float:
    """Largest peak to trough fall of the compounded equity curve, as a NEGATIVE fraction."""
    arr = _as_array(returns)
    return float(_drawdown_curve(arr).min())


def ulcer_index(returns: npt.ArrayLike) -> float:
    """Root mean squared drawdown, in return units (multiply by 100 for percentage points).

    Unlike max drawdown this penalises the time spent underwater, not just the deepest
    single point.
    """
    arr = _as_array(returns)
    dd = _drawdown_curve(arr)
    return math.sqrt(float(np.mean(dd**2)))


def cvar(returns: npt.ArrayLike, *, alpha: float = 0.05) -> float:
    """Empirical conditional value at risk: the mean of the worst `alpha` tail.

    Returned with the sign of the returns themselves, so a loss is negative. The tail is
    the set of observations at or below the empirical `alpha` quantile (linear
    interpolation), so with few observations it is a coarse estimate.
    """
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha!r}")
    arr = _as_array(returns)
    threshold = float(np.quantile(arr, float(alpha)))
    tail = arr[arr <= threshold]
    return float(tail.mean())


def annualized_return(returns: npt.ArrayLike, *, periods_per_year: float) -> float:
    """Geometric (compounded) annualized return, not the arithmetic mean times ppy."""
    arr = _as_array(returns)
    ppy = _check_periods_per_year(periods_per_year)
    total = float(_equity_curve(arr)[-1])
    return total ** (ppy / arr.size) - 1.0


def annualized_vol(returns: npt.ArrayLike, *, periods_per_year: float) -> float:
    """Sample standard deviation (ddof=1) scaled by sqrt(periods_per_year)."""
    arr = _as_array(returns)
    ppy = _check_periods_per_year(periods_per_year)
    return float(arr.std(ddof=1)) * math.sqrt(ppy)


def moments(returns: npt.ArrayLike) -> tuple[float, float]:
    """Method-of-moments skewness and NON-EXCESS kurtosis: (m3 / m2**1.5, m4 / m2**2).

    Neither is bias corrected, matching `scipy.stats.skew(bias=True)` and
    `scipy.stats.kurtosis(fisher=False, bias=True)`. A normal sample gives a kurtosis of
    about 3. These are exactly the estimators the PSR and DSR formulas expect.
    """
    arr = _as_array(returns)
    dev, m2 = _deviations(arr, "skewness and kurtosis")
    m3 = float(np.mean(dev**3))
    m4 = float(np.mean(dev**4))
    return m3 / m2**1.5, m4 / m2**2


def _psr_variance_term(*, sr: float, skew: float, kurt_nonexcess: float) -> float:
    """1 - skew*SR + ((kurt - 1) / 4) * SR**2, the variance of the Sharpe estimator."""
    term = 1.0 - float(skew) * sr + ((kurt_nonexcess - 1.0) / 4.0) * sr * sr
    if term <= _VARIANCE_TERM_FLOOR:
        raise ValueError(
            f"the Sharpe estimator variance term is not positive (got {term!r}) for sr={sr!r}, "
            f"skew={skew!r}, kurt_nonexcess={kurt_nonexcess!r}. Check that sr is a PER-PERIOD "
            "Sharpe rather than an annualized one, and that skew and kurtosis come from the "
            "same sample."
        )
    return term


def probabilistic_sharpe(
    *,
    sr: float,
    sr_benchmark: float,
    n_obs: float,
    skew: float,
    kurt_nonexcess: float,
) -> float:
    """Probabilistic Sharpe Ratio (Bailey and Lopez de Prado 2012).

    PSR = Phi[ (SR - SR*) * sqrt(T - 1) / sqrt(1 - skew*SR + ((kurt - 1) / 4) * SR**2) ]

    The probability that the true Sharpe exceeds `sr_benchmark`, given the length of the
    track record and its non-normality. Note sqrt(T - 1), not sqrt(T).

    `sr` and `sr_benchmark` are PER-PERIOD Sharpe ratios at the observation frequency.
    Divide an annualized Sharpe by sqrt(periods_per_year) before calling.

    `n_obs` is a bar count but is not rounded, so `min_track_record_length` inverts this
    function exactly rather than to the nearest bar.
    """
    n = float(n_obs)
    if not math.isfinite(n) or n < 2.0:
        raise ValueError(f"n_obs must be at least 2 for the sqrt(T - 1) term, got {n_obs!r}")
    kurt = _check_kurt(kurt_nonexcess)
    sr_f = float(sr)
    term = _psr_variance_term(sr=sr_f, skew=float(skew), kurt_nonexcess=kurt)
    z = (sr_f - float(sr_benchmark)) * math.sqrt(n - 1.0) / math.sqrt(term)
    return _NORM.cdf(z)


def expected_max_sharpe(*, n_trials: int, var_trial_sharpes: float) -> float:
    """Expected maximum Sharpe across N independent trials (Bailey and Lopez de Prado 2014, Eq. 1).

    SR_0 = sqrt(V) * ((1 - g) * Phi^-1[1 - 1/N] + g * Phi^-1[1 - (1/N) * e^-1])

    with g the Euler-Mascheroni constant and V the VARIANCE OF THE ESTIMATED SHARPES
    ACROSS THE N TRIALS, at the observation frequency. This is the extreme-value result
    used for both the DSR benchmark and the report's noise floor, so the two never
    disagree. The crude asymptotic sqrt(2 ln N) is not offered as an alternative: it
    overstates this expectation by roughly 15 to 40 percent at realistic N.

    N <= 1 means no selection took place, so the benchmark is 0.
    """
    n = int(n_trials)
    if n < 1:
        raise ValueError(f"n_trials must be at least 1, got {n_trials!r}")
    v = float(var_trial_sharpes)
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(
            f"var_trial_sharpes must be a non-negative finite number, got {var_trial_sharpes!r}"
        )
    if n <= 1:
        return 0.0
    if 1.0 - 1.0 / n >= 1.0:
        raise ValueError(
            f"n_trials={n} is too large to evaluate Phi^-1[1 - 1/N] in double precision"
        )
    g = _EULER_MASCHERONI
    upper = _NORM.inv_cdf(1.0 - 1.0 / n)
    upper_e = _NORM.inv_cdf(1.0 - (1.0 / n) * math.exp(-1.0))
    return math.sqrt(v) * ((1.0 - g) * upper + g * upper_e)


@dataclass(frozen=True)
class DSRResult:
    """Deflated Sharpe Ratio plus the inputs a reader needs in order to distrust it."""

    dsr: float
    expected_max_sharpe: float
    var_trial_sharpes: float
    variance_source: str
    n_trials: int
    n_obs: int
    warning: str | None = None


_IID_FALLBACK_WARNING = (
    "V[{SR_n}] was not supplied, so the iid-normal approximation 1/n_obs was used. That "
    "approximation assumes every trial had the same volatility and that the trials were "
    "independent; real strategy searches breach both, their Sharpe estimates are more "
    "dispersed, and the true deflation is therefore larger. Treat this DSR as an upper "
    "bound and pass trial_sharpes from the actual search."
)


def deflated_sharpe(
    *,
    sr: float,
    n_trials: int,
    n_obs: int,
    skew: float,
    kurt_nonexcess: float,
    trial_sharpes: npt.ArrayLike | None = None,
    var_trial_sharpes: float | None = None,
) -> DSRResult:
    """Deflated Sharpe Ratio (Bailey and Lopez de Prado 2014).

    DSR = PSR evaluated against SR_0 from `expected_max_sharpe`, that is, the probability
    that the selected strategy beats the best Sharpe you would expect from N trials of
    pure noise.

    Every quantity is at the OBSERVATION frequency. Divide an annualized Sharpe by
    sqrt(periods_per_year), and an annualized variance of trial Sharpes by
    periods_per_year, before calling.

    V[{SR_n}] is the variance of the ESTIMATED SHARPES ACROSS THE N TRIALS. Supply it
    either as `trial_sharpes` (the Sharpe of every configuration the search touched, from
    which the sample variance is taken) or as `var_trial_sharpes` (that variance already
    computed), never both. If neither is given the iid-normal 1/n_obs is used and the
    result carries a warning, because that fallback understates the deflation.
    """
    if trial_sharpes is not None and var_trial_sharpes is not None:
        raise ValueError(
            "pass either trial_sharpes or var_trial_sharpes, not both; they are two ways to "
            "specify the same quantity and supplying both hides which one was used"
        )

    n = int(n_obs)
    if n < 2:
        raise ValueError(f"n_obs must be at least 2, got {n_obs!r}")

    warning: str | None = None
    if trial_sharpes is not None:
        trials = _as_array(trial_sharpes, name="trial_sharpes", min_len=2)
        variance = float(trials.var(ddof=1))
        source = "trial_sharpes"
    elif var_trial_sharpes is not None:
        variance = float(var_trial_sharpes)
        if not math.isfinite(variance) or variance < 0.0:
            raise ValueError(
                f"var_trial_sharpes must be a non-negative finite number, got {var_trial_sharpes!r}"
            )
        source = "var_trial_sharpes"
    else:
        variance = 1.0 / n
        source = "iid_fallback"
        warning = _IID_FALLBACK_WARNING
        warnings.warn(_IID_FALLBACK_WARNING, UserWarning, stacklevel=2)

    sr_0 = expected_max_sharpe(n_trials=n_trials, var_trial_sharpes=variance)
    dsr = probabilistic_sharpe(
        sr=float(sr),
        sr_benchmark=sr_0,
        n_obs=n,
        skew=float(skew),
        kurt_nonexcess=kurt_nonexcess,
    )
    return DSRResult(
        dsr=dsr,
        expected_max_sharpe=sr_0,
        var_trial_sharpes=variance,
        variance_source=source,
        n_trials=int(n_trials),
        n_obs=n,
        warning=warning,
    )


def min_track_record_length(
    *,
    sr: float,
    sr_benchmark: float,
    skew: float,
    kurt_nonexcess: float,
    confidence: float = 0.95,
) -> float:
    """Observations needed before PSR reaches `confidence` (Bailey and Lopez de Prado 2012).

    MinTRL = 1 + (1 - skew*SR + ((kurt - 1) / 4) * SR**2) * (Phi^-1(confidence) / (SR - SR*))**2

    Returned as a float, at the OBSERVATION frequency; round up for a bar count. It is
    the exact inverse of `probabilistic_sharpe` in T, so PSR evaluated at this many
    observations equals `confidence`.
    """
    conf = float(confidence)
    if not 0.0 < conf < 1.0:
        raise ValueError(f"confidence must be strictly between 0 and 1, got {confidence!r}")
    kurt = _check_kurt(kurt_nonexcess)
    sr_f = float(sr)
    edge = sr_f - float(sr_benchmark)
    if edge <= 0.0:
        raise ValueError(
            f"sr ({sr!r}) must exceed sr_benchmark ({sr_benchmark!r}); no track record length "
            "makes a Sharpe at or below the benchmark significant"
        )
    term = _psr_variance_term(sr=sr_f, skew=float(skew), kurt_nonexcess=kurt)
    return 1.0 + term * (_NORM.inv_cdf(conf) / edge) ** 2


def _newey_west_lags(n_obs: int) -> int:
    """Standard automatic bandwidth: floor(4 * (T / 100)**(2/9))."""
    return math.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0))


def hac_tstat(returns: npt.ArrayLike, *, lags: int | None = None) -> float:
    """Newey-West (Bartlett kernel) t-statistic of the mean return.

    Serial correlation makes the naive t-statistic, and any Sharpe annualized by
    sqrt(periods_per_year), overstate significance (Lo 2002). This t-statistic corrects
    the standard error for it; comparing the two shows the size of the problem. `lags`
    defaults to floor(4 * (T / 100)**(2/9)) and is capped at T - 1.
    """
    arr = _as_array(returns, min_len=3)
    n = arr.size
    bandwidth = _newey_west_lags(n) if lags is None else int(lags)
    if bandwidth < 0:
        raise ValueError(f"lags must be non-negative, got {lags!r}")
    bandwidth = min(bandwidth, n - 1)

    mu = float(arr.mean())
    dev = arr - mu
    # Bartlett weights keep the long-run variance non-negative for any bandwidth.
    long_run = float(dev @ dev) / n
    for lag in range(1, bandwidth + 1):
        gamma = float(dev[lag:] @ dev[:-lag]) / n
        long_run += 2.0 * (1.0 - lag / (bandwidth + 1.0)) * gamma
    if long_run <= 0.0:
        return _degenerate_ratio(mu)
    return mu / math.sqrt(long_run / n)


def autocorr_1(returns: npt.ArrayLike) -> float:
    """Lag-1 sample autocorrelation, full-sample mean and denominator."""
    arr = _as_array(returns, min_len=3)
    dev, _ = _deviations(arr, "autocorrelation")
    return float(dev[1:] @ dev[:-1]) / float(dev @ dev)
