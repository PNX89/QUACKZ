"""Tests for quackz.metrics.

Golden values come from three places: the Bailey and Lopez de Prado (2014) worked
examples, arithmetic short enough to check by hand in the comment above the assertion,
and scipy, which is a development dependency for exactly this purpose.
"""

from __future__ import annotations

import math
from itertools import pairwise
from statistics import NormalDist

import numpy as np
import pytest
import scipy.stats as sps

from quackz import metrics

# A four-bar series with mean 0.01 and sample variance 0.001 / 3, used wherever a hand
# computation is short enough to be worth pinning.
HAND = np.array([0.02, -0.01, 0.03, 0.00])
HAND_PPY = 4.0

# Bailey and Lopez de Prado (2014) common inputs: annualized SR_hat = 2.5 at 250 periods
# per year over 1250 observations, with an annualized V[{SR_n}] of 0.5. Everything the
# DSR touches lives at the observation frequency, so both are de-annualized here, the
# Sharpe by sqrt(ppy) and the variance by ppy.
PAPER_PPY = 250.0
PAPER_N_OBS = 1250
PAPER_SR = 2.5 / math.sqrt(PAPER_PPY)
PAPER_VAR = 0.5 / PAPER_PPY


# --------------------------------------------------------------------------------------
# Bailey and Lopez de Prado golden values
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_trials", "skew", "kurt_nonexcess", "expected_dsr"),
    [
        (100, -3.0, 10.0, 0.900397),
        (46, -3.0, 10.0, 0.950502),
        (88, 0.0, 3.0, 0.950491),
    ],
)
def test_deflated_sharpe_reproduces_the_paper(n_trials, skew, kurt_nonexcess, expected_dsr):
    """The paper's worked example, and two cases derived from the same inputs, to 6 places.

    The first row is the worked example: 100 trials against a record with skew -3 and
    non-excess kurtosis 10. The other two are DERIVED here rather than quoted, which is the
    reason both land just above 0.95: they are the largest trial counts at which this record
    still clears that confidence, with the paper's non-normal moments and with normal ones.
    The test below pins the crossings that make them the largest.

    Passing this pins four independent choices at once: sqrt(T - 1) rather than sqrt(T),
    non-excess kurtosis, the de-annualization of both SR and V, and the e^-1 term inside
    the expected-maximum formula. Get any one of them wrong and at least one row moves.
    """
    result = metrics.deflated_sharpe(
        sr=PAPER_SR,
        n_trials=n_trials,
        n_obs=PAPER_N_OBS,
        skew=skew,
        kurt_nonexcess=kurt_nonexcess,
        var_trial_sharpes=PAPER_VAR,
    )
    assert result.dsr == pytest.approx(expected_dsr, abs=1e-6)
    assert round(result.dsr, 4) == round(expected_dsr, 4)


@pytest.mark.parametrize(
    ("last_clearing_count", "skew", "kurt_nonexcess"),
    [(46, -3.0, 10.0), (88, 0.0, 3.0)],
)
def test_the_two_derived_rows_are_the_last_trial_counts_that_clear_the_confidence(
    last_clearing_count, skew, kurt_nonexcess
):
    """What makes 46 and 88 the numbers they are, so the README can say what they are."""

    def dsr_at(n_trials: int) -> float:
        return metrics.deflated_sharpe(
            sr=PAPER_SR,
            n_trials=n_trials,
            n_obs=PAPER_N_OBS,
            skew=skew,
            kurt_nonexcess=kurt_nonexcess,
            var_trial_sharpes=PAPER_VAR,
        ).dsr

    assert dsr_at(last_clearing_count) >= 0.95
    assert dsr_at(last_clearing_count + 1) < 0.95


def test_expected_max_sharpe_reproduces_the_paper_threshold():
    """Example 1's SR_0, non-annualized."""
    sr_0 = metrics.expected_max_sharpe(n_trials=100, var_trial_sharpes=PAPER_VAR)
    assert sr_0 == pytest.approx(0.113172, abs=1e-6)


def test_deflated_sharpe_reports_the_threshold_it_used():
    result = metrics.deflated_sharpe(
        sr=PAPER_SR,
        n_trials=100,
        n_obs=PAPER_N_OBS,
        skew=-3.0,
        kurt_nonexcess=10.0,
        var_trial_sharpes=PAPER_VAR,
    )
    assert result.expected_max_sharpe == pytest.approx(0.113172, abs=1e-6)
    assert result.var_trial_sharpes == pytest.approx(PAPER_VAR)
    assert result.variance_source == "var_trial_sharpes"
    assert result.warning is None
    assert result.n_trials == 100
    assert result.n_obs == PAPER_N_OBS


# --------------------------------------------------------------------------------------
# Noise floor
# --------------------------------------------------------------------------------------


def test_expected_max_sharpe_noise_floor_golden():
    """Report noise floor: N = 200 trials, T = 1008 daily bars, ppy = 252.

    With no trial Sharpes supplied the variance falls back to the iid-normal 1 / T, so
    the floor is Bailey Eq. 1 evaluated at V = 1 / 1008. The annualized figure is the
    per-bar figure times sqrt(252), which is the only conversion the report performs.
    """
    per_bar = metrics.expected_max_sharpe(n_trials=200, var_trial_sharpes=1.0 / 1008)
    assert per_bar == pytest.approx(0.087106, abs=1e-6)
    assert per_bar * math.sqrt(252.0) == pytest.approx(1.382762, abs=1e-6)


def test_bailey_floor_sits_well_below_the_crude_asymptotic_bound():
    """sqrt(2 ln N) is not shipped as a second estimator, and this is why.

    It is the classical asymptotic for the expected maximum of N standard normals, and it
    overstates the finite-N expectation that Bailey Eq. 1 gives. Shipping both would put
    the report's noise floor in contradiction with its own DSR, which uses Eq. 1. The gap
    at the noise-floor inputs is pinned here so the difference stays visible.
    """
    variance = 1.0 / 1008
    bailey = metrics.expected_max_sharpe(n_trials=200, var_trial_sharpes=variance)
    crude = math.sqrt(variance) * math.sqrt(2.0 * math.log(200))
    assert crude == pytest.approx(0.102531, abs=1e-6)
    assert crude / bailey == pytest.approx(1.1771, abs=1e-4)


def test_expected_max_sharpe_is_zero_without_selection():
    assert metrics.expected_max_sharpe(n_trials=1, var_trial_sharpes=0.5) == 0.0


def test_expected_max_sharpe_grows_with_trials():
    floors = [
        metrics.expected_max_sharpe(n_trials=n, var_trial_sharpes=0.002) for n in (2, 10, 100, 1000)
    ]
    assert all(later > earlier for earlier, later in pairwise(floors))


def test_expected_max_sharpe_scales_with_the_square_root_of_variance():
    one = metrics.expected_max_sharpe(n_trials=50, var_trial_sharpes=0.001)
    four = metrics.expected_max_sharpe(n_trials=50, var_trial_sharpes=0.004)
    assert four == pytest.approx(2.0 * one, rel=1e-12)


# --------------------------------------------------------------------------------------
# PSR, DSR and MinTRL behaviour
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("sr", [-0.4, -0.05, 0.0, 0.05, 0.2, 0.9])
@pytest.mark.parametrize(("skew", "kurt"), [(0.0, 3.0), (-1.5, 8.0), (0.8, 4.5)])
def test_probabilistic_sharpe_is_a_probability(sr, skew, kurt):
    value = metrics.probabilistic_sharpe(
        sr=sr, sr_benchmark=0.0, n_obs=500, skew=skew, kurt_nonexcess=kurt
    )
    assert 0.0 <= value <= 1.0


def test_probabilistic_sharpe_is_one_half_at_the_benchmark():
    value = metrics.probabilistic_sharpe(
        sr=0.1, sr_benchmark=0.1, n_obs=500, skew=-0.5, kurt_nonexcess=6.0
    )
    assert value == pytest.approx(0.5, abs=1e-15)


def test_probabilistic_sharpe_rises_with_track_record_length():
    values = [
        metrics.probabilistic_sharpe(
            sr=0.05, sr_benchmark=0.0, n_obs=n, skew=0.0, kurt_nonexcess=3.0
        )
        for n in (50, 200, 800, 3200)
    ]
    assert all(later > earlier for earlier, later in pairwise(values))


def test_negative_skew_and_fat_tails_lower_the_psr():
    normal = metrics.probabilistic_sharpe(
        sr=0.1, sr_benchmark=0.0, n_obs=500, skew=0.0, kurt_nonexcess=3.0
    )
    ugly = metrics.probabilistic_sharpe(
        sr=0.1, sr_benchmark=0.0, n_obs=500, skew=-2.0, kurt_nonexcess=12.0
    )
    assert ugly < normal


def test_deflated_sharpe_decreases_with_more_trials():
    values = [
        metrics.deflated_sharpe(
            sr=PAPER_SR,
            n_trials=n,
            n_obs=PAPER_N_OBS,
            skew=-3.0,
            kurt_nonexcess=10.0,
            var_trial_sharpes=PAPER_VAR,
        ).dsr
        for n in (1, 10, 50, 100, 500, 1000)
    ]
    assert all(later < earlier for earlier, later in pairwise(values))


def test_deflated_sharpe_at_one_trial_equals_the_psr_against_zero():
    result = metrics.deflated_sharpe(
        sr=PAPER_SR,
        n_trials=1,
        n_obs=PAPER_N_OBS,
        skew=-3.0,
        kurt_nonexcess=10.0,
        var_trial_sharpes=PAPER_VAR,
    )
    psr = metrics.probabilistic_sharpe(
        sr=PAPER_SR, sr_benchmark=0.0, n_obs=PAPER_N_OBS, skew=-3.0, kurt_nonexcess=10.0
    )
    assert result.dsr == pytest.approx(psr, rel=1e-15)


def test_deflated_sharpe_takes_variance_from_trial_sharpes():
    rng = np.random.default_rng(7)
    trials = rng.normal(loc=0.02, scale=0.03, size=250)
    result = metrics.deflated_sharpe(
        sr=0.15,
        n_trials=trials.size,
        n_obs=1250,
        skew=0.0,
        kurt_nonexcess=3.0,
        trial_sharpes=trials,
    )
    assert result.variance_source == "trial_sharpes"
    assert result.var_trial_sharpes == pytest.approx(float(trials.var(ddof=1)), rel=1e-15)
    assert result.warning is None


def test_deflated_sharpe_warns_loudly_when_it_has_to_guess_the_variance():
    with pytest.warns(UserWarning, match="iid-normal"):
        result = metrics.deflated_sharpe(
            sr=PAPER_SR, n_trials=100, n_obs=PAPER_N_OBS, skew=-3.0, kurt_nonexcess=10.0
        )
    assert result.variance_source == "iid_fallback"
    assert result.warning is not None
    assert "placeholder, not a bound" in result.warning
    assert result.var_trial_sharpes == pytest.approx(1.0 / PAPER_N_OBS)


def test_deflated_sharpe_rejects_two_ways_of_saying_the_same_thing():
    with pytest.raises(ValueError, match="not both"):
        metrics.deflated_sharpe(
            sr=0.1,
            n_trials=10,
            n_obs=500,
            skew=0.0,
            kurt_nonexcess=3.0,
            trial_sharpes=[0.1, 0.2, 0.3],
            var_trial_sharpes=0.001,
        )


def test_iid_fallback_understates_the_deflation_for_a_realistic_search():
    """The warning claims the fallback is optimistic; this shows a case where it is."""
    with pytest.warns(UserWarning):
        optimistic = metrics.deflated_sharpe(
            sr=PAPER_SR, n_trials=200, n_obs=PAPER_N_OBS, skew=0.0, kurt_nonexcess=3.0
        ).dsr
    honest = metrics.deflated_sharpe(
        sr=PAPER_SR,
        n_trials=200,
        n_obs=PAPER_N_OBS,
        skew=0.0,
        kurt_nonexcess=3.0,
        var_trial_sharpes=4.0 / PAPER_N_OBS,
    ).dsr
    assert honest < optimistic


def test_min_track_record_length_inverts_the_psr_exactly():
    """MinTRL is the T at which PSR equals the confidence level, so round-tripping is exact."""
    n_needed = metrics.min_track_record_length(
        sr=0.1, sr_benchmark=0.0, skew=0.0, kurt_nonexcess=3.0, confidence=0.95
    )
    # 1 + 1.005 * (1.6448536269514722 / 0.1)**2
    assert n_needed == pytest.approx(272.907117, abs=1e-6)
    recovered = metrics.probabilistic_sharpe(
        sr=0.1, sr_benchmark=0.0, n_obs=n_needed, skew=0.0, kurt_nonexcess=3.0
    )
    assert recovered == pytest.approx(0.95, abs=1e-12)


def test_min_track_record_length_grows_as_the_edge_shrinks():
    close = metrics.min_track_record_length(
        sr=0.11, sr_benchmark=0.10, skew=0.0, kurt_nonexcess=3.0
    )
    clear = metrics.min_track_record_length(
        sr=0.30, sr_benchmark=0.10, skew=0.0, kurt_nonexcess=3.0
    )
    assert close > clear > 1.0


def test_min_track_record_length_refuses_an_impossible_question():
    with pytest.raises(ValueError, match="must exceed"):
        metrics.min_track_record_length(sr=0.05, sr_benchmark=0.05, skew=0.0, kurt_nonexcess=3.0)


# --------------------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------------------


def test_probabilistic_sharpe_needs_two_observations():
    with pytest.raises(ValueError, match="at least 2"):
        metrics.probabilistic_sharpe(
            sr=0.1, sr_benchmark=0.0, n_obs=1, skew=0.0, kurt_nonexcess=3.0
        )


@pytest.mark.parametrize("kurt", [0.0, 0.5, -3.0])
def test_excess_kurtosis_passed_by_mistake_is_rejected(kurt):
    with pytest.raises(ValueError, match="NON-EXCESS"):
        metrics.probabilistic_sharpe(
            sr=0.1, sr_benchmark=0.0, n_obs=500, skew=0.0, kurt_nonexcess=kurt
        )


def test_inconsistent_moments_raise_instead_of_returning_nan():
    """An annualized Sharpe with a strongly positive skew drives the variance term negative."""
    with pytest.raises(ValueError, match="PER-PERIOD"):
        metrics.probabilistic_sharpe(
            sr=2.5, sr_benchmark=0.0, n_obs=1250, skew=3.0, kurt_nonexcess=1.0
        )


def test_expected_max_sharpe_rejects_negative_variance():
    with pytest.raises(ValueError, match="non-negative"):
        metrics.expected_max_sharpe(n_trials=10, var_trial_sharpes=-0.1)


# --------------------------------------------------------------------------------------
# Descriptive metrics
# --------------------------------------------------------------------------------------


def test_sharpe_hand_computed():
    # mean 0.01, sample sd sqrt(0.001 / 3), annualized by sqrt(4): 0.01 / sd * 2 = sqrt(1.2).
    assert metrics.sharpe(HAND, periods_per_year=HAND_PPY) == pytest.approx(
        math.sqrt(1.2), rel=1e-14
    )


def test_sortino_hand_computed():
    # One downside bar of -0.01 over four bars: dd = sqrt(0.0001 / 4) = 0.005.
    # 0.01 / 0.005 * sqrt(4) = 4.0.
    assert metrics.sortino(HAND, periods_per_year=HAND_PPY) == pytest.approx(4.0, rel=1e-14)


def test_sortino_denominator_uses_the_full_sample():
    """Dividing by the count of downside bars only would give 0.01 / 0.01 * 2 = 2.0."""
    assert metrics.sortino(HAND, periods_per_year=HAND_PPY) > 2.0


def test_max_drawdown_hand_computed():
    # Equity 1, 1.02, 1.0098, 1.040094: the only fall is 1.0098 / 1.02 - 1 = -1 percent.
    assert metrics.max_drawdown(HAND) == pytest.approx(-0.01, rel=1e-14)


def test_max_drawdown_sees_a_fall_on_the_very_first_bar():
    assert metrics.max_drawdown(np.array([-0.10, 0.0, 0.0])) == pytest.approx(-0.10, rel=1e-14)


def test_max_drawdown_is_zero_for_a_monotone_climb():
    assert metrics.max_drawdown(np.array([0.01, 0.02, 0.03])) == 0.0


def test_ulcer_index_hand_computed():
    # Drawdown path 0.10, -0.20 (0.88 / 1.1), -0.12 (0.968 / 1.1): rms = sqrt(0.0544 / 3).
    value = metrics.ulcer_index(np.array([0.10, -0.20, 0.10]))
    assert value == pytest.approx(math.sqrt(0.0544 / 3.0), rel=1e-13)


def test_ulcer_index_penalises_time_underwater():
    shallow_but_long = np.array([-0.05, 0.0, 0.0, 0.0, 0.05])
    deep_but_brief = np.array([-0.10, 0.12, 0.0, 0.0, 0.0])
    assert metrics.max_drawdown(deep_but_brief) < metrics.max_drawdown(shallow_but_long)
    assert metrics.ulcer_index(shallow_but_long) > metrics.ulcer_index(deep_but_brief)


def test_calmar_hand_computed():
    # Annualized return over exactly one year is the compounded total, 1.040094 - 1,
    # divided by the 1 percent maximum drawdown.
    expected = (1.02 * 0.99 * 1.03 * 1.00 - 1.0) / 0.01
    assert metrics.calmar(HAND, periods_per_year=HAND_PPY) == pytest.approx(expected, rel=1e-12)


def test_cvar_is_the_mean_of_the_worst_tail():
    # Sorted: -0.02, 0.00, 0.01, 0.015, 0.03. The 20th percentile is -0.004, so the tail
    # holds the single worst bar.
    sample = np.array([0.01, -0.02, 0.03, 0.00, 0.015])
    assert metrics.cvar(sample, alpha=0.2) == pytest.approx(-0.02, rel=1e-14)


def test_cvar_is_never_above_the_mean():
    rng = np.random.default_rng(3)
    sample = rng.standard_normal(500) * 0.01
    assert metrics.cvar(sample, alpha=0.05) < float(sample.mean())


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_cvar_rejects_an_impossible_alpha(alpha):
    with pytest.raises(ValueError, match="strictly between"):
        metrics.cvar(HAND, alpha=alpha)


def test_annualized_return_compounds():
    assert metrics.annualized_return(np.full(252, 0.01), periods_per_year=252) == pytest.approx(
        1.01**252 - 1.0, rel=1e-12
    )


def test_annualized_vol_scales_by_root_time():
    rng = np.random.default_rng(11)
    sample = rng.standard_normal(1000) * 0.01
    assert metrics.annualized_vol(sample, periods_per_year=252) == pytest.approx(
        float(sample.std(ddof=1)) * math.sqrt(252), rel=1e-15
    )


def test_total_loss_is_rejected_rather_than_compounded_through_zero():
    with pytest.raises(ValueError, match="total loss"):
        metrics.max_drawdown(np.array([0.01, -1.5, 0.02]))


@pytest.mark.parametrize("bad_ppy", [0.0, -252, float("nan"), float("inf")])
def test_periods_per_year_must_be_positive_and_finite(bad_ppy):
    with pytest.raises(ValueError, match="positive finite"):
        metrics.sharpe(HAND, periods_per_year=bad_ppy)


def test_flat_returns_give_a_finite_zero_rather_than_a_crash():
    flat = np.zeros(10)
    assert metrics.sharpe(flat, periods_per_year=252) == 0.0
    assert metrics.sortino(flat, periods_per_year=252) == 0.0
    assert metrics.calmar(flat, periods_per_year=252) == 0.0


def test_constant_positive_returns_give_an_enormous_sharpe():
    # np.full leaves rounding dust in the sample standard deviation rather than an exact
    # zero, so the result is astronomically large rather than literally infinite.
    assert metrics.sharpe(np.full(10, 0.001), periods_per_year=252) > 1e6


def test_non_finite_returns_are_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        metrics.sharpe(np.array([0.01, np.nan, 0.02]), periods_per_year=252)


# --------------------------------------------------------------------------------------
# Moments
# --------------------------------------------------------------------------------------


def test_kurtosis_of_a_normal_sample_is_near_three_not_near_zero():
    """The single most common wiring error in a PSR implementation."""
    rng = np.random.default_rng(12345)
    sample = rng.standard_normal(200_000)
    skew, kurt = metrics.moments(sample)
    # Sampling standard errors at this size are about 0.0055 for skew and 0.011 for
    # kurtosis, so these tolerances are roughly five standard errors.
    assert skew == pytest.approx(0.0, abs=0.03)
    assert kurt == pytest.approx(3.0, abs=0.06)
    assert kurt > 2.0


def test_moments_are_not_bias_corrected():
    """Hand check on a tiny sample where the bias correction would be large."""
    sample = np.array([-1.0, 0.0, 0.0, 1.0])
    skew, kurt = metrics.moments(sample)
    # m2 = 0.5, m3 = 0, m4 = 0.5, so kurtosis is 0.5 / 0.25 = 2.
    assert skew == pytest.approx(0.0, abs=1e-15)
    assert kurt == pytest.approx(2.0, rel=1e-15)


def test_moments_reject_a_constant_series():
    """Rounding dust of order 1e-19 must not be amplified into a fake shape statistic."""
    with pytest.raises(ValueError, match="constant"):
        metrics.moments(np.full(20, 0.01))
    with pytest.raises(ValueError, match="constant"):
        metrics.moments(np.zeros(20))


# --------------------------------------------------------------------------------------
# Serial correlation
# --------------------------------------------------------------------------------------


def test_autocorr_1_hand_computed():
    # Deviations -1.5, -0.5, 0.5, 1.5: cross products sum to 1.25 over a variance sum of 5.
    assert metrics.autocorr_1(np.array([0.0, 1.0, 2.0, 3.0])) == pytest.approx(0.25, rel=1e-15)


def test_autocorr_1_recovers_a_known_ar1_coefficient():
    rng = np.random.default_rng(21)
    phi = 0.6
    noise = rng.standard_normal(200_000)
    series = np.empty_like(noise)
    series[0] = noise[0]
    for i in range(1, series.size):
        series[i] = phi * series[i - 1] + noise[i]
    assert metrics.autocorr_1(series) == pytest.approx(phi, abs=0.01)


def test_autocorr_1_rejects_a_constant_series():
    with pytest.raises(ValueError, match="constant"):
        metrics.autocorr_1(np.full(20, 0.02))


def test_hac_tstat_with_zero_lags_is_the_plain_t_statistic():
    mean = float(HAND.mean())
    m2 = float(((HAND - mean) ** 2).mean())
    expected = mean / math.sqrt(m2 / HAND.size)
    assert metrics.hac_tstat(HAND, lags=0) == pytest.approx(expected, rel=1e-14)


def test_hac_tstat_matches_the_plain_t_statistic_on_independent_data():
    rng = np.random.default_rng(4)
    sample = rng.standard_normal(5000) * 0.01 + 0.0005
    plain = metrics.hac_tstat(sample, lags=0)
    corrected = metrics.hac_tstat(sample)
    assert corrected == pytest.approx(plain, rel=0.15)


def test_hac_tstat_shrinks_when_returns_are_positively_autocorrelated():
    """Lo (2002): serial correlation inflates the naive significance of a mean return."""
    rng = np.random.default_rng(5)
    phi = 0.5
    noise = rng.standard_normal(4000) * 0.01
    series = np.empty_like(noise)
    series[0] = noise[0]
    for i in range(1, series.size):
        series[i] = phi * series[i - 1] + noise[i]
    series = series + 0.001
    assert metrics.autocorr_1(series) > 0.4
    assert abs(metrics.hac_tstat(series)) < abs(metrics.hac_tstat(series, lags=0))


def test_hac_tstat_rejects_negative_lags():
    with pytest.raises(ValueError, match="non-negative"):
        metrics.hac_tstat(HAND, lags=-1)


# --------------------------------------------------------------------------------------
# scipy equivalence, development dependency only
# --------------------------------------------------------------------------------------


def test_stdlib_normal_cdf_matches_scipy():
    grid = np.linspace(-8.0, 8.0, 641)
    theirs = sps.norm.cdf(grid)
    ours = np.array([NormalDist().cdf(float(x)) for x in grid])
    assert np.max(np.abs(ours - theirs)) < 1e-12


def test_stdlib_normal_inverse_cdf_matches_scipy():
    grid = np.linspace(1e-6, 1.0 - 1e-6, 501)
    theirs = sps.norm.ppf(grid)
    ours = np.array([NormalDist().inv_cdf(float(p)) for p in grid])
    assert np.max(np.abs(ours - theirs)) < 1e-12


def test_moments_match_scipy_biased_estimators():
    rng = np.random.default_rng(99)
    for sample in (
        rng.standard_normal(5000),
        rng.standard_t(df=4, size=5000),
        rng.lognormal(mean=0.0, sigma=0.7, size=5000),
    ):
        skew, kurt = metrics.moments(sample)
        assert skew == pytest.approx(float(sps.skew(sample, bias=True)), rel=1e-12)
        assert kurt == pytest.approx(
            float(sps.kurtosis(sample, fisher=False, bias=True)), rel=1e-12
        )


def test_probabilistic_sharpe_matches_a_scipy_reference_implementation():
    sr, benchmark, n_obs, skew, kurt = 0.12, 0.03, 900, -1.1, 7.5
    variance_term = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2
    reference = float(
        sps.norm.cdf((sr - benchmark) * math.sqrt(n_obs - 1) / math.sqrt(variance_term))
    )
    ours = metrics.probabilistic_sharpe(
        sr=sr, sr_benchmark=benchmark, n_obs=n_obs, skew=skew, kurt_nonexcess=kurt
    )
    assert ours == pytest.approx(reference, abs=1e-12)
