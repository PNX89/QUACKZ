# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `Thresholds` now rejects a probability, profit share or correlation cut-off outside the
  range that quantity can take, at construction. `dsr_warn` in particular is checked
  against the open interval `(0, 1)`: it used to accept `1.0`, pass construction, and only
  fail a hundred lines later inside `metrics.min_track_record_length`, with a bare
  `ValueError` naming `confidence`, a parameter the caller never passed.
- `checks.bootstrap` now checks the caller's own series for a total loss (a return at or
  below -1) before building any resample from it, and names the series and the index of
  the offending bar. It used to find such a value only after resampling and blame "a
  resampled path", which does not exist, since every resample is bars of that same series
  rearranged rather than a new value.
- `metrics.expected_max_sharpe`'s "too large" guard checked the wrong inverse-cdf argument.
  `(1/N) * e^-1` saturates to 1.0 at a lower `n_trials` than `1/N` does, about `e` times
  sooner, so the raw `statistics.StatisticsError` escaped over a real band above roughly
  6.6 quadrillion trials before the guard ever fired. It now checks the expression that
  actually saturates first.

## [0.1.0] - 2026-08-18

### Added

- `quackz.returns`: the position and cost conventions the whole library is built on. The
  position at bar t is decided at that bar's close and earns bar t+1's return; costs are
  charged on `|pos[t-1] - pos[t-2]|`, including the trade into the first position and the
  trade out of the last.
- `quackz.metrics`: Sharpe, Sortino, Calmar, maximum drawdown, ulcer index, CVaR,
  annualized return and volatility, method-of-moments skewness and non-excess kurtosis,
  the probabilistic Sharpe ratio, the expected maximum Sharpe of a search, the deflated
  Sharpe ratio, minimum track record length, a Newey-West t-statistic and the lag-1
  autocorrelation. Normal cdf and inverse cdf come from the standard library, so scipy is
  not a runtime dependency.
- `quackz.checks`: reconciliation against a claimed return stream, latency sensitivity
  graded on the level at zero delay and on how much of that level survives one bar of
  delay against what the holding period implies, a cost sweep with a closed-form
  break-even, a stationary bootstrap with a null-imposed p-value and a studentized
  interval, the noise floor of a declared search, subperiod stability with a noise-aware
  verdict, and profit concentration.
- `quackz.splits`: `WalkForward` and `EmbargoedKFold`, both following the scikit-learn
  splitter protocol without importing scikit-learn.
- `quackz.evaluate.evaluate`: one call that runs every check against a single price and
  position series and returns a frozen `Evaluation`, including the deflated Sharpe as a
  function of the number of trials and the break-even trial count.
- `quackz.report`: `text_report`, `markdown_report` and `json_report`, all built from one
  set of verdict sentences so the three formats cannot disagree.
- `quackz.cli`: `quackz report DATA.csv`, with additive `--json` and `--md` output and
  exit codes 0 for pass or warn, 1 for a failing verdict, 2 for anything the command could
  not run at all, including a file it could not open or decode. The dispersion of the
  search is reachable from the command line as well as from Python: `--trial-sharpes PATH`
  reads the annualized Sharpe of every configuration the search touched, and
  `--var-trial-sharpes` takes that spread already summarised.
- Configurable grading: every cut-off is a named constant with its reasoning beside it,
  gathered into `Thresholds` and overridable through `evaluate(thresholds=...)`.
- Examples: `examples/overfit_demo.py` and `examples/momentum_demo.py`, both seeded,
  offline and deterministic.

[0.1.0]: https://github.com/PNX89/QUACKZ/releases/tag/v0.1.0
