"""Cross-validation splitters for time-ordered data.

Both classes follow the scikit-learn splitter protocol (`get_n_splits`, `split`) without
importing scikit-learn, so they drop into an existing pipeline but do not add a dependency.
Following it means accepting the convention `cross_validate`, `cross_val_score` and
`GridSearchCV` actually call with, which is three positional arguments rather than one:
`split(X, y, groups)` and `get_n_splits(X, y, groups)`. The labels and the groups are
accepted and ignored, because a splitter that cuts a sample by time looks at nothing but
how many observations there are. `split` accepts anything with a length, or a plain integer
count of observations, and yields pairs of integer position arrays.

Which one to use. `WalkForward` is the honest default for trading research: every training
observation precedes every test observation, which is the only arrangement that matches how
the strategy would have been run. `EmbargoedKFold` exists because K-fold is what people
reach for, and it is here with its defect stated rather than hidden.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sized
from dataclasses import dataclass, field

import numpy as np

from quackz.returns import QuackzInputError

__all__ = ["EmbargoedKFold", "WalkForward"]


def _n_observations(data: object) -> int:
    """Number of observations in `data`, which may be a plain count."""
    if isinstance(data, bool):
        raise QuackzInputError("expected an observation count or a sized object, got a bool")
    if isinstance(data, (int, np.integer)):
        n_obs = int(data)
    elif isinstance(data, Sized):
        n_obs = len(data)
    else:
        raise QuackzInputError(
            f"expected an observation count or an object with a length, got {type(data).__name__}"
        )
    if n_obs < 2:
        raise QuackzInputError(f"need at least 2 observations to split, got {n_obs}")
    return n_obs


@dataclass(frozen=True)
class WalkForward:
    """Forward-only splitter: every training bar precedes every test bar.

    The observations are cut into `n_splits + 1` contiguous folds. Fold i + 1 is the test
    set of split i, and the training set is everything before it. Any remainder from the
    division lands in the first training set, so the test folds are equal in size and
    together cover the tail of the sample exactly once.

    With `expanding=True` (the default) the training set grows with each split, which is
    what a strategy that keeps refitting on all available history actually does. With
    `expanding=False` the training window has a fixed length, equal to the first split's
    training set, and slides forward; use that when the fitted quantity is meant to track a
    changing regime rather than average over all of history.

    This ordering is the point. It costs statistical power compared with K-fold, and the
    power is not worth having: a model trained partly on the future scores well and tells
    you nothing about whether the rule would have made money.
    """

    n_splits: int
    expanding: bool = field(default=True, kw_only=True)

    def __post_init__(self) -> None:
        if isinstance(self.n_splits, bool) or not isinstance(self.n_splits, (int, np.integer)):
            raise QuackzInputError(f"n_splits must be an integer, got {self.n_splits!r}")
        if int(self.n_splits) < 1:
            raise QuackzInputError(f"n_splits must be at least 1, got {self.n_splits}")
        if not isinstance(self.expanding, bool):
            raise QuackzInputError(f"expanding must be a bool, got {self.expanding!r}")

    def get_n_splits(self, data: object = None, y: object = None, groups: object = None) -> int:
        """Number of splits produced, which does not depend on the data."""
        return int(self.n_splits)

    def test_size(self, data: object) -> int:
        """Length of each test fold: `n_obs // (n_splits + 1)`."""
        n_obs = _n_observations(data)
        size = n_obs // (int(self.n_splits) + 1)
        if size < 1:
            raise QuackzInputError(
                f"n_splits={self.n_splits} needs at least {int(self.n_splits) + 1} "
                f"observations, got {n_obs}"
            )
        return size

    def split(
        self, data: object, y: object = None, groups: object = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield `(train_indices, test_indices)` in chronological order."""
        n_obs = _n_observations(data)
        size = self.test_size(n_obs)
        splits = int(self.n_splits)
        first_test = n_obs - splits * size
        indices = np.arange(n_obs)
        for step in range(splits):
            start = first_test + step * size
            test = indices[start : start + size]
            train = indices[:start] if self.expanding else indices[start - first_test : start]
            yield train, test


@dataclass(frozen=True)
class EmbargoedKFold:
    """Contiguous K-fold with a one-sided embargo after each test fold. NOT a purged K-fold.

    Purging, in the sense of Lopez de Prado's Advances in Financial Machine Learning, drops
    every training observation whose LABEL overlaps the test fold in time. That needs a
    label end time t1 for each observation, saying how far into the future the label looks.
    This library is given prices and positions, never labels, so it has no t1 and cannot
    purge. Calling this class purged would be a claim it cannot support, so it is not
    called that.

    What it does do is embargo: after each test fold, the `embargo_pct * n_obs`
    observations immediately following it are removed from the training set. The embargo is
    one-sided because serial correlation carries forward, so it is training data drawn from
    just after the test fold that most obviously overlaps it.

    The defect that remains, stated plainly. Every fold except the last trains on data that
    POSTDATES its test fold, and no embargo fixes that; an embargo trims the seam, not the
    direction. A model selected this way has seen the future in a way a live model never
    could. Use `WalkForward` when the question is whether the rule would have worked.
    """

    n_splits: int
    embargo_pct: float = field(kw_only=True)

    def __post_init__(self) -> None:
        if isinstance(self.n_splits, bool) or not isinstance(self.n_splits, (int, np.integer)):
            raise QuackzInputError(f"n_splits must be an integer, got {self.n_splits!r}")
        if int(self.n_splits) < 2:
            raise QuackzInputError(f"n_splits must be at least 2, got {self.n_splits}")
        pct = float(self.embargo_pct)
        if not np.isfinite(pct) or not 0.0 <= pct < 1.0:
            raise QuackzInputError(
                f"embargo_pct must be in [0, 1), got {self.embargo_pct!r}. It is a fraction of "
                "the TOTAL number of observations, not of a fold."
            )

    def get_n_splits(self, data: object = None, y: object = None, groups: object = None) -> int:
        """Number of folds produced, which does not depend on the data."""
        return int(self.n_splits)

    def embargo_width(self, data: object) -> int:
        """Embargoed observations after each test fold: `ceil(embargo_pct * n_obs)`.

        The width is a fraction of the WHOLE sample, not of a fold, so it does not change
        when `n_splits` does. Rounding up rather than truncating, as the AFML reference
        code does, keeps a small positive fraction on a short sample from quietly becoming
        no embargo at all; the error is on the side of removing training data.
        """
        return math.ceil(float(self.embargo_pct) * _n_observations(data))

    def split(
        self, data: object, y: object = None, groups: object = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield `(train_indices, test_indices)`, test folds in chronological order."""
        n_obs = _n_observations(data)
        splits = int(self.n_splits)
        if n_obs < splits:
            raise QuackzInputError(
                f"n_splits={splits} needs at least {splits} observations, got {n_obs}"
            )
        width = self.embargo_width(n_obs)
        indices = np.arange(n_obs)
        for fold in np.array_split(indices, splits):
            start = int(fold[0])
            stop = int(fold[-1]) + 1
            keep = np.ones(n_obs, dtype=bool)
            keep[start : min(stop + width, n_obs)] = False
            train = indices[keep]
            if train.size == 0:
                raise QuackzInputError(
                    f"embargo_pct={self.embargo_pct} leaves no training data for the fold "
                    f"starting at observation {start} of {n_obs}; lower the embargo or raise "
                    "n_splits"
                )
            yield train, fold
