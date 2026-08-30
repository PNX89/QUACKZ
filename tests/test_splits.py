"""Tests for quackz.splits.

A splitter that leaks is worse than no splitter, because the leak shows up as a good score.
The two properties worth pinning are therefore structural rather than statistical: what
`WalkForward` yields is always strictly ordered in time, and what `EmbargoedKFold` removes
is exactly the band it says it removes.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from quackz import splits
from quackz.returns import QuackzInputError


def as_sets(splitter, n_obs: int) -> list[tuple[set[int], set[int]]]:
    return [(set(train.tolist()), set(test.tolist())) for train, test in splitter.split(n_obs)]


# --------------------------------------------------------------------------------------
# WalkForward
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("expanding", [True, False])
@pytest.mark.parametrize("n_obs", [50, 100, 103, 997])
@pytest.mark.parametrize("n_splits", [1, 3, 5])
def test_every_training_bar_precedes_every_test_bar(expanding, n_obs, n_splits):
    """The one property the whole class exists for."""
    splitter = splits.WalkForward(n_splits, expanding=expanding)
    folds = list(splitter.split(n_obs))
    assert len(folds) == n_splits
    for train, test in folds:
        assert train.size > 0
        assert test.size > 0
        assert train.max() < test.min()


def test_test_folds_are_equal_sized_contiguous_and_cover_the_tail():
    splitter = splits.WalkForward(4)
    folds = [test for _, test in splitter.split(100)]
    assert [test.tolist() for test in folds] == [
        list(range(20, 40)),
        list(range(40, 60)),
        list(range(60, 80)),
        list(range(80, 100)),
    ]


def test_the_remainder_lands_in_the_first_training_set():
    splitter = splits.WalkForward(4)
    assert splitter.test_size(103) == 20
    first_train, first_test = next(iter(splitter.split(103)))
    assert first_train.size == 23
    assert first_test.tolist() == list(range(23, 43))


def test_test_folds_never_overlap():
    splitter = splits.WalkForward(5)
    seen: set[int] = set()
    for _, test in splitter.split(211):
        assert not (seen & set(test.tolist()))
        seen |= set(test.tolist())


def test_an_expanding_window_grows_and_a_rolling_window_does_not():
    expanding = [train.size for train, _ in splits.WalkForward(5).split(300)]
    rolling = [train.size for train, _ in splits.WalkForward(5, expanding=False).split(300)]
    assert all(earlier < later for earlier, later in pairwise(expanding))
    assert len(set(rolling)) == 1
    assert rolling[0] == expanding[0]


def test_a_rolling_window_slides_by_exactly_one_test_fold():
    starts = [train.min() for train, _ in splits.WalkForward(4, expanding=False).split(100)]
    assert starts == [0, 20, 40, 60]


def test_train_and_test_are_always_disjoint():
    for train, test in as_sets(splits.WalkForward(4), 137):
        assert not (train & test)


@pytest.mark.parametrize(
    "data",
    [
        60,
        np.zeros(60),
        list(range(60)),
        pd.Series(np.zeros(60)),
        pd.DataFrame({"a": np.zeros(60)}),
        pd.date_range("2020-01-01", periods=60, freq="B"),
    ],
)
def test_split_accepts_a_count_or_anything_with_a_length(data):
    folds = list(splits.WalkForward(3).split(data))
    assert [test.tolist() for test in (test for _, test in folds)] == [
        list(range(15, 30)),
        list(range(30, 45)),
        list(range(45, 60)),
    ]


def test_get_n_splits_does_not_need_the_data():
    assert splits.WalkForward(7).get_n_splits() == 7
    assert splits.EmbargoedKFold(3, embargo_pct=0.01).get_n_splits() == 3


def test_indices_are_integer_arrays_usable_for_positional_indexing():
    frame = pd.DataFrame({"a": np.arange(40.0)})
    train, test = next(iter(splits.WalkForward(3).split(frame)))
    assert train.dtype.kind == "i"
    assert frame.iloc[train].shape[0] == train.size
    assert frame.iloc[test].shape[0] == test.size


def test_walk_forward_rejects_a_split_count_it_cannot_honour():
    with pytest.raises(QuackzInputError, match="observations"):
        list(splits.WalkForward(10).split(8))


@pytest.mark.parametrize("n_splits", [0, -1])
def test_walk_forward_rejects_a_non_positive_split_count(n_splits):
    with pytest.raises(QuackzInputError, match="at least 1"):
        splits.WalkForward(n_splits)


@pytest.mark.parametrize("bad", [2.5, "3", None])
def test_walk_forward_rejects_a_non_integer_split_count(bad):
    with pytest.raises(QuackzInputError, match="must be an integer"):
        splits.WalkForward(bad)


def test_walk_forward_rejects_a_non_boolean_expanding_flag():
    with pytest.raises(QuackzInputError, match="expanding"):
        splits.WalkForward(3, expanding="yes")


def test_walk_forward_rejects_data_with_no_length():
    with pytest.raises(QuackzInputError, match="length"):
        list(splits.WalkForward(3).split(object()))


def test_walk_forward_is_comparable_and_reprs_its_configuration():
    assert splits.WalkForward(3) == splits.WalkForward(3)
    assert splits.WalkForward(3) != splits.WalkForward(3, expanding=False)
    assert "expanding=False" in repr(splits.WalkForward(3, expanding=False))


# --------------------------------------------------------------------------------------
# EmbargoedKFold
# --------------------------------------------------------------------------------------


def test_test_folds_partition_the_whole_sample():
    covered: list[int] = []
    for _, test in splits.EmbargoedKFold(5, embargo_pct=0.02).split(203):
        covered.extend(test.tolist())
    assert covered == list(range(203))


def test_train_and_test_are_disjoint_in_every_fold():
    for train, test in as_sets(splits.EmbargoedKFold(4, embargo_pct=0.05), 200):
        assert not (train & test)


def test_the_embargo_width_is_a_fraction_of_the_whole_sample():
    """Per AFML the embargo is a fraction of TOTAL observations, not of a fold."""
    splitter = splits.EmbargoedKFold(5, embargo_pct=0.01)
    assert splitter.embargo_width(1000) == 10
    assert splits.EmbargoedKFold(10, embargo_pct=0.01).embargo_width(1000) == 10


def test_a_requested_embargo_never_rounds_down_to_nothing():
    """Truncating 2.5 bars to 2 would quietly hand back training data that was asked for."""
    assert splits.EmbargoedKFold(5, embargo_pct=0.01).embargo_width(250) == 3
    assert splits.EmbargoedKFold(5, embargo_pct=0.001).embargo_width(250) == 1
    assert splits.EmbargoedKFold(5, embargo_pct=0.0).embargo_width(250) == 0


def test_exactly_the_embargoed_band_is_removed_from_training():
    """The removed set is the test fold plus the `width` bars immediately after it."""
    n_obs, width = 200, 10
    splitter = splits.EmbargoedKFold(4, embargo_pct=width / n_obs)
    assert splitter.embargo_width(n_obs) == width
    for train, test in splitter.split(n_obs):
        removed = set(range(n_obs)) - set(train.tolist())
        embargo_stop = min(int(test.max()) + 1 + width, n_obs)
        assert removed == set(range(int(test.min()), embargo_stop))


def test_nothing_before_the_test_fold_is_embargoed():
    """The embargo is one sided; bars before the fold stay in training."""
    splitter = splits.EmbargoedKFold(4, embargo_pct=0.10)
    for train, test in splitter.split(200):
        before = set(range(int(test.min())))
        assert before <= set(train.tolist())


def test_the_last_fold_has_nothing_after_it_to_embargo():
    splitter = splits.EmbargoedKFold(4, embargo_pct=0.05)
    folds = list(splitter.split(200))
    last_train, last_test = folds[-1]
    assert last_train.size == 200 - last_test.size
    assert folds[0][0].size == 200 - folds[0][1].size - splitter.embargo_width(200)


def test_a_zero_embargo_leaves_the_plain_kfold_complement():
    for train, test in as_sets(splits.EmbargoedKFold(5, embargo_pct=0.0), 100):
        assert train | test == set(range(100))
        assert len(train) == 80


def test_a_wider_embargo_removes_more_training_data():
    sizes = [
        next(iter(splits.EmbargoedKFold(4, embargo_pct=pct).split(400)))[0].size
        for pct in (0.0, 0.01, 0.05, 0.10)
    ]
    assert all(later < earlier for earlier, later in pairwise(sizes))


def test_every_fold_except_the_last_trains_on_data_from_after_its_test_fold():
    """The defect the docstring admits to, pinned so nobody can claim otherwise.

    An embargo trims the seam between train and test; it does not stop the training set
    from containing the future. That is why WalkForward is the recommended splitter.
    """
    folds = list(splits.EmbargoedKFold(4, embargo_pct=0.05).split(200))
    for train, test in folds[:-1]:
        assert train.max() > test.max()
    assert folds[-1][0].max() < folds[-1][1].min()


def test_embargoed_kfold_rejects_an_embargo_that_leaves_no_training_data():
    with pytest.raises(QuackzInputError, match="no training data"):
        list(splits.EmbargoedKFold(2, embargo_pct=0.9).split(100))


@pytest.mark.parametrize("pct", [-0.01, 1.0, 1.5, float("nan")])
def test_embargoed_kfold_rejects_an_impossible_embargo(pct):
    with pytest.raises(QuackzInputError, match=r"embargo_pct must be in \[0, 1\)"):
        splits.EmbargoedKFold(4, embargo_pct=pct)


@pytest.mark.parametrize("n_splits", [1, 0, -3])
def test_embargoed_kfold_needs_at_least_two_folds(n_splits):
    with pytest.raises(QuackzInputError, match="at least 2"):
        splits.EmbargoedKFold(n_splits, embargo_pct=0.01)


def test_embargoed_kfold_rejects_more_folds_than_observations():
    with pytest.raises(QuackzInputError, match="observations"):
        list(splits.EmbargoedKFold(20, embargo_pct=0.01).split(10))


def test_embargo_pct_must_be_passed_by_keyword():
    with pytest.raises(TypeError):
        splits.EmbargoedKFold(4, 0.01)


def test_the_embargoed_splitter_does_not_borrow_the_word_purged():
    """Purging needs label horizons this library never receives, so the name is not taken.

    A reviewer who knows AFML will look for exactly this, and a class called PurgedKFold
    that cannot purge is a claim that falls apart on the first question.
    """
    assert not hasattr(splits, "PurgedKFold")
    documentation = splits.EmbargoedKFold.__doc__ or ""
    assert "NOT a purged K-fold" in documentation
    assert "label" in documentation.lower()
    assert "t1" in documentation
    assert "POSTDATES" in documentation


def test_the_forward_only_splitter_is_documented_as_the_recommended_one():
    assert "honest default" in (splits.__doc__ or "")
    assert "Forward-only" in (splits.WalkForward.__doc__ or "")


def test_splitters_reject_a_sample_of_one():
    for splitter in (splits.WalkForward(1), splits.EmbargoedKFold(2, embargo_pct=0.0)):
        with pytest.raises(QuackzInputError, match="at least 2 observations"):
            list(splitter.split(1))


# The two splitters this module offers, named and counted here so that deleting one leaves
# the sklearn-convention test covering one case fewer and saying so, rather than passing.
SPLITTERS = {
    "WalkForward": splits.WalkForward(3),
    "EmbargoedKFold": splits.EmbargoedKFold(3, embargo_pct=0.02),
}


def test_both_splitters_are_named_here():
    assert set(SPLITTERS) == {"WalkForward", "EmbargoedKFold"}
    assert len(SPLITTERS) == 2


@pytest.mark.parametrize("name", sorted(SPLITTERS))
def test_a_splitter_takes_the_arguments_scikit_learn_actually_calls_it_with(name):
    """The module's first paragraph says these drop into a pipeline. They did not.

    `cross_validate`, `cross_val_score` and `GridSearchCV` all call a splitter as
    `split(X, y, groups)` and `get_n_splits(X, y, groups)`. Both methods took one argument,
    so every one of those raised TypeError on the first call, and no test could see it
    because the suite only ever called `split(n_obs)` and scikit-learn is not a dependency
    here. The extra arguments must be accepted and must change nothing: a splitter that cuts
    a sample by time reads neither labels nor groups.
    """
    splitter = SPLITTERS[name]
    features = np.zeros((60, 2))
    labels = np.arange(60)

    expected = [(train.tolist(), test.tolist()) for train, test in splitter.split(60)]
    assert len(expected) == 3

    positional = [
        (train.tolist(), test.tolist()) for train, test in splitter.split(features, labels, None)
    ]
    by_keyword = [
        (train.tolist(), test.tolist())
        for train, test in splitter.split(features, y=labels, groups=None)
    ]
    assert positional == expected
    assert by_keyword == expected

    assert splitter.get_n_splits(features, labels, None) == 3
    assert splitter.get_n_splits(features, y=labels, groups=None) == 3
