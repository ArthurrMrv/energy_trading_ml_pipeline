"""Chronological train/test split and rolling sessions."""

from datetime import date

import polars as pl
import pytest

from backend.pipeline.split import (
    chunk,
    cut,
    enabled,
    in_ts,
    make_test_sessions,
    settings,
    train_sessions,
)


def stamps(n=20):
    return list(range(n))


class TestSettings:
    def test_absent_split_is_disabled(self):
        assert not enabled({})
        assert settings({})["enabled"] is False

    def test_enabled_defaults_true_when_block_present(self):
        assert enabled({"split": {}})
        assert settings({"split": {}})["enabled"] is True

    def test_explicit_false_disables(self):
        assert not enabled({"split": {"enabled": False}})

    def test_purge_defaults_to_one(self):
        assert settings({"split": {}})["purge"] == 1

    def test_rejects_bad_mode(self):
        with pytest.raises(ValueError, match="mode"):
            settings({"split": {"train": {"mode": "kfold"}}})

    def test_rejects_bad_frac(self):
        with pytest.raises(ValueError, match="test_frac"):
            settings({"split": {"test_frac": 1.5}})


class TestCut:
    def test_train_ends_before_test(self):
        train, test = cut(stamps(20), test_frac=0.25, purge=0)
        assert train[-1] < test[0]
        assert len(train) + len(test) == 20

    def test_purge_opens_a_gap(self):
        plain_train, plain_test = cut(stamps(20), test_frac=0.25, purge=0)
        purged_train, purged_test = cut(stamps(20), test_frac=0.25, purge=2)
        assert purged_test == plain_test
        assert purged_train == plain_train[:-2]
        assert purged_train[-1] + 1 < purged_test[0]

    def test_test_start_override(self):
        train, test = cut(stamps(20), test_frac=0.5, test_start=15, purge=0)
        assert train[-1] == 14
        assert test[0] == 15


class TestSessions:
    def test_train_once_is_one_session(self):
        train, _ = cut(stamps(20), test_frac=0.25, purge=0)
        specs = train_sessions(train, mode="once", chunk_size=5)
        assert len(specs) == 1
        assert specs[0]["fit_ts"] == train

    def test_train_rolling_expands(self):
        train, _ = cut(stamps(20), test_frac=0.25, purge=0)
        specs = train_sessions(train, mode="rolling", chunk_size=5)
        lengths = [len(s["fit_ts"]) for s in specs]
        assert lengths == sorted(lengths)
        assert lengths[0] == 5
        assert lengths[-1] == len(train)

    def test_test_once_scores_all_test(self):
        train, test = cut(stamps(20), test_frac=0.25, purge=0)
        specs = make_test_sessions(train, test, mode="once", chunk_size=5, purge=0)
        assert len(specs) == 1
        assert specs[0]["score_ts"] == test
        assert specs[0]["fit_ts"] == train

    def test_test_rolling_expands_fit_and_scores_chunks(self):
        train, test = cut(stamps(40), test_frac=0.4, purge=0)
        parts = chunk(test, 4)
        specs = make_test_sessions(train, test, mode="rolling", chunk_size=4, purge=0)
        assert len(specs) == len(parts)
        assert specs[0]["fit_ts"] == train
        assert specs[0]["score_ts"] == parts[0]
        assert specs[1]["fit_ts"] == train + parts[0]
        assert specs[1]["score_ts"] == parts[1]
        # No overlap across scored chunks; union is the full test.
        scored = [ts for s in specs for ts in s["score_ts"]]
        assert scored == test


class TestInTs:
    def test_date_column_accepts_iso_string_stamps(self):
        frame = pl.DataFrame({
            "ts": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
            "x": [1, 2, 3],
        })
        out = in_ts(frame, ["2020-01-02", "2020-01-03"])
        assert out["ts"].to_list() == [date(2020, 1, 2), date(2020, 1, 3)]

    def test_date_column_accepts_date_stamps(self):
        frame = pl.DataFrame({
            "ts": [date(2020, 1, 1), date(2020, 1, 2)],
            "x": [1, 2],
        })
        out = in_ts(frame, [date(2020, 1, 1)])
        assert out.height == 1
