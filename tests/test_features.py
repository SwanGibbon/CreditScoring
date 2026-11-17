import numpy as np
import pandas as pd
import pytest

from credit_scoring.features import Aggregator, application_features, build_features, check_files, check_schema


def test_missing_files_lists_all_eight(tmp_path):
    with pytest.raises(FileNotFoundError) as error:
        check_files(tmp_path)
    assert str(error.value).count(".csv") == 8


def test_chunk_aggregation_matches_whole_table_with_nulls():
    x = pd.DataFrame({"SK_ID_CURR": [1, 2, 1, 2, 3],
                      "value": [np.nan, 4., 6., np.nan, np.nan]})
    whole, chunks = Aggregator("X"), Aggregator("X")
    whole.add(x)
    for i in range(len(x)):
        chunks.add(x.iloc[i:i+1])
    pd.testing.assert_frame_equal(whole.result(), chunks.result())
    assert whole.result().loc[1, "X_VALUE_MEAN"] == 6
    assert np.isnan(whole.result().loc[3, "X_VALUE_MEAN"])


def test_builder_alignment_target_exclusion_and_chunk_invariance(data_dir):
    a, y, test, ids, audit = build_features(data_dir, chunksize=3)
    b, _, _, _, _ = build_features(data_dir, chunksize=100)
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])
        assert not {"SK_ID_CURR", "SK_ID_PREV", "SK_ID_BUREAU", "TARGET", "key_0"} & set(a[name])
        assert len(a[name]) == len(y) == 10  # keep XNA applicants
    assert ids.tolist() == [110, 111]
    assert len(test["history"]) == 2
    assert a["history"].loc[0, "INS_AMT_PAYMENT_SUM"] == 3
    assert a["history"].loc[9, "INS_AMT_PAYMENT_SUM"] == 39
    assert a["application"].NAN_CNT.eq(1).all()
    assert audit["unique_application_ids"]
    raw = pd.read_csv(data_dir / "application_train.csv")
    changed = raw.copy()
    changed["TARGET"] = np.nan
    pd.testing.assert_frame_equal(application_features(raw), application_features(changed))


def test_future_history_is_excluded(data_dir):
    p = data_dir / "installments_payments.csv"
    frame = pd.read_csv(p)
    frame.loc[0, ["DAYS_ENTRY_PAYMENT", "AMT_PAYMENT"]] = [1, 1e9]
    frame.to_csv(p, index=False)
    a, _, _, _, audit = build_features(data_dir, chunksize=3)
    assert a["history"].loc[0, "INS_AMT_PAYMENT_SUM"] == 2
    assert audit["excluded_rows"]["future_payment_rows_ignored"] == 1


def test_conflicting_history_owner_is_rejected_across_chunks(data_dir):
    p = data_dir / "credit_card_balance.csv"
    frame = pd.read_csv(p)
    frame.loc[4, "SK_ID_PREV"] = frame.loc[0, "SK_ID_PREV"]
    frame.to_csv(p, index=False)
    with pytest.raises(ValueError, match="Conflicting borrower"):
        build_features(data_dir, chunksize=3)


def test_duplicate_application_id_is_rejected(data_dir):
    p = data_dir / "application_train.csv"
    frame = pd.read_csv(p)
    frame.loc[1, "SK_ID_CURR"] = frame.loc[0, "SK_ID_CURR"]
    frame.to_csv(p, index=False)
    with pytest.raises(ValueError, match="one application"):
        build_features(data_dir)


def test_bureau_status_dtype_is_stable_in_numeric_only_chunks(data_dir):
    path = data_dir / "bureau_balance.csv"
    bb = pd.read_csv(path)
    bb["STATUS"] = ["1", None, "C"] * (len(bb) // 3)
    bb.to_csv(path, index=False)
    a, _, _, _, _ = build_features(data_dir, chunksize=2)
    b, _, _, _, _ = build_features(data_dir, chunksize=100)
    pd.testing.assert_frame_equal(a["history"], b["history"])
    assert a["history"].loc[0, "BB_BAD_MEAN"] == 1.
    assert a["history"].loc[0, "BB_STATUS_UNKNOWN_MEAN"] == .5
    assert not {"BB_STATUS_MIN", "BB_STATUS_MAX", "BB_STATUS_MEAN", "BB_STATUS_SUM"} & set(a["history"])


def test_missing_installment_identity_is_caught_by_header_preflight(data_dir):
    path = data_dir / "installments_payments.csv"
    pd.read_csv(path).drop(columns="NUM_INSTALMENT_NUMBER").to_csv(path, index=False)
    with pytest.raises(ValueError, match="NUM_INSTALMENT_NUMBER"):
        check_schema(data_dir)


def test_train_test_schema_mismatch_fails_before_history_loading(data_dir):
    path = data_dir / "application_test.csv"
    pd.read_csv(path).assign(EXTRA_PREDICTOR=1).to_csv(path, index=False)
    with pytest.raises(ValueError, match="predictor columns differ"):
        check_schema(data_dir)


def test_invalid_target_rejected_before_history_loading(data_dir):
    path = data_dir / "application_train.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "TARGET"] = np.nan
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="complete binary"):
        build_features(data_dir)
