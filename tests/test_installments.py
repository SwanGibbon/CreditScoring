import numpy as np
import pandas as pd
import pytest

from credit_scoring.installments import InstallmentLedger


def ledger_result(tmp_path, rows, chunksize=1):
    ledger = InstallmentLedger(str(tmp_path / "payments.sqlite"))
    audit = {}
    try:
        for i in range(0, len(rows), chunksize):
            ledger.add(rows.iloc[i:i+chunksize], audit)
        result = pd.concat(list(ledger.installments(chunksize)), ignore_index=True)
        return result, audit
    finally:
        ledger.close()


def payments(amounts, days, due_amount=100., due_day=-10):
    return pd.DataFrame(dict(SK_ID_CURR=1, SK_ID_PREV=10,
        NUM_INSTALMENT_VERSION=1, NUM_INSTALMENT_NUMBER=1,
        DAYS_INSTALMENT=due_day, AMT_INSTALMENT=due_amount,
        DAYS_ENTRY_PAYMENT=days, AMT_PAYMENT=amounts))


def test_partial_payments_across_chunks_count_one_obligation(tmp_path):
    result, _ = ledger_result(tmp_path, payments([40., 60.], [-15., -8.]))
    assert len(result) == 1
    row = result.iloc[0]
    assert row.AMT_INSTALMENT == 100 and row.AMT_PAYMENT == 100
    assert row.PAY_RATIO == 1 and row.PAY_DIFF == 0
    assert row.COMPLETION_DAY == -8 and row.DPD == 2 and row.LATE == 1


def test_late_overpayment_does_not_move_completion_date(tmp_path):
    result, _ = ledger_result(tmp_path, payments([100., 20.], [-12., -2.]))
    row = result.iloc[0]
    assert row.COMPLETION_DAY == -12 and row.DPD == 0 and row.LATE == 0
    assert row.AMT_PAYMENT == 120 and row.PAY_DIFF == 0


def test_missed_installment_is_retained_with_ageing_delinquency(tmp_path):
    result, _ = ledger_result(tmp_path, payments([np.nan], [np.nan]))
    row = result.iloc[0]
    assert row.AMT_PAYMENT == 0 and row.PAY_DIFF == 100
    assert row.DPD == 10 and row.LATE == 1 and row.UNPAID == 1


def test_future_payment_is_not_cash_asof_application(tmp_path):
    result, audit = ledger_result(tmp_path, payments([30., 70.], [-12., 5.]))
    row = result.iloc[0]
    assert row.AMT_PAYMENT == 30 and row.PAY_DIFF == 70
    assert row.DPD == 10 and row.UNPAID == 1
    assert audit["future_payment_rows_ignored"] == 1


def test_unknown_payment_amount_does_not_become_false_zero(tmp_path):
    result, _ = ledger_result(tmp_path, payments([np.nan], [-8.]))
    row = result.iloc[0]
    assert row.PAYMENT_UNKNOWN == 1
    assert np.isnan(row.AMT_PAYMENT) and np.isnan(row.PAY_DIFF) and np.isnan(row.DPD)


def test_conflicting_obligation_fails_instead_of_double_counting(tmp_path):
    rows = payments([40., 60.], [-12., -8.])
    rows.loc[1, "AMT_INSTALMENT"] = 200
    with pytest.raises(ValueError, match="Conflicting scheduled amount"):
        ledger_result(tmp_path, rows)


def test_reordered_rows_and_chunks_give_identical_installments(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    rows = pd.concat([payments([40., 60.], [-15., -8.]),
                      payments([np.nan], [np.nan]).assign(NUM_INSTALMENT_NUMBER=2)], ignore_index=True)
    first, _ = ledger_result(a, rows, 1)
    second, _ = ledger_result(b, rows.iloc[::-1], 3)
    pd.testing.assert_frame_equal(first, second)
