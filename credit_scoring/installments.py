"""Reduce split payments to installments before computing repayment features.

SQLite holds the payment ledger on local temporary disk so an installment may
span arbitrary CSV chunks without keeping the complete ledger in RAM.
"""
import sqlite3

import numpy as np
import pandas as pd

KEYS = ["SK_ID_CURR", "SK_ID_PREV", "NUM_INSTALMENT_VERSION",
        "NUM_INSTALMENT_NUMBER", "DAYS_INSTALMENT"]
REQUIRED = KEYS + ["AMT_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_PAYMENT"]


class InstallmentLedger:
    def __init__(self, path):
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA temp_store = FILE")
        self.connection.execute("PRAGMA cache_size = -32768")  # 32 MiB page cache
        self.rows = 0

    def close(self):
        self.connection.close()

    def add(self, frame, audit):
        missing = set(REQUIRED) - set(frame)
        if missing:
            raise ValueError(f"installments_payments missing columns: {sorted(missing)}")
        if frame[KEYS].isna().any().any():
            raise ValueError("Installment identity/due date cannot be null")
        if not np.isfinite(frame[KEYS].to_numpy()).all():
            raise ValueError("Invalid installment identity/due date")
        audit["installment_due"] = audit.get("installment_due", 0) + int((frame.DAYS_INSTALMENT > 0).sum())
        frame = frame.loc[frame.DAYS_INSTALMENT <= 0, REQUIRED].copy()
        if frame.empty:
            return
        amounts = frame[["AMT_INSTALMENT", "AMT_PAYMENT"]]
        if np.isinf(amounts.to_numpy()).any() or (amounts < 0).any().any():
            raise ValueError("Installment/payment amounts must be nonnegative and finite or null")
        if np.isinf(frame.DAYS_ENTRY_PAYMENT.to_numpy()).any():
            raise ValueError("Invalid payment observation date")
        observed = frame.DAYS_ENTRY_PAYMENT.notna() & (frame.DAYS_ENTRY_PAYMENT <= 0)
        future = frame.DAYS_ENTRY_PAYMENT > 0
        audit["future_payment_rows_ignored"] = audit.get("future_payment_rows_ignored", 0) + int(future.sum())
        # An undated row with neither amount nor date denotes a missed payment.
        # An amount with unknown observation date cannot be used as known cash.
        unknown = (observed & frame.AMT_PAYMENT.isna()) | (
            frame.DAYS_ENTRY_PAYMENT.isna() & frame.AMT_PAYMENT.notna())
        audit["unknown_payment_rows"] = audit.get("unknown_payment_rows", 0) + int(unknown.sum())
        frame["paid_asof"] = frame.AMT_PAYMENT.where(observed, 0).fillna(0)
        frame["payment_day"] = frame.DAYS_ENTRY_PAYMENT.where(observed)
        frame["unknown_payment"] = unknown.astype(np.int8)
        frame = frame[KEYS + ["AMT_INSTALMENT", "paid_asof", "payment_day", "unknown_payment"]]
        frame.to_sql("payments", self.connection, if_exists="append", index=False, chunksize=10_000)
        self.rows += len(frame)

    def installments(self, chunksize=250_000):
        if not self.rows:
            return
        keys = ", ".join(KEYS)
        self.connection.execute(f"CREATE INDEX ledger_keys ON payments ({keys}, payment_day)")
        # Disagreement within the same version/number/due date is an ambiguous
        # obligation, not a license to sum its amount once per partial payment.
        conflict = self.connection.execute(f"""
            SELECT 1 FROM payments GROUP BY {keys}
            HAVING MAX(AMT_INSTALMENT) - MIN(AMT_INSTALMENT) >
                   0.000001 * MAX(1.0, ABS(MAX(AMT_INSTALMENT))) LIMIT 1
        """).fetchone()
        if conflict:
            raise ValueError("Conflicting scheduled amount for the same installment")
        query = f"""
            WITH totals AS (
                SELECT {keys}, MAX(AMT_INSTALMENT) AS due_amount,
                       SUM(paid_asof) AS paid, MAX(unknown_payment) AS unknown_payment
                FROM payments GROUP BY {keys}
            ), timeline AS (
                SELECT {keys}, payment_day,
                       SUM(paid_asof) OVER (
                           PARTITION BY {keys} ORDER BY payment_day
                           RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                       ) AS cumulative_paid
                FROM payments WHERE payment_day IS NOT NULL
            ), completion AS (
                SELECT {', '.join('t.' + key for key in KEYS)}, MIN(payment_day) AS completion_day
                FROM timeline t JOIN totals s USING ({keys})
                WHERE cumulative_paid + 0.000001 >= s.due_amount
                GROUP BY {', '.join('t.' + key for key in KEYS)}
            )
            SELECT {', '.join('s.' + key for key in KEYS)}, s.due_amount AS AMT_INSTALMENT,
                   CASE WHEN s.unknown_payment THEN NULL ELSE s.paid END AS AMT_PAYMENT,
                   s.unknown_payment AS PAYMENT_UNKNOWN, c.completion_day AS COMPLETION_DAY
            FROM totals s LEFT JOIN completion c USING ({keys})
            ORDER BY {', '.join('s.' + key for key in KEYS)}
        """
        for frame in pd.read_sql_query(query, self.connection, chunksize=chunksize):
            frame = frame.apply(pd.to_numeric)
            frame.loc[frame.PAYMENT_UNKNOWN > 0, "COMPLETION_DAY"] = np.nan
            due, paid = frame.AMT_INSTALMENT, frame.AMT_PAYMENT
            frame["PAY_RATIO"] = paid / due.where(due > 0)
            frame["PAY_DIFF"] = (due - paid).clip(lower=0)
            unfinished = paid + 1e-6 < due
            frame["DPD"] = (frame.COMPLETION_DAY - frame.DAYS_INSTALMENT).clip(lower=0)
            frame.loc[unfinished, "DPD"] = -frame.loc[unfinished, "DAYS_INSTALMENT"]
            frame.loc[(due == 0) & paid.notna(), "DPD"] = 0
            frame.loc[due.isna() | paid.isna(), "DPD"] = np.nan
            frame["LATE"] = (frame.DPD > 0).astype(float).where(frame.DPD.notna())
            frame["UNPAID"] = unfinished.astype(float).where(due.notna() & paid.notna())
            yield frame
