"""Only deterministic per-application / per-owner history features live here.

No target, population statistics, vocabulary fitting, or feature selection.
History is streamed; each chunk contributes sums/counts/min/max per owner.
"""
from pathlib import Path
import csv
import tempfile

import numpy as np
import pandas as pd

from .installments import InstallmentLedger, REQUIRED as INSTALLMENT_COLUMNS

ID, TARGET = "SK_ID_CURR", "TARGET"
HISTORY_FILES = ("bureau", "bureau_balance", "previous_application",
                 "POS_CASH_balance", "installments_payments", "credit_card_balance")
REQUIRED_FILES = ("application_train", "application_test") + HISTORY_FILES
EPS = 1e-6
APPLICATION_COLUMNS = {ID, "CODE_GENDER", "DAYS_EMPLOYED", "DAYS_BIRTH", "DAYS_REGISTRATION",
                       "DAYS_ID_PUBLISH", "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY",
                       "AMT_GOODS_PRICE", "CNT_FAM_MEMBERS", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"}
SCHEMA = {
    "application_train": APPLICATION_COLUMNS | {TARGET},
    "application_test": APPLICATION_COLUMNS,
    "bureau": {ID, "SK_ID_BUREAU", "DAYS_CREDIT", "DAYS_CREDIT_ENDDATE",
               "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM", "CREDIT_DAY_OVERDUE", "CREDIT_ACTIVE"},
    "bureau_balance": {"SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"},
    "previous_application": {ID, "SK_ID_PREV", "DAYS_DECISION", "AMT_APPLICATION",
                             "AMT_CREDIT", "AMT_ANNUITY", "NAME_CONTRACT_STATUS"},
    "installments_payments": set(INSTALLMENT_COLUMNS),
    "POS_CASH_balance": {ID, "SK_ID_PREV", "MONTHS_BALANCE", "SK_DPD"},
    "credit_card_balance": {ID, "SK_ID_PREV", "MONTHS_BALANCE", "SK_DPD",
                            "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"},
}


def check_files(directory, names=REQUIRED_FILES):
    directory = Path(directory)
    missing = [str(directory / f"{name}.csv") for name in names
               if not (directory / f"{name}.csv").is_file()]
    if missing:
        raise FileNotFoundError("Missing required data files:\n" + "\n".join(missing))


def check_schema(directory, names=REQUIRED_FILES):
    check_files(directory, names)
    headers = {}
    for name in names:
        required = SCHEMA[name]
        path = Path(directory) / f"{name}.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), [])
        if len(header) != len(set(header)):
            raise ValueError(f"{name}.csv has duplicate column names")
        missing = required - set(header)
        if missing:
            raise ValueError(f"{name}.csv missing required columns: {sorted(missing)}")
        headers[name] = set(header)
    if headers["application_train"] - {TARGET} != headers["application_test"]:
        raise ValueError("Application train/test predictor columns differ")


def application_features(raw):
    x = raw.drop(columns=[ID, TARGET], errors="ignore").copy()
    x["EMPLOYED_ANOM"] = (x.DAYS_EMPLOYED == 365243).astype(np.int8)
    x["DAYS_EMPLOYED"] = x.DAYS_EMPLOYED.replace(365243, np.nan)
    for c in ["DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH"]:
        x[c] = x[c].abs()
    for name, numerator, denominator in [
        ("CREDIT_INCOME", "AMT_CREDIT", "AMT_INCOME_TOTAL"),
        ("ANNUITY_INCOME", "AMT_ANNUITY", "AMT_INCOME_TOTAL"),
        ("ANNUITY_CREDIT_RATIO", "AMT_ANNUITY", "AMT_CREDIT"),
        ("CREDIT_GOODS", "AMT_CREDIT", "AMT_GOODS_PRICE"),
        ("INCOME_PER_PERSON", "AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"),
        ("EMPLOYED_RATIO", "DAYS_EMPLOYED", "DAYS_BIRTH"),
    ]:
        x[name] = x[numerator] / (x[denominator] + EPS)
    x["AGE"] = x.DAYS_BIRTH / 365.25
    ext = x[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    x["EXT_MEAN"], x["EXT_MIN"], x["EXT_STD"] = ext.mean(1), ext.min(1), ext.std(1)
    x["EXT_NAN"] = ext.isna().sum(1)
    x["EXT_X_ANNUITY_RATIO"] = x.EXT_MEAN * x.ANNUITY_CREDIT_RATIO
    # TARGET/test membership and identifiers must never enter this count.
    x["NAN_CNT"] = raw.drop(columns=[ID, TARGET], errors="ignore").isna().sum(1)
    return x.replace([np.inf, -np.inf], np.nan)


class Aggregator:
    """Associative chunk aggregation; means use non-null counts, not chunk means."""
    def __init__(self, prefix):
        self.prefix = prefix
        self.stats = None
        self.rows = None

    def add(self, frame):
        if frame.empty:
            return
        numeric = frame.select_dtypes("number").drop(
            columns=["SK_ID_PREV", "SK_ID_BUREAU", TARGET], errors="ignore")
        stats = numeric.groupby(ID).agg(["min", "max", "sum", "count"])
        rows = frame.groupby(ID).size()
        if self.stats is None:
            self.stats, self.rows = stats, rows
            return
        joined = pd.concat([self.stats, stats])
        self.stats = joined.groupby(level=0).agg(
            {c: c[1] if c[1] in {"min", "max"} else "sum" for c in joined.columns})
        self.rows = self.rows.add(rows, fill_value=0)

    def result(self):
        if self.stats is None:
            return pd.DataFrame(index=pd.Index([], name=ID))
        out = {}
        for name in self.stats.columns.get_level_values(0).unique():
            s = self.stats[name]
            for method in ["min", "max", "sum"]:
                out[f"{self.prefix}_{name}_{method}".upper()] = s[method]
            out[f"{self.prefix}_{name}_MEAN".upper()] = s["sum"] / s["count"].replace(0, np.nan)
        out[f"{self.prefix}_CNT"] = self.rows
        return pd.DataFrame(out).astype(np.float32)


def past(frame, column, audit, label):
    mask = frame[column].notna() & frame[column].le(0)
    audit[label] = audit.get(label, 0) + int((~mask).sum())
    return frame.loc[mask].copy()


def historical_features(directory, owners, chunksize=250_000):
    directory = Path(directory)
    audit = {"excluded_rows": {}, "unknown_owner_rows": {}}
    tables = []
    def chunks(name):
        dtype = {"STATUS": "string"} if name == "bureau_balance" else None
        for frame in pd.read_csv(directory / f"{name}.csv", chunksize=chunksize, dtype=dtype):
            if ID in frame:
                mask = frame[ID].isin(owners)
                counts = audit["unknown_owner_rows"]
                counts[name] = counts.get(name, 0) + int((~mask).sum())
                frame = frame.loc[mask].copy()
            yield frame

    # Bureau IDs are a foreign key, never model features. Enforce single ownership.
    bu = pd.read_csv(directory / "bureau.csv")
    if bu.SK_ID_BUREAU.isna().any() or bu.SK_ID_BUREAU.duplicated().any():
        raise ValueError("bureau.SK_ID_BUREAU must be unique and non-null")
    audit["unknown_owner_rows"]["bureau"] = int((~bu[ID].isin(owners)).sum())
    bu = bu.loc[bu[ID].isin(owners)].copy()
    bu = past(bu, "DAYS_CREDIT", audit["excluded_rows"], "bureau")
    owner_map = bu.set_index("SK_ID_BUREAU")[ID]
    bb_agg = Aggregator("BB")
    for bb in chunks("bureau_balance"):
        bb = past(bb, "MONTHS_BALANCE", audit["excluded_rows"], "bureau_balance")
        bb[ID] = bb.SK_ID_BUREAU.map(owner_map)
        audit["unknown_owner_rows"]["bureau_balance"] = (
            audit["unknown_owner_rows"].get("bureau_balance", 0) + int(bb[ID].isna().sum()))
        bb = bb.loc[bb[ID].notna()].copy()
        allowed = set("012345") | {"C", "X"}
        if not bb.STATUS.dropna().isin(allowed).all():
            raise ValueError("Unrecognized bureau_balance.STATUS")
        unknown_status = bb.STATUS.isna() | bb.STATUS.eq("X").fillna(False)
        bb["STATUS_UNKNOWN"] = unknown_status.astype(np.int8)
        bb["BAD"] = bb.STATUS.isin(list("12345")).astype(float).where(~unknown_status)
        # Aggregate monthly history directly to its verified owner.
        bb_agg.add(bb)
    tables.append(bb_agg.result())
    if "DAYS_ENDDATE_FACT" in bu:
        bu["DAYS_ENDDATE_FACT"] = bu.DAYS_ENDDATE_FACT.where(bu.DAYS_ENDDATE_FACT <= 0)
    bu["DURATION"] = bu.DAYS_CREDIT_ENDDATE - bu.DAYS_CREDIT  # scheduled end is allowed
    bu["DEBT_RATIO"] = bu.AMT_CREDIT_SUM_DEBT / (bu.AMT_CREDIT_SUM + EPS)
    bu["OVERDUE"] = (bu.CREDIT_DAY_OVERDUE > 0).astype(np.int8)
    for prefix, frame in [("BURO", bu), ("BURO_ACT", bu[bu.CREDIT_ACTIVE == "Active"]),
                          ("BURO_2Y", bu[bu.DAYS_CREDIT >= -730])]:
        a = Aggregator(prefix)
        a.add(frame)
        tables.append(a.result())
    del bu, owner_map

    # Validate SK_ID_PREV ownership across previous applications, POS, cards, payments.
    previous_owners = {}
    def verify_previous(frame):
        keys = frame[["SK_ID_PREV", ID]].drop_duplicates()
        if keys.isna().any().any() or keys.SK_ID_PREV.duplicated().any():
            raise ValueError("Invalid SK_ID_PREV ownership within a chunk")
        for key, owner in keys.itertuples(index=False, name=None):
            if key in previous_owners and previous_owners[key] != owner:
                raise ValueError(f"Conflicting borrower for SK_ID_PREV={key}")
            previous_owners[key] = owner

    for name, prefix in [("previous_application", "PREV"),
                         ("installments_payments", "INS"),
                         ("POS_CASH_balance", "POS"), ("credit_card_balance", "CC")]:
        aggregates = {prefix: Aggregator(prefix)}
        suffixes = ["OK", "NO"] if prefix == "PREV" else ["1Y"]
        aggregates.update({f"{prefix}_{s}": Aggregator(f"{prefix}_{s}") for s in suffixes})
        if prefix == "INS":
            with tempfile.TemporaryDirectory(prefix="creditscoring-installments-") as tmp:
                ledger = InstallmentLedger(str(Path(tmp) / "ledger.sqlite"))
                try:
                    for frame in chunks(name):
                        verify_previous(frame)
                        ledger.add(frame, audit["excluded_rows"])
                    for frame in ledger.installments(chunksize):
                        aggregates["INS"].add(frame)
                        aggregates["INS_1Y"].add(frame[frame.DAYS_INSTALMENT >= -365])
                finally:
                    ledger.close()
            tables.extend(a.result() for a in aggregates.values())
            continue
        for frame in chunks(name):
            verify_previous(frame)
            anchor = {"PREV": "DAYS_DECISION",
                      "POS": "MONTHS_BALANCE", "CC": "MONTHS_BALANCE"}[prefix]
            frame = past(frame, anchor, audit["excluded_rows"], name)
            if prefix == "PREV":
                for c in ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
                          "DAYS_LAST_DUE", "DAYS_TERMINATION"]:
                    if c in frame:
                        frame[c] = frame[c].replace(365243, np.nan)
                for c in ["DAYS_LAST_DUE", "DAYS_TERMINATION", "DAYS_FIRST_DRAWING"]:
                    if c in frame:
                        frame[c] = frame[c].where(frame[c] <= 0)
                frame["APP_CREDIT"] = frame.AMT_APPLICATION / (frame.AMT_CREDIT + EPS)
                frame["ANNUITY_CREDIT_RATIO"] = frame.AMT_ANNUITY / (frame.AMT_CREDIT + EPS)
                frame["REFUSED"] = (frame.NAME_CONTRACT_STATUS == "Refused").astype(np.int8)
                aggregates["PREV_OK"].add(frame[frame.NAME_CONTRACT_STATUS == "Approved"])
                aggregates["PREV_NO"].add(frame[frame.NAME_CONTRACT_STATUS == "Refused"])
            else:
                frame["DPD"] = (frame.SK_DPD > 0).astype(np.int8)
                if prefix == "CC":
                    frame["UTIL"] = frame.AMT_BALANCE / (frame.AMT_CREDIT_LIMIT_ACTUAL + EPS)
                aggregates[f"{prefix}_1Y"].add(frame[frame.MONTHS_BALANCE >= -12])
            frame = frame.replace([np.inf, -np.inf], np.nan)
            aggregates[prefix].add(frame)
        tables.extend(a.result() for a in aggregates.values())
    out = pd.DataFrame(index=pd.Index(sorted(owners), name=ID))
    for table in tables:
        out = out.join(table, how="left", validate="one_to_one")
    for numerator, denominator, name in [
        ("BURO_ACT_CNT", "BURO_CNT", "BURO_ACTIVE_SHARE"),
        ("PREV_NO_CNT", "PREV_CNT", "PREV_REFUSED_SHARE")]:
        if numerator in out and denominator in out:
            out[name] = out[numerator].fillna(0) / (out[denominator] + EPS)
    for name in ["DPD", "LATE"]:
        if f"INS_1Y_{name}_MEAN" in out:
            out[f"INS_{name}_TREND"] = out[f"INS_1Y_{name}_MEAN"] - out[f"INS_{name}_MEAN"]
    return out.replace([np.inf, -np.inf], np.nan), audit


def _read_applications(directory):
    directory = Path(directory)
    train = pd.read_csv(directory / "application_train.csv")
    test = pd.read_csv(directory / "application_test.csv")
    if TARGET not in train or TARGET in test:
        raise ValueError("Expected TARGET only in application_train.csv")
    if train.empty or test.empty or not train[TARGET].isin([0, 1]).all():
        raise ValueError("Need nonempty train/test and a complete binary TARGET")
    for name, frame in [("train", train), ("test", test)]:
        if frame[ID].isna().any() or frame[ID].duplicated().any():
            raise ValueError(f"{name}: expected one application per SK_ID_CURR")
    if set(train[ID]) & set(test[ID]):
        raise ValueError("Train/test application IDs overlap")
    metadata = {"BORROWER_ID", "CUSTOMER_ID", "APPLICATION_DATE", "APPLICATION_TIMESTAMP"}
    if metadata & set(train.columns):
        raise ValueError("Additional group/time metadata detected. Separate it from features and use "
                         "nested_evaluate(..., groups=..., times=..., config=Config(split_mode=...)).")
    return train, test


def build_application_features(directory):
    """Explicit partial experiment; never invents a history representation."""
    check_schema(directory, ("application_train", "application_test"))
    train, test = _read_applications(directory)
    def views(frame):
        return {"raw": frame.drop(columns=[ID, TARGET], errors="ignore").copy(),
                "application": application_features(frame)}
    audit = dict(train_rows=len(train), test_rows=len(test), history_available=False,
                 unique_application_ids=True, absolute_application_time_available=False,
                 split_assumption="IID applications; no application timestamps available")
    return views(train), train[TARGET].to_numpy(), views(test), test[ID].to_numpy(), audit


def build_features(directory, chunksize=250_000):
    if not isinstance(chunksize, int) or isinstance(chunksize, bool) or chunksize < 1:
        raise ValueError("chunksize must be a positive integer")
    check_schema(directory)
    train, test = _read_applications(directory)
    # Joining deterministic per-owner aggregates is safe across folds: no pooling
    # across owners, no target, no fitted categorical vocabulary.
    history, audit = historical_features(directory, set(train[ID]) | set(test[ID]), chunksize)
    def bundle(frame):
        raw = frame.drop(columns=[ID, TARGET], errors="ignore").copy()
        app = application_features(frame)
        aligned_history = history.reindex(frame[ID]).set_axis(app.index)
        engineered = pd.concat([app, aligned_history], axis=1)
        for col, base, name in [
            ("BURO_AMT_CREDIT_SUM_DEBT_SUM", "AMT_INCOME_TOTAL", "X_BURO_DEBT_INCOME"),
            ("INS_AMT_PAYMENT_MEAN", "AMT_INCOME_TOTAL", "X_INS_PAY_INCOME"),
            ("PREV_AMT_CREDIT_MEAN", "AMT_CREDIT", "X_PREV_CREDIT")]:
            if col in engineered:
                engineered[name] = engineered[col] / (app[base] + EPS)
        return {"raw": raw, "application": app,
                "history": engineered.replace([np.inf, -np.inf], np.nan)}
    audit.update(train_rows=len(train), test_rows=len(test), chunksize=chunksize,
                 unique_application_ids=True, absolute_application_time_available=False,
                 borrower_identity="SK_ID_CURR only; distinct applications cannot be linked to a hidden person",
                 split_assumption="IID applications; relative history dates are not application timestamps")
    return bundle(train), train[TARGET].to_numpy(), bundle(test), test[ID].to_numpy(), audit
