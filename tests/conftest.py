import numpy as np
import pandas as pd
import pytest

from credit_scoring.protocol import Config


@pytest.fixture
def config():
    return Config(outer_folds=2, inner_folds=3, max_rounds=16, patience=4,
                  probe_rounds=8, top_k=12, complexities=(7, 15), threads=1)


@pytest.fixture
def frames():
    rng = np.random.default_rng(42)
    n = 360
    raw = pd.DataFrame({"amount": rng.normal(size=n),
                        "type": rng.choice(["a", "b", None], n),
                        "missing": rng.normal(size=n)})
    raw.loc[raw.index[::5], "missing"] = np.nan
    y = (raw.amount + rng.normal(size=n) > .4).astype(np.int8).to_numpy()
    app = raw.assign(ratio=raw.amount ** 2)
    history = app.assign(history=rng.normal(size=n) + y * .2)
    return {"raw": raw, "application": app, "history": history}, y


@pytest.fixture
def data_dir(tmp_path, request):
    n = getattr(request, "param", 12)
    app = pd.DataFrame({"SK_ID_CURR": range(100, 100+n), "TARGET": [0, 1] * (n // 2),
        "CODE_GENDER": ["XNA", "F", "M"] * (n // 3),
        "DAYS_EMPLOYED": [-50, 365243] * (n // 2), "DAYS_BIRTH": [-10000] * n,
        "DAYS_REGISTRATION": [-100] * n, "DAYS_ID_PUBLISH": [-500] * n,
        "AMT_CREDIT": np.arange(n) + 100, "AMT_INCOME_TOTAL": [200] * n,
        "AMT_ANNUITY": [10] * n, "AMT_GOODS_PRICE": [90] * n,
        "CNT_FAM_MEMBERS": [2] * n, "EXT_SOURCE_1": [.3] * n,
        "EXT_SOURCE_2": [.5] * n, "EXT_SOURCE_3": [np.nan] * n})
    app.iloc[:-2].to_csv(tmp_path / "application_train.csv", index=False)
    app.iloc[-2:].drop(columns="TARGET").to_csv(tmp_path / "application_test.csv", index=False)
    ids = np.repeat(app.SK_ID_CURR, 2).to_numpy()
    prev = np.arange(len(ids)) + 1000
    bureau = np.arange(len(ids)) + 2000
    def save(name, fields):
        pd.DataFrame(fields).to_csv(tmp_path / f"{name}.csv", index=False)
    save("bureau", dict(SK_ID_CURR=ids, SK_ID_BUREAU=bureau, DAYS_CREDIT=-100,
        DAYS_CREDIT_ENDDATE=50, DAYS_ENDDATE_FACT=-1, AMT_CREDIT_SUM_DEBT=20,
        AMT_CREDIT_SUM=50, CREDIT_DAY_OVERDUE=0, CREDIT_ACTIVE="Active"))
    save("bureau_balance", dict(SK_ID_BUREAU=bureau, MONTHS_BALANCE=-2, STATUS="1"))
    save("previous_application", dict(SK_ID_CURR=ids, SK_ID_PREV=prev,
        DAYS_DECISION=-20, AMT_APPLICATION=90, AMT_CREDIT=100,
        AMT_ANNUITY=10, NAME_CONTRACT_STATUS="Approved"))
    save("installments_payments", dict(SK_ID_CURR=ids, SK_ID_PREV=prev,
        NUM_INSTALMENT_VERSION=1, NUM_INSTALMENT_NUMBER=1,
        DAYS_ENTRY_PAYMENT=-5, DAYS_INSTALMENT=-10, AMT_PAYMENT=np.arange(len(ids)) + 1,
        AMT_INSTALMENT=10))
    save("POS_CASH_balance", dict(SK_ID_CURR=ids, SK_ID_PREV=prev, MONTHS_BALANCE=-1, SK_DPD=0))
    save("credit_card_balance", dict(SK_ID_CURR=ids, SK_ID_PREV=prev,
        MONTHS_BALANCE=-1, SK_DPD=0, AMT_BALANCE=20, AMT_CREDIT_LIMIT_ACTUAL=100))
    return tmp_path
