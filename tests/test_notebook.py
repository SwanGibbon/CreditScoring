"""Execute the optional notebook export branch, which default smoke skips."""
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

from credit_scoring.evaluation import fit_policies


def test_notebook_optional_submission_writes_readable_json(config, frames, tmp_path):
    notebook = nbformat.read(Path(__file__).resolve().parents[1] / "main.ipynb", as_version=4)
    cell = next(c for c in notebook.cells if c.cell_type == "code" and "submission.to_csv" in c.source)
    x, y = frames
    namespace = dict(RUN_SUBMISSION=True, RUN_REAL=True, frames=x, y=y, CONFIG=config,
                     OUTPUT=tmp_path, fit_policies=fit_policies, pd=pd, json=json,
                     experiment={"policies": fit_policies(x, y, config)},
                     test_ids=np.arange(5), test_frames={k: v.iloc[:5] for k, v in x.items()})
    exec(compile(cell.source, "main.ipynb:submission", "exec"), namespace)
    assert namespace["final_policy"] is namespace["experiment"]["policies"]["selected"]
    assert pd.read_csv(tmp_path / "submission.csv").SK_ID_CURR.tolist() == list(range(5))
