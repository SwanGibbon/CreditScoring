"""Explicitly synthetic smoke data; no estimate of real Home Credit quality."""
import numpy as np
import pandas as pd


def synthetic_data(seed=42, n=480):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    raw = pd.DataFrame({"income": rng.lognormal(4, .5, n),
                        "credit": rng.lognormal(5, .5, n),
                        "category": rng.choice(["A", "B", None], n),
                        "ext": latent + rng.normal(size=n)})
    raw.loc[raw.index[::7], "income"] = np.nan
    app = raw.assign(ratio=raw.credit / (raw.income + 1e-6))
    history = app.assign(past_delinquency=latent + rng.normal(size=n))
    # Target and observed features share a latent factor; no target-derived feature.
    probability = 1 / (1 + np.exp(-(.7 * latent - 1)))
    y = rng.binomial(1, probability)
    frames = dict(raw=raw, application=app, history=history)
    return frames, y
