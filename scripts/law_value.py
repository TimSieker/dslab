"""Criterion (c) of the law tests: does conditioning on a law improve the base's mean beyond the encoder?

For the trained `afm` run, the encoder estimate E of pr and tasmax is conditioned on each law with the prior
N(E, sigma_max^2) (dslab.models.lcafm.LawConditioning; for the pointwise laws m = E + w (b - E),
w = s^2 / (s^2 + tau^2)), and the conditioned mean m is scored against the truth like E: RMSE, bias and June-August
bias, on the validation years and every `step`-th day of 1981-2000 and 2080-2099. Laws: T, Tfull, O (LC-AFM laws)
and the pointwise precipitation candidates U_lin, U_pos, M (oracle: their wind and humidity come from the RCM).
Needs the fits written by `dslab laws --candidates` (laws/candidate_fits.npz).

    python scripts/law_value.py [--step 10]      # -> work/eval/alps_esd/laws/law_value.csv
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import xarray as xr

from dslab import candidate_laws as CL
from dslab.data import load_grid, load_split
from dslab.experiment import Experiment
from dslab.laws import LAWS, CoarseOperator, fit_cached, months, phi_cached
from dslab.models import Downscaler
from dslab.models.lcafm import LawConditioning

POINTWISE = ("U_lin", "U_pos", "M")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="alps_esd")
    ap.add_argument("--step", type=int, default=10)
    args = ap.parse_args()
    exp = Experiment.load(args.exp)
    grid = load_grid(exp)
    op = CoarseOperator.from_grid(grid)
    t0 = time.time()
    law = fit_cached(load_split(exp, "train"), grid)
    out = exp.eval_dir / "laws"
    with xr.open_dataset(out / "candidates.nc") as ds:
        info = json.loads(ds.attrs["law_info"])
    fits = {}
    with np.load(out / "candidate_fits.npz") as f:
        for n in POINTWISE:
            fits[n] = {"coef": info[n]["coef"], "offset": f[f"{n}__offset"], "tol": f[f"{n}__tol"],
                       "clim": f[f"{n}__clim"], "beta": f[f"{n}__beta"] if f"{n}__beta" in f.files else None}

    afm = Downscaler.load(exp.runs_dir / "afm")
    afm.params["micro_batch"] = 16
    variables = list(exp.variables)
    smax = afm.net.sigma_max.numpy() * afm.y_std.ravel()  # prior noise in physical units
    cbs = {}
    for n, (var, hp) in LAWS.items():
        c = variables.index(var)
        kinds, tol = [None] * len(variables), [None] * len(variables)
        kinds[c], tol[c] = ("highpass" if hp else "full"), law.tol[n]
        cbs[n] = LawConditioning(op, kinds, tol)
    gen = torch.Generator().manual_seed(0)
    needs = set().union(*(CL.LAWS[n]["needs"] for n in POINTWISE)) | {"tasmax"}

    rows = []
    for split, step in [("val", 1), ("historical_perfect", args.step), ("end_century_perfect", args.step)]:
        full = load_split(exp, split)
        idx = np.arange(0, len(full), step)
        s = full.subset(idx)
        mon = months(s.time)
        jja = np.isin(mon, [5, 6, 7])
        E = afm._encoder_prediction(s, grid)
        fields = {"encoder E": E}
        b = {n: v[idx] for n, v in phi_cached(law, full, grid).items()}
        for n, cb in cbs.items():  # LC-AFM laws: exact posterior mean (Woodbury), as in the model
            ms = []
            for i in range(0, len(s), 64):
                bb = np.zeros_like(E[i:i + 64])
                bb[:, variables.index(LAWS[n][0])] = b[n][i:i + 64]
                m, _ = cb(torch.from_numpy(E[i:i + 64]), torch.from_numpy(bb), torch.from_numpy(mon[i:i + 64]).long(),
                          torch.from_numpy(smax), gen)
                ms.append(m.numpy())
            fields[f"conditioned on {n}"] = np.concatenate(ms)
        ctx = CL._Ctx(exp, s, grid, op, needs)  # pointwise precipitation candidates: m = E + w (b - E)
        c = variables.index("pr")
        for n in POINTWISE:
            _, rhs = CL._rhs(n, fits[n], ctx, slice(None))
            tau = fits[n]["tol"][mon]
            w = smax[c] ** 2 / (smax[c] ** 2 + tau**2)
            m = E.copy()
            m[:, c] = E[:, c] + np.where(np.isfinite(rhs), w * (rhs - E[:, c]), 0.0)
            fields[f"conditioned on {n}"] = m
        for name, f in fields.items():
            for c, v in enumerate(variables):
                err = f[:, c].astype(np.float64) - s.y[:, c]
                rows.append({"split": split, "field": name, "variable": v, "RMSE": float(np.sqrt((err**2).mean())),
                             "bias": float(err.mean()), "JJA bias": float(err[jja].mean())})
        print(f"[value] {split}: {len(s)} days ({time.time() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "law_value.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.pivot_table(index=["variable", "field"], columns="split", values=["RMSE", "JJA bias"], sort=False)
          .round(3).to_string())


if __name__ == "__main__":
    main()
