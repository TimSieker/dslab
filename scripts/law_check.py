"""Go/no-go check of the LC-AFM laws (plan step 2), on CPU, before any training.

Fits the laws on the training truth (cached), then scores, on val and (every `step`-th day of) the historical
and end-of-century tests: the RCM truth, the U-Net, one AFM member, the AFM encoder mean E, the white base
E + sigma_max eps, and the bases conditioned on each law (built from the trained `afm` encoder; an LC-AFM run
trains its own). Per field: each law's mean squared residual in tolerance units (truth = 1 on training) and the
bias / RMSE of the law's variable. A law is worth conditioning on if the white base scores well above 1.
Also warms the law caches for train / val (the GPU job then does not compute them).

    python scripts/law_check.py [--exp alps_esd] [--step 10]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import xarray as xr

from dslab.data import load_grid, load_split
from dslab.experiment import Experiment
from dslab.laws import LAWS, CoarseOperator, fit_cached, months, phi_cached
from dslab.models import Downscaler
from dslab.models.lcafm import LawConditioning


def member0(exp, run, split, idx, variables):
    with xr.open_dataset(exp.predictions_dir / run / f"{split}.nc") as ds:
        return np.stack([ds[v].isel(member=0, time=idx).transpose("time", ...).values for v in variables], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="alps_esd")
    ap.add_argument("--step", type=int, default=10, help="every step-th day of the 20-year test splits")
    args = ap.parse_args()
    torch.set_num_threads(max(torch.get_num_threads(), 1))
    exp = Experiment.load(args.exp)
    grid = load_grid(exp)
    t0 = time.time()
    train = load_split(exp, "train")
    law = fit_cached(train, grid)
    print(f"[laws] fitted ({time.time() - t0:.0f} s): {json.dumps(law.info, indent=1)}", flush=True)
    for sp in ("train", "val"):
        phi_cached(law, load_split(exp, sp), grid)
    print(f"[laws] train/val right-hand sides cached ({time.time() - t0:.0f} s)", flush=True)
    del train

    afm = Downscaler.load(exp.runs_dir / "afm")
    afm.params["micro_batch"] = 16
    variables = list(exp.variables)
    smax = afm.net.sigma_max.numpy() * afm.y_std.ravel()  # physical units
    op = CoarseOperator.from_grid(grid)
    cbs = {}
    for n, (var, hp) in LAWS.items():
        c = variables.index(var)
        kinds, tol = [None] * len(variables), [None] * len(variables)
        kinds[c], tol[c] = ("highpass" if hp else "full"), law.tol[n]
        cbs[n] = LawConditioning(op, kinds, tol)
    gen = torch.Generator().manual_seed(0)

    rows = []
    for split, step in [("val", 1), ("historical_perfect", args.step), ("end_century_perfect", args.step)]:
        full = load_split(exp, split)
        idx = np.arange(0, len(full), step)
        s = full.subset(idx)
        b = {n: v[idx] for n, v in phi_cached(law, full, grid).items()}
        mon = months(s.time)
        fields = {"truth": s.y, "unet": member0(exp, "unet", split, idx, variables),
                  "afm sample": member0(exp, "afm", split, idx, variables)}
        E = afm._encoder_prediction(s, grid)
        fields["afm encoder E"] = E
        fields["white base E + σε"] = E + smax[None, :, None, None] * np.random.default_rng(0).normal(size=E.shape)
        for n, cb in cbs.items():
            ms, xs = [], []
            for i in range(0, len(s), 64):
                bb = np.zeros_like(E[i:i + 64])
                bb[:, variables.index(LAWS[n][0])] = b[n][i:i + 64]
                m, eps = cb(torch.from_numpy(E[i:i + 64]), torch.from_numpy(bb), torch.from_numpy(mon[i:i + 64]).long(),
                            torch.from_numpy(smax), gen)
                ms.append(m.numpy())
                xs.append((m + torch.from_numpy(smax)[None, :, None, None] * eps).numpy())
            fields[f"mean | {n}"] = np.concatenate(ms)
            fields[f"base | {n}"] = np.concatenate(xs)
        jja = np.isin(pd.DatetimeIndex(s.time).month, [6, 7, 8])
        for name, f in fields.items():
            row = {"split": split, "field": name}
            for n in LAWS:
                c = variables.index(LAWS[n][0])
                row[f"law {n}"] = law.residual(n, f[:, c], b[n], mon, grid)
            for c, v in enumerate(variables):
                err = f[:, c].astype(np.float64) - s.y[:, c]
                row[f"{v} bias"] = float(err.mean())
                row[f"{v} RMSE"] = float(np.sqrt((err**2).mean()))
                row[f"{v} JJA bias"] = float(err[jja].mean())
            rows.append(row)
        print(f"[check] {split}: {len(s)} days ({time.time() - t0:.0f} s)", flush=True)

    df = pd.DataFrame(rows)
    out = exp.eval_dir / "laws"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "law_check.csv", index=False)
    (out / "law_info.json").write_text(json.dumps(law.info, indent=1))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for split, d in df.groupby("split", sort=False):
        print(f"\n== {split}\n" + d.drop(columns="split").set_index("field").round(3).to_string())
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
