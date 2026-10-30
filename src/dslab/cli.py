"""Command-line entry point.

    dslab prepare  --exp alps_esd                         # build the .npy caches, print a summary
    dslab fit      --exp alps_esd --model linear          # -> work/runs/alps_esd/linear/
    dslab predict  --exp alps_esd --run linear            # -> work/predictions/alps_esd/linear/<split>.nc
    dslab evaluate --exp alps_esd --run linear            # -> work/eval/alps_esd/linear/<split>/

--model takes a name in configs/models/ or a YAML path; --set overrides params (--set window=15).
Splits default to the experiment's default_splits.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from .data import load_grid, load_split
from .experiment import CONFIG_DIR, REPO_ROOT, Experiment


def git_commit() -> str:
    head = REPO_ROOT / ".git" / "HEAD"
    try:
        ref = head.read_text().strip()
        if ref.startswith("ref: "):
            ref = (REPO_ROOT / ".git" / ref[5:]).read_text().strip()
        return ref
    except OSError:
        return "unknown"


def provenance() -> dict:
    return {"time": datetime.now().isoformat(timespec="seconds"), "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "git_commit": git_commit()}


def _model_cfg(model: str, overrides: list[str]) -> dict:
    path = Path(model) if model.endswith((".yaml", ".yml")) else CONFIG_DIR / "models" / f"{model}.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg.setdefault("params", {})
    for item in overrides:
        key, value = item.split("=", 1)
        cfg["params"][key] = yaml.safe_load(value)
    return cfg


def cmd_prepare(args):
    exp = Experiment.load(args.exp)
    grid = load_grid(exp)
    print(f"{exp.name}: target grid {grid.shape} dims {grid.dims}, predictor grid "
          f"{len(grid.pred_lat)}x{len(grid.pred_lon)}, orog {grid.orog.min():.0f}..{grid.orog.max():.0f} m")
    for split in args.splits or exp.splits:
        t0 = time.time()
        s = load_split(exp, split)
        rng = f"{str(s.time[0])[:10]}..{str(s.time[-1])[:10]}"
        msg = f"  {split:<24} {len(s):>6} days {rng}  x{tuple(s.x.shape[1:])}"
        if s.y is not None:
            msg += "  " + "  ".join(f"{v} mean {s.y[:, i].mean():.2f}" for i, v in enumerate(s.variables))
        print(msg + f"  ({time.time() - t0:.1f} s)", flush=True)


def cmd_fit(args):
    from .models import build_model

    exp = Experiment.load(args.exp)
    cfg = _model_cfg(args.model, args.set or [])
    run_dir = exp.runs_dir / (args.name or cfg["model"])
    if (run_dir / "model.yaml").exists() and not args.overwrite:
        raise SystemExit(f"{run_dir} exists; use --overwrite or --name")
    grid = load_grid(exp)
    train, val = load_split(exp, "train"), load_split(exp, "val")
    model = build_model(cfg["model"], exp.variables, exp.predictors, **cfg["params"])
    print(f"[fit] {exp.name} / {run_dir.name}: {cfg['model']} {cfg['params']}", flush=True)
    t0 = time.time()
    info = model.fit(train, val, grid)
    info["wall_seconds"] = round(time.time() - t0, 1)
    model.save(run_dir)
    run = {"experiment": exp.name, "run": run_dir.name, "config": cfg, **provenance()}
    (run_dir / "run.yaml").write_text(yaml.safe_dump(run, sort_keys=False))
    (run_dir / "fit_info.json").write_text(json.dumps(info, indent=1))
    print(f"[fit] done in {info['wall_seconds']} s -> {run_dir}", flush=True)
    if "selected" in info:
        print(f"[fit] selected: {info['selected']}")


def cmd_predict(args):
    from .io import write_prediction
    from .models import Downscaler

    exp = Experiment.load(args.exp)
    run_dir = exp.runs_dir / args.run
    model = Downscaler.load(run_dir)
    grid = load_grid(exp)
    members = args.members or model.params.get("n_members", 10 if model.stochastic else 1)
    for split in args.splits or exp.default_splits:
        t0 = time.time()
        s = load_split(exp, split)
        pred = model.predict(s, grid, n_members=members, seed=args.seed)
        out = exp.predictions_dir / args.run / f"{split}.nc"
        attrs = {"experiment": exp.name, "run": args.run, "model": model.name, "split": split,
                 "n_members": pred.shape[0], **{k: str(v) for k, v in provenance().items()}}
        write_prediction(out, pred, s, grid, attrs)
        print(f"[predict] {args.run} {split}: {pred.shape} in {time.time() - t0:.1f} s -> {out}", flush=True)


def cmd_evaluate(args):
    from .evaluate import evaluate_run

    exp = Experiment.load(args.exp)
    runs = args.run or sorted(p.name for p in exp.predictions_dir.iterdir() if p.is_dir())
    for run in runs:
        for split in args.splits or exp.default_eval_splits:
            evaluate_run(exp, run, split, overwrite=args.overwrite)


def main(argv=None):
    p = argparse.ArgumentParser(prog="dslab", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prepare", help="build data caches")
    sp.add_argument("--exp", required=True)
    sp.add_argument("--splits", nargs="*")
    sp.set_defaults(func=cmd_prepare)

    sp = sub.add_parser("fit", help="fit a model")
    sp.add_argument("--exp", required=True)
    sp.add_argument("--model", required=True)
    sp.add_argument("--name", help="run name (default: model name)")
    sp.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="override model params")
    sp.add_argument("--overwrite", action="store_true")
    sp.set_defaults(func=cmd_fit)

    sp = sub.add_parser("predict", help="write predictions for splits")
    sp.add_argument("--exp", required=True)
    sp.add_argument("--run", required=True)
    sp.add_argument("--splits", nargs="*")
    sp.add_argument("--members", type=int, help="ensemble size (default: the model's n_members, 1 if deterministic)")
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_predict)

    sp = sub.add_parser("evaluate", help="compute metrics for predictions")
    sp.add_argument("--exp", required=True)
    sp.add_argument("--run", nargs="*", help="runs (default: all with predictions)")
    sp.add_argument("--splits", nargs="*")
    sp.add_argument("--overwrite", action="store_true")
    sp.set_defaults(func=cmd_evaluate)

    args = p.parse_args(argv)
    np.set_printoptions(precision=4, suppress=True)
    args.func(args)


if __name__ == "__main__":
    main()
