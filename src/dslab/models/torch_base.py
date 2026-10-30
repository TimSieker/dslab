"""Shared machinery for neural regression models trained with PyTorch.

Targets are transformed (pr: sqrt by default) and standardised per variable and grid point with
training statistics; the network predicts that standardised field and is trained with MSE. The whole
training split lives on the GPU (ALPS ESD: ~1 GB). Early stopping on the validation loss; the best
epoch's weights are kept. Subclasses implement build_network() and, if needed, set_geometry().
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..data import Grid, Split
from ..preprocessing import PredictorScaler, TargetTransform
from .base import Downscaler

TRAIN_DEFAULTS = dict(transform={"pr": "sqrt"}, lr=1e-3, weight_decay=0.0, batch_size=32, max_epochs=100,
                      patience=10, lr_patience=4, lr_factor=0.5, min_lr=1e-6, amp=True, seed=0,
                      predict_batch=128)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TorchRegressor(Downscaler):
    arch_defaults: dict = {}

    def __init__(self, variables, predictors, **params):
        p = {**TRAIN_DEFAULTS, **self.arch_defaults, **params}
        super().__init__(variables, predictors, **p)
        self.tf = TargetTransform(self.variables, p["transform"])
        self.scaler = None
        self.y_mean = self.y_std = None  # (V, H, W), transformed space
        self.net: nn.Module | None = None

    # ---- to implement
    def build_network(self, n_in: int, in_hw, n_out: int, out_hw) -> nn.Module:
        raise NotImplementedError

    def set_geometry(self, net: nn.Module, grid: Grid) -> None:
        """Copy grid-dependent buffers (e.g. interpolation weights, orography) into the network."""

    # ---- helpers
    def _standardise_y(self, y: np.ndarray) -> np.ndarray:
        return ((self.tf.forward(y) - self.y_mean) / self.y_std).astype(np.float32)

    def _autocast(self, dev):
        return torch.autocast(dev.type, dtype=torch.bfloat16, enabled=bool(self.params["amp"]) and dev.type == "cuda")

    def _loss(self, net, xb, yb, dev):
        with self._autocast(dev):
            pred = net(xb)
        return torch.mean((pred.float() - yb) ** 2)

    def fit(self, train: Split, val: Split, grid: Grid) -> dict:
        p = self.params
        torch.manual_seed(p["seed"])
        np.random.seed(p["seed"])
        dev = device()
        self.scaler = PredictorScaler().fit(train.x)
        ztr = self.tf.forward(train.y)
        self.y_mean = ztr.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.y_std = np.maximum(ztr.std(axis=0, dtype=np.float64), 1e-2).astype(np.float32)
        del ztr
        X = torch.from_numpy(self.scaler.transform(train.x)).to(dev)
        Y = torch.from_numpy(self._standardise_y(train.y)).to(dev)
        Xv = torch.from_numpy(self.scaler.transform(val.x)).to(dev)
        Yv = torch.from_numpy(self._standardise_y(val.y)).to(dev)

        net = self.build_network(X.shape[1], tuple(X.shape[2:]), Y.shape[1], tuple(Y.shape[2:]))
        self.set_geometry(net, grid)
        net = net.to(dev)
        n_params = sum(q.numel() for q in net.parameters())
        print(f"[fit] {type(net).__name__}: {n_params / 1e6:.2f} M parameters on {dev}", flush=True)
        opt = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=p["lr_factor"], patience=p["lr_patience"],
                                                           min_lr=p["min_lr"])
        bs = p["batch_size"]
        best, best_state, best_epoch, wait, history = np.inf, None, -1, 0, []
        t0 = time.time()
        for epoch in range(p["max_epochs"]):
            net.train()
            perm = torch.randperm(len(X), device=dev)
            tot = 0.0
            for i in range(0, len(X) - bs + 1, bs):  # drop the last incomplete batch
                idx = perm[i:i + bs]
                loss = self._loss(net, X[idx], Y[idx], dev)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += loss.item() * bs
            train_loss = tot / (len(X) // bs * bs)
            val_loss = self._evaluate_loss(net, Xv, Yv, dev)
            sched.step(val_loss)
            lr = opt.param_groups[0]["lr"]
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr,
                            "seconds": round(time.time() - t0, 1)})
            improved = val_loss < best - 1e-5
            if improved:
                best, best_epoch, wait = val_loss, epoch, 0
                best_state = copy.deepcopy(net.state_dict())
            else:
                wait += 1
            if epoch % 5 == 0 or improved or wait >= p["patience"]:
                print(f"[fit] epoch {epoch:4d}  train {train_loss:.4f}  val {val_loss:.4f}  lr {lr:.1e}"
                      f"{'  *' if improved else ''}  ({time.time() - t0:.0f} s)", flush=True)
            if wait >= p["patience"]:
                break
        net.load_state_dict(best_state)
        self.net = net
        info = {"n_params": n_params, "device": torch.cuda.get_device_name() if dev.type == "cuda" else "cpu",
                "epochs": len(history), "best_epoch": best_epoch, "best_val_loss": best,
                "train_seconds": round(time.time() - t0, 1), "history": history}
        pred = self.predict(val, grid)[0]
        info["selected"] = {v: {"best_epoch": best_epoch,
                                "val_rmse": float(np.sqrt(np.mean((pred[:, i] - val.y[:, i]) ** 2)))}
                            for i, v in enumerate(self.variables)}
        return info

    @torch.no_grad()
    def _evaluate_loss(self, net, Xv, Yv, dev) -> float:
        net.eval()
        bs, tot = self.params["predict_batch"], 0.0
        for i in range(0, len(Xv), bs):
            tot += self._loss(net, Xv[i:i + bs], Yv[i:i + bs], dev).item() * len(Xv[i:i + bs])
        return tot / len(Xv)

    @torch.no_grad()
    def predict(self, split: Split, grid: Grid, n_members: int = 1, seed: int = 0) -> np.ndarray:
        dev = device()
        net = self.net.to(dev).eval()
        X = torch.from_numpy(self.scaler.transform(split.x))
        out = np.empty((1, len(split), len(self.variables), *grid.shape), np.float32)
        mean, std = torch.from_numpy(self.y_mean).to(dev), torch.from_numpy(self.y_std).to(dev)
        bs = self.params["predict_batch"]
        for i in range(0, len(X), bs):
            with self._autocast(dev):
                z = net(X[i:i + bs].to(dev))
            z = (z.float() * std + mean).cpu().numpy()
            out[0, i:i + bs] = self.tf.inverse(z)
        return out

    # ---- persistence: model weights + normalisation in one torch file
    def save_state(self, run_dir: Path) -> None:
        torch.save({"net": self.net.state_dict(),
                    "shapes": {"n_in": len(self.predictors), "in_hw": list(self.scaler.mean.shape[1:]),
                               "n_out": len(self.variables), "out_hw": list(self.y_mean.shape[1:])},
                    "x_mean": torch.from_numpy(self.scaler.mean), "x_std": torch.from_numpy(self.scaler.std),
                    "y_mean": torch.from_numpy(self.y_mean), "y_std": torch.from_numpy(self.y_std)},
                   run_dir / "state.pt")

    def load_state(self, run_dir: Path) -> None:
        st = torch.load(run_dir / "state.pt", map_location="cpu", weights_only=True)
        s = st["shapes"]
        self.net = self.build_network(s["n_in"], tuple(s["in_hw"]), s["n_out"], tuple(s["out_hw"]))
        self.net.load_state_dict(st["net"])
        self.scaler = PredictorScaler(st["x_mean"].numpy(), st["x_std"].numpy())
        self.y_mean, self.y_std = st["y_mean"].numpy(), st["y_std"].numpy()
