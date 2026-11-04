"""Adaptive flow matching (AFM; Fotiadis et al. 2025, ICML) as specified in Debeire et al. (2026,
"Physics-Constrained Adaptive Flow Matching for Climate Downscaling", arXiv:2604.03459, Sect. 2.2, 3.1, 3.4):
their "AFM-baseline", i.e. *without* the physics-constraint (conservation) loss and gradient surgery.

Paper -> code
  encoder E (Sect. 3.1): SongUNet-L (channel_mult [1, 2, 2, 4, 4]; 32 base channels, ~12 M parameters in
      Fotiadis et al.), no noise embedding, no positional embedding; input = coarse predictors bilinearly
      interpolated to the target grid + static fields; output = first high-resolution estimate.
  denoiser D: same SongUNet, Fourier noise embedding, input = [x_t, interpolated predictors + statics].
  interpolant (Eq. 2): x_t = (1 - t) E + t x1 + sigma_t eps, sigma_t = (1 - t) sigma_max, sigma_t ~ U[0, sigma_max].
  sigma_max: per channel, initialised at 1.0, updated as an EMA of the encoder's RMS error, the
      maximum-likelihood noise estimate of Fotiadis et al. (their "adaptive" noise scaling).
  loss (Eq. 3): L = E[w(sigma) ||D - x1||^2] + 0.25 ||E - x1||^2, w = (sigma^2 + sd^2) / (sigma sd)^2, sd = 0.5;
      gradients of both terms reach the encoder (Algorithm 1 of Fotiadis et al.).
  denoiser parameterisation: EDM preconditioning (Karras et al. 2022) per channel.
  training (Sect. 3.4): Adam lr 2e-4, betas (0.9, 0.999), batch 256, EMA of the denoiser weights with a
      half-life of 5e5 samples; checkpoint selection on the validation split.
  sampling (Eq. 4): Karras schedule (rho 7, sigma_min 0.002), N = 50 Euler steps from E + sigma_max eps;
      12 ensemble members.
Choices the papers leave open are marked "choice:" in the code.
"""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..data import Grid, Split
from ..preprocessing import PredictorScaler, TargetTransform
from .base import Downscaler, register
from .networks import sample_grid
from .torch_base import device

SIGMA_DATA = 0.5


def _songunet(img_hw, cin, cout, embedding, kw, amp):
    from physicsnemo.models.diffusion_unets import SongUNet

    return SongUNet(img_resolution=list(img_hw), in_channels=cin, out_channels=cout, embedding_type=embedding,
                    model_channels=kw["model_channels"], channel_mult=list(kw["channel_mult"]),
                    num_blocks=kw["num_blocks"], attn_resolutions=list(kw["attn_resolutions"]),
                    dropout=kw["dropout"], amp_mode=amp)


class AFMNet(nn.Module):
    """Encoder + denoiser + the fixed geometry (interpolation grid, static fields) and sigma_max."""

    def __init__(self, n_pred: int, n_out: int, out_hw, n_static: int, net_kw: dict, amp: bool):
        super().__init__()
        H, W = out_hw
        self.register_buffer("grid", torch.zeros(1, H, W, 2))
        self.register_buffer("static", torch.zeros(1, n_static, H, W))
        self.register_buffer("sigma_max", torch.ones(n_out))
        n_cond = n_pred + n_static
        self.encoder = _songunet(out_hw, n_cond, n_out, "zero", net_kw, amp)
        self.denoiser = _songunet(out_hw, n_out + n_cond, n_out, "fourier", net_kw, amp)
        self._fn = {}  # torch.compile'd encoder/denoiser (kept out of the state dict)
        self._mf = torch.contiguous_format

    def use_compiled(self, channels_last: bool = False) -> None:
        """Run both networks through torch.compile; channels-last layout is faster for sampling,
        slightly slower for training (scripts/bench_afm.py). Weights and results are unchanged."""
        mf = torch.channels_last if channels_last else torch.contiguous_format
        if self._fn and mf == self._mf:
            return
        self.encoder.to(memory_format=mf)
        self.denoiser.to(memory_format=mf)
        # static shapes: sampling pads its last batch, so each member count compiles once
        self._fn = {"encoder": torch.compile(self.encoder, dynamic=False),
                    "denoiser": torch.compile(self.denoiser, dynamic=False)}
        self._mf = mf

    def cond(self, x):
        """Coarse predictors (B, C, h, w) -> interpolated to the target grid + static fields."""
        up = nn.functional.grid_sample(x, self.grid.expand(len(x), -1, -1, -1), mode="bilinear",
                                       align_corners=True)
        return torch.cat([up, self.static.expand(len(x), -1, -1, -1)], dim=1)

    def encode(self, cond):
        fn = self._fn.get("encoder", self.encoder)
        return fn(cond.contiguous(memory_format=self._mf), torch.zeros(len(cond), device=cond.device)).float()

    def precond(self, s):
        """EDM coefficients for relative noise level s = sigma / sigma_max (B,), per channel."""
        sigma = s[:, None, None, None] * self.sigma_max[None, :, None, None]
        norm = torch.sqrt(sigma**2 + SIGMA_DATA**2)
        # choice: one scalar noise label per sample (the channels share t); EDM's c_noise = ln(sigma) / 4
        # with sigma = s * RMS over channels of sigma_max
        c_noise = torch.log(s.clamp_min(1e-4) * self.sigma_max.pow(2).mean().sqrt()) / 4
        return sigma, SIGMA_DATA**2 / norm**2, sigma * SIGMA_DATA / norm, 1 / norm, c_noise

    def denoise_raw(self, x_t, s, cond, net=None):
        """Raw network output F and the preconditioning coefficients."""
        sigma, c_skip, c_out, c_in, c_noise = self.precond(s)
        inp = torch.cat([c_in * x_t, cond], dim=1).contiguous(memory_format=self._mf)
        F = (net or self._fn.get("denoiser", self.denoiser))(inp, c_noise).float()
        return F, sigma, c_skip, c_out

    def denoise(self, x_t, s, cond, net=None):
        F, _, c_skip, c_out = self.denoise_raw(x_t, s, cond, net)
        return c_skip * x_t + c_out * F


@register("afm")
class AFM(Downscaler):
    summary = "adaptive flow matching (Debeire et al. 2026 without conservation loss): SongUNet encoder + flow-matching denoiser"
    stochastic = True
    defaults = dict(
        # data (Sect. 2.2: z-score per variable with training statistics, no transform)
        transform={}, normalize="variable", statics=["orog", "lat"],
        # networks: SongUNet-L of Fotiadis et al. (attention at the lowest resolution, dropout 0.13)
        model_channels=32, channel_mult=[1, 2, 2, 4, 4], num_blocks=4, attn_resolutions=[8], dropout=0.13,
        # objective
        lambda_enc=0.25, sigma_max_init=1.0, sigma_max_beta=0.01,
        # optimisation (Sect. 3.4); train_samples: 6e6 in the paper, 2e6 in its ablation
        lr=2e-4, betas=[0.9, 0.999], batch_size=256, micro_batch=256, train_samples=2_000_000,
        ema_halflife=500_000, ema_rampup=0.05, eval_every=100_000, amp=True, compile=True, seed=0,
        # sampling (Eq. 4); members_per_split overrides n_members for single splits
        n_members=12, members_per_split={}, n_steps=50, rho=7.0, sigma_min=0.002, predict_batch=64,
    )

    def __init__(self, variables, predictors, **params):
        p = {**self.defaults, **params}
        super().__init__(variables, predictors, **p)
        self.tf = TargetTransform(self.variables, p["transform"])
        self.scaler = None
        self.y_mean = self.y_std = None
        self.net: AFMNet | None = None

    # ---- data
    def _statics(self, grid: Grid) -> np.ndarray:
        fields = {"orog": grid.orog, "lat": grid.lat}
        out = [(fields[k] - fields[k].mean()) / fields[k].std() for k in self.params["statics"]]
        return np.stack(out).astype(np.float32)[None]

    def _fit_normalisation(self, y: np.ndarray) -> None:
        z = self.tf.forward(y)
        axes = (0, 2, 3) if self.params["normalize"] == "variable" else (0,)
        self.y_mean = z.mean(axis=axes, dtype=np.float64, keepdims=True)[0].astype(np.float32)
        self.y_std = np.maximum(z.std(axis=axes, dtype=np.float64, keepdims=True)[0], 1e-6).astype(np.float32)

    def _standardise_y(self, y):
        return ((self.tf.forward(y) - self.y_mean) / self.y_std).astype(np.float32)

    def _physical(self, z: np.ndarray) -> np.ndarray:
        y = self.tf.inverse(z * self.y_std + self.y_mean)
        if "pr" in self.variables:  # choice: no negative precipitation
            i = self.variables.index("pr")
            y[..., i, :, :] = np.maximum(y[..., i, :, :], 0)
        return y

    def build_network(self, n_in, n_out, out_hw) -> AFMNet:
        p = self.params
        kw = {k: p[k] for k in ("model_channels", "channel_mult", "num_blocks", "attn_resolutions", "dropout")}
        return AFMNet(n_in, n_out, out_hw, len(p["statics"]), kw, bool(p["amp"]) and torch.cuda.is_available())

    def _autocast(self, dev):
        return torch.autocast(dev.type, dtype=torch.bfloat16, enabled=bool(self.params["amp"]) and dev.type == "cuda")

    # ---- base distribution (subclasses, e.g. LC-AFM, replace it)
    def _aux(self, split: Split, grid: Grid, dev) -> dict:
        """Per-day tensors the base needs besides the predictors (none for the white base)."""
        return {}

    def _base(self, net, e, aux, gen=None):
        """Base x0 = m + sigma_max * eps: returns (m, eps). AFM: m = E, eps white."""
        return e, torch.randn(e.shape, device=e.device, generator=gen)

    # ---- loss (Eq. 3)
    def _losses(self, net, x_lr, x1, dev, gen=None, ema=None, aux=None):
        """Returns L_AFM, L_enc and the encoder's per-channel squared error (for sigma_max)."""
        B = len(x1)
        cond = net.cond(x_lr)  # fp32 (bilinear interpolation)
        with self._autocast(dev):
            e = net.encode(cond)
        t = torch.rand(B, device=dev, generator=gen)
        s = 1 - t  # sigma_t / sigma_max in (0, 1]
        m, eps = self._base(net, e, aux, gen)
        smax = net.sigma_max[None, :, None, None]
        n = (m - x1) / smax + eps  # x_t = x1 + sigma n  (Eq. 2 rewritten; x0 = m + sigma_max eps)
        sigma = s[:, None, None, None] * smax
        x_t = x1 + sigma * n
        with self._autocast(dev):
            F, sigma, c_skip, c_out = net.denoise_raw(x_t, s, cond, ema)
        # w(sigma) ||D - x1||^2 with D = c_skip x_t + c_out F equals ||F - target||^2 (w c_out^2 = 1), with the
        # target written without cancellation so that small sigma stays numerically exact
        norm = torch.sqrt(sigma**2 + SIGMA_DATA**2)
        target = sigma * x1 / (SIGMA_DATA * norm) - SIGMA_DATA * n / norm
        l_afm = ((F - target) ** 2).mean()
        l_enc = ((e - x1) ** 2).mean()
        enc_sq = ((e - x1).detach() ** 2).mean(dim=(0, 2, 3))
        return l_afm, l_enc, enc_sq

    @torch.no_grad()
    def _validate(self, net, ema, Xv, Yv, dev, aux=None):
        gen = torch.Generator(device=dev).manual_seed(12345)  # same t, eps every evaluation
        tot_afm = tot_enc = 0.0
        mb = self.params["micro_batch"]
        net.eval()
        ema.eval()
        for i in range(0, len(Xv), mb):
            ab = {k: v[i:i + mb] for k, v in (aux or {}).items()}
            la, le, _ = self._losses(net, Xv[i:i + mb], Yv[i:i + mb], dev, gen, ema, ab)
            n = len(Xv[i:i + mb])
            tot_afm += la.item() * n
            tot_enc += le.item() * n
        net.train()
        l_afm, l_enc = tot_afm / len(Xv), tot_enc / len(Xv)
        return l_afm + self.params["lambda_enc"] * l_enc, l_afm, l_enc

    # ---- training
    def fit(self, train: Split, val: Split, grid: Grid) -> dict:
        p = self.params
        torch.manual_seed(p["seed"])
        np.random.seed(p["seed"])
        dev = device()
        self.scaler = PredictorScaler().fit(train.x)
        self._fit_normalisation(train.y)
        X = torch.from_numpy(self.scaler.transform(train.x)).to(dev)
        Y = torch.from_numpy(self._standardise_y(train.y)).to(dev)
        Xv = torch.from_numpy(self.scaler.transform(val.x)).to(dev)
        Yv = torch.from_numpy(self._standardise_y(val.y)).to(dev)
        aux_tr, aux_va = self._aux(train, grid, dev), self._aux(val, grid, dev)

        net = self.build_network(X.shape[1], Y.shape[1], grid.shape)
        net.grid.copy_(sample_grid(grid.lat, grid.lon, grid.pred_lat, grid.pred_lon))
        net.static.copy_(torch.from_numpy(self._statics(grid)))
        net.sigma_max.fill_(p["sigma_max_init"])
        net = net.to(dev).train()
        ema = copy.deepcopy(net.denoiser).eval().requires_grad_(False)
        if p["compile"] and dev.type == "cuda":
            torch.set_float32_matmul_precision("high")
            net.use_compiled(channels_last=False)
        n_enc = sum(q.numel() for q in net.encoder.parameters())
        n_den = sum(q.numel() for q in net.denoiser.parameters())
        print(f"[fit] AFM: encoder {n_enc / 1e6:.2f} M + denoiser {n_den / 1e6:.2f} M parameters on {dev}",
              flush=True)
        opt = torch.optim.Adam(list(net.encoder.parameters()) + list(net.denoiser.parameters()), lr=p["lr"],
                               betas=tuple(p["betas"]))

        bs, mb = p["batch_size"], p["micro_batch"]
        n_steps = math.ceil(p["train_samples"] / bs)
        seen, best, best_state, history = 0, np.inf, None, []
        acc = {"loss": 0.0, "l_afm": 0.0, "l_enc": 0.0, "n": 0}
        t0 = t_last = time.time()
        next_eval = p["eval_every"]
        for step in range(n_steps):
            idx = torch.randint(len(X), (bs,), device=dev)  # choice: sampling with replacement
            opt.zero_grad(set_to_none=True)
            for j in range(0, bs, mb):
                b = idx[j:j + mb]
                l_afm, l_enc, enc_sq = self._losses(net, X[b], Y[b], dev, aux={k: v[b] for k, v in aux_tr.items()})
                loss = l_afm + p["lambda_enc"] * l_enc
                (loss * len(b) / bs).backward()
                with torch.no_grad():  # adaptive noise: EMA of the encoder's RMS error per channel
                    net.sigma_max.mul_(1 - p["sigma_max_beta"]).add_(p["sigma_max_beta"] * enc_sq.sqrt())
                acc["loss"] += loss.item() * len(b)
                acc["l_afm"] += l_afm.item() * len(b)
                acc["l_enc"] += l_enc.item() * len(b)
                acc["n"] += len(b)
            opt.step()
            seen += bs
            with torch.no_grad():  # EMA of the denoiser weights (half-life in samples, EDM-style ramp-up)
                halflife = min(p["ema_halflife"], seen * p["ema_rampup"])
                beta = 0.5 ** (bs / max(halflife, 1e-8))
                for pe, pn in zip(ema.parameters(), net.denoiser.parameters()):
                    pe.copy_(pn.detach().lerp(pe, beta))
                for be, bn in zip(ema.buffers(), net.denoiser.buffers()):
                    be.copy_(bn)
            if seen >= next_eval or step == n_steps - 1:
                next_eval += p["eval_every"]
                vl, va, ve = self._validate(net, ema, Xv, Yv, dev, aux_va)
                now = time.time()
                rec = {"epoch": seen / len(X), "samples": seen, "train_loss": acc["loss"] / acc["n"],
                       "train_l_afm": acc["l_afm"] / acc["n"], "train_l_enc": acc["l_enc"] / acc["n"],
                       "val_loss": vl, "val_l_afm": va, "val_l_enc": ve,
                       "sigma_max": net.sigma_max.tolist(), "seconds": round(now - t0, 1),
                       "samples_per_s": round(acc["n"] / (now - t_last), 1)}
                history.append(rec)
                acc = {"loss": 0.0, "l_afm": 0.0, "l_enc": 0.0, "n": 0}
                t_last = now
                improved = vl < best
                if improved:
                    best = vl
                    best_state = {"net": copy.deepcopy(net.state_dict()), "ema": copy.deepcopy(ema.state_dict()),
                                  "samples": seen}
                print(f"[fit] {seen / 1e6:6.2f} M samples  train {rec['train_loss']:.4f}  val {vl:.4f} "
                      f"(afm {va:.4f}, enc {ve:.4f})  sigma_max {np.round(rec['sigma_max'], 3).tolist()}  "
                      f"{rec['samples_per_s']:.0f}/s{'  *' if improved else ''}  ({now - t0:.0f} s)", flush=True)

        net.load_state_dict(best_state["net"])
        net.denoiser.load_state_dict(best_state["ema"])  # inference uses the EMA denoiser
        self.net = net.eval()
        info = {"n_params": n_enc + n_den, "n_params_encoder": n_enc, "n_params_denoiser": n_den,
                "device": torch.cuda.get_device_name() if dev.type == "cuda" else "cpu",
                "loss_name": "L_AFM + 0.25 L_enc", "train_samples": seen, "epochs": seen / len(X),
                "best_samples": best_state["samples"], "best_val_loss": best,
                "sigma_max": net.sigma_max.tolist(), "train_seconds": round(time.time() - t0, 1),
                "max_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1) if dev.type == "cuda" else None,
                "history": history}
        enc = self._encoder_prediction(val, grid)
        info["selected"] = {v: {"best_samples": best_state["samples"],
                                "val_rmse_encoder": float(np.sqrt(np.mean((enc[:, i] - val.y[:, i]) ** 2)))}
                            for i, v in enumerate(self.variables)}
        return info

    # ---- inference
    @torch.no_grad()
    def _encoder_prediction(self, split: Split, grid: Grid) -> np.ndarray:
        """Deterministic first estimate E(x_LR) in physical units, (T, V, H, W)."""
        dev = device()
        net = self.net.to(dev).eval()
        X = torch.from_numpy(self.scaler.transform(split.x))
        out = np.empty((len(split), len(self.variables), *grid.shape), np.float32)
        bs = self.params["micro_batch"]
        for i in range(0, len(X), bs):
            cond = net.cond(X[i:i + bs].to(dev))
            with self._autocast(dev):
                e = net.encode(cond)
            out[i:i + bs] = self._physical(e.cpu().numpy())
        return out

    def schedule(self) -> torch.Tensor:
        """Relative noise levels s_n = sigma_n / sigma_max (Eq. 4), n = 0..N-1, followed by 0.
        choice: sigma_min is applied relative to the RMS of sigma_max over channels."""
        p = self.params
        N, rho = p["n_steps"], p["rho"]
        s_min = min(p["sigma_min"] / float(self.net.sigma_max.pow(2).mean().sqrt()), 1.0)
        n = torch.arange(N, dtype=torch.float64)
        s = (1 + n / (N - 1) * (s_min ** (1 / rho) - 1)) ** rho
        return torch.cat([s, torch.zeros(1, dtype=torch.float64)]).float()

    @torch.no_grad()
    def sample(self, x_lr: torch.Tensor, n_members: int, gen: torch.Generator, aux=None) -> torch.Tensor:
        """Euler integration of the flow from the base x0 = E + sigma_max eps (s = 1) to s = 0; returns
        (M, B, V, H, W) in standardised units. Step: x <- x + (D - x)(s_n - s_{n+1}) / s_n (Sect. 3.1).
        With params["base_only"] the base draws x0 themselves are returned (diagnostics: what the flow adds)."""
        dev = x_lr.device
        net = self.net
        cond = net.cond(x_lr)
        with self._autocast(dev):
            e = net.encode(cond)
        cond = cond.repeat(n_members, 1, 1, 1)
        e = e.repeat(n_members, 1, 1, 1)
        aux = {k: v.repeat(n_members, *([1] * (v.dim() - 1))) for k, v in (aux or {}).items()}
        m, eps = self._base(net, e, aux, gen)
        x = m + net.sigma_max[None, :, None, None] * eps
        if self.params.get("base_only"):
            return x.reshape(n_members, len(x_lr), *x.shape[1:])
        s_all = self.schedule().to(dev)
        for s, s_next in zip(s_all[:-1], s_all[1:]):
            with self._autocast(dev):
                D = net.denoise(x, s.expand(len(x)), cond)
            x = D + (x - D) * (s_next / s)
        return x.reshape(n_members, len(x_lr), *x.shape[1:])

    def predict(self, split: Split, grid: Grid, n_members: int = 1, seed: int = 0) -> np.ndarray:
        dev = device()
        self.net = self.net.to(dev).eval()
        if self.params["compile"] and dev.type == "cuda":
            torch.set_float32_matmul_precision("high")
            self.net.use_compiled(channels_last=True)
        X = torch.from_numpy(self.scaler.transform(split.x))
        aux = {k: v.cpu() for k, v in self._aux(split, grid, dev).items()}
        out = np.empty((n_members, len(split), len(self.variables), *grid.shape), np.float32)
        bs = self.params["predict_batch"]
        t0 = time.time()

        def pad(a, nb):  # pad the last batch to the full size (no recompilation); padding is discarded
            return a if nb == bs else torch.cat([a, a[-1:].expand(bs - nb, *a.shape[1:])])

        for k, i in enumerate(range(0, len(X), bs)):
            gen = torch.Generator(device=dev).manual_seed(seed * 1_000_003 + k)
            nb = len(X[i:i + bs])
            xb = pad(X[i:i + bs], nb)
            ab = {key: pad(v[i:i + bs], nb).to(dev) for key, v in aux.items()}
            z = self.sample(xb.to(dev), n_members, gen, ab)[:, :nb]
            out[:, i:i + nb] = self._physical(z.cpu().numpy())
            if k % 50 == 0:
                print(f"[predict] {split.name}: {i + len(z[0])}/{len(X)} days ({time.time() - t0:.0f} s)", flush=True)
        return out

    # ---- persistence
    def save_state(self, run_dir: Path) -> None:
        torch.save({"net": self.net.state_dict(),
                    "shapes": {"n_in": len(self.predictors), "n_out": len(self.variables),
                               "out_hw": list(self.net.grid.shape[1:3])},
                    "x_mean": torch.from_numpy(self.scaler.mean), "x_std": torch.from_numpy(self.scaler.std),
                    "y_mean": torch.from_numpy(self.y_mean), "y_std": torch.from_numpy(self.y_std)},
                   run_dir / "state.pt")

    def load_state(self, run_dir: Path) -> None:
        st = torch.load(run_dir / "state.pt", map_location="cpu", weights_only=True)
        s = st["shapes"]
        self.net = self.build_network(s["n_in"], s["n_out"], tuple(s["out_hw"]))
        self.net.load_state_dict(st["net"])
        self.net.eval()
        self.scaler = PredictorScaler(st["x_mean"].numpy(), st["x_std"].numpy())
        self.y_mean, self.y_std = st["y_mean"].numpy(), st["y_std"].numpy()
