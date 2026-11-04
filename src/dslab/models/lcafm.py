"""Law-conditioned adaptive flow matching (LC-AFM): AFM whose base distribution is a Gaussian conditioned
on linear physical laws (plan "pcAFM: from Burgers to 12 km climate downscaling"; laws in dslab.laws).

Everything else is AFM (models/afm.py): networks, loss, joint training of encoder and denoiser, sigma_max
adaptation, sampler. Only the base changes, per output channel c:

    white (AFM)   x0 = E + sigma_max eps
    LC-AFM        x0 = m + n,  the exact posterior of the prior N(E, S), S = sigma_max^2 I, given the law
                  J0 x = b + tau eta (eta ~ N(0, I)); with J = D_tau^-1 J0, J0 = I (law Tfull) or
                  H = I - U A (laws T, O):
                      m = E + S J^T (J S J^T + I)^-1 (D_tau^-1 b - J E)          (posterior mean)
                      n = e + S J^T (J S J^T + I)^-1 (eta - J e),  e ~ N(0, S)   (Matheron's rule)

For J0 = I the posterior is elementwise. For J0 = H = I - Q Q^T (Q: orthonormal basis of the 2°-resolved
fields, rank r = number of 2° cells in the domain, 92 for ALPS), J S J^T + I is a diagonal minus a rank-r
matrix and is inverted exactly with the Woodbury identity; its r x r core I - s^2 Q^T diag(1/(tau^2+s^2)) Q is
positive definite, so one Cholesky factorisation per calendar month (tau varies by month) suffices and no CG
is needed. m is linear in E and differentiable, so the flow gradients still reach the encoder (joint
training, as in AFM).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..data import Grid, Split
from ..laws import LAWS, CoarseOperator, LawSet, fit_cached, months, phi_cached
from .afm import AFM
from .base import register


class LawConditioning(nn.Module):
    """Conditioned Gaussian base for (B, V, H, W) fields in standardised units. kinds[c] is None (white),
    "full" (J0 = I) or "highpass" (J0 = H); tol[c] is (12, H, W) in standardised units or None."""

    def __init__(self, op: CoarseOperator, kinds: list, tol: list):
        super().__init__()
        self.kinds = list(kinds)
        Q = torch.from_numpy(op.Q)
        self.register_buffer("Q", Q)  # (N, r), orthonormal: P = Q Q^T, H = I - P (symmetric, idempotent)
        self.register_buffer("Q64", Q.double())
        for c, t in enumerate(tol):
            if t is not None:
                self.register_buffer(f"tol{c}", torch.from_numpy(np.ascontiguousarray(t, np.float32)).flatten(1))

    def _H(self, x):  # x (B, N); H^T = H
        return x - (x @ self.Q) @ self.Q.T

    def _cholesky(self, tau_m: torch.Tensor, s: float):
        """Cholesky factors of the Woodbury cores K_m = I - s^2 Q^T diag(1 / (tau_m^2 + s^2)) Q (12, r, r),
        positive definite because s^2 / (tau^2 + s^2) < 1 and Q^T Q = I."""
        g = 1.0 / (tau_m.double() ** 2 + s**2)  # (12, N)
        eye = torch.eye(self.Q.shape[1], dtype=torch.float64, device=self.Q.device)
        K = eye - s**2 * torch.einsum("mn,ni,nj->mij", g, self.Q64, self.Q64)
        return torch.linalg.cholesky(K)

    def _solve(self, y, tau, s, chol, month):
        """(J S J^T + I)^-1 y for rows y (B, N), tolerances tau (B, N): with D = diag(tau) and
        Delta = I + s^2 D^-2, J S J^T + I = Delta - V V^T, V = s D^-1 Q, and by Woodbury
        (Delta - V V^T)^-1 = Delta^-1 + Delta^-1 V K^-1 V^T Delta^-1."""
        g = 1.0 / (tau**2 + s**2)
        a, tg = tau**2 * g, tau * g  # Delta^-1, and Delta^-1 D^-1
        z = ((tg * y).double() @ self.Q64)[..., None]  # (B, r, 1)
        sol = torch.cholesky_solve(z, chol[month])[..., 0]
        return a * y + s**2 * tg * (sol @ self.Q64.T).float()

    def forward(self, m0, b, month, sigma_max, gen=None):
        """m0: encoder output E (B, V, H, W); b: law right-hand sides (B, V, H, W), standardised; month (B,).
        Returns (m, eps) with x0 = m + sigma_max * eps (eps = n / sigma_max, white for unconstrained channels)."""
        B, V = m0.shape[:2]
        shape = m0.shape
        eps_e = torch.randn(shape, device=m0.device, generator=gen)
        eta = torch.randn(shape, device=m0.device, generator=gen)
        ms, eps = [], []
        for c in range(V):
            kind = self.kinds[c]
            if kind is None:
                ms.append(m0[:, c])
                eps.append(eps_e[:, c])
                continue
            s = float(sigma_max[c])
            x0 = m0[:, c].flatten(1)
            bc = b[:, c].flatten(1)
            tau = getattr(self, f"tol{c}")[month]  # (B, N)
            e = s * eps_e[:, c].flatten(1)
            et = eta[:, c].flatten(1)
            if kind == "full":
                w = s**2 / (s**2 + tau**2)
                m = x0 + w * (bc - x0)
                n = e + w * (tau * et - e)
            else:
                chol = self._cholesky(getattr(self, f"tol{c}"), s)
                rhs = torch.cat([bc / tau - self._H(x0) / tau, et - self._H(e) / tau])
                y = self._solve(rhs, tau.repeat(2, 1), s, chol, month.repeat(2))
                corr = s**2 * self._H(y / tau.repeat(2, 1))  # S J^T y = s^2 H D^-1 y
                m, n = x0 + corr[:B], e + corr[B:]
            ms.append(m.view(B, *shape[2:]))
            eps.append((n / s).view(B, *shape[2:]))
        return torch.stack(ms, 1), torch.stack(eps, 1)


@register("lcafm")
class LCAFM(AFM):
    summary = ("law-conditioned adaptive flow matching: AFM whose base noise is a Gaussian conditioned on "
               "linear physical laws (lapse rate, orographic precipitation)")
    defaults = {**AFM.defaults, "laws": ["T", "O"]}

    def __init__(self, variables, predictors, **params):
        super().__init__(variables, predictors, **params)
        for n in self.params["laws"]:
            if n not in LAWS:
                raise KeyError(f"unknown law {n!r}; available: {sorted(LAWS)}")
        per_var = [LAWS[n][0] for n in self.params["laws"]]
        if len(set(per_var)) != len(per_var):
            raise ValueError(f"at most one law per variable, got {self.params['laws']}")
        self.laws: LawSet | None = None
        self.cb: LawConditioning | None = None

    # ---- laws: fitted on the training truth (cached), right-hand sides from each split's predictors
    def fit(self, train: Split, val: Split, grid: Grid) -> dict:
        self.laws = fit_cached(train, grid, self.params["laws"])
        info = super().fit(train, val, grid)
        info["laws"] = self.laws.info
        return info

    def _conditioning(self, grid: Grid, dev) -> LawConditioning:
        if self.cb is None:
            kinds, tol = [None] * len(self.variables), [None] * len(self.variables)
            for n in self.laws.names:
                c = self.variables.index(LAWS[n][0])
                kinds[c] = "highpass" if LAWS[n][1] else "full"
                tol[c] = self.laws.tol[n] / self.y_std[c]
            self.cb = LawConditioning(CoarseOperator.from_grid(grid), kinds, tol)
        return self.cb.to(dev)

    def _aux(self, split: Split, grid: Grid, dev) -> dict:
        self._conditioning(grid, dev)
        phi = phi_cached(self.laws, split, grid)
        b = np.zeros((len(split), len(self.variables), *grid.shape), np.float32)
        for n, rhs in phi.items():
            c = self.variables.index(LAWS[n][0])
            mean = 0.0 if LAWS[n][1] else self.y_mean[c]  # H removes constants; the identity does not
            b[:, c] = (rhs - mean) / self.y_std[c]
        return {"b": torch.from_numpy(b).to(dev), "month": torch.from_numpy(months(split.time)).long().to(dev)}

    def _base(self, net, e, aux, gen=None):
        return self.cb(e, aux["b"], aux["month"], net.sigma_max, gen)

    # ---- persistence: the fitted laws next to state.pt
    def save_state(self, run_dir: Path) -> None:
        super().save_state(run_dir)
        np.savez(run_dir / "laws.npz", **self.laws.state())

    def load_state(self, run_dir: Path) -> None:
        super().load_state(run_dir)
        with np.load(run_dir / "laws.npz") as f:
            self.laws = LawSet.from_state({k: f[k] for k in f.files})
