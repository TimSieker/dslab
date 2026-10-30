"""Throughput of the AFM networks (training step and sampling) on one GPU, for a few execution settings.
Random inputs; nothing is trained or saved.  Usage: python scripts/bench_afm.py [model_channels] [num_blocks]
"""

from __future__ import annotations

import sys
import time

import torch

from dslab.models.afm import AFMNet

torch.backends.cudnn.benchmark = True
dev = torch.device("cuda")
mc = int(sys.argv[1]) if len(sys.argv) > 1 else 32
nb = int(sys.argv[2]) if len(sys.argv) > 2 else 4
KW = dict(model_channels=mc, channel_mult=[1, 2, 2, 4, 4], num_blocks=nb, attn_resolutions=[8], dropout=0.13)
H = W = 128


def make(channels_last=False):
    net = AFMNet(15, 2, (H, W), 2, KW, amp=True).to(dev)
    net.grid.uniform_(-1, 1)
    if channels_last:
        net = net.to(memory_format=torch.channels_last)
    return net


def timed(fn, n=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n


def train_step(net, opt, mb, channels_last):
    x = torch.randn(mb, 15, 16, 16, device=dev)
    x1 = torch.randn(mb, 2, H, W, device=dev)
    mf = torch.channels_last if channels_last else torch.contiguous_format

    def step():
        cond = net.cond(x).contiguous(memory_format=mf)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e = net.encode(cond)
            s = 1 - torch.rand(mb, device=dev)
            x_t = (x1 + s[:, None, None, None] * (e - x1 + torch.randn_like(x1))).contiguous(memory_format=mf)
            F, *_ = net.denoise_raw(x_t, s, cond)
        loss = (F.float() ** 2).mean() + 0.25 * ((e - x1) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return step


def sample_step(net, b, channels_last):
    x = torch.randn(b, 2, H, W, device=dev)
    cond = torch.randn(b, 17, H, W, device=dev)
    if channels_last:
        x, cond = x.contiguous(memory_format=torch.channels_last), cond.contiguous(memory_format=torch.channels_last)
    s = torch.full((b,), 0.5, device=dev)

    @torch.no_grad()
    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            net.denoise(x, s, cond)
    return step


print(torch.cuda.get_device_name(), f"model_channels={mc} num_blocks={nb}", flush=True)
for cl in (False, True):
    for compiled in (False, True):
        net = make(cl)
        if compiled:
            net.encoder = torch.compile(net.encoder)
            net.denoiser = torch.compile(net.denoiser)
        opt = torch.optim.Adam(net.parameters(), lr=2e-4)
        for mb in (64, 128, 256):
            try:
                torch.cuda.reset_peak_memory_stats()
                dt = timed(train_step(net, opt, mb, cl), n=5, warmup=3 if not compiled else 4)
                mem = torch.cuda.max_memory_allocated() / 1e9
                print(f"train   channels_last={cl!s:5} compile={compiled!s:5} micro_batch={mb:4d}: "
                      f"{mb / dt:7.0f} samples/s  ({mem:.0f} GB)", flush=True)
            except torch.OutOfMemoryError:
                print(f"train   channels_last={cl} compile={compiled} micro_batch={mb}: out of memory", flush=True)
                torch.cuda.empty_cache()
        net.eval()
        for b in (192, 384, 768):
            try:
                dt = timed(sample_step(net, b, cl), n=10, warmup=3 if not compiled else 4)
                print(f"sample  channels_last={cl!s:5} compile={compiled!s:5} batch={b:4d}: "
                      f"{b / dt:7.0f} denoiser evaluations/s", flush=True)
            except torch.OutOfMemoryError:
                print(f"sample  channels_last={cl} compile={compiled} batch={b}: out of memory", flush=True)
                torch.cuda.empty_cache()
        del net, opt
        torch.cuda.empty_cache()
