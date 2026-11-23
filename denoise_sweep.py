# -*- coding: utf-8 -*-
"""Denoise sweep: post-hoc SNR boost on EXISTING traces (no new simulations).
Denoisers: none / moving-average 5,15 / Savitzky-Golay 15,31 / Haar wavelet
soft-threshold / total-variation (Chambolle projection).
For each: (a) retrain chronos crop identity reader on homopolymers, report T1;
(b) 5-fold CV of the 12-way order classifier (chronos, whole-trace, reversal
TTA) on the 25 labelled 2-block strands. Baseline 'none' must reproduce
T1~0.88 / order~0.45.
Run: HF_ENDPOINT=https://hf-mirror.com C:/Users/huangnian/pyenvs/esm/Scripts/python.exe denoise_sweep.py --epochs 30 --oepochs 40
"""
import os, json, argparse, random
import numpy as np
import torch
import torch.nn.functional as F
from bench_new import load, engaged, ALPH, C2I, TARGET
from fm_block2 import crop512
from fm_chronos import ChronosBlock
from fm_mix2 import OrderHead, PAIRS, P2I
from fm_mix3 import tta_probs


def ma(x, w):
    pad = w // 2
    xp = np.pad(x, pad, mode="reflect")
    k = np.ones(w) / w
    return np.convolve(xp, k, mode="valid")


def sg(x, w, poly=3):
    h = w // 2
    t = np.arange(-h, h + 1, dtype=float)
    V = np.vander(t, poly + 1, increasing=True)
    coef = np.linalg.pinv(V)[0]
    xp = np.pad(x, h, mode="reflect")
    return np.convolve(xp, coef[::-1], mode="valid")


def haar_soft(x, levels=3):
    n = len(x)
    # pad to power of two
    m = 1
    while m < n:
        m *= 2
    xp = np.pad(x, (0, m - n), mode="reflect")
    coeffs = []
    cur = xp.copy()
    L = m
    for _ in range(levels):
        L //= 2
        a = (cur[0::2][:L] + cur[1::2][:L]) / np.sqrt(2)
        d = (cur[0::2][:L] - cur[1::2][:L]) / np.sqrt(2)
        coeffs.append(d)
        cur = a
    # sigma from finest details
    sig = np.median(np.abs(coeffs[0])) / 0.6745 + 1e-9
    thr = sig * np.sqrt(2 * np.log(m))
    for li in range(levels):
        t = thr / (2 ** (li * 0.5))
        d = coeffs[li]
        coeffs[li] = np.sign(d) * np.maximum(np.abs(d) - t, 0)
    rec = cur
    for li in range(levels - 1, -1, -1):
        d = coeffs[li]
        L = len(rec)
        out = np.empty(2 * L)
        out[0::2] = (rec + d) / np.sqrt(2)
        out[1::2] = (rec - d) / np.sqrt(2)
        rec = out
    return rec[:n]


def tv1d(x, lam=0.05, iters=300):
    # PDHG for min_u 0.5||u-x||^2 + lam*||Du||_1 (1D total variation)
    n = len(x)
    u = x.copy()
    ub = x.copy()
    p = np.zeros(n - 1)
    tau = sig_ = 0.49

    def Dt(q):
        d = np.zeros(n)
        d[0] = -q[0]
        d[1:-1] = q[:-1] - q[1:]
        d[-1] = q[-1]
        return d

    for _ in range(iters):
        p = np.clip(p + sig_ * np.diff(ub), -lam, lam)
        un = (u + tau * (x - Dt(p))) / (1 + tau)
        ub = 2 * un - u
        u = un
    return u


DENOISERS = {
    "none": lambda s: s,
    "ma5": lambda s: np.stack([ma(s[c], 5) for c in range(3)]),
    "ma15": lambda s: np.stack([ma(s[c], 15) for c in range(3)]),
    "sg15": lambda s: np.stack([sg(s[c], 15) for c in range(3)]),
    "sg31": lambda s: np.stack([sg(s[c], 31) for c in range(3)]),
    "haar": lambda s: np.stack([haar_soft(s[c]) for c in range(3)]),
    "tv": lambda s: np.stack([tv1d(s[c]) for c in range(3)]),
}


def eng_of(sig):
    out = []
    for i in range(len(sig)):
        idx = np.where(engaged(sig[i]))[0]
        out.append((int(idx[0]), int(idx[-1]) + 1) if len(idx) else (0, TARGET))
    return out


def train_ident(sig, eng, tr, meta, epochs, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    model = ChronosBlock()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    for ep in range(epochs):
        model.train()
        random.shuffle(tr)
        for s in range(0, len(tr), 16):
            batch = tr[s:s + 16]
            xs, ys = [], []
            for i in batch:
                e0, e1 = eng[i]
                L = e1 - e0
                if L > 160 and random.random() < 0.8:
                    ln = random.randint(120, min(400, L))
                    st = random.randint(e0, e1 - ln)
                    xs.append(crop512(sig[i], st, st + ln).T)
                else:
                    xs.append(crop512(sig[i], e0, e1).T)
                ys.append(C2I[meta[i]["base"]])
            xb = torch.from_numpy(np.stack(xs))
            xb = xb * (0.9 + 0.2 * torch.rand(len(batch), 1, 1)) + 0.02 * torch.randn(len(batch), 1, 1)
            loss = F.cross_entropy(model(xb), torch.tensor(ys))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_order(tr, sig, eng, lab, epochs, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    model = OrderHead(ChronosBlock())
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    for ep in range(epochs):
        model.train()
        random.shuffle(tr)
        for s in range(0, len(tr), 8):
            batch = tr[s:s + 8]
            xs, ys = [], []
            for i in batch:
                e0, e1 = eng[i]
                x = crop512(sig[i], e0, e1).T
                y = lab[i]
                if random.random() < 0.5:
                    x = x[::-1].copy()
                    y = P2I[PAIRS[y][::-1]]
                xs.append(x)
                ys.append(y)
            xb = torch.from_numpy(np.stack(xs))
            xb = xb * (0.9 + 0.2 * torch.rand(len(batch), 1, 1)) + 0.02 * torch.randn(len(batch), 1, 1)
            loss = F.cross_entropy(model(xb), torch.tensor(ys))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--oepochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    os.makedirs("runs_dn", exist_ok=True)
    sig0, meta = load()
    homo = [i for i, m in enumerate(meta) if not m["mixed"] and m["base"] in C2I]
    random.seed(a.seed)
    hs = homo[:]
    random.shuffle(hs)
    nval = max(1, int(len(hs) * 0.2))
    va, tr = hs[:nval], hs[nval:]
    z = np.load("runs_mix2/dataset.npz")
    msig0 = z["sig"]
    mmeta = json.load(open("runs_mix2/meta.json", encoding="utf-8"))
    mlab = np.array([P2I[m["seq"][::10]] for m in mmeta])
    fam = np.array([m["block_label"] for m in mmeta])
    random.seed(a.seed)
    folds = [[] for _ in range(5)]
    for f in sorted(set(fam)):
        ii = [i for i in range(len(mmeta)) if fam[i] == f]
        random.shuffle(ii)
        for k, i in enumerate(ii):
            folds[k % 5].append(i)
    res = {}
    names = [a.only] if a.only else list(DENOISERS)
    for name in names:
        dn = DENOISERS[name]
        sig = np.stack([dn(sig0[i]) for i in range(len(sig0))]).astype(np.float32)
        msig = np.stack([dn(msig0[i]) for i in range(len(msig0))]).astype(np.float32)
        eng = eng_of(sig0)          # windows fixed from raw signal
        meng = eng_of(msig0)
        im = train_ident(sig, eng, tr, meta, a.epochs, a.seed)
        ok = 0
        with torch.no_grad():
            for i in va:
                x = torch.from_numpy(crop512(sig[i], *eng[i]).T)[None]
                b = ALPH[int(im(x)[0].argmax())]
                ok += b == meta[i]["base"]
        t1 = ok / len(va)
        oa, ca = [], []
        for fold in range(5):
            te = folds[fold]
            trn = [i for i in range(len(mmeta)) if i not in te]
            m = train_order(trn, msig, meng, mlab, a.oepochs, a.seed + fold)
            ok = comp = 0
            for i in te:
                p = tta_probs([m], msig, i, *meng[i])
                pr = PAIRS[int(p.argmax())]
                t = PAIRS[mlab[i]]
                ok += pr == t
                comp += sorted(pr) == sorted(t)
            oa.append(ok / len(te))
            ca.append(comp / len(te))
        res[name] = dict(T1=t1, order=float(np.mean(oa)), comp=float(np.mean(ca)),
                         order_folds=oa)
        print("%-5s T1=%.3f order=%.3f comp=%.3f" % (name, t1, np.mean(oa), np.mean(ca)))
        with open("runs_dn/sweep.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
    print("saved runs_dn/sweep.json")


if __name__ == "__main__":
    main()