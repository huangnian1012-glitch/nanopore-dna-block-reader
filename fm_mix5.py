# -*- coding: utf-8 -*-
"""Order-model accuracy push A: multi-seed ensemble + richer test-time averaging
+ label smoothing. No new simulation needed.
TTA views: original / time-reversed (label-flipped) / phase rolls +-10 samples.
3 seeds x 2 encoders averaged per fold. 5-fold CV on the 25 labelled strands.
Run: HF_ENDPOINT=https://hf-mirror.com C:/Users/huangnian/pyenvs/esm/Scripts/python.exe fm_mix5.py --epochs 80 --seeds 3
"""
import os, json, argparse, random
import numpy as np
import torch
import torch.nn.functional as F
from bench_new import engaged, TARGET
from fm_block2 import crop512, FMCrop
from fm_chronos import ChronosBlock
from fm_mix2 import OrderHead, PAIRS, P2I
from denoise_sweep import tv1d


def train_enc(enc_name, tr, sig, eng, lab, epochs, seed, ls=0.05):
    random.seed(seed)
    torch.manual_seed(seed)
    enc = FMCrop() if enc_name == "patchtst" else ChronosBlock()
    model = OrderHead(enc)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=3e-4 if enc_name == "patchtst" else 1e-4)
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
            loss = F.cross_entropy(model(xb), torch.tensor(ys), label_smoothing=ls)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def tta5(models, sig, i, e0, e1):
    x = crop512(sig[i], e0, e1).T
    views = []
    for roll in (0, -10, 10):
        xr = np.roll(x, roll, axis=0)
        views.append((xr, False))
        views.append((xr[::-1].copy(), True))
    ps = []
    with torch.no_grad():
        for m in models:
            acc = np.zeros(12)
            for xv, rev in views:
                p = F.softmax(m(torch.from_numpy(xv)[None])[0], -1).numpy()
                if rev:
                    p = p[[P2I[q[::-1]] for q in PAIRS]]
                acc += p
            ps.append(acc / len(views))
    return np.mean(ps, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ls", type=float, default=0.05)
    ap.add_argument("--tv", action="store_true", help="TV-denoise signals (edge-preserving)")
    ap.add_argument("--tag", default="v5")
    a = ap.parse_args()
    os.makedirs("runs_mix2", exist_ok=True)
    z = np.load("runs_mix2/dataset.npz")
    sig = z["sig"]
    meta = json.load(open("runs_mix2/meta.json", encoding="utf-8"))
    eng = []
    for i in range(len(meta)):
        idx = np.where(engaged(sig[i]))[0]
        eng.append((int(idx[0]), int(idx[-1]) + 1) if len(idx) else (0, TARGET))
    if a.tv:
        sig = np.stack([np.stack([tv1d(sig[i, c]) for c in range(3)])
                        for i in range(len(sig))]).astype(np.float32)
    lab = np.array([P2I[m["seq"][::10]] for m in meta])
    fam = np.array([m["block_label"] for m in meta])
    random.seed(a.seed)
    folds = [[] for _ in range(a.folds)]
    for f in sorted(set(fam)):
        ii = [i for i in range(len(meta)) if fam[i] == f]
        random.shuffle(ii)
        for k, i in enumerate(ii):
            folds[k % a.folds].append(i)
    res = []
    for fold in range(a.folds):
        te = folds[fold]
        tr = [i for i in range(len(meta)) if i not in te]
        models = []
        for sd in range(a.seeds):
            for e in ("chronos", "patchtst"):
                models.append(train_enc(e, tr, sig, eng, lab, a.epochs,
                                        a.seed + 100 * sd + fold, a.ls))
        ok = comp = dirb = 0
        preds = []
        for i in te:
            p = tta5(models, sig, i, *eng[i])
            pr = PAIRS[int(p.argmax())]
            preds.append(pr)
            t = PAIRS[lab[i]]
            ok += pr == t
            comp += sorted(pr) == sorted(t)
            dirb += pr == t or pr[::-1] == t
        n = len(te)
        res.append(dict(fold=fold, n=n, order_acc=ok / n, comp_acc=comp / n,
                        dirbest_acc=dirb / n, preds=preds,
                        truth=[PAIRS[lab[i]] for i in te], ids=[meta[i]["id"] for i in te]))
        print("fold%d n=%d order=%.3f comp=%.3f dirbest=%.3f" % (fold, n, ok / n, comp / n, dirb / n))
    summ = dict(order_acc=float(np.mean([r["order_acc"] for r in res])),
                comp_acc=float(np.mean([r["comp_acc"] for r in res])),
                dirbest_acc=float(np.mean([r["dirbest_acc"] for r in res])),
                chance_order=1 / 12, chance_comp=1 / 6)
    print("SUMMARY", json.dumps(summ))
    with open("runs_mix2/order_%s.json" % a.tag, "w", encoding="utf-8") as f:
        json.dump(dict(summary=summ, folds=res, tv=a.tv), f, indent=1)
    print("saved runs_mix2/order_%s.json" % a.tag)


if __name__ == "__main__":
    main()