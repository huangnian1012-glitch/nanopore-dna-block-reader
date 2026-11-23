# -*- coding: utf-8 -*-
"""Trace-level mixed read-order classifier trained on the 25 LABELLED 2-block
strands (runs_mix2). No boundary estimation needed: whole engaged trace ->
12-way ordered base-pair order (e.g. 'CA' means C-block first, A-block second).
Legitimate augmentation: time reversal flips the order label (x[::-1], label
reversed), plus scale/offset/noise and random engaged sub-crops.
5-fold cross-validation stratified by family; metrics: 12-way order acc,
composition (unordered pair) acc, direction-best acc, chance levels reported.
Encoder: chronos-bolt-mini partial finetune (or patchtst).
Run: HF_ENDPOINT=https://hf-mirror.com C:/Users/huangnian/pyenvs/esm/Scripts/python.exe fm_mix2.py --enc chronos
"""
import os, json, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bench_new import engaged, ALPH, TARGET
from fm_block2 import crop512, FMCrop
from fm_chronos import ChronosBlock

PAIRS = [a + b for a in ALPH for b in ALPH if a != b]
P2I = {p: i for i, p in enumerate(PAIRS)}


class OrderHead(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
        d = 3 * 128 if isinstance(enc, FMCrop) else enc.m.config.d_model
        self.head = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 12))

    def forward(self, x):
        if isinstance(self.enc, FMCrop):
            h = self.enc.enc(past_values=x).last_hidden_state[:, :, 1:, :]
            B, C, P, d = h.shape
            pool = h.permute(0, 2, 1, 3).reshape(B, P, C * d).mean(1)
        else:
            B = x.shape[0]
            xu = x.permute(0, 2, 1).reshape(B * 3, x.shape[1])
            h = self.enc.m.encode(xu)[0]
            if self.enc.use_reg:
                h = h[:, :-1]
            pool = h.mean(1).reshape(B, 3, -1).mean(1)
        return self.head(pool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="chronos", choices=["chronos", "patchtst"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs("runs_mix2", exist_ok=True)
    z = np.load("runs_mix2/dataset.npz")
    sig = z["sig"]
    meta = json.load(open("runs_mix2/meta.json", encoding="utf-8"))
    eng = []
    for i in range(len(meta)):
        idx = np.where(engaged(sig[i]))[0]
        eng.append((int(idx[0]), int(idx[-1]) + 1) if len(idx) else (0, TARGET))
    lab = np.array([P2I[m["seq"][::10]] for m in meta])
    fam = np.array([m["block_label"] for m in meta])
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
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
        enc = FMCrop() if a.enc == "patchtst" else ChronosBlock()
        model = OrderHead(enc)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=3e-4 if a.enc == "patchtst" else 1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-5)
        bs = 8
        for ep in range(a.epochs):
            model.train()
            random.shuffle(tr)
            tot, nb = 0.0, 0
            for s in range(0, len(tr), bs):
                batch = tr[s:s + bs]
                xs, ys = [], []
                for i in batch:
                    e0, e1 = eng[i]
                    L = e1 - e0
                    if L > 200 and random.random() < 0.5:
                        ln = random.randint(200, L)
                        st = random.randint(e0, e1 - ln)
                        x = crop512(sig[i], st, st + ln).T
                    else:
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
                tot += float(loss.item())
                nb += 1
            sched.step()
        model.eval()
        ok = comp = dirb = 0
        preds = []
        with torch.no_grad():
            for i in te:
                x = torch.from_numpy(crop512(sig[i], *eng[i]).T)[None]
                p = F.softmax(model(x)[0], -1).numpy()
                pr = PAIRS[int(p.argmax())]
                preds.append(pr)
                t = PAIRS[lab[i]]
                ok += pr == t
                comp += sorted(pr) == sorted(t)
                dirb += pr == t or pr[::-1] == t
        n = len(te)
        r = dict(fold=fold, n=n, order_acc=ok / n, comp_acc=comp / n, dirbest_acc=dirb / n,
                 preds=preds, truth=[PAIRS[lab[i]] for i in te], ids=[meta[i]["id"] for i in te])
        res.append(r)
        print("fold%d n=%d order=%.3f comp=%.3f dirbest=%.3f" % (fold, n, ok / n, comp / n, dirb / n))
    summ = dict(enc=a.enc,
                order_acc=float(np.mean([r["order_acc"] for r in res])),
                comp_acc=float(np.mean([r["comp_acc"] for r in res])),
                dirbest_acc=float(np.mean([r["dirbest_acc"] for r in res])),
                chance_order=1 / 12, chance_comp=1 / 6, chance_dirbest=1 / 6)
    print("SUMMARY", json.dumps(summ))
    with open("runs_mix2/order_%s.json" % a.enc, "w", encoding="utf-8") as f:
        json.dump(dict(summary=summ, folds=res), f, indent=1)
    print("saved runs_mix2/order_%s.json" % a.enc)


if __name__ == "__main__":
    main()