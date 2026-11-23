# -*- coding: utf-8 -*-
"""Mixed read-order v3: longer training + test-time reversal averaging (TTA) +
chronos/patchtst probability ensemble, 5-fold CV on the 25 labelled 2-block
strands. Also trains a FINAL model on all 25 and saves weights for recognize3.
Run: HF_ENDPOINT=https://hf-mirror.com C:/Users/huangnian/pyenvs/esm/Scripts/python.exe fm_mix3.py --epochs 100
"""
import os, json, argparse, random
import numpy as np
import torch
import torch.nn.functional as F
from bench_new import engaged, TARGET
from fm_block2 import crop512, FMCrop
from fm_chronos import ChronosBlock
from fm_mix2 import OrderHead, PAIRS, P2I


def train_enc(enc_name, tr, sig, meta, eng, lab, epochs, seed):
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
            loss = F.cross_entropy(model(xb), torch.tensor(ys))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def tta_probs(models, sig, i, e0, e1):
    x = crop512(sig[i], e0, e1).T
    xf = torch.from_numpy(x)[None]
    xr = torch.from_numpy(x[::-1].copy())[None]
    ps = []
    with torch.no_grad():
        for m in models:
            pf = F.softmax(m(xf)[0], -1).numpy()
            pr = F.softmax(m(xr)[0], -1).numpy()
            pr = pr[[P2I[p[::-1]] for p in PAIRS]]
            ps.append((pf + pr) / 2)
    return np.mean(ps, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
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
        models = [train_enc(e, tr, sig, meta, eng, lab, a.epochs, a.seed + fold)
                  for e in ("chronos", "patchtst")]
        ok = comp = dirb = 0
        preds = []
        for i in te:
            p = tta_probs(models, sig, i, *eng[i])
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
    with open("runs_mix2/order_v4.json", "w", encoding="utf-8") as f:
        json.dump(dict(summary=summ, folds=res), f, indent=1)
    print("training FINAL models on all 25 ...")
    allidx = list(range(len(meta)))
    for e in ("chronos", "patchtst"):
        m = train_enc(e, allidx, sig, meta, eng, lab, a.epochs, a.seed + 99)
        torch.save(m.state_dict(), "runs_mix2/order_final_%s.pt" % e)
        print("saved runs_mix2/order_final_%s.pt" % e)
    print("saved runs_mix2/order_v4.json")


if __name__ == "__main__":
    main()