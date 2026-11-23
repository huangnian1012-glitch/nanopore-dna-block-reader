# -*- coding: utf-8 -*-
"""FM block-reader v3: leave-one-mixed-out with block-crop training.
The depth->base map is context dependent (mixed-block core depths are shifted
and re-ranked vs homopolymer references; per-channel affine calibration was
tested and rejected numerically), so homopolymer-only training cannot read
mixed blocks. The only real mixed labels are the 2 register-truth strands.
v3 trains on homopolymer crops PLUS heavily-augmented core crops of the OTHER
mixed strand's register blocks (leave-one-out), then reads the held-out strand
by DP segmentation + per-block crop identity.
--enc patchtst | chronos | both  (both = probability ensemble)
Run: HF_ENDPOINT=https://hf-mirror.com C:/Users/huangnian/pyenvs/esm/Scripts/python.exe fm_block3.py --enc both
"""
import os, json, argparse, random
import numpy as np
import torch
import torch.nn.functional as F
from bench_new import load, engaged, lev, ALPH, C2I, TARGET
from fm_block import seg_dp
from fm_block2 import crop512, FMCrop
from fm_chronos import ChronosBlock


def reg_blocks(m, min_len=60):
    rl = m["register_labels"]
    runs, cur, n, st = [], None, 0, 0
    for j, b in enumerate(rl):
        if b == cur:
            n += 1
        else:
            if cur is not None and cur != "-" and n >= min_len:
                runs.append((cur, st, st + n))
            cur, n, st = b, 1, j
    if cur is not None and cur != "-" and n >= min_len:
        runs.append((cur, st, st + n))
    return runs


def make_model(enc):
    return FMCrop() if enc == "patchtst" else ChronosBlock()


def train_one(enc, tr_homo, mix_train_blocks, sig, meta, eng, epochs, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model(enc)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=3e-4 if enc == "patchtst" else 1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    bs = 32 if enc == "patchtst" else 16
    pool = list(tr_homo) + [b for b in mix_train_blocks for _ in range(10)]
    for ep in range(epochs):
        model.train()
        random.shuffle(pool)
        tot, nb = 0.0, 0
        for s in range(0, len(pool), bs):
            batch = pool[s:s + bs]
            xs, ys = [], []
            for item in batch:
                if isinstance(item, int):
                    i = item
                    e0, e1 = eng[i]
                    L = e1 - e0
                    if L > 160 and random.random() < 0.8:
                        ln = random.randint(120, min(400, L))
                        st = random.randint(e0, e1 - ln)
                        xs.append(crop512(sig[i], st, st + ln).T)
                    else:
                        xs.append(crop512(sig[i], e0, e1).T)
                    ys.append(C2I[meta[i]["base"]])
                else:
                    i, s0, s1, b = item
                    c0, c1 = s0 + (s1 - s0) // 4, s1 - (s1 - s0) // 4
                    L = c1 - c0
                    ln = random.randint(max(40, L // 2), L)
                    st = random.randint(c0, c1 - ln)
                    xs.append(crop512(sig[i], st, st + ln).T)
                    ys.append(C2I[b])
            xb = torch.from_numpy(np.stack(xs))
            xb = xb * (0.9 + 0.2 * torch.rand(len(batch), 1, 1)) + 0.02 * torch.randn(len(batch), 1, 1)
            lg = model(xb)
            loss = F.cross_entropy(lg, torch.tensor(ys))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
        sched.step()
        if (ep + 1) % 10 == 0:
            print("  [%s] ep %d loss=%.4f" % (enc, ep + 1, tot / nb))
    model.eval()
    return model


def probs_of(model, sig, i, s0, s1):
    x = torch.from_numpy(crop512(sig[i], s0, s1).T)[None]
    with torch.no_grad():
        return F.softmax(model(x)[0], -1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="both", choices=["patchtst", "chronos", "both"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs("runs_alt", exist_ok=True)
    sig, meta = load()
    random.seed(a.seed)
    homo = [i for i, m in enumerate(meta) if not m["mixed"] and m["base"] in C2I]
    random.shuffle(homo)
    nval = max(1, int(len(homo) * 0.2))
    va, tr = homo[:nval], homo[nval:]
    mixed = [i for i, m in enumerate(meta) if m["mixed"] and m.get("register_labels")]
    eng = []
    for i in range(len(meta)):
        idx = np.where(engaged(sig[i]))[0]
        eng.append((int(idx[0]), int(idx[-1]) + 1) if len(idx) else (0, TARGET))
    encs = ["patchtst", "chronos"] if a.enc == "both" else [a.enc]
    res = {}
    for fold, ti in enumerate(mixed):
        other = [j for j in mixed if j != ti]
        mtb = [(j, s0, s1, b) for j in other for (b, s0, s1) in reg_blocks(meta[j])]
        print("fold%d test=%s train_mixed_blocks=%d" % (fold, meta[ti]["id"], len(mtb)))
        models = [train_one(e, tr, mtb, sig, meta, eng, a.epochs, a.seed + fold) for e in encs]
        with torch.no_grad():
            ok = 0
            for i in va:
                pr = np.mean([probs_of(m, sig, i, *eng[i]) for m in models], axis=0)
                ok += ALPH[int(pr.argmax())] == meta[i]["base"]
            t1 = ok / len(va)
        m = meta[ti]
        a0, a1 = eng[ti]
        y = sig[ti][2, a0:a1].astype(float)
        segs = seg_dp(y, kmax=6, min_len=30)
        kbic = min(segs, key=lambda k: segs[k][2])
        entry = {"K_bic": kbic, "T1": t1}
        for tag, K in (("bic", kbic), ("K4", 4)):
            if K not in segs:
                continue
            cuts = segs[K][0]
            bases, probs = [], []
            for u in range(len(cuts) - 1):
                s0 = max(0, a0 + cuts[u] - 8)
                s1 = min(TARGET, a0 + cuts[u + 1] + 8)
                pr = np.mean([probs_of(md, sig, ti, s0, s1) for md in models], axis=0)
                bases.append(ALPH[int(pr.argmax())])
                probs.append(round(float(pr.max()), 3))
            order = "".join(bases)
            pred40 = "".join(b * 10 for b in order)
            rl = m["register_labels"]
            first = next((b for b in rl if b != "-"), None)
            truth = m["seq"] if first == m["seq"][0] else m["seq"][::-1]
            torder = truth[::10]
            cer = lev(pred40, truth) / len(truth)
            cer_rev = lev(pred40[::-1], truth) / len(truth)
            co = lev(order, torder) / max(1, len(torder))
            cor = lev(order[::-1], torder) / max(1, len(torder))
            lab800 = np.array([C2I.get(b, -100) for b in rl])
            assign = np.full(TARGET, -100, dtype=np.int64)
            for u in range(len(cuts) - 1):
                assign[a0 + cuts[u]:a0 + cuts[u + 1]] = C2I[bases[u]]
            msk = lab800 != -100
            t3 = float((assign[msk] == lab800[msk]).mean())
            entry[tag] = dict(order=order, conf=probs, CER=cer, CER_rev=cer_rev,
                              CER_bestdir=min(cer, cer_rev), CER_order=co,
                              CER_order_rev=cor, truth_order=torder, T3=t3)
            print("  %s %s K=%d order=%s conf=%s CER=%.3f CERrev=%.3f CERord=%.3f CERordrev=%.3f T3=%.3f" % (
                m["id"], tag, K, order, probs, cer, cer_rev, co, cor, t3))
        res["fold%d_%s" % (fold, m["id"])] = entry
        print("  T1=%.3f" % t1)
    with open("runs_alt/fm_block3.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("saved runs_alt/fm_block3.json")


if __name__ == "__main__":
    main()