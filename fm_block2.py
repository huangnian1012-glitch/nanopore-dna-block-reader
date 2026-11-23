# -*- coding: utf-8 -*-
"""FM block-reader v2: distribution-matched block identity.
v1 pooled a sub-span of patch embeddings inside the full-trace forward pass;
attention context drowned local base identity -> all-A collapse.
v2 crops each segment and RESAMPLES IT TO A FULL 512-sample trace before the
encoder, exactly matching training (random engaged crops of homopolymers
resampled to 512). Segmentation = piecewise-constant DP + BIC (blind) / K=4.
Metrics: T1 whole-engaged identity; T2 / T2_order CER vs register-direction and
reverse-direction truth; T3 per-sample base from segmentation.
Run: HF_ENDPOINT=https://hf-mirror.com C:/Users/huangnian/pyenvs/esm/Scripts/python.exe fm_block2.py
"""
import os, json, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from bench_new import load, engaged, lev, ALPH, C2I, TARGET
from fm_block import seg_dp, MID, CTX, NPATCH


def crop512(sig3, s0, s1):
    seg = sig3[:, s0:s1]
    x0 = np.linspace(0, 1, seg.shape[1])
    x1 = np.linspace(0, 1, CTX)
    return np.stack([np.interp(x1, x0, seg[c]) for c in range(3)]).astype(np.float32)


class FMCrop(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = AutoConfig.from_pretrained(MID)
        cfg.num_input_channels = 3
        cfg.context_length = CTX
        self.enc = AutoModel.from_pretrained(MID, config=cfg)
        d = cfg.d_model
        self.ident = nn.Sequential(nn.Linear(3 * d, 128), nn.GELU(), nn.Linear(128, 4))

    def forward(self, x):                       # (B,512,3) whole-crop trace
        h = self.enc(past_values=x).last_hidden_state[:, :, 1:, :]
        B, C, P, d = h.shape
        hc = h.permute(0, 2, 1, 3).reshape(B, P, C * d)
        return self.ident(hc.mean(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs("runs_alt", exist_ok=True)
    sig, meta = load()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    homo = [i for i, m in enumerate(meta) if not m["mixed"] and m["base"] in C2I]
    random.shuffle(homo)
    nval = max(1, int(len(homo) * 0.2))
    va, tr = homo[:nval], homo[nval:]
    mixed = [i for i, m in enumerate(meta) if m["mixed"] and m.get("register_labels")]
    eng = []
    for i in range(len(meta)):
        idx = np.where(engaged(sig[i]))[0]
        eng.append((int(idx[0]), int(idx[-1]) + 1) if len(idx) else (0, TARGET))
    model = FMCrop()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-5)
    for ep in range(a.epochs):
        model.train()
        random.shuffle(tr)
        tot, nb = 0.0, 0
        for s in range(0, len(tr), 32):
            batch = tr[s:s + 32]
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
            yb = torch.tensor(ys)
            lg = model(xb)
            loss = F.cross_entropy(lg, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
        sched.step()
        if (ep + 1) % 10 == 0:
            print("ep %d loss=%.4f" % (ep + 1, tot / nb))
    model.eval()

    def ident_of(i, s0, s1):
        x = torch.from_numpy(crop512(sig[i], s0, s1).T)[None]
        with torch.no_grad():
            pr = F.softmax(model(x)[0], -1).numpy()
        return ALPH[int(pr.argmax())], pr

    res = {}
    ok = 0
    for i in va:
        b, _ = ident_of(i, *eng[i])
        ok += b == meta[i]["base"]
    res["T1"] = ok / len(va)
    print("T1=%.3f" % res["T1"])
    for i in mixed:
        m = meta[i]
        a0, a1 = eng[i]
        y = sig[i][2, a0:a1].astype(float)
        segs = seg_dp(y, kmax=6, min_len=30)
        kbic = min(segs, key=lambda k: segs[k][2])
        entry = {"K_bic": kbic}
        for tag, K in (("bic", kbic), ("K4", 4)):
            if K not in segs:
                continue
            cuts = segs[K][0]
            bases, probs = [], []
            for u in range(len(cuts) - 1):
                s0 = max(0, a0 + cuts[u] - 8)
                s1 = min(TARGET, a0 + cuts[u + 1] + 8)
                b, pr = ident_of(i, s0, s1)
                bases.append(b)
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
                              CER_order_rev=cor, truth_order=torder, T3=t3,
                              cuts=[int(c) for c in cuts])
            print("%s %s K=%d order=%s conf=%s CER=%.3f CERrev=%.3f CERord=%.3f CERordrev=%.3f T3=%.3f" % (
                m["id"], tag, K, order, probs, cer, cer_rev, co, cor, t3))
        res[m["id"]] = entry
    with open("runs_alt/fm_block2.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("saved runs_alt/fm_block2.json")


if __name__ == "__main__":
    main()