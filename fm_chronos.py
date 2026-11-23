# -*- coding: utf-8 -*-
"""Second foundation model: chronos-bolt-mini (T5 encoder, ~9M params) as a
block reader, same distribution-matched crop protocol as fm_block2.
Critical adaptation: chronos' per-trace InstanceNorm would subtract each crop's
mean depth and divide by its std, ERASING the depth magnitude that encodes base
identity. Replace it with a FIXED global scale (0.15) so relative depths across
traces survive while inputs stay in the pretrained numeric range.
Univariate model: the 3 blockade channels are fed as 3 batch items and their
logits averaged. Partial finetune: last 2 encoder blocks + patch embedding + head.
Run: HF_ENDPOINT=https://hf-mirror.com C:/Users/huangnian/pyenvs/esm/Scripts/python.exe fm_chronos.py
"""
import os, json, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from chronos import ChronosBoltPipeline
from bench_new import load, engaged, lev, ALPH, C2I, TARGET
from fm_block import seg_dp
from fm_block2 import crop512

MID = "amazon/chronos-bolt-mini"
CTX = 512
FIXED_SCALE = 0.15


class FixedNorm(nn.Module):
    def forward(self, x, loc_scale=None):
        loc = torch.zeros(x.shape[0], 1, dtype=torch.float32)
        scale = torch.full((x.shape[0], 1), FIXED_SCALE, dtype=torch.float32)
        return x.to(torch.float32) / FIXED_SCALE, (loc, scale)


class ChronosBlock(nn.Module):
    def __init__(self, unfreeze_last=2):
        super().__init__()
        pipe = ChronosBoltPipeline.from_pretrained(MID, device_map="cpu", dtype=torch.float32)
        self.m = pipe.model
        self.m.instance_norm = FixedNorm()
        self.use_reg = bool(self.m.chronos_config.use_reg_token)
        for p in self.m.parameters():
            p.requires_grad = False
        for p in self.m.input_patch_embedding.parameters():
            p.requires_grad = True
        blocks = self.m.encoder.block
        for bl in blocks[-unfreeze_last:]:
            for p in bl.parameters():
                p.requires_grad = True
        d = self.m.config.d_model
        self.ident = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 4))

    def forward(self, x):                        # (B,512,3)
        B = x.shape[0]
        xu = x.permute(0, 2, 1).reshape(B * 3, CTX)
        h = self.m.encode(xu)[0]
        if self.use_reg:
            h = h[:, :-1]
        pool = h.mean(1)
        lg = self.ident(pool).reshape(B, 3, 4).mean(1)
        return lg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
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
    model = ChronosBlock()
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable params(M)=%.2f" % (ntr / 1e6))
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-5)
    for ep in range(a.epochs):
        model.train()
        random.shuffle(tr)
        tot, nb = 0.0, 0
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
            lg = model(xb)
            loss = F.cross_entropy(lg, torch.tensor(ys))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
        sched.step()
        if (ep + 1) % 5 == 0:
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
    with open("runs_alt/fm_chronos.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("saved runs_alt/fm_chronos.json")


if __name__ == "__main__":
    main()