# -*- coding: utf-8 -*-
"""FM block-reader: matches the ACTUAL data structure (4-block copolymer strands,
e.g. A10C10G10T10), not per-base alternation.
Pipeline: (1) piecewise-constant DP change-point segmentation of the engaged
depth trace, K chosen by BIC (blind) and K=4 (oracle-K variant);
(2) each segment -> mean-pooled PatchTST patch embeddings over its span ->
identity head (trained on random engaged spans of homopolymers, so block-level
pooling matches training distribution); (3) segment bases in time order = read
order. Metrics: T1 whole-trace identity; T2 CER of expanded block read vs
register-direction truth AND best-direction truth (direction is physically
ambiguous blind); T3 per-sample base from segmentation vs register truth.
Run: HF_ENDPOINT=https://hf-mirror.com C:/Users/huangnian/pyenvs/esm/Scripts/python.exe fm_block.py
"""
import os, json, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from bench_new import load, engaged, lev, ALPH, C2I, TARGET

MID = "ibm-granite/granite-timeseries-patchtst"
CTX = 512
PATCH = 12
NPATCH = (CTX - PATCH) // PATCH + 1


def resample512(sig):
    x0 = np.linspace(0, 1, TARGET)
    x1 = np.linspace(0, 1, CTX)
    return np.stack([np.interp(x1, x0, sig[c]) for c in range(3)]).astype(np.float32)


class FMBlock(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = AutoConfig.from_pretrained(MID)
        cfg.num_input_channels = 3
        cfg.context_length = CTX
        self.enc = AutoModel.from_pretrained(MID, config=cfg)
        d = cfg.d_model
        self.ident = nn.Sequential(nn.Linear(3 * d, 128), nn.GELU(), nn.Linear(128, 4))

    def embed(self, x):                          # (B,512,3) -> (B,P,3d)
        h = self.enc(past_values=x).last_hidden_state[:, :, 1:, :]
        B, C, P, d = h.shape
        return h.permute(0, 2, 1, 3).reshape(B, P, C * d)

    def forward(self, x, p0, p1):                # pooled span -> logits
        hc = self.embed(x)
        pool = torch.stack([hc[b, p0[b]:p1[b]].mean(0) for b in range(hc.shape[0])])
        return self.ident(pool)


def seg_dp(y, kmax=6, min_len=30):
    n = len(y)
    S1 = np.concatenate([[0], np.cumsum(y)])
    S2 = np.concatenate([[0], np.cumsum(y * y)])

    def sse(i, j):
        cnt = j - i
        return S2[j] - S2[i] - (S1[j] - S1[i]) ** 2 / cnt

    INF = 1e18
    dp = np.full((kmax + 1, n + 1), INF)
    bp = np.zeros((kmax + 1, n + 1), dtype=np.int64)
    dp[0, 0] = 0.0
    for k in range(1, kmax + 1):
        for j in range(k * min_len, n + 1):
            i = np.arange((k - 1) * min_len, j - min_len + 1)
            if len(i) == 0:
                continue
            cand = dp[k - 1, i] + sse(i, j)
            t = int(np.argmin(cand))
            dp[k, j] = cand[t]
            bp[k, j] = i[t]
    out = {}
    for k in range(1, kmax + 1):
        if dp[k, n] >= INF / 2:
            continue
        cuts, j = [n], n
        for kk in range(k, 0, -1):
            j = int(bp[kk, j])
            cuts.append(j)
        cuts = cuts[::-1]
        sse_v = dp[k, n]
        bic = n * np.log(max(sse_v, 1e-9) / n) + 2 * k * np.log(n)
        out[k] = (cuts, float(sse_v), float(bic))
    return out


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
    X = np.stack([resample512(sig[i]).T for i in range(len(meta))])
    xt = torch.from_numpy(X)
    # engaged patch spans per trace
    spans = []
    for i in range(len(meta)):
        em = engaged(sig[i])
        idx = np.where(em)[0]
        if len(idx) == 0:
            spans.append((0, NPATCH))
            continue
        e0, e1 = idx[0] * CTX / TARGET, (idx[-1] + 1) * CTX / TARGET
        spans.append((max(0, int(e0 // PATCH)), min(NPATCH, int(np.ceil(e1 / PATCH)))))
    model = FMBlock()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-5)
    yb = torch.tensor([C2I[meta[i]["base"]] for i in tr])
    ti = torch.tensor(tr, dtype=torch.long)
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(tr))
        tot, nb = 0.0, 0
        for s in range(0, len(tr), 32):
            bi = perm[s:s + 32]
            p0, p1 = [], []
            for j in bi:
                i = tr[int(j)]
                q0, q1 = spans[i]
                L = q1 - q0
                if L > 8 and random.random() < 0.8:
                    ln = random.randint(6, min(30, L))
                    st = random.randint(q0, q1 - ln)
                    p0.append(st)
                    p1.append(st + ln)
                else:
                    p0.append(q0)
                    p1.append(q1)
            xb = xt[ti[bi]].clone()
            xb = xb * (0.9 + 0.2 * torch.rand(len(bi), 1, 1)) + 0.02 * torch.randn(len(bi), 1, 1)
            lg = model(xb, torch.tensor(p0), torch.tensor(p1))
            loss = F.cross_entropy(lg, yb[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
        sched.step()
        if (ep + 1) % 10 == 0:
            print("ep %d loss=%.4f" % (ep + 1, tot / nb))
    model.eval()
    res = {}
    with torch.no_grad():
        ok = 0
        for i in va:
            q0, q1 = spans[i]
            lg = model(xt[i:i + 1], torch.tensor([q0]), torch.tensor([q1]))
            ok += ALPH[int(lg.argmax())] == meta[i]["base"]
        res["T1"] = ok / len(va)
        print("T1=%.3f" % res["T1"])
        for i in mixed:
            m = meta[i]
            em = engaged(sig[i])
            idx = np.where(em)[0]
            a0, a1 = int(idx[0]), int(idx[-1]) + 1
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
                    s0, s1 = a0 + cuts[u], a0 + cuts[u + 1]
                    p0 = max(0, int(s0 * CTX / TARGET // PATCH))
                    p1 = min(NPATCH, int(np.ceil(s1 * CTX / TARGET / PATCH)))
                    if p1 <= p0:
                        p1 = p0 + 1
                    lg = model(xt[i:i + 1], torch.tensor([p0]), torch.tensor([p1]))
                    pr = F.softmax(lg[0], -1).numpy()
                    bases.append(ALPH[int(pr.argmax())])
                    probs.append(round(float(pr.max()), 3))
                order = "".join(bases)
                pred40 = "".join(b * 10 for b in order)
                rl = m["register_labels"]
                first = next((b for b in rl if b != "-"), None)
                truth = m["seq"] if first == m["seq"][0] else m["seq"][::-1]
                cer = lev(pred40, truth) / len(truth)
                cer_rev = lev(pred40[::-1], truth) / len(truth)
                torder = truth[::10]
                cer_order = lev(order, torder) / max(1, len(torder))
                cer_order_rev = lev(order[::-1], torder) / max(1, len(torder))
                lab800 = np.array([C2I.get(b, -100) for b in rl])
                assign = np.full(TARGET, -100, dtype=np.int64)
                for u in range(len(cuts) - 1):
                    assign[a0 + cuts[u]:a0 + cuts[u + 1]] = C2I[bases[u]]
                msk = lab800 != -100
                t3 = float((assign[msk] == lab800[msk]).mean())
                entry[tag] = dict(order=order, conf=probs, CER=cer, CER_rev=cer_rev,
                                  CER_bestdir=min(cer, cer_rev), T3=t3,
                                  CER_order=cer_order, CER_order_rev=cer_order_rev,
                                  truth_order=torder,
                                  cuts=[int(c) for c in cuts])
                print("%s %s K=%d order=%s conf=%s CER=%.3f CERrev=%.3f CERord=%.3f CERordrev=%.3f T3=%.3f" % (
                    m["id"], tag, K, order, probs, cer, cer_rev, cer_order, cer_order_rev, t3))
            res[m["id"]] = entry
    with open("runs_alt/fm_block.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("saved runs_alt/fm_block.json")


if __name__ == "__main__":
    main()