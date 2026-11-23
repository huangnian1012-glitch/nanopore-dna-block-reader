# -*- coding: utf-8 -*-
"""FM block-reader v4: two remaining cheap levers.
(1) segmentation cost summed over ALL 3 blockade channels (v2/v3 used ch2 only,
    which can miss boundaries visible only in ch0/ch1);
(2) identity crop = middle 50% CORE of each segment (v2/v3 used the whole
    segment +/-8 samples, contaminating the crop with transition regions).
Encoder: chronos-bolt-mini (best honest T1) or ensemble with PatchTST.
Run: HF_ENDPOINT=https://hf-mirror.com C:/Users/huangnian/pyenvs/esm/Scripts/python.exe fm_block4.py --enc chronos
"""
import os, json, argparse, random
import numpy as np
import torch
import torch.nn.functional as F
from bench_new import load, engaged, lev, ALPH, C2I, TARGET
from fm_block2 import crop512, FMCrop
from fm_chronos import ChronosBlock


def seg_dp_mc(Y, kmax=6, min_len=30, pen=2.0):
    n = Y.shape[0]
    S1 = np.concatenate([np.zeros((1, Y.shape[1])), np.cumsum(Y, axis=0)])
    S2 = np.concatenate([np.zeros((1, Y.shape[1])), np.cumsum(Y * Y, axis=0)])

    def sse(i, j):
        cnt = (j - i).astype(float)
        d1 = S1[j] - S1[i]
        return (S2[j] - S2[i] - d1 * d1 / cnt[:, None]).sum(axis=1)

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
        bic = n * np.log(max(dp[k, n], 1e-9) / n) + pen * k * np.log(n)
        out[k] = (cuts, float(dp[k, n]), float(bic))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="chronos", choices=["patchtst", "chronos", "both"])
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
    encs = ["patchtst", "chronos"] if a.enc == "both" else [a.enc]
    models = []
    for e in encs:
        model = FMCrop() if e == "patchtst" else ChronosBlock()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=3e-4 if e == "patchtst" else 1e-4)
        epochs = a.epochs if e == "patchtst" else a.epochs
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
        bs = 32 if e == "patchtst" else 16
        for ep in range(epochs):
            model.train()
            random.shuffle(tr)
            tot, nb = 0.0, 0
            for s in range(0, len(tr), bs):
                batch = tr[s:s + bs]
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
                tot += float(loss.item())
                nb += 1
            sched.step()
            if (ep + 1) % 10 == 0:
                print("  [%s] ep %d loss=%.4f" % (e, ep + 1, tot / nb))
        model.eval()
        models.append(model)
        torch.save(model.state_dict(), "runs_alt/fm_block4_%s.pt" % e)
        print("  saved runs_alt/fm_block4_%s.pt" % e)

    def probs_of(i, s0, s1):
        x = torch.from_numpy(crop512(sig[i], s0, s1).T)[None]
        with torch.no_grad():
            return np.mean([F.softmax(m(x)[0], -1).numpy() for m in models], axis=0)

    res = {}
    ok = 0
    for i in va:
        pr = probs_of(i, *eng[i])
        ok += ALPH[int(pr.argmax())] == meta[i]["base"]
    res["T1"] = ok / len(va)
    print("T1=%.3f" % res["T1"])
    for i in mixed:
        m = meta[i]
        a0, a1 = eng[i]
        Y = sig[i][:, a0:a1].T.astype(float)
        segs = seg_dp_mc(Y, kmax=6, min_len=30)
        kbic = min(segs, key=lambda k: segs[k][2])
        rl = m["register_labels"]
        first = next((b for b in rl if b != "-"), None)
        truth = m["seq"] if first == m["seq"][0] else m["seq"][::-1]
        torder = truth[::10]
        entry = {"K_bic": kbic, "truth_order": torder}
        for tag, K in (("bic", kbic), ("K4", 4)):
            if K not in segs:
                continue
            cuts = segs[K][0]
            bases, probs = [], []
            for u in range(len(cuts) - 1):
                s0, s1 = a0 + cuts[u], a0 + cuts[u + 1]
                c0, c1 = s0 + (s1 - s0) // 4, s1 - (s1 - s0) // 4
                pr = probs_of(i, c0, c1)
                bases.append(ALPH[int(pr.argmax())])
                probs.append(round(float(pr.max()), 3))
            order = "".join(bases)
            pred40 = "".join(b * 10 for b in order)
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
                              CER_order_rev=cor, T3=t3, cuts=[int(c + a0) for c in cuts])
            print("%s %s K=%d order=%s conf=%s CER=%.3f CERrev=%.3f CERord=%.3f CERordrev=%.3f T3=%.3f cuts=%s" % (
                m["id"], tag, K, order, probs, cer, cer_rev, co, cor, t3, [int(c + a0) for c in cuts]))
        res[m["id"]] = entry
    with open("runs_alt/fm_block4.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("saved runs_alt/fm_block4.json")


if __name__ == "__main__":
    main()