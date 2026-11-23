# -*- coding: utf-8 -*-
"""Foundation-model finetune v2: fixes + any-method accuracy push.
Fix vs v1: v1 frame head sat on patch granularity (~21 engaged patches for a
40-base strand) -> resolution floor CER>=0.475 even with perfect predictions.
v2 interpolates patch embeddings back to 512 samples before the frame head.
Additions: (a) leave-one-mixed-out training WITH register truth (the only real
mixed labels we have), (b) augmentation (scale/offset/noise, roll for homo),
(c) per-channel concat head instead of channel-mean, (d) cosine lr, 60 epochs,
(e) structured Viterbi decode with a dwell prior, (f) T2b = CER minimised over
read direction and cyclic phase (separates 'base pattern recovered' from
'direction/phase unknown').
Run: HF_ENDPOINT=https://hf-mirror.com C:/Users/huangnian/pyenvs/esm/Scripts/python.exe fm_finetune2.py
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


def lab512_of(m, sig_i):
    if m["mixed"]:
        lab800 = np.array([C2I.get(b, -100) for b in (m.get("register_labels") or [])])
        if len(lab800) != TARGET:
            lab800 = np.full(TARGET, -100)
    else:
        em = engaged(sig_i)
        lab800 = np.full(TARGET, -100, dtype=np.int64)
        lab800[em] = C2I[m["base"]]
    v = np.interp(np.linspace(0, TARGET - 1, CTX), np.arange(TARGET), lab800.astype(float))
    return np.where(v < 0, -100, np.round(v)).astype(np.int64)


class FMReader2(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = AutoConfig.from_pretrained(MID)
        cfg.num_input_channels = 3
        cfg.context_length = CTX
        self.enc = AutoModel.from_pretrained(MID, config=cfg)
        d = cfg.d_model
        self.frame = nn.Sequential(nn.Linear(3 * d, 128), nn.GELU(), nn.Linear(128, 4))
        self.ident = nn.Sequential(nn.Linear(3 * d, 128), nn.GELU(), nn.Linear(128, 4))

    def forward(self, x):                       # (B,512,3)
        h = self.enc(past_values=x).last_hidden_state   # (B,3,P+1,d)
        h = h[:, :, 1:, :]                              # (B,3,P,d)
        B, C, P, d = h.shape
        hc = h.permute(0, 2, 1, 3).reshape(B, P, C * d)         # (B,P,3d)
        hf = F.interpolate(hc.transpose(1, 2), size=CTX, mode="linear",
                           align_corners=False).transpose(1, 2)  # (B,512,3d)
        return self.frame(hf), self.ident(hc.mean(1))


def augment(x, roll_ok):
    x = x * (0.9 + 0.2 * torch.rand(x.shape[0], 1, 1))
    x = x + 0.03 * torch.randn(x.shape[0], 1, 1)
    x = x + 0.01 * torch.randn_like(x)
    if roll_ok:
        for b in range(x.shape[0]):
            s = int(torch.randint(-20, 21, (1,)))
            x[b] = torch.roll(x[b], s, dims=0)
    return x


def collapse(seq, min_len=3):
    blocks, cur, n = [], None, 0
    for c in seq:
        if c == cur:
            n += 1
        else:
            if cur is not None and n >= min_len:
                blocks.append(cur)
            cur, n = c, 1
    if cur is not None and n >= min_len:
        blocks.append(cur)
    return "".join(blocks)


def viterbi(logp, p_stay):
    T, K = logp.shape
    stay = np.log(p_stay)
    sw = np.log((1 - p_stay) / (K - 1))
    dp = np.full((T, K), -1e9)
    bp = np.zeros((T, K), dtype=np.int64)
    dp[0] = logp[0]
    for t in range(1, T):
        for k in range(K):
            same = dp[t - 1, k] + stay
            other = dp[t - 1].max() + sw
            j = int(dp[t - 1].argmax())
            if same >= other:
                dp[t, k] = same + logp[t, k]
                bp[t, k] = k
            else:
                dp[t, k] = other + logp[t, k]
                bp[t, k] = j
    out = [int(dp[-1].argmax())]
    for t in range(T - 1, 0, -1):
        out.append(int(bp[t, out[-1]]))
    return np.array(out[::-1])


def cer_best_phase(pred, truth):
    best = 1e9
    L = len(truth)
    for rev in (pred, pred[::-1]):
        for s in range(L):
            t = truth[s:] + truth[:s]
            best = min(best, lev(rev, t) / max(1, L))
    return best


def train_eval(tr_idx, va_idx, test_mixed, X, Y, Yi, meta, epochs, seed, p_stay):
    random.seed(seed)
    torch.manual_seed(seed)
    model = FMReader2()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    xt = torch.from_numpy(X)
    yt = torch.from_numpy(Y)
    yit = torch.from_numpy(Yi)
    n = len(tr_idx)
    trt = torch.tensor(tr_idx, dtype=torch.long)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot, nb = 0.0, 0
        for s in range(0, n, 16):
            bi = perm[s:s + 16]
            gi = trt[bi]
            xb = xt[gi].clone()
            roll_ok = all(not meta[int(i)]["mixed"] for i in gi)
            xb = augment(xb, roll_ok)
            fl, il = model(xb)
            loss = F.cross_entropy(fl.transpose(1, 2), yt[gi], ignore_index=-100) \
                + F.cross_entropy(il, yit[gi], ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
        sched.step()
        if (ep + 1) % 20 == 0:
            print("  ep %d loss=%.4f" % (ep + 1, tot / nb))
    model.eval()
    out = {}
    with torch.no_grad():
        ok = nv = 0
        for i in va_idx:
            _, il = model(xt[i:i + 1])
            ok += ALPH[int(il.argmax())] == meta[i]["base"]
            nv += 1
        out["T1"] = ok / max(1, nv)
        if test_mixed is not None:
            i = test_mixed
            m = meta[i]
            fl, _ = model(xt[i:i + 1])
            lp = fl[0].log_softmax(-1).numpy()
            pr = lp.argmax(-1)
            msk = Y[i] != -100
            out["T3"] = float((pr[msk] == Y[i][msk]).mean()) if msk.any() else 0.0
            first = next((b for b in m["register_labels"] if b != "-"), None)
            truth = m["seq"] if first == m["seq"][0] else m["seq"][::-1]
            pred = collapse("".join(ALPH[c] for c in pr))
            pv = collapse("".join(ALPH[c] for c in viterbi(lp, p_stay)))
            out["T2"] = lev(pred, truth) / len(truth)
            out["T2_viterbi"] = lev(pv, truth) / len(truth)
            out["T2b"] = cer_best_phase(pred, truth)
            out["T2b_viterbi"] = cer_best_phase(pv, truth)
            out["pred"] = pred
            out["pred_viterbi"] = pv
            out["truth"] = truth
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs("runs_alt", exist_ok=True)
    sig, meta = load()
    mixed = [i for i, m in enumerate(meta) if m["mixed"] and m.get("register_labels")]
    homo = [i for i, m in enumerate(meta) if not m["mixed"] and m["base"] in C2I]
    random.seed(a.seed)
    random.shuffle(homo)
    nval = max(1, int(len(homo) * 0.2))
    va, htr = homo[:nval], homo[nval:]
    X = np.stack([resample512(sig[i]).T for i in range(len(meta))])
    Y = np.stack([lab512_of(m, sig[i]) for i, m in enumerate(meta)])
    Yi = np.array([C2I.get(m["base"], -100) for m in meta])
    # dwell prior from register block lengths (in 512-sample units)
    lens = []
    for i in mixed:
        rl = meta[i]["register_labels"]
        cur, n = None, 0
        for b in rl:
            if b == cur:
                n += 1
            else:
                if cur is not None and cur != "-" and n > 0:
                    lens.append(n * CTX / TARGET)
                cur, n = b, 1
        if cur is not None and cur != "-" and n > 0:
            lens.append(n * CTX / TARGET)
    mean_len = float(np.mean(lens)) if lens else 6.0
    p_stay = float(np.clip(1 - 1 / max(2.0, mean_len), 0.5, 0.98))
    print("mean block len(512 units)=%.2f p_stay=%.3f" % (mean_len, p_stay))
    res = {}
    print("mode homo_only_aug")
    res["homo_only_aug"] = train_eval(htr, va, None, X, Y, Yi, meta, a.epochs, a.seed, p_stay)
    print("  T1=%.3f" % res["homo_only_aug"]["T1"])
    for fold, ti in enumerate(mixed):
        other = [j for j in mixed if j != ti]
        tr = htr + other
        print("mode loo fold%d test=%s" % (fold, meta[ti]["id"]))
        r = train_eval(tr, va, ti, X, Y, Yi, meta, a.epochs, a.seed + fold + 1, p_stay)
        res["loo_fold%d" % fold] = r
        print("  T1=%.3f T3=%.3f T2=%.3f T2v=%.3f T2b=%.3f T2bv=%.3f pred=%s" % (
            r["T1"], r["T3"], r["T2"], r["T2_viterbi"], r["T2b"], r["T2b_viterbi"], r["pred"]))
    with open("runs_alt/fm_finetune2.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("saved runs_alt/fm_finetune2.json")


if __name__ == "__main__":
    main()