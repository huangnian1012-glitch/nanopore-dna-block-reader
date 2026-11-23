# -*- coding: utf-8 -*-
"""Time-series foundation-model fine-tune (user authorized network).
Model: ibm-granite/granite-timeseries-patchtst (PatchTST, 0.6M params, CPU-ok),
downloaded via hf-mirror.com (HF direct unreachable). Config adapted to 3 channels,
context 512 (pretrained pos-enc buffer fixes length); our 800-sample trace is
resampled to 512. Heads: per-patch frame head (4 bases) + pooled identity head.
Two regimes: full fine-tune vs linear-probe (frozen encoder).
Tasks vs benchmark: T1 identity, T2 mixed blind read CER, T3 mixed frame acc.
Run: HF_ENDPOINT=https://hf-mirror.com C:/Users/huangnian/pyenvs/esm/Scripts/python.exe fm_finetune.py --epochs 30
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
NPATCH = (CTX - PATCH) // PATCH + 1   # 42


def resample512(sig):
    x0 = np.linspace(0, 1, TARGET)
    x1 = np.linspace(0, 1, CTX)
    return np.stack([np.interp(x1, x0, sig[c]) for c in range(3)]).astype(np.float32)


def patch_labels(lab_ctx):
    out = []
    for p in range(NPATCH):
        seg = lab_ctx[p * PATCH:(p + 1) * PATCH]
        seg = seg[seg != -100]
        if len(seg) == 0:
            out.append(-100)
        else:
            vals, cnts = np.unique(seg, return_counts=True)
            out.append(int(vals[np.argmax(cnts)]))
    return np.array(out, dtype=np.int64)


class FMReader(nn.Module):
    def __init__(self, freeze=False):
        super().__init__()
        cfg = AutoConfig.from_pretrained(MID)
        cfg.num_input_channels = 3
        cfg.context_length = CTX
        self.enc = AutoModel.from_pretrained(MID, config=cfg)
        if freeze:
            for p in self.enc.parameters():
                p.requires_grad = False
        d = cfg.d_model
        self.frame = nn.Linear(d, 4)
        self.ident = nn.Linear(d, 4)

    def forward(self, x):           # x (B,512,3)
        h = self.enc(past_values=x).last_hidden_state   # (B,3,P+1,d)
        h = h[:, :, 1:, :].mean(1)                      # drop cls, mean channels -> (B,P,d)
        return self.frame(h), self.ident(h.mean(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs("runs_alt", exist_ok=True)
    sig, meta = load()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    homo = [i for i, m in enumerate(meta) if not m["mixed"] and m["base"] in C2I]
    random.shuffle(homo)
    nval = max(1, int(len(homo) * 0.2))
    va, tr = homo[:nval], homo[nval:]
    X = np.stack([resample512(sig[i]).T for i in range(len(meta))])   # (M,512,3)
    Yf, Yi = [], []
    for i, m in enumerate(meta):
        if m["mixed"]:
            lab800 = np.array([C2I.get(b, -100) for b in (m.get("register_labels") or [])])
            if len(lab800) != TARGET:
                lab800 = np.full(TARGET, -100)
        else:
            em = engaged(sig[i])
            lab800 = np.full(TARGET, -100, dtype=np.int64)
            lab800[em] = C2I[m["base"]]
        lab512 = np.interp(np.linspace(0, TARGET - 1, CTX), np.arange(TARGET), lab800.astype(float))
        lab512 = np.where(lab512 < 0, -100, np.round(lab512)).astype(np.int64)
        Yf.append(patch_labels(lab512))
        Yi.append(C2I.get(m["base"], -100))
    Yf = np.stack(Yf)
    Yi = np.array(Yi)
    xt = torch.from_numpy(X)
    yft = torch.from_numpy(Yf)
    yit = torch.from_numpy(Yi)
    res = {}
    for freeze in [False, True]:
        model = FMReader(freeze=freeze)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=1e-4 if not freeze else 1e-3)
        n = len(tr)
        trt = torch.tensor(tr, dtype=torch.long)
        for ep in range(a.epochs):
            model.train()
            perm = torch.randperm(n)
            tot, nb = 0.0, 0
            for s in range(0, n, 16):
                bi = perm[s:s + 16]
                gi = trt[bi]
                fl, il = model(xt[gi])
                lf = F.cross_entropy(fl.transpose(1, 2), yft[gi], ignore_index=-100)
                li = F.cross_entropy(il, yit[gi], ignore_index=-100)
                loss = lf + li
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss.item())
                nb += 1
            if (ep + 1) % 10 == 0:
                print("freeze=%s ep %d loss=%.4f" % (freeze, ep + 1, tot / nb))
        model.eval()
        with torch.no_grad():
            ok = nv = 0
            for i in va:
                _, il = model(xt[i:i + 1])
                pred = ALPH[int(il.argmax())]
                ok += pred == meta[i]["base"]
                nv += 1
            acc = ok / max(1, nv)
            accs, cers, preds = [], [], []
            for i, m in enumerate(meta):
                if not m["mixed"] or not m.get("register_labels"):
                    continue
                fl, _ = model(xt[i:i + 1])
                pr = fl[0].argmax(-1).numpy()
                msk = Yf[i] != -100
                accs.append(float((pr[msk] == Yf[i][msk]).mean()) if msk.any() else 0.0)
                seq = "".join(ALPH[c] for c in pr)
                blocks = []
                cur = None
                for c in seq:
                    if c != cur:
                        blocks.append(c)
                        cur = c
                pred = "".join(blocks)
                preds.append(pred)
                first = next((b for b in m["register_labels"] if b != "-"), None)
                truth = m["seq"] if first == m["seq"][0] else m["seq"][::-1]
                cers.append(lev(pred, truth) / max(1, len(truth)))
        key = "fm_probe" if freeze else "fm_finetune"
        res[key] = dict(T1=acc, T3=float(np.mean(accs)), T2=float(np.mean(cers)), mixed_pred=preds)
        print("%s T1=%.3f T3=%.3f T2_CER=%.3f preds=%s" % (key, acc, np.mean(accs), np.mean(cers), preds))
    with open("runs_alt/fm_finetune.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("saved runs_alt/fm_finetune.json")


if __name__ == "__main__":
    main()