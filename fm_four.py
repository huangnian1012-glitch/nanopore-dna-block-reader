# -*- coding: utf-8 -*-
"""Incorporate the 3 new labelled 4-block 40-mers (Desktop/test) into the
per-block identity reader.
Baseline: deployed runs_alt/fm_block4_chronos.pt, forced K=4, best-direction
per-block top1/top2 acc on the 3 new + 2 janus strands.
Finetune: homopolymer crops + janus register-core crops (hard labels) + new
strands' forced-K4 core crops with DIRECTION-AGNOSTIC soft labels
(0.5/0.5 over {fwd[i], rev[i]}), leave-one-out over the 3 new strands.
If LOO beats baseline, train final on all and save runs_four/ident_four.pt.
Run: HF_ENDPOINT=https://hf-mirror.com C:/Users/huangnian/pyenvs/esm/Scripts/python.exe fm_four.py --epochs 30
"""
import os, json, argparse, random
import numpy as np
import torch
import torch.nn.functional as F
from bench_new import load, engaged, ALPH, C2I, TARGET
from recognize import load_current
from fm_block2 import crop512
from fm_block4 import seg_dp_mc
from fm_chronos import ChronosBlock

_TEST = os.environ.get("TEST_DATA_ROOT", "C:/Users/huangnian/Desktop/test")
NEW = [
    dict(id="new1", csv=_TEST + "/electric-t-1.csv",
         cj=_TEST + "/cutoff-1.json", blocks="ACGT"),
    dict(id="new2", csv=_TEST + "/electric-t-2.csv",
         cj=_TEST + "/cutoff-2.json", blocks="ACGT"),
    dict(id="new3", csv=_TEST + "/electric-t-3.csv",
         cj=_TEST + "/cutoff-3.json", blocks="ACGT"),
]
OVER = 6


def core(sig, s0, s1):
    c0, c1 = s0 + (s1 - s0) // 4, s1 - (s1 - s0) // 4
    return crop512(sig, c0, c1).T


def cuts4(sig, a0, a1):
    Y = sig[:, a0:a1].T.astype(float)
    segs = seg_dp_mc(Y, kmax=4, min_len=30)
    return segs[4][0]


def eng_of(sig):
    idx = np.where(engaged(sig))[0]
    return (int(idx[0]), int(idx[-1]) + 1) if len(idx) else (0, TARGET)


def register_blocks(rl):
    out = []
    cur, st = None, 0
    for i, b in enumerate(rl + ["-"]):
        if b != cur:
            if cur is not None and cur != "-" and i - st >= 30:
                out.append((cur, st, i))
            cur, st = b, i
    return out


def soft_loss(lg, targets):
    lp = F.log_softmax(lg, -1)
    return -(targets * lp).sum(-1).mean()


def evaluate(model, items):
    rows = []
    with torch.no_grad():
        for it in items:
            sig, (a0, a1), truth = it["sig"], it["eng"], it["blocks"]
            cuts = cuts4(sig, a0, a1)
            tops, top2 = [], []
            for u in range(4):
                x = torch.from_numpy(core(sig, a0 + cuts[u], a0 + cuts[u + 1]))[None]
                p = F.softmax(model(x)[0], -1).numpy()
                tops.append(ALPH[int(p.argmax())])
                top2.append(set(np.argsort(-p)[:2].tolist()))
            af = np.mean([tops[i] == truth[i] for i in range(4)])
            ar = np.mean([tops[i] == truth[::-1][i] for i in range(4)])
            t2f = np.mean([C2I[truth[i]] in top2[i] for i in range(4)])
            t2r = np.mean([C2I[truth[::-1][i]] in top2[i] for i in range(4)])
            rows.append(dict(id=it["id"], pred="".join(tops), truth=truth,
                             acc_bestdir=float(max(af, ar)),
                             top2_bestdir=float(max(t2f, t2r))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs("runs_four", exist_ok=True)
    sig0, meta = load()
    homo = [i for i, m in enumerate(meta) if not m["mixed"] and m["base"] in C2I]
    janus = [i for i, m in enumerate(meta) if m["mixed"] and m.get("register_labels")]
    items = []
    for i in janus:
        m = meta[i]
        rl = m["register_labels"]
        first = next((b for b in rl if b != "-"), None)
        truth = m["seq"][::10] if first == m["seq"][0] else m["seq"][::10][::-1]
        items.append(dict(id=m["id"], sig=sig0[i], eng=eng_of(sig0[i]), blocks=truth,
                          hard=[(b, s, e) for b, s, e in register_blocks(rl)]))
    for nd in NEW:
        sig, tw, I0, orient, factor, dt, note = load_current(nd["csv"], nd["cj"])
        items.append(dict(id=nd["id"], sig=sig, eng=eng_of(sig), blocks=nd["blocks"],
                          hard=None))
    base = ChronosBlock()
    base.load_state_dict(torch.load("runs_alt/fm_block4_chronos.pt", map_location="cpu"))
    base.eval()
    brows = evaluate(base, items)
    for r in brows:
        print("BASE %s pred=%s truth=%s acc=%.2f top2=%.2f" % (
            r["id"], r["pred"], r["truth"], r["acc_bestdir"], r["top2_bestdir"]))
    bmean = float(np.mean([r["acc_bestdir"] for r in brows if r["id"].startswith("new")]))
    bmean2 = float(np.mean([r["top2_bestdir"] for r in brows if r["id"].startswith("new")]))

    def train_model(train_items, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        model = ChronosBlock()
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-5)
        pool = []
        for i in homo:
            pool.append(("homo", i, None))
        for it in train_items:
            if it["hard"]:
                for b, s, e in it["hard"]:
                    for _ in range(OVER):
                        pool.append(("hard", it, (b, s, e)))
            else:
                a0, a1 = it["eng"]
                cuts = cuts4(it["sig"], a0, a1)
                for u in range(4):
                    tg = np.zeros(4)
                    for b in {it["blocks"][u], it["blocks"][::-1][u]}:
                        tg[C2I[b]] += 0.5
                    for _ in range(OVER):
                        pool.append(("soft", it, (u, cuts, tg)))
        for ep in range(a.epochs):
            model.train()
            random.shuffle(pool)
            for s in range(0, len(pool), 16):
                batch = pool[s:s + 16]
                xs, ts = [], []
                for kind, obj, extra in batch:
                    if kind == "homo":
                        i = obj
                        e0, e1 = eng_of(sig0[i])
                        L = e1 - e0
                        if L > 160 and random.random() < 0.8:
                            ln = random.randint(120, min(400, L))
                            st = random.randint(e0, e1 - ln)
                            xs.append(crop512(sig0[i], st, st + ln).T)
                        else:
                            xs.append(crop512(sig0[i], e0, e1).T)
                        t = np.zeros(4)
                        t[C2I[meta[i]["base"]]] = 1.0
                    elif kind == "hard":
                        b, s0, s1 = extra
                        xs.append(core(obj["sig"], s0, s1))
                        t = np.zeros(4)
                        t[C2I[b]] = 1.0
                    else:
                        u, cuts, tg = extra
                        a0, a1 = obj["eng"]
                        xs.append(core(obj["sig"], a0 + cuts[u], a0 + cuts[u + 1]))
                        t = tg
                    ts.append(t)
                xb = torch.from_numpy(np.stack(xs))
                xb = xb * (0.9 + 0.2 * torch.rand(len(batch), 1, 1)) + 0.02 * torch.randn(len(batch), 1, 1)
                loss = soft_loss(model(xb), torch.from_numpy(np.stack(ts)).float())
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
        model.eval()
        return model

    newitems = [it for it in items if it["id"].startswith("new")]
    loo = []
    for hold in range(3):
        tr_items = [it for it in items if it["id"] != newitems[hold]["id"]]
        m = train_model(tr_items, a.seed + hold)
        r = evaluate(m, [newitems[hold]])[0]
        loo.append(r)
        print("LOO  %s pred=%s truth=%s acc=%.2f top2=%.2f" % (
            r["id"], r["pred"], r["truth"], r["acc_bestdir"], r["top2_bestdir"]))
    lmean = float(np.mean([r["acc_bestdir"] for r in loo]))
    lmean2 = float(np.mean([r["top2_bestdir"] for r in loo]))
    print("SUMMARY baseline_new_acc=%.3f top2=%.3f | LOO_new_acc=%.3f top2=%.3f" % (
        bmean, bmean2, lmean, lmean2))
    out = dict(baseline=brows, loo=loo,
               summary=dict(base_acc=bmean, base_top2=bmean2, loo_acc=lmean, loo_top2=lmean2))
    if lmean > bmean or lmean2 > bmean2:
        m = train_model(items, a.seed + 99)
        torch.save(m.state_dict(), "runs_four/ident_four.pt")
        out["saved"] = "runs_four/ident_four.pt"
        print("saved runs_four/ident_four.pt")
    with open("runs_four/four.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("saved runs_four/four.json")


if __name__ == "__main__":
    main()