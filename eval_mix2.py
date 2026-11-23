# -*- coding: utf-8 -*-
"""Blind-read evaluation on the 25 labelled 2-block mixed strands (runs_mix2).
Model: chronos-bolt-mini crop-identity reader (runs_alt/fm_block4_chronos.pt),
trained ONLY on homopolymers -> these 25 strands are fully held-out mixed data.
Read: engaged region -> 2-segment DP (multi-channel) -> middle-50% core crops ->
per-block base probs -> order. Direction is physically ambiguous blind, so we
report composition accuracy (both bases right), direction-best order accuracy,
and fixed-direction order accuracy, plus CER of the expanded 20-char read.
Run: HF_ENDPOINT=https://hf-mirror.com C:/Users/huangnian/pyenvs/esm/Scripts/python.exe eval_mix2.py
"""
import os, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from bench_new import engaged, lev, ALPH, C2I, TARGET
from fm_block2 import crop512
from fm_block4 import seg_dp_mc
from fm_chronos import ChronosBlock


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs_alt/fm_block4_chronos.pt")
    a = ap.parse_args()
    z = np.load("runs_mix2/dataset.npz")
    sig = z["sig"]
    meta = json.load(open("runs_mix2/meta.json", encoding="utf-8"))
    model = ChronosBlock()
    model.load_state_dict(torch.load(a.model, map_location="cpu"))
    model.eval()

    def probs(i, s0, s1):
        x = torch.from_numpy(crop512(sig[i], s0, s1).T)[None]
        with torch.no_grad():
            return F.softmax(model(x)[0], -1).numpy()

    rows = []
    for i, m in enumerate(meta):
        idx = np.where(engaged(sig[i]))[0]
        if len(idx) < 60:
            print("skip no engaged", m["id"])
            continue
        a0, a1 = int(idx[0]), int(idx[-1]) + 1
        Y = sig[i][:, a0:a1].T.astype(float)
        segs = seg_dp_mc(Y, kmax=4, min_len=30)
        K = 2 if 2 in segs else min(segs, key=lambda k: segs[k][2])
        cuts = segs[K][0]
        bases, confs = [], []
        for u in range(len(cuts) - 1):
            s0, s1 = a0 + cuts[u], a0 + cuts[u + 1]
            c0, c1 = s0 + (s1 - s0) // 4, s1 - (s1 - s0) // 4
            p = probs(i, c0, c1)
            bases.append(ALPH[int(p.argmax())])
            confs.append(float(p.max()))
        order = "".join(bases)
        torder = m["seq"][::10]
        comp_ok = sorted(bases) == sorted(m["bases"]) and len(bases) == 2
        dirbest = order == torder or order[::-1] == torder
        dirfix = order == torder
        pred20 = "".join(b * 10 for b in bases)
        cer = min(lev(pred20, m["seq"]), lev(pred20[::-1], m["seq"])) / len(m["seq"])
        rows.append(dict(id=m["id"], truth=torder, order=order, K=K,
                         conf=[round(c, 3) for c in confs], comp_ok=bool(comp_ok),
                         dirbest=bool(dirbest), dirfix=bool(dirfix), cer=cer,
                         cuts=[int(c + a0) for c in cuts]))
        print("%s truth=%s order=%s K=%d conf=%s comp=%s dirbest=%s cer=%.2f" % (
            m["id"], torder, order, K, [round(c, 2) for c in confs], comp_ok, dirbest, cer))
    n = len(rows)
    summ = dict(n=n,
                composition_acc=float(np.mean([r["comp_ok"] for r in rows])),
                order_dirbest_acc=float(np.mean([r["dirbest"] for r in rows])),
                order_dirfix_acc=float(np.mean([r["dirfix"] for r in rows])),
                mean_CER_bestdir=float(np.mean([r["cer"] for r in rows])),
                mean_conf=float(np.mean([c for r in rows for c in r["conf"]])))
    print("SUMMARY", json.dumps(summ))
    with open("runs_mix2/eval.json", "w", encoding="utf-8") as f:
        json.dump(dict(summary=summ, rows=rows), f, indent=1)
    print("saved runs_mix2/eval.json")


if __name__ == "__main__":
    main()