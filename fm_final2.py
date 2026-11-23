# -*- coding: utf-8 -*-
"""Train FINAL order models: TV-denoised whole-engaged traces, 3 seeds x 2
encoders, label smoothing 0.05, 80 epochs (protocol of fm_mix5 --tv, CV order
0.603 / comp 0.637 on the 25 labelled 2-block strands).
Run: HF_ENDPOINT=https://hf-mirror.com C:/Users/huangnian/pyenvs/esm/Scripts/python.exe fm_final2.py
"""
import os, json
import numpy as np
import torch
from bench_new import engaged, TARGET
from fm_mix2 import PAIRS, P2I
from fm_mix5 import train_enc
from denoise_sweep import tv1d


def main():
    z = np.load("runs_mix2/dataset.npz")
    sig = z["sig"]
    meta = json.load(open("runs_mix2/meta.json", encoding="utf-8"))
    eng = []
    for i in range(len(meta)):
        idx = np.where(engaged(sig[i]))[0]
        eng.append((int(idx[0]), int(idx[-1]) + 1) if len(idx) else (0, TARGET))
    sig = np.stack([np.stack([tv1d(sig[i, c]) for c in range(3)])
                    for i in range(len(sig))]).astype(np.float32)
    lab = np.array([P2I[m["seq"][::10]] for m in meta])
    allidx = list(range(len(meta)))
    for sd in range(3):
        for e in ("chronos", "patchtst"):
            m = train_enc(e, allidx, sig, eng, lab, 80, 900 + 100 * sd, 0.05)
            p = "runs_mix2/order_v6_%s_%d.pt" % (e, sd)
            torch.save(m.state_dict(), p)
            print("saved", p)
    print("done")


if __name__ == "__main__":
    main()