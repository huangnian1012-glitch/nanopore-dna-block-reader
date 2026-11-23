# -*- coding: utf-8 -*-
"""Final recognizer v3.1. Input: ONLY unknown-strand current csv (+cutoff.json).
Paths:
  * single engaged block  -> homopolymer identity (chronos crop reader, ~0.88)
  * two engaged blocks    -> 12-way ordered base-pair read (TV-denoised trace,
    ensemble of 3 seeds x chronos+patchtst order classifiers trained on the 25
    labelled 2-block strands, with reversal+phase test-time averaging).
    CV order acc ~0.60 vs chance 0.083; composition acc ~0.64 vs chance 0.167.
  * >2 blocks             -> segments reported, order flagged OUT-OF-SCOPE
    (trained only on 2-block strands).
Direction note: for a 2-block strand the reverse order is the opposite physical
entry direction; both are listed with their probabilities.
Run: HF_ENDPOINT=https://hf-mirror.com C:/Users/huangnian/pyenvs/esm/Scripts/python.exe recognize3.py --current <csv> --cutoff <json> --out runs_mix2/rec3_<name>
"""
import os, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bench_new import engaged, ALPH, TARGET
from recognize import load_current
from fm_block2 import crop512, FMCrop
from fm_block4 import seg_dp_mc
from fm_chronos import ChronosBlock
from fm_mix2 import OrderHead, PAIRS, P2I
from fm_mix5 import tta5
from denoise_sweep import tv1d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", required=True)
    ap.add_argument("--cutoff", default=None)
    ap.add_argument("--out", default="runs_mix2/rec3")
    ap.add_argument("--k", type=int, default=None,
                    help="force number of blocks if known (e.g. 4); default BIC auto")
    ap.add_argument("--lengths", default=None,
                    help="comma-separated base counts per block in PASSAGE order, "
                         "e.g. 10,10,10,10; expands block guesses to a full string")
    a = ap.parse_args()
    sig, tw, I0, orient, factor, dt, note = load_current(a.current, a.cutoff)
    idx = np.where(engaged(sig))[0]
    if len(idx) < 60:
        print("no engaged region; nothing to read")
        return
    a0, a1 = int(idx[0]), int(idx[-1]) + 1
    t0, t1 = float(tw[0]), float(tw[-1])
    span = t1 - t0

    def ns(i):
        return t0 + span * i / (TARGET - 1)

    ident_model = ChronosBlock()
    ident_model.load_state_dict(torch.load("runs_alt/fm_block4_chronos.pt", map_location="cpu"))
    ident_model.eval()
    with torch.no_grad():
        x = torch.from_numpy(crop512(sig, a0, a1).T)[None]
        p_id = F.softmax(ident_model(x)[0], -1).numpy()
    ident = ALPH[int(p_id.argmax())]

    Y = sig[:, a0:a1].T.astype(float)
    kmax = max(4, a.k or 4)
    segs = seg_dp_mc(Y, kmax=kmax, min_len=30)
    kbic = min(segs, key=lambda k: segs[k][2])
    K = a.k if a.k else kbic
    if K not in segs:
        print("requested K=%d not available (max %d)" % (K, max(segs)))
        return
    cuts = segs[K][0]
    seg_model = ident_model
    if K >= 3 and os.path.exists("runs_four/ident_four.pt"):
        seg_model = ChronosBlock()
        seg_model.load_state_dict(torch.load("runs_four/ident_four.pt", map_location="cpu"))
        seg_model.eval()
    segments = []
    for u in range(len(cuts) - 1):
        s0, s1 = a0 + cuts[u], a0 + cuts[u + 1]
        c0, c1 = s0 + (s1 - s0) // 4, s1 - (s1 - s0) // 4
        with torch.no_grad():
            xx = torch.from_numpy(crop512(sig, c0, c1).T)[None]
            p = F.softmax(seg_model(xx)[0], -1).numpy()
        segments.append({"start_ns": round(ns(s0), 3), "end_ns": round(ns(s1), 3),
                         "base_guess": ALPH[int(p.argmax())],
                         "conf": round(float(p.max()), 3),
                         "reliability": ("4-block reader LOO acc ~0.58 / top2 ~0.92"
                                         if K >= 3 else "LOW (mixed-context shift)")})
    out = {"note": note, "K_bic": K,
           "hypothesis_homopolymer": {"identity": ident,
                                      "conf": round(float(p_id.max()), 3)},
           "engaged_ns": [round(ns(a0), 3), round(ns(a1), 3)], "segments": segments}
    models = []
    for e in ("chronos", "patchtst"):
        for sd in range(3):
            enc = FMCrop() if e == "patchtst" else ChronosBlock()
            m = OrderHead(enc)
            m.load_state_dict(torch.load("runs_mix2/order_v6_%s_%d.pt" % (e, sd),
                                         map_location="cpu"))
            m.eval()
            models.append(m)
    sig_tv = np.stack([tv1d(sig[c]) for c in range(3)]).astype(np.float32)
    p = tta5(models, sig_tv[None], 0, a0, a1)
    top = np.argsort(-p)[:3]
    orders = [{"order": PAIRS[t], "prob": round(float(p[t]), 3)} for t in top]
    best = PAIRS[int(p.argmax())]
    out["hypothesis_two_block"] = {"order_fwd": best, "order_rev": best[::-1],
                                   "top3": orders,
                                   "composition": {b: 0.5 for b in sorted(best)}}
    out["type_gate"] = ("NOT decidable from current alone (2-means depth statistic "
                        "AUC=0.50; learned gate mixed-recall=0.20): T/A and C/A depth "
                        "levels nearly coincide. Both hypotheses are reported; supply "
                        "strand type if known.")
    out["reliability"] = ("homopolymer identity ~0.88; two-block order CV acc ~0.60 "
                          "(chance 0.083), composition ~0.64 (chance 0.167); "
                          "TV denoise + 6-model ensemble + reversal/phase TTA")
    print("hypothesis_homopolymer: identity=%s conf=%.3f" % (ident, p_id.max()))
    print("hypothesis_two_block: order_fwd=%s order_rev=%s" % (best, best[::-1]))
    for o in orders:
        print("  order=%s prob=%.3f" % (o["order"], o["prob"]))
    print("type gate: not decidable from current alone; both hypotheses reported")
    print("segments (K_bic=%d, per-block guess; LOW reliability on mixed contexts):" % K)
    for s in segments:
        print("  %s  %.2f - %.2f ns  conf=%.3f" % (
            s["base_guess"], s["start_ns"], s["end_ns"], s["conf"]))
    guess = "".join(s["base_guess"] for s in segments)
    print("BLOCK-ORDER GUESS (passage order): %s   [K=%d; trained on 2-block strands, treat as reference]" % (guess, K))
    if a.lengths:
        lens = [int(v) for v in a.lengths.split(",")]
        if len(lens) == K:
            full = "".join(b * n for b, n in zip(guess, lens))
            out["full_string_guess"] = full
            out["full_string_note"] = ("block guesses x user-supplied lengths; "
                                       "letters carry the block-reader accuracy, "
                                       "lengths are NOT read from current")
            print("FULL-STRING GUESS (guesses x your lengths): %s" % full)
        else:
            print("--lengths has %d values but K=%d; ignored" % (len(lens), K))
    with open(a.out + ".json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1]})
    tt = np.linspace(t0, t1, TARGET)
    for c, name in enumerate(["i_c", "i_a", "i_tot"]):
        ax[0].plot(tt, sig[c], lw=0.6, label=name)
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_ylabel("fractional blockade")
    for s in segments:
        ax[0].axvline(s["start_ns"], color="k", lw=0.6, alpha=0.5)
        ax[1].axvspan(s["start_ns"], s["end_ns"], alpha=0.25,
                      color={"A": "tab:green", "C": "tab:blue", "G": "tab:orange", "T": "tab:red"}[s["base_guess"]])
        ax[1].text((s["start_ns"] + s["end_ns"]) / 2, 0.5, s["base_guess"],
                   ha="center", va="center", fontsize=12)
    ax[1].set_yticks([])
    ax[1].set_xlabel("time (ns)")
    ax[0].set_title("K=%d | homo=%s | 2-block fwd=%s rev=%s" % (
        K, ident, best, best[::-1]))
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=130)
    print("saved %s.json / %s.png" % (a.out, a.out))


if __name__ == "__main__":
    main()