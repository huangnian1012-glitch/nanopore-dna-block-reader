# -*- coding: utf-8 -*-
"""FINAL-GOAL INTERFACE: unknown-strand current -> base order.
Input : one electric-t.csv (time_ns,I_cat_nA,I_an_nA,I_tot_nA) from a strand you
        have never seen; optional cutoff.json to fix the translocation window.
Output: predicted base order in READ order (order bases passed the pore neck),
        per-block time boundaries, and strand composition vote.
No trajectory, no label needed at inference.
Run: C:/Users/huangnian/pyenvs/esm/Scripts/python.exe recognize.py --current <csv> [--cutoff <json>]
"""
import os, json, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sup_read import WinNet, subwindows, collapse, TARGET, WIN, ALPH, engaged_mask
from prep_supervised import downsample


def load_current(cpath, cjpath):
    tj = np.loadtxt(cpath, delimiter=",", skiprows=1, usecols=(0, 1, 2, 3))
    t, ch = tj[:, 0], tj[:, 1:4]
    dt = float(np.median(np.diff(t)))
    if cjpath and os.path.exists(cjpath):
        j = json.load(open(cjpath, encoding="utf-8"))
        cut = j.get("cutoff_ns")
        t_first = j.get("t_first_ns", t[0])
        if j.get("fully_translocated") and cut is not None:
            m = (t >= t_first) & (t <= cut)
            win = ch[m]
            tw = t[m]
            tail = ch[t > cut]
            I0 = tail.mean(axis=0) if len(tail) >= 10 else win[-max(10, len(win) // 10):].mean(axis=0)
            note = "window from cutoff.json"
        else:
            win, tw = ch, t
            I0 = ch[-max(10, len(ch) // 10):].mean(axis=0)
            note = "cutoff present but not fully translocated; using whole trace, tail baseline"
    else:
        win, tw = ch, t
        I0 = ch[-max(10, len(ch) // 10):].mean(axis=0)
        note = "no cutoff.json; whole trace, last-10% as open-pore baseline (approx)"
    orient = 1.0 if float(win[:, 2].mean()) >= I0[2] else -1.0
    frac = (win - I0) / np.abs(I0) * orient
    sig, factor = downsample(frac[:, 0], TARGET)
    sig = np.stack([downsample(frac[:, c], TARGET)[0] for c in range(3)])  # (3,800)
    return sig, tw, I0, orient, factor, dt, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", required=True)
    ap.add_argument("--cutoff", default=None)
    ap.add_argument("--model", default="runs_sup/sup_read.pt")
    ap.add_argument("--out", default="runs_sup/recognize")
    a = ap.parse_args()
    model = WinNet()
    model.load_state_dict(torch.load(a.model, map_location="cpu"))
    model.eval()
    sig, tw, I0, orient, factor, dt, note = load_current(a.current, a.cutoff)
    print("note:", note)
    print("open-pore I0 (cat,an,tot) nA:", np.round(I0, 3).tolist(), " orient:", orient)
    ws = subwindows(sig)
    if not ws:
        print("NO engaged window detected: strand may not be in pore; cannot read.")
        return
    x = torch.from_numpy(np.stack([w for _, w in ws]))
    with torch.no_grad():
        probs = torch.softmax(model(x), 1).numpy()
    bases = [ALPH[i] for i in probs.argmax(1)]
    conf = probs.max(1)
    lab = np.full(TARGET, "-", dtype="<U1")
    for (s, _), b in zip(ws, bases):
        lab[s:s + WIN] = b
    blocks = collapse(lab)
    read_order = "".join(b for b, _, _ in blocks)
    vals, cnts = np.unique(bases, return_counts=True)
    comp = {v: int(c) for v, c in zip(vals, cnts)}
    print("PREDICTED READ ORDER (passage order):", read_order)
    print("opposite-end convention (reverse):   ", read_order[::-1])
    print("composition vote (windows):", comp)
    print("blocks (base, t_start_ns, t_end_ns, mean_conf):")
    out_blocks = []
    for b, s, e in blocks:
        sel = [k for k, (st, _) in enumerate(ws) if s <= st < e]
        mc = float(conf[sel].mean()) if sel else 0.0
        t0 = tw[min(len(tw) - 1, int(s * factor))]
        t1 = tw[min(len(tw) - 1, int(e * factor))]
        print("   %s  %.2f  %.2f  conf=%.2f" % (b, t0, t1, mc))
        out_blocks.append(dict(base=b, t_start_ns=float(t0), t_end_ns=float(t1), conf=mc))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    res = dict(current=a.current, read_order=read_order, reverse_order=read_order[::-1],
               composition=comp, blocks=out_blocks, note=note)
    with open(a.out + ".json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    colmap = {"A": "#1f77b4", "C": "#ff7f0e", "G": "#2ca02c", "T": "#d62728", "-": "0.6"}
    depth = sig[2]
    cols = [colmap[b] for b in lab]
    fig, ax = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
    xs = np.arange(TARGET) * factor * dt
    ax[0].scatter(xs, depth, s=3, c=cols)
    ax[0].set_ylabel("oriented fractional blockade I_tot")
    ax[0].set_title("predicted read order = %s" % read_order)
    ax[1].plot(xs, [{"A": 0, "C": 1, "G": 2, "T": 3, "-": -1}[b] for b in lab], lw=0.8, color="k")
    ax[1].set_yticks([-1, 0, 1, 2, 3])
    ax[1].set_yticklabels(["-", "A", "C", "G", "T"])
    ax[1].set_ylabel("predicted base")
    ax[1].set_xlabel("time ns")
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=130)
    print("saved", a.out + ".json", a.out + ".png")


if __name__ == "__main__":
    main()