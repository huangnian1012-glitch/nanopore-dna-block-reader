# -*- coding: utf-8 -*-
"""Build the supervised reading dataset from real nanopore replicas.
Physical normalization that PRESERVES base identity:
  fractional blockade f = (I_window - I0) / |I0| per channel, I0 = open-pore tail.
  (z-score would erase the mean blockade depth, which is the main base signature.)
Salt conc computed from gro box volume + K/CL ion counts (NOTE-prescribed method):
  conc_M = min(n_K, n_CL) / (0.6022 * V_nm3).
Homopolymer replicas -> (subwindow, base) supervision pairs (engaged frames only,
  selected by blockade depth). Mixed block replicas -> register read-order truth.
Run with base python (stdlib+numpy): python prep_supervised.py
Output: runs_sup/dataset.npz + runs_sup/meta.json
"""
import os, json, csv
import numpy as np

DESK = os.environ.get("NANOPORE_DATA_ROOT", "C:/Users/huangnian/Desktop")
FOLDERS = ["nanopore_dataset", "mixed_dataset", "other_membranes_dataset", "janus_dna_dataset"]
OUT = "runs_sup"
TARGET = 800
WIN = 100          # subwindow length in downsampled samples
STEP = 50


def downsample(x, target):
    factor = max(1, int(round(len(x) / float(target))))
    n = (len(x) // factor) * factor
    y = x[:n].reshape(-1, factor).mean(axis=1)
    if len(y) >= target:
        y = y[:target]
    else:
        y = np.pad(y, (0, target - len(y)), mode="edge")
    return y.astype(np.float32), factor


def gro_conc(gro):
    with open(gro) as f:
        f.readline()
        n = int(f.readline().strip())
        nk = ncl = 0
        for i in range(n):
            line = f.readline()
            rn = line[5:10].strip()
            if rn == "K" or rn == "NA":
                nk += 1
            elif rn == "CL":
                ncl += 1
        box = f.readline().split()
    v = float(box[0]) * float(box[1]) * float(box[2])
    pairs = min(nk, ncl)
    return pairs / (0.6022 * v), v, nk, ncl


def eligible():
    out = []
    for fol in FOLDERS:
        mp = os.path.join(DESK, fol, "manifest.csv")
        if not os.path.exists(mp):
            continue
        for r in csv.DictReader(open(mp, encoding="utf-8")):
            if r.get("has_current") != "True" or r.get("has_dna_xtc") != "True":
                continue
            s = (r.get("sequence") or "").strip()
            if not s:
                continue
            d = os.path.join(DESK, fol, r["dir"])
            cj = os.path.join(d, "tail_analysis", "cutoff.json")
            cp = os.path.join(d, "current_analysis", "electric-t.csv")
            if not os.path.exists(cj) or not os.path.exists(cp):
                continue
            j = json.load(open(cj, encoding="utf-8"))
            if not (j.get("fully_translocated") and j.get("cutoff_ns") is not None):
                continue
            sd = os.path.join(d, "structures")
            gro = os.path.join(sd, "md.gro")
            if not os.path.exists(gro):
                gs = [x for x in os.listdir(sd) if x.endswith(".gro")] if os.path.isdir(sd) else []
                if not gs:
                    continue
                gro = os.path.join(sd, gs[0])
            out.append(dict(fol=fol, id=r["sample_id"], base=r.get("base", ""), seq=s,
                            membrane=r.get("membrane", ""), d=d, cj=cj, cp=cp, gro=gro, j=j))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = eligible()
    print("eligible", len(rows))
    sigs, metas = [], []
    for k, e in enumerate(rows):
        j = e["j"]
        tj = np.loadtxt(e["cp"], delimiter=",", skiprows=1, usecols=(0, 1, 2, 3))
        t = tj[:, 0]
        ch = tj[:, 1:4]
        cut = j["cutoff_ns"]
        t_first = j.get("t_first_ns", t[0])
        m = (t >= t_first) & (t <= cut)
        win = ch[m]
        if len(win) < 50:
            continue
        tail = ch[t > cut]
        I0 = tail.mean(axis=0) if len(tail) >= 10 else win[-max(10, len(win) // 10):].mean(axis=0)
        orient = 1.0 if float(win[:, 2].mean()) >= I0[2] else -1.0
        frac = (win - I0) / np.abs(I0) * orient        # (W,3) fractional blockade, depth kept
        sig, factor = downsample(frac[:, 0], TARGET)      # per-channel downsample
        sig = np.stack([downsample(frac[:, c], TARGET)[0] for c in range(3)])  # (3,800)
        conc, v, nk, ncl = gro_conc(e["gro"])
        # engaged mask: blockade depth well above open-pore noise
        depth = np.abs(sig[2])
        thr = 0.5 * np.percentile(depth, 75)
        engaged = depth > thr
        meta = dict(id=e["id"], fol=e["fol"], base=e["base"], seq=e["seq"],
                    membrane=e["membrane"], conc=round(conc, 3), I0=I0.tolist(),
                    orient=orient, factor=factor, n_win=len(win),
                    engaged_frac=float(engaged.mean()), mixed=len(set(e["seq"])) > 1)
        # register truth for mixed strands
        if meta["mixed"]:
            rp = os.path.join("runs_real", "register_%s.csv" % e["id"])
            if os.path.exists(rp):
                rt = np.loadtxt(rp, delimiter=",", skiprows=1, usecols=(0,))
                rb = np.loadtxt(rp, delimiter=",", skiprows=1, usecols=(1,), dtype=str)
                # map register frame time -> downsampled index
                idx = np.clip(np.round(rt / (j["dt_ns"] * factor)).astype(int), 0, TARGET - 1)
                lab = np.full(TARGET, "-", dtype="<U1")
                lab[idx] = rb
                meta["register_labels"] = lab.tolist()
            else:
                meta["register_labels"] = None
        sigs.append(sig)
        metas.append(meta)
        if (k + 1) % 20 == 0:
            print("  processed %d/%d" % (k + 1, len(rows)))
    np.savez_compressed(os.path.join(OUT, "dataset.npz"),
                        sig=np.stack(sigs).astype(np.float32))
    with open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(metas, f, indent=1)
    from collections import Counter
    print("saved", len(sigs), "replicas")
    print("base counts", dict(Counter(m["base"] for m in metas)))
    print("conc by base (4-4):")
    for b in "ACGT":
        cs = [m["conc"] for m in metas if m["base"] == b and m["membrane"] == "4-4"]
        if cs:
            print("   %s n=%d conc median=%.3f range=%.3f..%.3f" % (b, len(cs), np.median(cs), min(cs), max(cs)))
    print("mixed with register:", [m["id"] for m in metas if m["mixed"] and m.get("register_labels")])


if __name__ == "__main__":
    main()