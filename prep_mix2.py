# -*- coding: utf-8 -*-
"""Prepare the 25 labelled 2-block mixed strands from mixed_dataset/train.
These were NOT in runs_sup (which only had the 2 janus 4-block strands).
Labels come from manifest.csv sequence/block_label (real file-derived truth).
Current csv: <dir>/current_analysis/electric-t.csv ; cutoff: <dir>/tail_analysis/cutoff.json
Output: runs_mix2/dataset.npz (sig (M,3,800)) + meta.json
Run: C:/Users/huangnian/pyenvs/esm/Scripts/python.exe prep_mix2.py
"""
import os, csv, json
import numpy as np
from recognize import load_current

ROOT = os.environ.get("MIXED_DATA_ROOT", "C:/Users/huangnian/Desktop/mixed_dataset")


def main():
    os.makedirs("runs_mix2", exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(ROOT, "manifest.csv"), encoding="utf-8")))
    sigs, meta = [], []
    for r in rows:
        d = os.path.join(ROOT, r["dir"])
        cp = os.path.join(d, "current_analysis", "electric-t.csv")
        cj = os.path.join(d, "tail_analysis", "cutoff.json")
        if not os.path.exists(cp):
            print("skip missing csv", r["sample_id"])
            continue
        sig, tw, I0, orient, factor, dt, note = load_current(cp, cj if os.path.exists(cj) else None)
        sigs.append(sig)
        meta.append(dict(id=r["sample_id"], seq=r["sequence"], block_label=r["block_label"],
                         bases=sorted(set(r["sequence"])), n_blocks=int(r["n_blocks"]),
                         membrane=r["membrane_measured"], conc=float(r["conc_M"]),
                         orient=orient, factor=factor, note=note, dir=r["dir"]))
        print("%s %s conc=%.4f note=%s" % (r["sample_id"], r["block_label"], float(r["conc_M"]), note))
    np.savez_compressed("runs_mix2/dataset.npz", sig=np.stack(sigs))
    with open("runs_mix2/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print("saved runs_mix2: %d traces" % len(sigs))


if __name__ == "__main__":
    main()