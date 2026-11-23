# -*- coding: utf-8 -*-
"""Supervised base-identity reader trained on homopolymer subwindows.
Idea: every engaged subwindow of a homopolymer strand = that strand's base, so
homopolymers supply many (current-window, base) pairs. A window classifier is
then slid across an UNKNOWN strand to read its base order (block level).
Mixed block strands are held out entirely = blind read-order test vs register.
Normalization = fractional blockade (depth preserved), see prep_supervised.py.
Run: C:/Users/huangnian/pyenvs/esm/Scripts/python.exe sup_read.py --out runs_sup
"""
import os, json, argparse, random
import numpy as np
import torch
import torch.nn as nn

ALPH = "ACGT"
C2I = {c: i for i, c in enumerate(ALPH)}
TARGET, WIN, STEP = 800, 100, 50


def downsample_idx(factor):
    return factor


class WinNet(nn.Module):
    def __init__(self, nch=3, ncls=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(nch, 32, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Linear(64, ncls)

    def forward(self, x):
        h = self.cnn(x).squeeze(-1)
        return self.fc(h)


def engaged_mask(sig):
    depth = np.abs(sig[2])
    thr = 0.5 * np.percentile(depth, 75)
    return depth > thr


def subwindows(sig, keep_engaged=True):
    em = engaged_mask(sig)
    out = []
    for s in range(0, TARGET - WIN + 1, STEP):
        if keep_engaged and em[s:s + WIN].mean() < 0.8:
            continue
        out.append((s, sig[:, s:s + WIN]))
    return out


def collapse(labels):
    """labels: array of base chars or '-' -> list of (base, start, end) ignoring '-'."""
    blocks = []
    cur, s = None, None
    for i, b in enumerate(labels):
        if b == "-":
            if cur is not None:
                blocks.append((cur, s, i))
                cur, s = None, None
            continue
        if b != cur:
            if cur is not None:
                blocks.append((cur, s, i))
            cur, s = b, i
    if cur is not None:
        blocks.append((cur, s, len(labels)))
    return blocks


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def load():
    z = np.load("runs_sup/dataset.npz")
    sig = z["sig"]
    meta = json.load(open("runs_sup/meta.json", encoding="utf-8"))
    return sig, meta


def build_pairs(sig, meta, idxs):
    X, Y, owner = [], [], []
    for i in idxs:
        m = meta[i]
        if m["mixed"] or m["base"] not in C2I:
            continue
        for s, w in subwindows(sig[i]):
            X.append(w)
            Y.append(C2I[m["base"]])
            owner.append(i)
    return np.stack(X).astype(np.float32), np.array(Y), np.array(owner)


@torch.no_grad()
def predict_windows(model, sig):
    model.eval()
    ws = subwindows(sig)
    if not ws:
        return [], []
    x = torch.from_numpy(np.stack([w for _, w in ws]))
    p = model(x).argmax(1).numpy()
    return [s for s, _ in ws], [ALPH[c] for c in p]


def read_strand(model, sig):
    starts, bases = predict_windows(model, sig)
    lab = np.full(TARGET, "-", dtype="<U1")
    for s, b in zip(starts, bases):
        lab[s:s + WIN] = b
    return collapse(lab)


def replica_identity(model, sig, meta, idxs):
    correct = 0
    rows = []
    for i in idxs:
        m = meta[i]
        if m["mixed"] or m["base"] not in C2I:
            continue
        _, bases = predict_windows(model, sig[i])
        if not bases:
            pred = "-"
        else:
            vals, cnts = np.unique(bases, return_counts=True)
            pred = vals[np.argmax(cnts)]
        ok = pred == m["base"]
        correct += ok
        rows.append(dict(id=m["id"], true=m["base"], pred=pred, ok=bool(ok),
                         membrane=m["membrane"], conc=m["conc"]))
    return correct / max(1, len(rows)), rows


def train(args):
    sig, meta = load()
    homo = [i for i, m in enumerate(meta) if not m["mixed"] and m["base"] in C2I]
    random.seed(args.seed)
    random.shuffle(homo)
    nval = max(1, int(len(homo) * 0.2))
    val_idx, train_idx = homo[:nval], homo[nval:]
    X, Y, owner = build_pairs(sig, meta, train_idx)
    print("homopolymer replicas train=%d val=%d  subwindows=%d" % (len(train_idx), len(val_idx), len(X)))
    model = WinNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    xt = torch.from_numpy(X)
    yt = torch.from_numpy(Y)
    n = len(X)
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        tot, nb = 0.0, 0
        for s in range(0, n, args.batch):
            bidx = perm[s:s + args.batch]
            logits = model(xt[bidx])
            loss = lossf(logits, yt[bidx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
        if ep % 10 == 0 or ep == 1:
            acc, _ = replica_identity(model, sig, meta, val_idx)
            print("ep %2d/%d loss=%.4f val_replica_acc=%.3f" % (ep, args.epochs, tot / nb, acc))
    os.makedirs(args.out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out, "sup_read.pt"))

    results = {}
    acc, rows = replica_identity(model, sig, meta, val_idx)
    results["val_replica_identity_acc"] = acc
    print("VAL replica identity acc=%.3f" % acc)
    for r in rows:
        print("   %s true=%s pred=%s ok=%s memb=%s conc=%.2f" % (r["id"], r["true"], r["pred"], r["ok"], r["membrane"], r["conc"]))
    # matched-condition subsets
    for name, filt in [("4-4_conc~1.0", lambda m: m["membrane"] == "4-4" and 0.9 <= m["conc"] <= 1.1),
                       ("2-6", lambda m: m["membrane"] == "2-6")]:
        sub = [i for i in val_idx if filt(meta[i])]
        if sub:
            a, _ = replica_identity(model, sig, meta, sub)
            results["val_acc_" + name] = a
            print("VAL subset %s n=%d acc=%.3f" % (name, len(sub), a))
    # blind mixed read-order test
    mixed_res = []
    for i, m in enumerate(meta):
        if not m["mixed"] or not m.get("register_labels"):
            continue
        pred_blocks = read_strand(model, sig[i])
        pred_seq = "".join(b for b, _, _ in pred_blocks)
        true_blocks = collapse(np.array(m["register_labels"]))
        true_seq = "".join(b for b, _, _ in true_blocks)
        cer = levenshtein(pred_seq, true_seq) / max(1, len(true_seq))
        mixed_res.append(dict(id=m["id"], true_read_order=true_seq, pred_read_order=pred_seq,
                              block_CER=cer,
                              pred_blocks=[(b, int(s), int(e)) for b, s, e in pred_blocks],
                              true_blocks=[(b, int(s), int(e)) for b, s, e in true_blocks]))
        print("MIXED %s true=%s pred=%s block_CER=%.3f" % (m["id"], true_seq, pred_seq, cer))
    results["mixed_blind"] = mixed_res
    with open(os.path.join(args.out, "metrics_sup.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs_sup")
    train(ap.parse_args())