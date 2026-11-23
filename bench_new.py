# -*- coding: utf-8 -*-
"""Benchmark: current results frozen as baseline; newer-than-CTC architectures
compared under IDENTICAL preprocessing (fractional blockade) and splits.
Tasks:
  T1 strand identity (homopolymer 4-class), replica-level acc
  T2 mixed blind read-order CER (2 mixed strands held out entirely)
  T3 within-strand per-frame base acc vs register truth (alignment KNOWN)
Models: cnn (CNN+BiLSTM, CTC-era) vs conformer (conv+MHSA, newer);
heads: CTC, frame-CE, attention-seq2seq (alignment-free, newer than CTC).
Run: C:/Users/huangnian/pyenvs/esm/Scripts/python.exe bench_new.py --epochs 40
"""
import os, json, argparse, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ALPH = "ACGT"
C2I = {c: i for i, c in enumerate(ALPH)}
BLANK = 4
TARGET = 800


def load():
    z = np.load("runs_sup/dataset.npz")
    sig = z["sig"]
    meta = json.load(open("runs_sup/meta.json", encoding="utf-8"))
    return sig, meta


def engaged(sig):
    d = np.abs(sig[2])
    return d > 0.5 * np.percentile(d, 75)


def frame_labels_homopolymer(sig, base):
    em = engaged(sig)
    lab = np.full(TARGET, -100, dtype=np.int64)
    lab[em] = C2I[base]
    return lab


def to_frames(lab800, F):
    idx = (np.arange(F) * TARGET // F)
    return lab800[idx]


class CNNEncoder(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(3, 32, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU())
        self.rnn = nn.LSTM(64, d // 2, batch_first=True, bidirectional=True)
        self.d = d

    def forward(self, x):
        h = self.cnn(x).transpose(1, 2)
        h, _ = self.rnn(h)
        return h


class ConfBlock(nn.Module):
    def __init__(self, d, heads=4, ff=128, k=15):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.pw2 = nn.Conv1d(d, d, 1)
        self.ln3 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, ff)
        self.ff2 = nn.Linear(ff, d)
        self.ln4 = nn.LayerNorm(d)

    def forward(self, x):
        h = self.ln1(x)
        h, _ = self.att(h, h, h, need_weights=False)
        x = x + h
        h = self.ln2(x).transpose(1, 2)
        h = F.glu(self.pw1(h), dim=1)
        h = self.dw(h)
        h = F.silu(self.bn(h))
        h = self.pw2(h).transpose(1, 2)
        x = x + h
        h = self.ln3(x)
        h = self.ff2(F.silu(self.ff1(h))) * 0.5
        x = x + h
        return self.ln4(x)


class ConfEncoder(nn.Module):
    def __init__(self, d=64, nblk=2):
        super().__init__()
        self.sub = nn.Sequential(
            nn.Conv1d(3, d, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(d, d, 5, stride=2, padding=2), nn.ReLU())
        self.blocks = nn.ModuleList([ConfBlock(d) for _ in range(nblk)])
        self.d = d

    def forward(self, x):
        h = self.sub(x).transpose(1, 2)
        for b in self.blocks:
            h = b(h)
        return h


def make_enc(kind):
    return CNNEncoder() if kind == "cnn" else ConfEncoder()


class CTCHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 5)

    def forward(self, h):
        return self.fc(h)


class FrameHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 4)

    def forward(self, h):
        return self.fc(h)


class AttnDecoder(nn.Module):
    """alignment-free seq2seq (Bahdanau), newer paradigm than CTC."""
    def __init__(self, d, ncls=4):
        super().__init__()
        self.emb = nn.Embedding(ncls + 1, d)   # +1 = SOS
        self.gru = nn.GRUCell(d + d, d)
        self.Wa = nn.Linear(d, d)
        self.Va = nn.Linear(d, 1)
        self.out = nn.Linear(d * 2, ncls + 1)  # +1 = EOS
        self.ncls = ncls
        self.d = d

    def forward(self, enc, targets=None, max_len=60):
        B, Fe, d = enc.shape
        SOS = self.ncls
        EOS = self.ncls
        h = enc.mean(1)
        inp = torch.full((B,), SOS, dtype=torch.long, device=enc.device)
        logits_all = []
        L = targets.shape[1] + 1 if targets is not None else max_len
        for t in range(L):
            e = self.emb(inp)
            h = self.gru(torch.cat([e, h], 1), h)
            score = self.Va(torch.tanh(self.Wa(enc) + h.unsqueeze(1))).squeeze(-1)
            w = torch.softmax(score, 1)
            ctx = (w.unsqueeze(-1) * enc).sum(1)
            lo = self.out(torch.cat([h, ctx], 1))
            logits_all.append(lo)
            if targets is not None:
                inp = targets[:, t] if t < targets.shape[1] else torch.full((B,), EOS, dtype=torch.long, device=enc.device)
            else:
                inp = lo.argmax(1)
        return torch.stack(logits_all, 1)


def greedy_ctc(logp):
    ids = logp.argmax(-1)
    out = []
    prev = -1
    for i in ids:
        i = int(i)
        if i != prev and i != BLANK:
            out.append(ALPH[i])
        prev = i
    return "".join(out)


def lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def true_read_full(m):
    lab = m.get("register_labels")
    if not lab:
        return None
    # first non-'-' base
    first = next((b for b in lab if b != "-"), None)
    seq = m["seq"]
    return seq if first == seq[0] else seq[::-1]


def batch_sig(sig, idxs):
    return torch.from_numpy(np.stack([sig[i] for i in idxs]))


def run_task(enc_kind, head_kind, sig, meta, tr, va, mixed, epochs, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    enc = make_enc(enc_kind)
    d = enc.d
    if head_kind == "ctc":
        head = CTCHead(d)
    elif head_kind == "frame":
        head = FrameHead(d)
    else:
        head = AttnDecoder(d)
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    homo_tr = [i for i in tr if not meta[i]["mixed"]]
    for ep in range(epochs):
        enc.train()
        head.train()
        random.shuffle(homo_tr)
        for s in range(0, len(homo_tr), 16):
            bd = homo_tr[s:s + 16]
            x = batch_sig(sig, bd)
            h = enc(x)
            if head_kind == "ctc":
                logp = head(h).log_softmax(2).transpose(0, 1)
                texts = [meta[i]["base"] * len(meta[i]["seq"]) for i in bd]
                Lmax = max(len(t) for t in texts)
                tgt = torch.full((len(bd), Lmax), BLANK, dtype=torch.long)
                for i, t in enumerate(texts):
                    for k, c in enumerate(t):
                        tgt[i, k] = C2I[c]
                in_lens = torch.full((len(bd),), h.shape[1], dtype=torch.long)
                tgt_lens = torch.tensor([len(t) for t in texts], dtype=torch.long)
                loss = F.ctc_loss(logp, tgt, in_lens, tgt_lens, blank=BLANK, zero_infinity=True)
            elif head_kind == "frame":
                lab = np.stack([to_frames(frame_labels_homopolymer(sig[i], meta[i]["base"]), h.shape[1]) for i in bd])
                loss = F.cross_entropy(head(h).transpose(1, 2), torch.from_numpy(lab), ignore_index=-100)
            else:
                texts = [meta[i]["base"] * len(meta[i]["seq"]) for i in bd]
                Lmax = max(len(t) for t in texts)
                tgt = torch.full((len(bd), Lmax), 4, dtype=torch.long)
                for i, t in enumerate(texts):
                    for k, c in enumerate(t):
                        tgt[i, k] = C2I[c]
                lo = head(h, targets=tgt)
                tgt_ext = torch.cat([tgt, torch.full((len(bd), 1), 4, dtype=torch.long)], 1)
                loss = F.cross_entropy(lo.transpose(1, 2), tgt_ext)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
    enc.eval()
    head.eval()
    res = {}
    with torch.no_grad():
        if head_kind == "ctc":
            ok = 0
            for i in va:
                if meta[i]["mixed"]:
                    continue
                h = enc(batch_sig(sig, [i]))
                pred = greedy_ctc(head(h)[0].numpy())
                vals, cnts = np.unique(list(pred), return_counts=True)
                p = vals[np.argmax(cnts)] if len(vals) else "-"
                ok += p == meta[i]["base"]
            nv = len([i for i in va if not meta[i]["mixed"]])
            res["T1_identity_acc"] = ok / max(1, nv)
            cer = []
            for m in mixed:
                i = m["idx"]
                h = enc(batch_sig(sig, [i]))
                pred = greedy_ctc(head(h)[0].numpy())
                truth = m["true_full"]
                cer.append(lev(pred, truth) / max(1, len(truth)))
            res["T2_mixed_CER"] = float(np.mean(cer)) if cer else None
        elif head_kind == "frame":
            ok = 0
            for i in va:
                if meta[i]["mixed"]:
                    continue
                h = enc(batch_sig(sig, [i]))
                lab = to_frames(frame_labels_homopolymer(sig[i], meta[i]["base"]), h.shape[1])
                pr = head(h)[0].argmax(-1).numpy()
                msk = lab != -100
                ok += (pr[msk] == lab[msk]).mean()
            nv = len([i for i in va if not meta[i]["mixed"]])
            res["T1_frame_acc_homopolymer"] = ok / max(1, nv)
            accs = []
            for m in mixed:
                i = m["idx"]
                h = enc(batch_sig(sig, [i]))
                lab800 = np.array([C2I.get(b, -100) for b in m["register_labels"]])
                lab = to_frames(lab800, h.shape[1])
                pr = head(h)[0].argmax(-1).numpy()
                msk = lab != -100
                accs.append(float((pr[msk] == lab[msk]).mean()) if msk.any() else 0.0)
            res["T3_mixed_frame_acc"] = float(np.mean(accs)) if accs else None
        else:
            ok = 0
            for i in va:
                if meta[i]["mixed"]:
                    continue
                h = enc(batch_sig(sig, [i]))
                lo = head(h)[0]
                ids = lo.argmax(-1).numpy()
                pred = "".join(ALPH[c] for c in ids if c < 4)
                vals, cnts = np.unique(list(pred), return_counts=True)
                p = vals[np.argmax(cnts)] if len(vals) else "-"
                ok += p == meta[i]["base"]
            nv = len([i for i in va if not meta[i]["mixed"]])
            res["T1_identity_acc"] = ok / max(1, nv)
            cer = []
            for m in mixed:
                i = m["idx"]
                h = enc(batch_sig(sig, [i]))
                lo = head(h)[0]
                ids = lo.argmax(-1).numpy()
                pred = "".join(ALPH[c] for c in ids if c < 4)
                truth = m["true_full"]
                cer.append(lev(pred, truth) / max(1, len(truth)))
            res["T2_mixed_CER"] = float(np.mean(cer)) if cer else None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs_bench")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sig, meta = load()
    homo = [i for i, m in enumerate(meta) if not m["mixed"] and m["base"] in C2I]
    random.seed(a.seed)
    random.shuffle(homo)
    nval = max(1, int(len(homo) * 0.2))
    va, tr = homo[:nval], homo[nval:]
    mixed = []
    for i, m in enumerate(meta):
        if m["mixed"] and m.get("register_labels"):
            mixed.append(dict(idx=i, id=m["id"], register_labels=m["register_labels"],
                              true_full=true_read_full(m)))
    print("train=%d val=%d mixed=%d" % (len(tr), len(va), len(mixed)))
    bench = {}
    for enc_kind in ["cnn", "conformer"]:
        for head_kind in ["ctc", "frame", "seq2seq"]:
            key = "%s+%s" % (enc_kind, head_kind)
            r = run_task(enc_kind, head_kind, sig, meta, tr, va, mixed, a.epochs, a.seed)
            bench[key] = r
            print("%-18s %s" % (key, json.dumps(r)))
    bench["_baseline_history"] = dict(
        ctc_zscore_CER=1.0, ssl_ctc_zscore_CER=1.0,
        cnn_win_frac_identity=0.848, cnn_win_frac_mixed_CER=0.85)
    with open(os.path.join(a.out, "benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(bench, f, indent=1)
    print("saved", os.path.join(a.out, "benchmark.json"))


if __name__ == "__main__":
    main()