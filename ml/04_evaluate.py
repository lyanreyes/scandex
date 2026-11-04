"""
Step 4 - Model training, experiments and evaluation.

Task: given a camera capture of a card, retrieve the correct card out of the
full gallery (~4.8k Pokemon cards). Retrieval is k-nearest
neighbour search over the fingerprints from features.py.

Because collecting thousands of labelled phone photos is out of reach for this
project, query images are produced by a camera-capture simulator (perspective,
rotation, framing error, lighting, white balance, glare, blur, sensor noise,
JPEG). Real photos can be evaluated separately with 05_evaluate_real.py.

Experiments
  E1  single fingerprints vs. fusion        (aHash, dHash, pHash-64, pHash-256, colour, HSV hist)
  E2  hash size (pHash 64 vs 256 bits)     (parameter experiment)
  E3  capture severity 1 / 2 / 3            (robustness)
  E4  confidence threshold                  (precision vs. auto-accept coverage)

Fusion model (two logistic regressions, both fitted on TRAIN cards only):
  ranker     - pairwise logistic regression (RankNet-style) on the difference of
               distance vectors D(neg) - D(pos); its weights give the score
               s = w . D used to sort the gallery (lower = better).
  calibrator - logistic regression on [s1, s2 - s1] of a query's best two
               candidates -> probability that the top-1 answer is correct.
               Fitted on a separate group of train cards, not the ranker's.
TEST cards are never seen while fitting. Exported to ../web/model/fusion.json.
"""
import json
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression

from features import extract, hex_to_bits

ROOT = Path(__file__).parent
IMG, PROC, OUT = ROOT / "data" / "images", ROOT / "data" / "processed", ROOT / "outputs"
WEB_MODEL = ROOT.parent / "web" / "model"
OUT.mkdir(exist_ok=True)
SEED = 42
N_TRAIN_CARDS, N_CALIB_CARDS, N_TEST_CARDS = 600, 300, 1200   # test is capped by the 30 % split
FEATS = ["a", "d", "p", "P", "c", "h"]
LABELS = {"a": "aHash-64", "d": "dHash-64", "p": "pHash-64", "P": "pHash-256", "c": "Colour layout",
          "h": "HSV histogram", "sum": "Equal-weight sum", "lr": "Pairwise-LR fusion"}

SEVERITY = {
    1: dict(persp=.02, rot=2, frame=.02, blur=.5, noise=2, light=.10, tint=.03, glare=.0, jpeg=90),
    2: dict(persp=.05, rot=5, frame=.04, blur=1.2, noise=5, light=.20, tint=.06, glare=.4, jpeg=75),
    3: dict(persp=.09, rot=9, frame=.07, blur=2.2, noise=9, light=.32, tint=.10, glare=.7, jpeg=60),
}


def simulate_capture(card, sev, rng):
    """Render the card as a phone camera would see it inside the scan guide."""
    P = SEVERITY[sev]
    h, w = card.shape[:2]
    H, W = int(h * 1.6), int(w * 1.6)
    bg = np.full((H, W, 3), rng.integers(30, 220, 3), dtype=np.float32)
    bg += rng.normal(0, 12, (H, W, 1))
    cx, cy = W / 2, H / 2
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src - [w / 2, h / 2] + rng.uniform(-P["persp"], P["persp"], (4, 2)) * [w, h]
    t = np.deg2rad(rng.uniform(-P["rot"], P["rot"]))
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    dst = (dst @ R.T + [cx, cy]).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(card.astype(np.float32), M, (W, H))
    mask = cv2.warpPerspective(np.ones((h, w), np.float32), M, (W, H))[..., None]
    img = warped * mask + bg * (1 - mask)

    # the user aligns the card with the on-screen guide: framing error
    s = 1 + rng.uniform(-P["frame"], P["frame"])
    gw, gh = w * s, h * s
    ox, oy = rng.uniform(-P["frame"], P["frame"], 2) * [w, h]
    x0, y0 = int(cx - gw / 2 + ox), int(cy - gh / 2 + oy)
    img = img[max(y0, 0):y0 + int(gh), max(x0, 0):x0 + int(gw)]
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    # lighting: exposure, contrast, gamma, white balance
    img = img * (1 + rng.uniform(-P["light"], P["light"]))
    img = (img - 128) * (1 + rng.uniform(-P["light"], P["light"]) / 2) + 128
    img *= 1 + rng.uniform(-P["tint"], P["tint"], 3)
    img = 255 * (np.clip(img, 0, 255) / 255) ** rng.uniform(1 - P["light"], 1 + P["light"])
    # specular glare from holo foil / sleeves
    if rng.random() < P["glare"]:
        gx, gy = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(.15, .35) * w
        yy, xx = np.mgrid[0:h, 0:w]
        blob = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * r * r))
        img += blob[..., None] * rng.uniform(80, 170)
    sigma = rng.uniform(0, P["blur"])
    if sigma > .2:
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    img += rng.normal(0, P["noise"], img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, P["jpeg"]])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1]


class Gallery:
    def __init__(self):
        z = np.load(PROC / "features.npz", allow_pickle=True)
        self.game, self.key = z["game"], z["key"]
        self.bits = {f: z[f].astype(np.uint8) for f in "adpP"}
        self.vec = {k: z[k] / (np.linalg.norm(z[k], axis=1, keepdims=True) + 1e-9) for k in "ch"}
        # equivalence classes: pixel-identical images count as the same answer
        groups = json.loads((PROC / "dup_groups.json").read_text())
        self.cls = np.arange(len(self.key))
        idx = {k: i for i, k in enumerate(self.key)}
        for g in groups:
            ids = [idx[k] for k in g if k in idx]
            self.cls[ids] = min(ids)

    def distances(self, f):
        """(6, N) matrix of normalised distances from one query to all cards."""
        D = [(self.bits[k] != hex_to_bits(f[k])[None, :]).mean(1) for k in "adpP"]
        for k in "ch":
            q = np.asarray(f[k], np.float64)
            q /= np.linalg.norm(q) + 1e-9
            D.append((1 - self.vec[k] @ q) / 2)
        return np.stack(D)


def build_queries(G, idx, sevs, rng, per_card=1):
    Q = []
    for i in idx:
        card = np.asarray(Image.open(IMG / G.game[i] / f"{G.key[i]}.jpg").convert("RGB"))
        for sev in sevs:
            for _ in range(per_card):
                img = simulate_capture(card, sev, rng)
                t = time.perf_counter()
                D = G.distances(extract(img))           # feature extraction + full gallery scan
                Q.append((i, sev, D, (time.perf_counter() - t) * 1000))
    return Q


def baseline_sum(D):
    return D[2:4].mean(0) + D[1] * .5 + D[4]


def fit_ranker(G, Q, rng):
    """Pairwise logistic regression: learn w so that w.D(pos) < w.D(neg)."""
    X = []
    for i, _, D, _ in Q:
        pos = G.cls == G.cls[i]
        hard = [j for j in np.argsort(baseline_sum(D))[:60] if not pos[j]][:30]  # look-alikes
        rand = rng.choice(np.flatnonzero(~pos), 20, replace=False)
        for j in list(hard) + list(rand):
            X.append(D[:, j] - D[:, i])
    X = np.array(X)
    X, y = np.vstack([X, -X]), np.r_[np.ones(len(X)), np.zeros(len(X))]
    lr = LogisticRegression(fit_intercept=False, C=10, max_iter=5000).fit(X, y)
    return lr.coef_[0]


def top2(w, D):
    s = w @ D
    o = np.argpartition(s, 2)[:2]
    o = o[np.argsort(s[o])]
    return s, o[0], s[o[0]], s[o[1]] - s[o[0]]


def fit_calibrator(G, Q, w):
    X, y = [], []
    for i, _, D, _ in Q:
        _, top, s1, gap = top2(w, D)
        X.append([s1, gap]); y.append(G.cls[top] == G.cls[i])
    return LogisticRegression(max_iter=2000).fit(np.array(X), np.array(y))


def prob(cal, s1, gap):
    z = cal.intercept_[0] + cal.coef_[0] @ [s1, gap]
    return 1 / (1 + np.exp(-z))


def rank_of_truth(G, i, score):
    """1-based rank of the correct class when gallery is sorted by ascending score."""
    true_best = score[G.cls == G.cls[i]].min()
    return int((score < true_best).sum()) + 1


def evaluate(G, Q, w, cal):
    rows, conf = [], []
    for i, sev, D, ms in Q:
        scores = {f: D[k] for k, f in enumerate(FEATS)}
        scores["sum"] = baseline_sum(D)
        t = time.perf_counter()
        s, top, s1, gap = top2(w, D)
        p = prob(cal, s1, gap)
        lat = ms + (time.perf_counter() - t) * 1000
        scores["lr"] = s
        for m, sc in scores.items():
            r = rank_of_truth(G, i, sc)
            rows.append(dict(card=G.key[i], game=G.game[i], severity=sev, method=m, rank=r,
                             top1=r == 1, top5=r <= 5, rr=1 / r))
        conf.append(dict(severity=sev, game=G.game[i], prob=p,
                         correct=G.cls[top] == G.cls[i], ms=lat))
    return pd.DataFrame(rows), pd.DataFrame(conf)


def figures(res, conf, G, rng):
    order = ["a", "d", "p", "P", "c", "h", "sum", "lr"]
    s = res.groupby("method")[["top1", "top5"]].mean().loc[order] * 100
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    x = np.arange(len(order))
    ax.bar(x - .2, s.top1, .4, label="Top-1", color="#743BD2")
    ax.bar(x + .2, s.top5, .4, label="Top-5", color="#9CFF27", edgecolor="#555")
    ax.set_xticks(x, [LABELS[m] for m in order], rotation=25, ha="right")
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100); ax.legend()
    for xi, v in zip(x, s.top1):
        ax.text(xi - .2, v + 1, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_title("Accuracy per method (test cards, all severities)")
    fig.tight_layout(); fig.savefig(OUT / "eval_methods.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    for m in ["P", "p", "d", "c", "h", "sum", "lr"]:
        v = res[res.method == m].groupby("severity").top1.mean() * 100
        ax.plot(v.index, v.values, marker="o", label=LABELS[m], lw=2.5 if m == "lr" else 1.2)
    ax.set_xticks([1, 2, 3], ["Mild", "Moderate", "Harsh"])
    ax.set_xlabel("Capture severity"); ax.set_ylabel("Top-1 accuracy (%)")
    ax.legend(fontsize=7); ax.set_title("Robustness to capture conditions")
    fig.tight_layout(); fig.savefig(OUT / "eval_severity.png", dpi=200); plt.close(fig)

    th = np.linspace(0, 1, 101)
    cov = [(conf.prob >= t).mean() * 100 for t in th]
    prec = [conf[conf.prob >= t].correct.mean() * 100 if (conf.prob >= t).any() else np.nan
            for t in th]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(th, prec, label="Precision of auto-accepted scans", color="#743BD2")
    ax.plot(th, cov, label="Coverage (scans auto-accepted)", color="#E3350D")
    ax.set_xlabel("Calibrated confidence threshold"); ax.set_ylabel("%"); ax.legend(fontsize=7)
    ax.set_title("Confidence threshold trade-off")
    fig.tight_layout(); fig.savefig(OUT / "eval_threshold.png", dpi=200); plt.close(fig)

    keys = rng.choice(len(G.key), 3, replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(6, 6.4))
    for row, i in zip(axes, keys):
        card = np.asarray(Image.open(IMG / G.game[i] / f"{G.key[i]}.jpg").convert("RGB"))
        row[0].imshow(card); row[0].set_title("Reference", fontsize=7)
        for sev, ax in zip([1, 2, 3], row[1:]):
            ax.imshow(simulate_capture(card, sev, rng)); ax.set_title(f"Severity {sev}", fontsize=7)
        for ax in row:
            ax.axis("off")
    fig.suptitle("Simulated camera captures"); fig.tight_layout()
    fig.savefig(OUT / "eval_capture_examples.png", dpi=200); plt.close(fig)


def main():
    rng = np.random.default_rng(SEED)
    G = Gallery()
    n = len(G.key)
    perm = rng.permutation(n)
    split = int(n * .7)
    train_idx = perm[:split][:N_TRAIN_CARDS]
    calib_idx = perm[:split][N_TRAIN_CARDS:N_TRAIN_CARDS + N_CALIB_CARDS]
    test_idx = perm[split:][:N_TEST_CARDS]
    print(f"Gallery {n} cards | train {len(train_idx)} | calibration {len(calib_idx)} | test {len(test_idx)}")

    t = time.time()
    Qtr = build_queries(G, train_idx, [1, 2, 3], rng)
    w = fit_ranker(G, Qtr, rng)
    Qcal = build_queries(G, calib_idx, [1, 2, 3], rng)
    cal = fit_calibrator(G, Qcal, w)
    print(f"Ranker weights {dict(zip(FEATS, w.round(3)))}  calibrator {cal.coef_[0].round(3)} "
          f"b={cal.intercept_[0]:.3f} ({time.time() - t:.0f}s)")

    t = time.time()
    Qte = build_queries(G, test_idx, [1, 2, 3], rng)
    res, conf = evaluate(G, Qte, w, cal)
    print(f"Evaluated {len(Qte)} test queries ({time.time() - t:.0f}s)")
    print(f"Latency (extract + search {n} cards + rank, Python): mean {conf.ms.mean():.1f} ms, p95 {conf.ms.quantile(.95):.1f} ms")
    (OUT / "eval_latency.txt").write_text(f"{conf.ms.mean():.2f},{conf.ms.quantile(.95):.2f}")

    summary = (res.groupby("method")[["top1", "top5", "rr"]].mean()
               .rename(columns={"rr": "MRR"}).mul([100, 100, 1]))
    by_sev = res.pivot_table(index="method", columns="severity", values="top1", aggfunc="mean") * 100
    by_game = res.pivot_table(index="method", columns="game", values="top1", aggfunc="mean") * 100
    summary.index = [LABELS[m] for m in summary.index]
    by_sev.index = [LABELS[m] for m in by_sev.index]
    by_game.index = [LABELS[m] for m in by_game.index]
    print("\n", summary.round(2), "\n\nTop-1 by severity\n", by_sev.round(2),
          "\n\nTop-1 by game\n", by_game.round(2))

    for t_ in [.5, .7, .8, .9]:
        sel = conf[conf.prob >= t_]
        print(f"threshold {t_}: coverage {len(sel) / len(conf):.1%}  precision {sel.correct.mean():.1%}")

    res.to_csv(OUT / "eval_queries.csv", index=False)
    summary.round(3).to_csv(OUT / "eval_summary.csv")
    by_sev.round(3).to_csv(OUT / "eval_by_severity.csv")
    by_game.round(3).to_csv(OUT / "eval_by_game.csv")
    thr = pd.DataFrame([dict(threshold=t_, coverage=(conf.prob >= t_).mean(),
                             precision=conf[conf.prob >= t_].correct.mean())
                        for t_ in np.round(np.arange(.3, 1.0, .05), 2)])
    thr.round(4).to_csv(OUT / "eval_threshold.csv", index=False)
    figures(res, conf, G, rng)

    # auto-accept threshold: lowest confidence whose precision reaches 97 %
    ok = thr[thr.precision >= .97]
    auto = float(ok.threshold.min()) if len(ok) else 0.95
    print(f"auto-accept threshold = {auto}")
    fusion = {"features": FEATS, "weights": w.tolist(),
              "calibration": {"bias": float(cal.intercept_[0]), "s1": float(cal.coef_[0][0]),
                              "gap": float(cal.coef_[0][1])},
              "autoAccept": auto,
              "trainedOn": f"{len(Qtr)} simulated captures of {len(train_idx)} cards; "
                           f"calibrated on {len(Qcal)} captures of {len(calib_idx)} other cards",
              "seed": SEED}
    (WEB_MODEL / "fusion.json").write_text(json.dumps(fusion, indent=2))
    print("wrote web/model/fusion.json")


if __name__ == "__main__":
    main()
