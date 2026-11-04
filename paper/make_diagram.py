"""Figure 1 - ScanDex system architecture (saved to paper/fig_architecture.png)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).parent / "fig_architecture.png"
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")


def box(x, y, w, h, title, sub="", fc="#EFE9FF", ec="#743BD2"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2 + (1.6 if sub else 0), title, ha="center", va="center",
            fontsize=8.2, weight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 2.2, sub, ha="center", va="center", fontsize=6.6, color="#333")


def arrow(x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", lw=1, color="#444"))


ax.text(1, 60, "Offline (Python)", fontsize=8.5, weight="bold", color="#743BD2")
box(1, 46, 21, 10, "Data collection", "Pokémon TCG API\nOPTCG API")
box(27, 46, 21, 10, "Preprocessing\n& EDA", "duplicates, missing,\noutliers")
box(53, 46, 21, 10, "Feature extraction", "aHash, dHash, pHash,\ncolour layout")
box(78, 46, 21, 10, "LR fusion training", "simulated captures,\nhard negatives")
arrow(22.6, 51, 26.4, 51); arrow(48.6, 51, 52.4, 51); arrow(74.6, 51, 77.4, 51)

ax.text(1, 38, "Online (browser, phone or PC)", fontsize=8.5, weight="bold", color="#3A7D00")
G = dict(fc="#F2FFE0", ec="#5AA800")
box(1, 20, 17, 12, "Camera / photo", "getUserMedia,\nscan guide crop", **G)
box(22, 20, 17, 12, "Fingerprints", "same code as\nPython (parity)", **G)
box(43, 20, 17, 12, "k-NN retrieval", "~9k cards,\nLR probability", **G)
box(64, 20, 15, 12, "OCR check", "collector no.\nTesseract.js", **G)
box(83, 20, 16, 12, "Live price", "TCGplayer USD\nCardmarket EUR", **G)
arrow(18.6, 26, 21.4, 26); arrow(39.6, 26, 42.4, 26); arrow(60.6, 26, 63.4, 26); arrow(79.6, 26, 82.4, 26)
box(43, 2, 56, 10, "Digital binder", "stored on device · total value · price refresh · CSV export", **G)
arrow(91, 19.4, 91, 12.6)

# model artefacts flow down into the browser
ax.annotate("", (51.5, 32.6), (63.5, 45.4), arrowprops=dict(arrowstyle="-|>", lw=1, ls="--", color="#743BD2"))
ax.annotate("", (52.5, 32.6), (88.5, 45.4), arrowprops=dict(arrowstyle="-|>", lw=1, ls="--", color="#743BD2"))
ax.text(84, 36.5, "index.json + fusion.json", fontsize=6.8, color="#743BD2", ha="center", style="italic")

fig.tight_layout()
fig.savefig(OUT, dpi=250)
print(OUT)
