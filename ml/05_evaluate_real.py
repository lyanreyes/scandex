"""
Step 5 (optional) - accuracy on real phone/webcam photos.

Put photos in data/real/, cropped to the card like the app's scan guide, and
name each file after its Pokemon TCG API card id: sv3pt5-25.jpg, sv8-238.jpg.
A suffix after "__" is ignored, so several shots of one card can be added:
sv3pt5-25__glare.jpg

Output: outputs/real_eval.csv and a printed top-1 / top-5 summary.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from features import extract

ROOT = Path(__file__).parent
REAL, OUT = ROOT / "data" / "real", ROOT / "outputs"


def main():
    G = __import__("04_evaluate").Gallery()
    fusion = json.loads((ROOT.parent / "web" / "model" / "fusion.json").read_text())
    w, b = np.array(fusion["weights"]), fusion["bias"]
    idx = {k: i for i, k in enumerate(G.key)}
    rows = []
    photos = sorted(p for p in REAL.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not photos:
        print(f"No photos in {REAL} - see the docstring for naming.")
        return
    for p in photos:
        key = p.stem.split("__")[0]
        if key not in idx:
            print(f"skip {p.name}: {key} is not in the index")
            continue
        im = ImageOps.exif_transpose(Image.open(p)).convert("RGB").resize((200, 280), Image.LANCZOS)
        t = time.perf_counter()
        D = G.distances(extract(np.asarray(im)))
        logit = w @ D + b
        ms = (time.perf_counter() - t) * 1000
        i = idx[key]
        score = -logit
        rank = int((score < score[G.cls == G.cls[i]].min()).sum()) + 1
        top = int(np.argmax(logit))
        rows.append(dict(photo=p.name, card=key, rank=rank, top1=rank == 1, top5=rank <= 5,
                         predicted=G.key[top], confidence=1 / (1 + np.exp(-logit[top])), ms=ms))
    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "real_eval.csv", index=False)
    print(df.to_string(index=False))
    print(f"\n{len(df)} photos | top-1 {df.top1.mean():.1%} | top-5 {df.top5.mean():.1%} | "
          f"MRR {(1 / df['rank']).mean():.3f} | {df.ms.mean():.1f} ms/query")


if __name__ == "__main__":
    main()
