"""Writes tests/parity_fixture.json: sample images + the fingerprints Python computes.
Uses real card thumbnails when available, otherwise synthetic images."""
import base64, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).parents[1] / "ml"))
from features import extract

rng = np.random.default_rng(0)
imgs = []
for p in sorted((Path(__file__).parents[1] / "ml/data/images").glob("*/*.jpg"))[::400][:12]:
    imgs.append(np.asarray(Image.open(p).convert("RGB")))
for w, h in [(200, 280), (630, 880), (157, 233)]:
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([xx * 255 / w, yy * 255 / h, (xx + yy) % 256], -1)
    imgs.append(np.clip(base + rng.normal(0, 30, base.shape), 0, 255).astype(np.uint8))
out = []
for im in imgs:
    h, w = im.shape[:2]
    rgba = np.concatenate([im, np.full((h, w, 1), 255, np.uint8)], -1)
    out.append({"w": w, "h": h, "rgba": base64.b64encode(rgba.tobytes()).decode(), "expected": extract(im)})
(Path(__file__).parent / "parity_fixture.json").write_text(json.dumps(out))
print(f"{len(out)} fixtures")
