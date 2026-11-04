"""
Step 3 - Feature extraction / visual index.

Computes the fingerprints from features.py for every clean card image and
writes them twice:
  data/processed/features.npz   gallery matrices used by 04_evaluate.py
  ../web/model/pokemon.json     compact index loaded by the web app
"""
import base64
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from features import extract, hex_to_bits

ROOT = Path(__file__).parent
RAW, IMG, PROC = ROOT / "data" / "raw", ROOT / "data" / "images", ROOT / "data" / "processed"
WEB_MODEL = ROOT.parent / "web" / "model"
WEB_MODEL.mkdir(parents=True, exist_ok=True)


def raw_lookup():
    pk = {c["id"]: c for c in json.loads((RAW / "pokemon_cards.json").read_text(encoding="utf-8"))}
    return {"pokemon": pk}


def main():
    cards = pd.read_csv(PROC / "cards.csv", dtype={"number": str})
    raw = raw_lookup()
    feats, t0 = [], time.time()
    for g, k in zip(cards.game, cards.card_key):
        feats.append(extract(np.asarray(Image.open(IMG / g / f"{k}.jpg").convert("RGB"))))
    dt = (time.time() - t0) / len(cards)
    print(f"Extracted {len(cards)} fingerprints ({dt * 1000:.1f} ms/card)")

    np.savez_compressed(
        PROC / "features.npz",
        game=cards.game.values, key=cards.card_key.values,
        a=np.stack([hex_to_bits(f["a"]) for f in feats]),
        d=np.stack([hex_to_bits(f["d"]) for f in feats]),
        p=np.stack([hex_to_bits(f["p"]) for f in feats]),
        P=np.stack([hex_to_bits(f["P"]) for f in feats]),
        c=np.array([f["c"] for f in feats], dtype=np.float64),
        h=np.array([f["h"] for f in feats], dtype=np.float64),
    )

    for game in ["pokemon"]:
        rows = []
        for (_, r), f in zip(cards.iterrows(), feats):
            if r.game != game:
                continue
            src = raw[game][r.card_key]
            s = src["set"]
            meta = [r.card_key, src["name"], s["name"], src["number"], s.get("printedTotal"),
                    s.get("ptcgoCode") or "", src.get("rarity") or "", src["images"]["small"]]
            price = None if pd.isna(r.market_usd) else round(float(r.market_usd), 2)
            vec = [base64.b64encode(np.array(f[k], dtype=np.int8).tobytes()).decode() for k in "ch"]
            rows.append(meta + [price, f["a"], f["d"], f["p"], f["P"], *vec])
        out = {
            "game": game, "built": time.strftime("%Y-%m-%d"), "count": len(rows),
            "fields": ["key", "name", "set", "number", "printedTotal", "setCode", "rarity",
                       "image", "priceSnapshot", "a", "d", "p", "P", "c", "h"],
            "cards": rows,
        }
        path = WEB_MODEL / f"{game}.json"
        path.write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
        print(f"  {path.name}: {len(rows)} cards, {path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
