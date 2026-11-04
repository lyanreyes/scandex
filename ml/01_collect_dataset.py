"""
Step 1 - Dataset acquisition.

Downloads card metadata (incl. market prices) and card images for the sets in
scope from the Pokemon TCG API  https://api.pokemontcg.io/v2
(TCGplayer USD + Cardmarket EUR prices).

Outputs
  data/raw/pokemon_cards.json            raw API payload
  data/images/pokemon/<card_id>.jpg      200x280 thumbnails

Usage:  python 01_collect_dataset.py            (optionally set POKEMONTCG_API_KEY)
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
IMG = ROOT / "data" / "images"
THUMB = (200, 280)

# Pokemon scope: Scarlet & Violet and Mega Evolution era sets (2023 onwards)
POKEMON_SERIES = {"Scarlet & Violet", "Mega Evolution"}
POKEMON_API = "https://api.pokemontcg.io/v2"

session = requests.Session()
if os.environ.get("POKEMONTCG_API_KEY"):
    session.headers["X-Api-Key"] = os.environ["POKEMONTCG_API_KEY"]


def get_json(url, params=None, tries=12):
    for attempt in range(tries):
        try:
            r = session.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # the free APIs occasionally return 5xx
            print(f"  retry {attempt + 1} {url} {params}: {e}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed: {url} {params}")


def collect_pokemon():
    sets = get_json(f"{POKEMON_API}/sets", {"pageSize": 250})["data"]
    sets = [s for s in sets if s["series"] in POKEMON_SERIES]
    print(f"Pokemon: {len(sets)} sets in scope")
    cards = []
    cache = RAW / "pokemon_sets"
    cache.mkdir(exist_ok=True)
    for s in sorted(sets, key=lambda s: s["releaseDate"]):
        f = cache / f"{s['id']}.json"
        if f.exists():  # resume: sets already fetched are not requested again
            set_cards = json.loads(f.read_text(encoding="utf-8"))
        else:
            set_cards, page = [], 1
            while True:  # small pages: the free API times out on pageSize=250
                d = get_json(f"{POKEMON_API}/cards",
                             {"q": f"set.id:{s['id']}", "pageSize": 60, "page": page})
                set_cards += d["data"]
                if page * 60 >= d["totalCount"]:
                    break
                page += 1
            f.write_text(json.dumps(set_cards), encoding="utf-8")
        cards += set_cards
        print(f"  {s['id']:<12} {s['name']:<40} total so far {len(cards)}")
    return cards


def download(job):
    url, out = job
    if out.exists():
        return True
    try:
        r = session.get(url, timeout=60)
        r.raise_for_status()
        im = Image.open(BytesIO(r.content)).convert("RGB").resize(THUMB, Image.LANCZOS)
        im.save(out, quality=92)
        return True
    except Exception as e:
        print(f"  image failed {url}: {e}")
        return False


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    jobs = []
    if "--skip-pokemon" not in sys.argv:
        pk = collect_pokemon()
        (RAW / "pokemon_cards.json").write_text(json.dumps(pk), encoding="utf-8")
    pk = json.loads((RAW / "pokemon_cards.json").read_text(encoding="utf-8"))
    (IMG / "pokemon").mkdir(parents=True, exist_ok=True)
    jobs += [(c["images"]["small"], IMG / "pokemon" / f"{c['id']}.jpg") for c in pk]

    print(f"Downloading {len(jobs)} images ...")
    with ThreadPoolExecutor(16) as ex:
        ok = sum(ex.map(download, jobs))
    print(f"done: {ok}/{len(jobs)} images available")


if __name__ == "__main__":
    main()
