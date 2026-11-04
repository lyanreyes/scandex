"""
Step 2 - Data preprocessing and exploratory data analysis (EDA).

Input : data/raw/*.json, data/images/<game>/*.jpg
Output: data/processed/cards.csv        one clean row per card
        data/processed/dup_groups.json  cards whose images are pixel-identical
        outputs/eda_*.png               figures used in the paper
        outputs/eda_summary.txt         descriptive statistics
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

ROOT = Path(__file__).parent
RAW, IMG = ROOT / "data" / "raw", ROOT / "data" / "images"
PROC, OUT = ROOT / "data" / "processed", ROOT / "outputs"
PROC.mkdir(parents=True, exist_ok=True)
OUT.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", context="paper")

# TCGplayer price variant chosen as "the" market value of a Pokemon card.
# Reverse holo is only used when the card has no other printing.
VARIANT_PRIORITY = ["normal", "holofoil", "1stEditionHolofoil", "unlimitedHolofoil",
                    "reverseHolofoil"]

log = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.append(s)


def pokemon_frame():
    cards = json.loads((RAW / "pokemon_cards.json").read_text(encoding="utf-8"))
    rows = []
    for c in cards:
        prices = (c.get("tcgplayer") or {}).get("prices") or {}
        variant = next((v for v in VARIANT_PRIORITY if v in prices), next(iter(prices), None))
        market = (prices.get(variant) or {}).get("market") if variant else None
        cm = ((c.get("cardmarket") or {}).get("prices") or {}).get("trendPrice")
        s = c["set"]
        rows.append({
            "game": "pokemon", "card_key": c["id"], "name": c["name"],
            "set_id": s["id"], "set_name": s["name"], "number": c["number"],
            "printed_total": s.get("printedTotal"), "rarity": c.get("rarity"),
            "supertype": c.get("supertype"), "hp": c.get("hp"),
            "release_date": s["releaseDate"], "price_variant": variant,
            "market_usd": market, "cardmarket_eur": cm,
            "price_updated": (c.get("tcgplayer") or {}).get("updatedAt"),
        })
    return pd.DataFrame(rows)


def image_md5(path):
    return hashlib.md5(np.asarray(Image.open(path).convert("RGB")).tobytes()).hexdigest()


def main():
    df = pokemon_frame()
    say(f"Raw rows: {len(df)}")

    # --- duplicates -------------------------------------------------------
    dup_keys = df.duplicated(["game", "card_key"], keep="first")
    say(f"Duplicate card_key rows removed: {dup_keys.sum()}")
    df = df[~dup_keys].copy()

    # --- missing images ---------------------------------------------------
    df["image_path"] = [str(IMG / g / f"{k}.jpg") for g, k in zip(df.game, df.card_key)]
    has_img = df.image_path.map(lambda p: Path(p).exists())
    say(f"Rows without a downloadable image (dropped): {(~has_img).sum()}")
    df = df[has_img].copy()

    # --- inconsistent formats ---------------------------------------------
    df["hp"] = pd.to_numeric(df.hp, errors="coerce")
    df["market_usd"] = pd.to_numeric(df.market_usd, errors="coerce")
    df["release_date"] = pd.to_datetime(df.release_date, format="%Y/%m/%d", errors="coerce")
    df["rarity"] = df.rarity.fillna("Unknown").str.strip()
    num = pd.to_numeric(df.number.str.extract(r"(\d+)")[0], errors="coerce")
    df["secret_rare"] = (df.game == "pokemon") & (num > df.printed_total)
    df.loc[df.market_usd <= 0, "market_usd"] = np.nan

    # --- missing values -----------------------------------------------------
    for g, sub in df.groupby("game"):
        say(f"[{g}] cards={len(sub)}  missing market price={sub.market_usd.isna().sum()} "
            f"({sub.market_usd.isna().mean():.1%})  missing rarity="
            f"{(sub.rarity == 'Unknown').sum()}")
    # Missing prices are NOT imputed: an invented price is worse than "no data"
    # for a valuation app. Those cards stay in the visual index (they can still
    # be recognised) but are excluded from price statistics.

    # --- outliers -----------------------------------------------------------
    df["log_price"] = np.log10(df.market_usd)
    for g, sub in df.dropna(subset=["log_price"]).groupby("game"):
        q1, q3 = sub.log_price.quantile([.25, .75])
        hi = q3 + 1.5 * (q3 - q1)
        n_hi = (sub.log_price > hi).sum()
        say(f"[{g}] price outliers above log10 IQR fence (>${10 ** hi:,.2f}): {n_hi} "
            f"- kept, they are genuine chase cards")

    # --- identical images (indistinguishable by any visual method) ---------
    md5 = df.image_path.map(image_md5)
    groups = [list(v) for v in df.groupby(md5).card_key.apply(list) if len(v) > 1]
    say(f"Groups of pixel-identical card images: {len(groups)} "
        f"covering {sum(map(len, groups))} cards (treated as equivalent at evaluation)")
    (PROC / "dup_groups.json").write_text(json.dumps(groups))

    df.drop(columns=["image_path"]).to_csv(PROC / "cards.csv", index=False)
    say(f"Clean rows saved: {len(df)}")

    # ---------------- descriptive statistics ----------------
    priced = df.dropna(subset=["market_usd"])
    say("\nMarket price (USD):")
    say(priced.groupby("game").market_usd.describe(percentiles=[.5, .9, .99]).round(2).to_string())

    # ---------------- figures ----------------
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(priced.log_price, bins=50, color="#E3350D", edgecolor="#B71C1C",
            label=f"priced cards (n={len(priced)})")
    ax.set_xlabel("Market price, log10(USD)")
    ax.set_ylabel("Cards")
    ax.set_xticks([-2, -1, 0, 1, 2, 3])
    ax.set_xticklabels(["$0.01", "$0.10", "$1", "$10", "$100", "$1,000"])
    ax.legend()
    ax.set_title("Distribution of card market prices")
    fig.tight_layout(); fig.savefig(OUT / "eda_price_distribution.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    order = priced.groupby("rarity").market_usd.median().sort_values(ascending=False)
    order = order[priced.rarity.value_counts()[order.index] >= 10]
    sns.barplot(x=order.values, y=order.index, ax=ax, color="#3B4CCA")
    ax.set_xscale("log"); ax.set_xlabel("Median market price (USD, log)")
    ax.set_ylabel(""); ax.set_title("Median price by rarity (rarities with >= 10 cards)")
    fig.tight_layout(); fig.savefig(OUT / "eda_price_by_rarity.png", dpi=200); plt.close(fig)

    pkp = priced[priced.game == "pokemon"]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.scatter(pkp.release_date, pkp.market_usd, s=4, alpha=.35,
               c=np.where(pkp.secret_rare, "#E3350D", "#888888"))
    ax.set_yscale("log"); ax.set_ylabel("Market price (USD, log)")
    ax.set_title("Pokemon price vs. set release (red = secret rare)")
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(OUT / "eda_price_vs_release.png", dpi=200); plt.close(fig)

    pkp = pkp.assign(age_days=(pd.Timestamp("2026-11-04") - pkp.release_date).dt.days,
                     secret=pkp.secret_rare.astype(int),
                     is_pokemon=(pkp.supertype == "Pokémon").astype(int))
    corr = pkp[["log_price", "hp", "age_days", "secret", "is_pokemon"]].corr()
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation (Pokemon)")
    fig.tight_layout(); fig.savefig(OUT / "eda_correlation.png", dpi=200); plt.close(fig)
    say("\nPokemon correlation matrix:\n" + corr.round(3).to_string())

    fig, ax = plt.subplots(figsize=(6, 3.4))
    counts = (df.groupby(["set_name", "release_date"]).size().reset_index(name="n")
              .sort_values("release_date"))
    ax.bar(range(len(counts)), counts.n, color="#FFCB05", edgecolor="#3B4CCA")
    ax.set_xticks(range(len(counts)), counts.set_name, rotation=70, ha="right", fontsize=6)
    ax.set_ylabel("Cards"); ax.set_title("Cards per set (in release order)")
    fig.tight_layout(); fig.savefig(OUT / "eda_set_sizes.png", dpi=200); plt.close(fig)

    (OUT / "eda_summary.txt").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()
