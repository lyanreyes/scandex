// Live market data from the Pokémon TCG API (TCGplayer USD + Cardmarket EUR).
// It sends Access-Control-Allow-Origin: *, so the browser calls it directly.

const PTCG = "https://api.pokemontcg.io/v2";
const cache = new Map();

async function getJSON(url, tries = 4) {
  if (cache.has(url)) return cache.get(url);
  let err;
  for (let a = 0; a < tries; a++) {
    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const j = await r.json();
      cache.set(url, j);
      return j;
    } catch (e) {
      err = e; // the free Pokémon TCG API intermittently answers 500/502
      await new Promise((res) => setTimeout(res, 600 * (a + 1)));
    }
  }
  throw err;
}

const VARIANT_LABEL = {
  normal: "Normal", holofoil: "Holofoil", reverseHolofoil: "Reverse Holo",
  "1stEditionHolofoil": "1st Ed. Holo", unlimitedHolofoil: "Unlimited Holo",
};

function pokemonPrices(c) {
  const variants = Object.entries(c.tcgplayer?.prices || {}).map(([k, v]) => ({
    id: k, label: VARIANT_LABEL[k] || k, market: v.market ?? v.mid ?? null,
    low: v.low, mid: v.mid, high: v.high,
  }));
  const cm = c.cardmarket?.prices;
  return {
    variants,
    cardmarket: cm ? { trend: cm.trendPrice, avg30: cm.avg30, low: cm.lowPrice, url: c.cardmarket.url } : null,
    tcgplayerUrl: c.tcgplayer?.url,
    updated: c.tcgplayer?.updatedAt,
  };
}

function normalizePokemon(c) {
  return {
    game: "pokemon", key: c.id, name: c.name, set: c.set.name, number: c.number,
    printedTotal: c.set.printedTotal, setCode: c.set.ptcgoCode || "", rarity: c.rarity || "",
    image: c.images.small, imageLarge: c.images.large, ...pokemonPrices(c),
  };
}

// card: an index entry {key, ...}
export async function livePrice(card) {
  const { data } = await getJSON(`${PTCG}/cards/${encodeURIComponent(card.key)}`);
  return normalizePokemon(data);
}

// cards outside the visual index, found by the OCR'd collector number
export async function lookupByCode(code) {
  if (!code.number) return [];
  const n = String(parseInt(code.number, 10));
  let q = `number:${n}`;
  if (code.total) q += ` set.printedTotal:${parseInt(code.total, 10)}`;
  const run = async (query) => (await getJSON(`${PTCG}/cards?q=${encodeURIComponent(query)}&pageSize=12`)).data;
  let data = code.setCode ? await run(`${q} set.ptcgoCode:${code.setCode}`).catch(() => []) : [];
  if (!data.length) data = await run(q); // set code misread by OCR -> number/total only
  return data.map(normalizePokemon);
}

// full-catalogue name search (any era, not only the indexed sets)
export async function searchPokemon(name) {
  const q = `name:"${name.replace(/"/g, "")}*"`;
  const { data } = await getJSON(`${PTCG}/cards?q=${encodeURIComponent(q)}&pageSize=36&orderBy=-set.releaseDate`);
  return data.map(normalizePokemon);
}

export function bestMarket(p) {
  const v = p.variants?.find((x) => x.market != null);
  return v ? v.market : null;
}

// USD -> PHP / EUR rate, cached for 12 h
export async function fxRates() {
  const KEY = "scandex.fx";
  try {
    const c = JSON.parse(localStorage.getItem(KEY) || "null");
    if (c && Date.now() - c.t < 12 * 3600e3) return c.rates;
  } catch {}
  try {
    const j = await getJSON("https://open.er-api.com/v6/latest/USD", 2);
    const rates = { USD: 1, PHP: j.rates.PHP, EUR: j.rates.EUR };
    try { localStorage.setItem(KEY, JSON.stringify({ t: Date.now(), rates })); } catch {}
    return rates;
  } catch {
    return { USD: 1, PHP: 58, EUR: 0.92, fallback: true };
  }
}
