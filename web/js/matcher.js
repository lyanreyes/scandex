// k-nearest-neighbour card retrieval over the visual index built by
// ml/03_build_index.py. Cards are sorted by the pairwise-LR ranking score
// s = w . D (lower = more similar) and the top answer gets a calibrated
// probability from [s1, s2 - s1], both trained in ml/04_evaluate.py
// (web/model/fusion.json).

const GAMES = ["pokemon"];
const HASHES = ["a", "d", "p", "P"];
const VECTORS = ["c", "h"];

function hexToWords(hex) {
  const w = new Uint32Array(hex.length / 8);
  for (let i = 0; i < w.length; i++) w[i] = parseInt(hex.slice(i * 8, i * 8 + 8), 16) >>> 0;
  return w;
}

function popcount(x) {
  x -= (x >>> 1) & 0x55555555;
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  return (((x + (x >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

function b64ToInt8(b64) {
  const bin = atob(b64), out = new Int8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = (bin.charCodeAt(i) << 24) >> 24;
  return out;
}

function unit(v) {
  let n = 0;
  for (const x of v) n += x * x;
  n = Math.sqrt(n) + 1e-9;
  return Float32Array.from(v, (x) => x / n);
}

export class Matcher {
  constructor() {
    this.cards = [];   // metadata objects
    this.hash = {};    // hash feature -> Uint32Array (cards x words)
    this.words = {};
    this.vec = {};     // vector feature -> Float32Array (cards x dim), L2-normalised
    this.dims = {};
    this.fusion = null;
  }

  async load(base = "model", onProgress = () => {}) {
    const get = async (f) => {
      const r = await fetch(`${base}/${f}`);
      if (!r.ok) throw new Error(`${f} missing - run the ml/ pipeline (03_build_index.py, 04_evaluate.py)`);
      return r.json();
    };
    this.fusion = await get("fusion.json");
    const parts = [];
    for (const g of GAMES) {
      onProgress("Loading Pokémon card index…");
      parts.push(await get(`${g}.json`));
    }
    const rows = parts.flatMap((p) => p.cards.map((r) => ({ game: p.game, r })));
    const n = rows.length, f = parts[0].fields, at = (k) => f.indexOf(k);
    for (const h of HASHES) {
      this.words[h] = rows[0].r[at(h)].length / 8;
      this.hash[h] = new Uint32Array(n * this.words[h]);
    }
    for (const v of VECTORS) {
      this.dims[v] = b64ToInt8(rows[0].r[at(v)]).length;
      this.vec[v] = new Float32Array(n * this.dims[v]);
    }
    rows.forEach(({ game, r }, i) => {
      this.cards.push({
        game, key: r[at("key")], name: r[at("name")], set: r[at("set")], number: r[at("number")],
        printedTotal: r[at("printedTotal")], setCode: r[at("setCode")], rarity: r[at("rarity")],
        image: r[at("image")], priceSnapshot: r[at("priceSnapshot")],
      });
      for (const h of HASHES) this.hash[h].set(hexToWords(r[at(h)]), i * this.words[h]);
      for (const v of VECTORS) this.vec[v].set(unit(b64ToInt8(r[at(v)])), i * this.dims[v]);
    });
    this.built = parts[0].built;
    return this;
  }

  get size() { return this.cards.length; }

  confidence(s1, gap) {
    const c = this.fusion.calibration;
    return 1 / (1 + Math.exp(-(c.bias + c.s1 * s1 + c.gap * gap)));
  }

  // Distance vector D (aHash, dHash, pHash-64, pHash-256, colour, HSV hist) of
  // the query to every card -> ranking score.
  rank(q, k = 8) {
    const t0 = performance.now();
    const qw = Object.fromEntries(HASHES.map((h) => [h, hexToWords(q[h])]));
    const qv = Object.fromEntries(VECTORS.map((v) => [v, unit(q[v])]));
    const w = this.fusion.weights;
    const out = [];
    for (let i = 0; i < this.cards.length; i++) {
      let s = 0, j = 0;
      for (const h of HASHES) {
        const W = this.words[h], arr = this.hash[h], o = i * W;
        let d = 0;
        for (let x = 0; x < W; x++) d += popcount((arr[o + x] ^ qw[h][x]) >>> 0);
        s += w[j++] * (d / (W * 32));
      }
      for (const v of VECTORS) {
        const dim = this.dims[v], arr = this.vec[v], o = i * dim, qq = qv[v];
        let dot = 0;
        for (let x = 0; x < dim; x++) dot += arr[o + x] * qq[x];
        s += w[j++] * ((1 - dot) / 2);
      }
      out.push({ i, s });
    }
    out.sort((x, y) => x.s - y.s);
    const prob = out.length > 1 ? this.confidence(out[0].s, out[1].s - out[0].s) : null;
    return {
      ms: performance.now() - t0,
      results: out.slice(0, k).map(({ i }, r) => ({ ...this.cards[i], prob: r === 0 ? prob : null })),
      all: out, // full ranking, used for OCR re-ranking
    };
  }

  // OCR re-ranking: cards matching the printed collector number / code.
  // OCR often misreads the 3-letter set code, so it only narrows the result
  // when it is a real code and still leaves at least one card.
  findByCode({ number, total, setCode }) {
    if (!number) return [];
    const n = String(parseInt(number, 10)), t = total ? parseInt(total, 10) : null;
    const hits = [];
    this.cards.forEach((c, i) => {
      const sameNum = String(parseInt(c.number, 10)) === n || c.number === number;
      if (sameNum && (!t || c.printedTotal === t)) hits.push(i);
    });
    const inSet = setCode ? hits.filter((i) => this.cards[i].setCode === setCode) : [];
    return inSet.length ? inSet : hits;
  }

  searchText(q, limit = 60) {
    const s = q.trim().toLowerCase();
    if (!s) return [];
    return this.cards
      .filter((c) => `${c.name} ${c.set} ${c.number}/${c.printedTotal} ${c.setCode}`.toLowerCase().includes(s))
      .slice(0, limit);
  }
}
