// ScanDex visual fingerprints - JavaScript mirror of ml/features.py.
// Pure functions over RGBA pixel arrays so they run the same in the browser
// and in Node (see tools/parity_test.mjs). Keep in sync with the Python file.

export const BASE_W = 64, BASE_H = 88, TRIM = 0.05, COLOR_W = 4, COLOR_H = 6;

// Box-filter resize: each source pixel is added to the cell its centre falls in.
// src: Float64Array of h*w*ch values.
export function areaResize(src, sw, sh, ch, tw, th) {
  const out = new Float64Array(tw * th * ch), cnt = new Float64Array(tw * th);
  for (let y = 0; y < sh; y++) {
    const ty = Math.floor((y * th) / sh);
    for (let x = 0; x < sw; x++) {
      const t = ty * tw + Math.floor((x * tw) / sw), s = (y * sw + x) * ch;
      for (let c = 0; c < ch; c++) out[t * ch + c] += src[s + c];
      cnt[t]++;
    }
  }
  for (let t = 0; t < tw * th; t++) {
    const n = Math.max(cnt[t], 1);
    for (let c = 0; c < ch; c++) out[t * ch + c] /= n;
  }
  return out;
}

// RGBA (Uint8ClampedArray) -> trimmed RGB Float64Array
function trimRGB(rgba, w, h, frac = TRIM) {
  const dy = Math.round(h * frac), dx = Math.round(w * frac);
  const tw = w - 2 * dx, th = h - 2 * dy, out = new Float64Array(tw * th * 3);
  for (let y = 0; y < th; y++)
    for (let x = 0; x < tw; x++) {
      const s = ((y + dy) * w + x + dx) * 4, t = (y * tw + x) * 3;
      out[t] = rgba[s]; out[t + 1] = rgba[s + 1]; out[t + 2] = rgba[s + 2];
    }
  return { data: out, w: tw, h: th };
}

function toGray(rgb, n) {
  const g = new Float64Array(n);
  for (let i = 0; i < n; i++) g[i] = rgb[i * 3] * 0.299 + rgb[i * 3 + 1] * 0.587 + rgb[i * 3 + 2] * 0.114;
  return g;
}

function bitsToHex(bits) {
  let s = "";
  for (let i = 0; i < bits.length; i += 4)
    s += ((bits[i] << 3) | (bits[i + 1] << 2) | (bits[i + 2] << 1) | bits[i + 3]).toString(16);
  return s;
}

const DCT = {};
function dctMatrix(n) {
  if (DCT[n]) return DCT[n];
  const m = new Float64Array(n * n);
  for (let k = 0; k < n; k++)
    for (let i = 0; i < n; i++) {
      let v = Math.cos((Math.PI * (2 * i + 1) * k) / (2 * n));
      if (k === 0) v *= 1 / Math.sqrt(2);
      m[k * n + i] = v * Math.sqrt(2 / n);
    }
  return (DCT[n] = m);
}

function median(arr) {
  const s = Float64Array.from(arr).sort(), n = s.length;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

function aHash(gray) {
  const g = areaResize(gray, BASE_W, BASE_H, 1, 8, 8);
  const mean = g.reduce((a, b) => a + b, 0) / g.length;
  return Array.from(g, (v) => (v > mean ? 1 : 0));
}

function dHash(gray) {
  const g = areaResize(gray, BASE_W, BASE_H, 1, 9, 8), bits = [];
  for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) bits.push(g[y * 9 + x + 1] > g[y * 9 + x] ? 1 : 0);
  return bits;
}

function pHash(gray, size) {
  const n = size * 4, g = areaResize(gray, BASE_W, BASE_H, 1, n, n), M = dctMatrix(n);
  // low = (M @ g @ M.T)[:size, :size]
  const tmp = new Float64Array(size * n); // (M[:size] @ g)
  for (let k = 0; k < size; k++)
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += M[k * n + i] * g[i * n + j];
      tmp[k * n + j] = s;
    }
  const low = new Float64Array(size * size);
  for (let k = 0; k < size; k++)
    for (let l = 0; l < size; l++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += tmp[k * n + j] * M[l * n + j];
      low[k * size + l] = s;
    }
  const med = median(low);
  return Array.from(low, (v) => (v > med ? 1 : 0));
}

function colorLayout(base) {
  const c = areaResize(base, BASE_W, BASE_H, 3, COLOR_W, COLOR_H);
  const mean = c.reduce((a, b) => a + b, 0) / c.length;
  let norm = 0;
  for (let i = 0; i < c.length; i++) { c[i] -= mean; norm += c[i] * c[i]; }
  norm = Math.sqrt(norm) + 1e-9;
  return Array.from(c, (v) => v / norm);
}

const HIST_BINS = 51;
function hsvHist(base, n) {
  const hist = new Float64Array(HIST_BINS);
  for (let i = 0; i < n; i++) {
    const r = base[i * 3], g = base[i * 3 + 1], b = base[i * 3 + 2];
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    const v = mx / 255, s = mx === 0 ? 0 : (mx - mn) / mx;
    const vb = Math.min(Math.floor(v * 3), 2);
    if (s < 0.2) { hist[48 + vb]++; continue; }
    const d = mx - mn;
    let h;
    if (mx === r) h = ((((g - b) / d) % 6) + 6) % 6;
    else if (mx === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    const hb = Math.min(Math.floor((h / 6) * 8), 7), sb = s < 0.6 ? 0 : 1;
    hist[(hb * 2 + sb) * 3 + vb]++;
  }
  return Array.from(hist, (c) => Math.sqrt(c / n));
}

// rgba: Uint8ClampedArray (ImageData.data), w, h -> fingerprints
export function extract(rgba, w, h) {
  const t = trimRGB(rgba, w, h);
  const base = areaResize(t.data, t.w, t.h, 3, BASE_W, BASE_H);
  const gray = toGray(base, BASE_W * BASE_H);
  return {
    a: bitsToHex(aHash(gray)),
    d: bitsToHex(dHash(gray)),
    p: bitsToHex(pHash(gray, 8)),
    P: bitsToHex(pHash(gray, 16)),
    c: colorLayout(base),
    h: hsvHist(base, BASE_W * BASE_H),
  };
}
