// Optional text check with Tesseract.js (loaded lazily from a CDN on first use):
//   bottom strip  "MEW EN 025/165"        ->  { number: "025", total: "165", setCode: "MEW" }
//   top strip     "BASIC Pikachu HP 60"   ->  { name: "Pikachu", suffix: null }
//                 "Rayquaza VMAX HP 320"  ->  { name: "Rayquaza", suffix: "VMAX" }

const TESSERACT_URL = "https://cdn.jsdelivr.net/npm/tesseract.js@5.1.1/dist/tesseract.min.js";
let workerPromise = null;

function loadScript(src) {
  return new Promise((res, rej) => {
    const s = document.createElement("script");
    s.src = src; s.onload = res; s.onerror = () => rej(new Error("OCR library failed to load"));
    document.head.appendChild(s);
  });
}

export function warmUpOCR() {
  if (!workerPromise) {
    workerPromise = (async () => {
      if (!window.Tesseract) await loadScript(TESSERACT_URL);
      const w = await window.Tesseract.createWorker("eng");
      await w.setParameters({ tessedit_pageseg_mode: "11" }); // sparse text
      return w;
    })().catch((e) => { workerPromise = null; throw e; });
  }
  return workerPromise;
}

// a horizontal strip of the card, upscaled and contrast-stretched for OCR
function strip(canvas, y0, y1, x1 = 1) {
  const sy = Math.round(canvas.height * y0), sh = Math.round(canvas.height * (y1 - y0));
  const sw = Math.round(canvas.width * x1), out = document.createElement("canvas");
  out.width = sw * 3; out.height = sh * 3;
  const ctx = out.getContext("2d");
  ctx.filter = "grayscale(1) contrast(1.6)";
  ctx.drawImage(canvas, 0, sy, sw, sh, 0, 0, out.width, out.height);
  return out;
}

const NAME_STOP = new Set(["BASIC", "STAGE", "STAGE1", "STAGE2", "HP", "EVOLVES", "FROM", "TRAINER",
  "ITEM", "SUPPORTER", "STADIUM", "TOOL", "ENERGY", "POKEMON", "POKÉMON", "TERA", "MEGA", "DYNAMAX",
  "GIGANTAMAX", "RAPID", "SINGLE", "STRIKE", "FUSION", "RADIANT", "ABILITY", "PRISM", "STAR"]);
const SUFFIX = ["VMAX", "VSTAR", "V-UNION", "GX", "EX", "V"];

export function parseName(text) {
  // "Evolves from Rayquaza V" names the previous stage: use it only as a last resort
  const lines = text.split(/\n+/);
  const evo = lines.find((l) => /evolves\s+from/i.test(l));
  const main = lines.filter((l) => l !== evo).join(" ");
  const r = nameFrom(main);
  if (r.name || !evo) return r;
  return { name: nameFrom(evo.replace(/.*evolves\s+from/i, "")).name, suffix: null, fromEvolves: true };
}

function nameFrom(text) {
  const words = text.replace(/[^A-Za-zÀ-ÿ'.\- ]+/g, " ").split(/\s+/).filter(Boolean);
  let suffix = null;
  const name = [];
  for (const w of words) {
    const up = w.toUpperCase();
    if (SUFFIX.includes(up)) { if (name.length) { suffix = up; break; } continue; }
    if (NAME_STOP.has(up) || w.length < 3) { if (name.length) break; continue; }
    if (/^[A-Z][a-zà-ÿ'.\-]+$/.test(w) || /^[A-Z][A-Z'.\-]+$/.test(w)) name.push(w);
    else if (name.length) break;
    if (name.length === 3) break;
  }
  return { name: name.join(" ") || null, suffix };
}

export function parseCode(text) {
  const t = text.toUpperCase().replace(/[|]/g, "/").replace(/O(?=\d)/g, "0");
  const num = t.match(/\b(\d{1,3})\s?\/\s?(\d{2,3})\b/);
  if (num) {
    const set = t.match(/\b([A-Z]{3})\s+(EN|JP)\b/) || t.match(/\b([A-Z]{3})\b(?=[^A-Z]*\d{1,3}\s?\/)/);
    return { number: num[1], total: num[2], setCode: set ? set[1] : null, raw: text };
  }
  return { raw: text };
}

// -> { number, total, setCode, name, suffix }  (fields missing when unreadable)
export async function readCard(cardCanvas, timeoutMs = 12000) {
  const worker = await warmUpOCR();
  const job = (async () => {
    const bottom = await worker.recognize(strip(cardCanvas, 0.87, 1));
    const top = await worker.recognize(strip(cardCanvas, 0.02, 0.12, 0.78));
    return { ...parseCode(bottom.data.text || ""), ...parseName(top.data.text || "") };
  })();
  const timeout = new Promise((res) => setTimeout(() => res({ raw: "", timeout: true }), timeoutMs));
  return Promise.race([job, timeout]);
}
