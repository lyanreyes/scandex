// Optional text check: reads the collector number printed at the bottom-left
// of the card with Tesseract.js (loaded lazily from a CDN on first use).
//   "MEW EN 025/165"  ->  { number: "025", total: "165", setCode: "MEW" }

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

// bottom strip of the card, upscaled and contrast-stretched for OCR
function bottomStrip(canvas) {
  const sh = Math.round(canvas.height * 0.13), out = document.createElement("canvas");
  out.width = canvas.width * 3; out.height = sh * 3;
  const ctx = out.getContext("2d");
  ctx.filter = "grayscale(1) contrast(1.6)";
  ctx.drawImage(canvas, 0, canvas.height - sh, canvas.width, sh, 0, 0, out.width, out.height);
  return out;
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

export async function readCode(cardCanvas, timeoutMs = 9000) {
  const worker = await warmUpOCR();
  const job = worker.recognize(bottomStrip(cardCanvas)).then((r) => parseCode(r.data.text || ""));
  const timeout = new Promise((res) => setTimeout(() => res({ raw: "", timeout: true }), timeoutMs));
  return Promise.race([job, timeout]);
}
