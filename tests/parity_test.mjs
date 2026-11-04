// node tests/parity_test.mjs  - JS fingerprints must equal the Python ones
import { readFileSync } from "node:fs";
import { extract } from "../web/js/features.js";

const fx = JSON.parse(readFileSync(new URL("./parity_fixture.json", import.meta.url)));
const ham = (a, b) => [...a].reduce((s, ch, i) => s + popc(parseInt(ch, 16) ^ parseInt(b[i], 16)), 0);
const popc = (n) => n.toString(2).split("1").length - 1;
let worst = 0, colorErr = 0;
for (const f of fx) {
  const rgba = new Uint8ClampedArray(Buffer.from(f.rgba, "base64"));
  const got = extract(rgba, f.w, f.h);
  for (const k of ["a", "d", "p", "P"]) worst = Math.max(worst, ham(got[k], f.expected[k]));
  for (const k of ["c", "h"])
    got[k].forEach((v, i) => (colorErr = Math.max(colorErr, Math.abs(Math.round(v * 127) - f.expected[k][i]))));
}
console.log(`${fx.length} images | max differing hash bits: ${worst} | max colour error: ${colorErr}/127`);
process.exit(worst === 0 && colorErr <= 1 ? 0 : 1);
