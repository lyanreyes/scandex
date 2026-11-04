// Digital binder - stored on this device (localStorage).

const KEY = "scandex.binder.v1";
let mem = null; // fallback when storage is blocked (private mode)

function read() {
  try { return JSON.parse(localStorage.getItem(KEY) || "[]"); } catch { return mem || []; }
}
function write(items) {
  mem = items;
  try { localStorage.setItem(KEY, JSON.stringify(items)); } catch {}
  window.dispatchEvent(new CustomEvent("binder-change"));
}

export const binder = {
  all: read,
  add(card, { variant, condition = "NM", qty = 1, price }) {
    const items = read();
    const id = `${card.game}:${card.key}:${variant}:${condition}`;
    const ex = items.find((x) => x.id === id);
    if (ex) { ex.qty += qty; ex.price = price ?? ex.price; ex.priceAt = Date.now(); }
    else items.unshift({
      id, game: card.game, key: card.key, name: card.name, set: card.set, number: card.number,
      rarity: card.rarity, image: card.image, variant, condition, qty, price: price ?? null,
      priceAt: Date.now(), addedAt: Date.now(),
    });
    write(items);
  },
  update(id, patch) {
    const items = read(), it = items.find((x) => x.id === id);
    if (it) Object.assign(it, patch);
    write(items);
  },
  remove(id) { write(read().filter((x) => x.id !== id)); },
  replaceAll(items) { write(items); },
  stats(items = read()) {
    const value = items.reduce((s, x) => s + (x.price || 0) * x.qty, 0);
    const count = items.reduce((s, x) => s + x.qty, 0);
    const top = [...items].sort((a, b) => (b.price || 0) - (a.price || 0))[0];
    return { value, count, unique: items.length, top };
  },
  toCSV(items = read()) {
    const head = ["game", "name", "set", "number", "rarity", "variant", "condition", "qty", "price_usd", "total_usd"];
    const esc = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
    return [head.join(","), ...items.map((x) => [x.game, x.name, x.set, x.number, x.rarity, x.variant,
      x.condition, x.qty, x.price ?? "", ((x.price || 0) * x.qty).toFixed(2)].map(esc).join(","))].join("\n");
  },
};
