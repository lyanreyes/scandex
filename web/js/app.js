import { extract } from "./features.js";
import { Matcher } from "./matcher.js";
import { readCard, warmUpOCR } from "./ocr.js";
import { livePrice, lookupByCode, searchPokemon, bestMarket, fxRates } from "./api.js";
import { binder } from "./binder.js";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const CONDITIONS = ["NM", "LP", "MP", "HP", "DMG"];
// below this calibrated confidence the visual match is treated as "not recognised"
// (about 1 in 7 such top answers is right; see ml/outputs/eval_threshold.csv)
const LOW_CONF = 0.3;

const state = {
  binderFilter: "", currency: "USD", rates: { USD: 1 },
  matcher: new Matcher(), ready: false,
  stream: null, facing: "environment", source: "video",
  img: { s: 1, x: 0, y: 0 }, scanId: 0,
};

/* ---------------- helpers ---------------- */
function money(usd) {
  if (usd == null || Number.isNaN(usd)) return "—";
  const v = usd * (state.rates[state.currency] || 1);
  return new Intl.NumberFormat(undefined, { style: "currency", currency: state.currency,
    maximumFractionDigits: v >= 1000 ? 0 : 2 }).format(v);
}
function eur(v) {
  return v == null ? "—" : new Intl.NumberFormat(undefined, { style: "currency", currency: "EUR" }).format(v);
}
function toast(msg) {
  const t = $("#toast");
  t.innerHTML = `<span class="ball-icon" aria-hidden="true"></span>${esc(msg)}`;
  t.classList.add("show");
  clearTimeout(toast.t); toast.t = setTimeout(() => t.classList.remove("show"), 2200);
}
function setStatus(msg) { $("#status").textContent = msg; }
function download(name, text, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

/* ---------------- routing ---------------- */
function route() {
  const tab = (location.hash || "#scan").slice(1);
  const name = ["scan", "binder", "search"].includes(tab) ? tab : "scan";
  $$(".view").forEach((v) => (v.hidden = v.id !== `view-${name}`));
  $$(".tabs a").forEach((a) => a.classList.toggle("active", a.dataset.tab === name));
  if (name === "binder") renderBinder();
}
window.addEventListener("hashchange", route);

/* ---------------- camera ---------------- */
async function startCamera() {
  stopCamera();
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus(window.isSecureContext ? "This browser has no camera API. Use Upload instead."
      : "Camera needs HTTPS (or localhost). Use Upload, or open the HTTPS address.");
    return;
  }
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: state.facing }, width: { ideal: 1920 }, height: { ideal: 1080 } },
    });
    const v = $("#video");
    v.srcObject = state.stream;
    await v.play();
    useSource("video");
    $("#camMsg").hidden = true;
    $("#capture").disabled = !state.ready;
    setStatus(state.ready ? "Ready — tap the button to scan" : "Loading card index…");
  } catch (e) {
    setStatus(`Camera unavailable (${e.name}). You can upload a photo instead.`);
  }
}
function stopCamera() {
  state.stream?.getTracks().forEach((t) => t.stop());
  state.stream = null;
}
function useSource(src) {
  state.source = src;
  $("#video").hidden = src !== "video";
  $("#still").hidden = src !== "image";
  $("#zoomRow").hidden = src !== "image";
  $("#hint").textContent = src === "image" ? "Drag and zoom so the card fills the frame" : "Fit the card inside the frame";
}

// inner (padding) box of the viewport: video/photo are positioned inside the bezel
function innerBox(el) {
  const r = el.getBoundingClientRect();
  return { left: r.left + el.clientLeft, top: r.top + el.clientTop, width: el.clientWidth, height: el.clientHeight };
}

/* ---------------- uploaded photo: pan & zoom ---------------- */
function applyImgTransform() {
  const { s, x, y } = state.img;
  $("#still").style.transform = `translate(${x}px, ${y}px) scale(${s})`;
}
function loadPhoto(file) {
  const img = $("#still");
  img.onload = () => {
    stopCamera();
    useSource("image");
    $("#camMsg").hidden = true;
    const vp = innerBox($("#viewport")), g = $("#guide").getBoundingClientRect();
    // initial fit: assume the card fills most of the photo -> fit photo into the guide
    const s = Math.min(g.width / img.naturalWidth, g.height / img.naturalHeight) * 1.02;
    state.img = { s, x: (vp.width - img.naturalWidth * s) / 2, y: (vp.height - img.naturalHeight * s) / 2 };
    $("#zoom").value = 1; state.baseScale = s;
    applyImgTransform();
    $("#capture").disabled = !state.ready;
    setStatus("Adjust the photo, then tap scan");
  };
  img.src = URL.createObjectURL(file);
}
function zoomTo(s, cx, cy) {
  const k = s / state.img.s;
  state.img.x = cx - (cx - state.img.x) * k;
  state.img.y = cy - (cy - state.img.y) * k;
  state.img.s = s;
  applyImgTransform();
}
function initPanZoom() {
  const vp = $("#viewport");
  let drag = null;
  vp.addEventListener("pointerdown", (e) => {
    if (state.source !== "image") return;
    drag = { x: e.clientX, y: e.clientY, ox: state.img.x, oy: state.img.y };
    vp.setPointerCapture(e.pointerId);
  });
  vp.addEventListener("pointermove", (e) => {
    if (!drag) return;
    state.img.x = drag.ox + e.clientX - drag.x;
    state.img.y = drag.oy + e.clientY - drag.y;
    applyImgTransform();
  });
  vp.addEventListener("pointerup", () => (drag = null));
  vp.addEventListener("wheel", (e) => {
    if (state.source !== "image") return;
    e.preventDefault();
    const r = innerBox(vp);
    zoomTo(state.img.s * (e.deltaY < 0 ? 1.08 : 1 / 1.08), e.clientX - r.left, e.clientY - r.top);
    $("#zoom").value = state.img.s / state.baseScale;
  }, { passive: false });
  $("#zoom").addEventListener("input", (e) => {
    const r = innerBox(vp);
    zoomTo(state.baseScale * Number(e.target.value), r.width / 2, r.height / 2);
  });
}

/* ---------------- capture -> crop of the guide region ---------------- */
function cropGuide(w, h) {
  const vp = innerBox($("#viewport")), g = $("#guide").getBoundingClientRect();
  const gx = g.left - vp.left, gy = g.top - vp.top;
  let src, sx, sy, sw, sh;
  if (state.source === "video") {
    src = $("#video");
    const vw = src.videoWidth, vh = src.videoHeight;
    const scale = Math.max(vp.width / vw, vp.height / vh);
    const ox = (vp.width - vw * scale) / 2, oy = (vp.height - vh * scale) / 2;
    [sx, sy, sw, sh] = [(gx - ox) / scale, (gy - oy) / scale, g.width / scale, g.height / scale];
  } else {
    src = $("#still");
    const { s, x, y } = state.img;
    [sx, sy, sw, sh] = [(gx - x) / s, (gy - y) / s, g.width / s, g.height / s];
  }
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, w, h);
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(src, sx, sy, sw, sh, 0, 0, w, h);
  return c;
}

async function scan() {
  if (!state.ready) return;
  const guide = $("#guide"), ball = $("#capture");
  guide.classList.add("busy");
  ball.classList.remove("caught");
  ball.classList.add("wobble");
  ball.disabled = true;
  const minWobble = new Promise((r) => setTimeout(r, 650)); // let the ball shake once
  try {
    const sid = ++state.scanId;
    const t0 = performance.now();
    const small = cropGuide(200, 280);                  // same size as the index thumbnails
    const hi = cropGuide(630, 880);                      // for OCR + preview
    const px = small.getContext("2d").getImageData(0, 0, 200, 280);
    const q = extract(px.data, 200, 280);
    const ranked = state.matcher.rank(q, 12);
    const ms = performance.now() - t0;
    const yours = hi.toDataURL("image/jpeg", 0.8);
    const top = ranked.results[0];
    await minWobble;
    const unsure = top.prob != null && top.prob < LOW_CONF;
    setStatus(`${unsure ? "No confident match" : "Matched"} in ${ms.toFixed(0)} ms against ${state.matcher.size.toLocaleString()} cards · reading card text…`);
    if (unsure) showUnsure(ranked.results.slice(0, 8), yours);
    else showMatch(top, { prob: top.prob, alts: ranked.results.slice(1, 9), yours, how: "visual" });
    // text check runs in the background and can confirm or correct the match
    readCard(hi).then((code) => sid === state.scanId && applyOCR(code, ranked, yours, unsure))
      .catch(() => setStatus(`Matched in ${ms.toFixed(0)} ms · text check unavailable`));
  } catch (e) {
    console.error(e);
    setStatus(`Scan failed: ${e.message}`);
  } finally {
    guide.classList.remove("busy");
    ball.classList.remove("wobble");
    ball.classList.add("caught");
    ball.disabled = false;
  }
}

const sameName = (card, code) => code.name && card.name.toLowerCase().startsWith(code.name.toLowerCase());
const suffixOk = (card, code) => !code.suffix || new RegExp(`\\b${code.suffix}\\b`, "i").test(card.name);

async function applyOCR(code, ranked, yours, unsure) {
  const numLabel = code.number ? `${code.number}${code.total ? "/" + code.total : ""}` : null;
  const nameLabel = code.name ? `${code.name}${code.suffix ? " " + code.suffix : ""}` : null;
  const label = numLabel || nameLabel;
  if (!label) {
    setStatus(unsure ? "Couldn't read the card's name or number — pick a candidate or use Search"
      : "Visual match only — no collector number could be read");
    return;
  }
  const top = ranked.results[0];
  // 1) printed number found in the offline index -> confirm or correct the visual match
  let ids = state.matcher.findByCode(code);
  if (ids.length && code.name) {
    const named = ids.filter((i) => sameName(state.matcher.cards[i], code));
    if (named.length) ids = named;
  }
  if (ids.length) {
    const set = new Set(ids);
    const best = ranked.all.find((r) => set.has(r.i));
    const card = state.matcher.cards[best.i];
    if (card.key === top.key && !unsure) {
      setStatus(`Verified: printed number ${numLabel} matches`);
      $("#verify")?.replaceChildren(Object.assign(document.createElement("span"), { className: "tag ok", textContent: `✓ text ${numLabel}` }));
    } else {
      setStatus(`Identified using printed number ${numLabel}`);
      showMatch(card, { prob: null, alts: ranked.results.slice(0, 8).filter((c) => c.key !== card.key),
        yours, how: "text", label: numLabel });
    }
    return;
  }
  // 2) confident visual match whose name agrees with the printed name
  if (!unsure && code.name && sameName(top, code)) {
    setStatus(`Verified: printed name "${nameLabel}" matches`);
    $("#verify")?.replaceChildren(Object.assign(document.createElement("span"), { className: "tag ok", textContent: `✓ name ${code.name}` }));
    return;
  }
  if (!unsure && !code.name) { setStatus(`Read ${numLabel}, which is not in the offline index — showing visual match`); }
  // 3) not in the offline index (e.g. an older set) -> ask the online catalogue
  try {
    let found = [];
    if (code.number) {
      found = await lookupByCode(code);
      if (code.name) {
        const named = found.filter((c) => sameName(c, code));
        if (named.length) found = named;
      }
    }
    if (!found.length && code.name) {
      found = (await searchPokemon(code.name)).filter((c) => sameName(c, code) && suffixOk(c, code));
    }
    if (!found.length) {
      setStatus(`Read "${label}" but found no matching card online`);
      return;
    }
    const by = code.number && found.some((c) => c.number.replace(/^0+/, "") === code.number.replace(/^0+/, "")) ? numLabel : nameLabel;
    setStatus(`Found ${found.length} card${found.length > 1 ? "s" : ""} for "${by}" in the online catalogue`);
    if (unsure || found.length === 1 || code.number) {
      showMatch(found[0], { prob: null, alts: found.slice(1, 12).concat(ranked.results.slice(0, 3)), yours,
        how: "online", label: by, priced: found[0], pickHint: found.length > 1 });
    }
  } catch {
    setStatus(`Read "${label}"; online lookup failed — check your connection or use Search`);
  }
}

function showUnsure(cands, yours) {
  const root = $("#result");
  root.innerHTML = `<div class="unsure">
      <img src="${yours}" alt="Your scan">
      <div>
        <span class="tag warn">Not recognised</span>
        <h2>Hmm… who's that card?</h2>
        <p class="meta">This card may be from a set that isn't in the offline index yet, or the photo is blurry or not filling the frame.
          ScanDex is reading the card's name and number to search the full catalogue…</p>
      </div>
    </div>
    <div class="alts"><h3>Or is it one of these?</h3><div class="alt-list">${cands.map((c, i) => `<button class="alt" data-i="${i}" title="${esc(c.name)} · ${esc(c.set)}">
      <img src="${esc(c.image)}" alt="" loading="lazy" referrerpolicy="no-referrer"><span>${esc(c.name)}</span></button>`).join("")}</div></div>`;
  $$(".alt", root).forEach((b) => (b.onclick = () => {
    state.scanId++;
    const c = cands[+b.dataset.i];
    showMatch(c, { prob: null, how: "manual", yours, alts: cands.filter((x) => x !== c) });
  }));
  if (matchMedia("(max-width: 900px)").matches) root.scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ---------------- result panel ---------------- */
function priceTable(p) {
  const rows = (p.variants || []).map((v) => `<tr><td>${esc(v.label)}</td><td>${money(v.low)}</td>
    <td>${v.mid != null ? money(v.mid) : "—"}</td><td><b>${money(v.market)}</b></td></tr>`).join("");
  const cm = p.cardmarket ? `<tr><td>Cardmarket trend</td><td>${eur(p.cardmarket.low)}</td><td>${eur(p.cardmarket.avg30)}</td><td><b>${eur(p.cardmarket.trend)}</b></td></tr>` : "";
  return `<table class="prices"><thead><tr><th>TCGplayer</th><th>Low</th><th>Mid</th><th>Market</th></tr></thead>
    <tbody>${rows || `<tr><td colspan="4">No price data</td></tr>`}${cm}</tbody></table>`;
}

function cardBlock(card, { prob, yours, how, label } = {}) {
  const auto = state.matcher.fusion?.autoAccept ?? 0.8;
  const confident = prob == null || prob >= auto;
  const confHtml = prob == null ? "" : `<div class="conf ${confident ? "" : "low"}">
      <div class="bar"><b style="width:${Math.round(prob * 100)}%"></b></div>
      <small>${confident ? "Confident match" : "Best guess — tap an alternative if this is wrong"} · ${(prob * 100).toFixed(0)}%</small></div>`;
  const howTag = how === "text" ? `<span class="tag ok">✓ corrected by text ${esc(label)}</span>`
    : how === "online" ? `<span class="tag ok">✓ found online: ${esc(label)}</span>` : "";
  return `<div class="match">
    <div class="art"><img src="${esc(card.image)}" alt="${esc(card.name)}" referrerpolicy="no-referrer">
      ${yours ? `<img class="yours" src="${yours}" alt="Your scan">` : ""}</div>
    <div>
      ${card.setCode ? `<span class="tag set">${esc(card.setCode)}</span>` : ""} ${card.rarity ? `<span class="tag">${esc(card.rarity)}</span>` : ""} <span id="verify">${howTag}</span>
      <h2>${esc(card.name)}</h2>
      <div class="meta">${esc(card.set)} · #${esc(card.number)}${card.printedTotal ? "/" + card.printedTotal : ""}</div>
      ${confHtml}
      <div class="big-price" id="bigPrice"><span class="skeleton">$00.00</span><small>Fetching live market price…</small></div>
    </div>
  </div>
  <div id="priceArea"></div>`;
}

function addRow(card, p) {
  const variants = (p?.variants || []).filter((v) => v.market != null);
  const opts = (variants.length ? variants : [{ id: "normal", label: "Normal", market: null }])
    .map((v) => `<option value="${esc(v.id)}" data-price="${v.market ?? ""}">${esc(v.label)} · ${money(v.market)}</option>`).join("");
  return `<div class="add-row">
    <select id="addVariant" aria-label="Variant">${opts}</select>
    <select id="addCond" aria-label="Condition">${CONDITIONS.map((c) => `<option>${c}</option>`).join("")}</select>
    <input id="addQty" type="number" min="1" value="1" aria-label="Quantity">
    <button class="btn primary" id="addBtn">Add to binder</button>
  </div>`;
}

async function fillPrices(card, root, priced) {
  const big = $("#bigPrice", root), area = $("#priceArea", root);
  let p = priced;
  try {
    p = p || (await livePrice(card));
  } catch {
    p = { variants: [{ id: "normal", label: "Snapshot", market: card.priceSnapshot ?? null }], snapshot: true };
  }
  const m = bestMarket(p);
  big.innerHTML = `${money(m)}<small>${p.snapshot ? "Offline snapshot price (live API unreachable)"
    : `Market price · TCGplayer${p.updated ? " · updated " + esc(p.updated) : ""}`}</small>`;
  area.innerHTML = priceTable(p) + addRow(card, p) + `<div class="links">
    ${p.tcgplayerUrl ? `<a href="${esc(p.tcgplayerUrl)}" target="_blank" rel="noopener">TCGplayer listing ↗</a>` : ""}
    ${p.cardmarket?.url ? `<a href="${esc(p.cardmarket.url)}" target="_blank" rel="noopener">Cardmarket ↗</a>` : ""}</div>`;
  $("#addBtn", area).onclick = () => {
    const sel = $("#addVariant", area), opt = sel.selectedOptions[0];
    binder.add(card, { variant: opt.textContent.split(" · ")[0], condition: $("#addCond", area).value,
      qty: Math.max(1, parseInt($("#addQty", area).value, 10) || 1),
      price: opt.dataset.price ? Number(opt.dataset.price) : null });
    toast(`Gotcha! ${card.name} was added to your binder`);
  };
  return p;
}

function showMatch(card, opts) {
  const root = $("#result");
  const alts = (opts.alts || []).map((c, i) => `<button class="alt" data-i="${i}" title="${esc(c.name)} · ${esc(c.set)}">
      <img src="${esc(c.image)}" alt="" loading="lazy" referrerpolicy="no-referrer"><span>${esc(c.name)}</span></button>`).join("");
  const altTitle = opts.pickHint ? "Several printings match — pick yours" : "Not it? Other candidates";
  root.innerHTML = cardBlock(card, opts) + (alts ? `<div class="alts"><h3>${altTitle}</h3><div class="alt-list">${alts}</div></div>` : "");
  $$(".alt", root).forEach((b) => (b.onclick = () => {
    const c = opts.alts[+b.dataset.i];
    state.scanId++; // a manual pick wins over a late OCR result
    showMatch(c, { ...opts, prob: null, how: "manual", alts: opts.alts.filter((x) => x !== c).concat(card) });
  }));
  fillPrices(card, root, opts.priced);
  if (matchMedia("(max-width: 900px)").matches) root.scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ---------------- detail dialog (search + binder) ---------------- */
function openDetail(card, item) {
  const d = $("#detail");
  d.innerHTML = `<button class="close" aria-label="Close">×</button>${cardBlock(card)}
    ${item ? `<div class="add-row"><label class="muted">Qty <input id="editQty" type="number" min="1" value="${item.qty}"></label>
      <span class="tag">${esc(item.variant)} · ${esc(item.condition)}</span>
      <button class="btn ghost small" id="removeBtn">Remove from binder</button></div>` : ""}`;
  $(".close", d).onclick = () => d.close();
  d.showModal();
  fillPrices(card, d).then((p) => {
    if (!item) return;
    const v = p.variants?.find((x) => x.label === item.variant) || p.variants?.[0];
    if (v?.market != null && !p.snapshot) binder.update(item.id, { price: v.market, priceAt: Date.now() });
  });
  if (item) {
    $("#editQty", d).onchange = (e) => binder.update(item.id, { qty: Math.max(1, +e.target.value || 1) });
    $("#removeBtn", d).onclick = () => { binder.remove(item.id); d.close(); toast("Removed"); };
  }
}
$("#detail").addEventListener("close", () => route());

/* ---------------- binder ---------------- */
function renderBinder() {
  let items = binder.all();
  const s = binder.stats(items);
  $("#stats").innerHTML = `
    <div class="stat hero"><small>Collection value</small><b>${money(s.value)}</b></div>
    <div class="stat"><small>Cards</small><b>${s.count}</b></div>
    <div class="stat"><small>Unique</small><b>${s.unique}</b></div>
    <div class="stat"><small>Most valuable</small><b style="font-size:17px">${s.top ? esc(s.top.name) + " · " + money(s.top.price) : "—"}</b></div>`;
  const f = state.binderFilter.trim().toLowerCase();
  if (f) items = items.filter((x) => `${x.name} ${x.set} ${x.number}`.toLowerCase().includes(f));
  const sort = $("#sort").value;
  if (sort === "value") items.sort((a, b) => (b.price || 0) * b.qty - (a.price || 0) * a.qty);
  if (sort === "name") items.sort((a, b) => a.name.localeCompare(b.name));
  $("#pockets").innerHTML = items.length ? items.map((x, i) => `<button class="pocket" data-i="${i}">
      ${x.qty > 1 ? `<span class="qty">×${x.qty}</span>` : ""}
      <img src="${esc(x.image)}" alt="${esc(x.name)}" loading="lazy" referrerpolicy="no-referrer">
      <div class="pp"><span>${esc(x.name)}</span><b>${money(x.price)}</b></div></button>`).join("")
    : `<div class="none"><span class="ball-icon" aria-hidden="true"></span><br>${f ? "No cards match that filter." : "Your binder is empty. Scan a card and tap <b>Add to binder</b>."}</div>`;
  $$(".pocket").forEach((b) => (b.onclick = () => openDetail(items[+b.dataset.i], items[+b.dataset.i])));
}

async function refreshPrices() {
  const items = binder.all();
  let n = 0;
  toast(`Refreshing ${items.length} prices…`);
  for (const it of items) {
    try {
      const p = await livePrice(it);
      const v = p.variants.find((x) => x.label === it.variant) || p.variants.find((x) => x.market != null);
      if (v?.market != null) { binder.update(it.id, { price: v.market, priceAt: Date.now() }); n++; }
    } catch {}
  }
  toast(`Updated ${n} of ${items.length} prices`);
}

/* ---------------- search ---------------- */
async function doSearch(e) {
  e.preventDefault();
  const q = $("#q").value.trim();
  if (!q) return;
  const local = state.matcher.searchText(q);
  const render = (list) => {
    $("#searchResults").innerHTML = list.map((c, i) => `<button class="pocket" data-i="${i}">
      <img src="${esc(c.image)}" alt="${esc(c.name)}" loading="lazy" referrerpolicy="no-referrer">
      <div class="pp"><span>${esc(c.name)} · ${esc(c.number)}</span><b>${c.priceSnapshot != null ? money(c.priceSnapshot) : ""}</b></div></button>`).join("");
    $$("#searchResults .pocket").forEach((b) => (b.onclick = () => openDetail(list[+b.dataset.i])));
  };
  render(local);
  $("#searchInfo").textContent = `${local.length} in offline index · searching full Pokémon catalogue…`;
  if (q.length < 3 || /^\d/.test(q)) return;
  try {
    const online = await searchPokemon(q);
    const seen = new Set(local.map((c) => c.key));
    const merged = local.concat(online.filter((c) => !seen.has(c.key)));
    render(merged);
    $("#searchInfo").textContent = `${local.length} in offline index · ${merged.length - local.length} more from the Pokémon TCG API`;
  } catch {
    $("#searchInfo").textContent = `${local.length} in offline index · online catalogue unreachable`;
  }
}

/* ---------------- boot ---------------- */
function bind() {
  $("#startCam").onclick = startCamera;
  $("#flip").onclick = () => { state.facing = state.facing === "environment" ? "user" : "environment"; startCamera(); };
  $("#backToCam").onclick = startCamera;
  $("#capture").onclick = scan;
  $("#upload").onchange = (e) => e.target.files[0] && loadPhoto(e.target.files[0]);
  document.addEventListener("keydown", (e) => {
    if (e.code === "Space" && !$("#view-scan").hidden && document.activeElement === document.body) { e.preventDefault(); scan(); }
  });
  $("#binderFilter").oninput = (e) => { state.binderFilter = e.target.value; renderBinder(); };
  $("#sort").onchange = renderBinder;
  $("#refresh").onclick = refreshPrices;
  $("#export").onclick = () => download("scandex-binder.csv", binder.toCSV(), "text/csv");
  $("#backup").onclick = () => download("scandex-binder.json", JSON.stringify(binder.all(), null, 1), "application/json");
  $("#restore").onchange = async (e) => {
    try { binder.replaceAll(JSON.parse(await e.target.files[0].text())); toast("Binder restored"); }
    catch { toast("That file is not a ScanDex backup"); }
  };
  $("#searchForm").onsubmit = doSearch;
  $("#currency").onchange = (e) => {
    state.currency = e.target.value;
    try { localStorage.setItem("scandex.currency", state.currency); } catch {}
    route();
  };
  window.addEventListener("binder-change", () => { if (!$("#view-binder").hidden) renderBinder(); });
  initPanZoom();
}

async function boot() {
  bind();
  try { state.currency = localStorage.getItem("scandex.currency") || "USD"; } catch {}
  $("#currency").value = state.currency;
  route();
  fxRates().then((r) => { state.rates = r; route(); });
  try {
    await state.matcher.load("model", setStatus);
    state.ready = true;
    $("#indexCount").textContent = state.matcher.size.toLocaleString();
    $("#capture").disabled = !(state.stream || state.source === "image");
    setStatus(`Index ready · ${state.matcher.size.toLocaleString()} cards (built ${state.matcher.built})`);
    setTimeout(() => warmUpOCR().catch(() => {}), 1500);
  } catch (e) {
    setStatus(`Could not load the card index: ${e.message}`);
  }
  // start automatically when permission was granted before
  try {
    const perm = await navigator.permissions?.query({ name: "camera" });
    if (perm?.state === "granted") startCamera();
  } catch {}
}
boot();

if ("serviceWorker" in navigator && location.protocol === "https:") {
  navigator.serviceWorker.register("sw.js").catch(() => {});
}
