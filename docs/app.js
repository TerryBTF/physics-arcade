"use strict";

const W = 640, H = 360;
const GAME_TEXT = {
  catapult: { noun: "robot", prompt: "Click every robot that will still be alive once everything has come to rest.", event: "survive" },
  pool: { noun: "ball", prompt: "Click every numbered ball that will end up in a pocket.", event: "potted" },
  dominoes: { noun: "domino", prompt: "Click every domino that will fall.", event: "fall" },
};
const DIFFS = ["easy", "medium", "hard"];

const $ = (sel) => document.querySelector(sel);
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else e.setAttribute(k, v);
  }
  for (const k of kids) e.append(k);
  return e;
};
const store = {
  get(k) { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* storage unavailable */ } },
};
const pct = (x) => `${(x * 100).toFixed(0)}%`;

async function load(name) {
  const r = await fetch(`data/${name}.json`);
  if (!r.ok) throw new Error(`could not load ${name}.json`);
  return r.json();
}

/* ---------------- Leaderboard ---------------- */

function segmented(container, options, current, onPick) {
  container.replaceChildren();
  for (const [value, label] of options) {
    const b = el("button", { type: "button", role: "tab", "aria-selected": String(value === current), text: label });
    b.addEventListener("click", () => onPick(value));
    container.append(b);
  }
}

function scoreCell(s) {
  const lo = 0.3, hi = 1.0, x = (v) => `${((Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo)) * 100}%`;
  const track = el("div", { class: "track" });
  const ci = el("div", { class: "ci" });
  ci.style.left = x(s.ci[0]);
  ci.style.width = `calc(${x(s.ci[1])} - ${x(s.ci[0])})`;
  const pt = el("div", { class: "pt" });
  pt.style.left = x(s.overall);
  const chance = el("div", { class: "chance", title: "chance (0.50)" });
  chance.style.left = x(0.5);
  track.append(ci, chance, pt);
  return el("div", { class: "score" }, el("b", { text: s.overall.toFixed(3) }), track);
}

function renderLeaderboard(board, state) {
  segmented($("#lb-game"), board.games.map((g) => [g.name, g.title]), state.game, (v) => { state.game = v; renderLeaderboard(board, state); });
  segmented($("#lb-style"), [["flat", "Flat"], ["cartoon", "Cartoon"]], state.style, (v) => { state.style = v; renderLeaderboard(board, state); });
  const rows = board.entries.map((e) => ({ e, s: e.results[state.game][state.style] }));
  const vlm = rows.filter((r) => r.e.kind === "vlm").sort((a, b) => b.s.overall - a.s.overall);
  const refs = rows.filter((r) => r.e.kind !== "vlm");
  const table = $("#lb-table");
  table.replaceChildren();
  table.append(el("thead", {}, el("tr", {},
    el("th", { text: "Model" }), el("th", { text: "Stable balanced accuracy" }),
    ...DIFFS.map((d) => el("th", { class: "num", text: d })), el("th", { class: "num", text: "Brier ↓" }))));
  const body = el("tbody");
  const addRow = ({ e, s }, cls) => {
    const name = el("td", {}, (e.names && e.names[state.game]) || e.name);
    if (e.note) name.append(el("span", { class: "note", text: e.note }));
    body.append(el("tr", { class: cls }, name, el("td", {}, scoreCell(s)),
      ...DIFFS.map((d) => el("td", { class: "num", text: s.tiers[d].toFixed(3) })), el("td", { class: "num", text: s.brier.toFixed(3) })));
  };
  vlm.forEach((r) => addRow(r, ""));
  body.append(el("tr", { class: "ref" }, el("td", { colspan: "6", class: "small", text: "Non-vision references" })));
  refs.forEach((r) => addRow(r, "ref"));
  table.append(body);
}

/* ---------------- Challenge ---------------- */

const play = { rounds: [], game: null, i: 0, picks: new Set(), tally: null, locked: false };

function chips(ids, truth, all) {
  const box = el("span", { class: "ids" });
  if (!ids.length) box.append(el("span", { class: "muted", text: "none" }));
  for (const id of ids) {
    const cls = truth ? (truth.includes(id) ? "id ok" : "id no") : "id";
    box.append(el("span", { class: cls, text: String(id) }));
  }
  return box;
}

function roundScore(picked, truth, targets) {
  return targets.filter((t) => picked.includes(t.id) === truth.includes(t.id)).length;
}

function renderPicker(rounds) {
  const box = $("#game-picker");
  box.replaceChildren();
  for (const game of Object.keys(GAME_TEXT)) {
    const first = rounds.find((r) => r.game === game);
    if (!first) continue;
    const best = store.get(`pa-best-${game}`);
    const card = el("button", { class: "pick-card", type: "button" },
      el("img", { src: first.still, alt: "", loading: "lazy" }),
      el("div", {}, el("strong", { text: game[0].toUpperCase() + game.slice(1) }),
        el("div", { class: "best", text: best != null ? `Your best: ${pct(best)}` : "8 rounds" })));
    card.addEventListener("click", () => startGame(game));
    box.append(card);
  }
}

function startGame(game) {
  play.game = game;
  play.list = play.rounds.filter((r) => r.game === game);
  play.i = 0;
  play.tally = { you: [0, 0], models: {} };
  $("#summary").hidden = true;
  $("#round").hidden = false;
  showRound();
  $("#round").scrollIntoView({ behavior: "smooth", block: "start" });
}

function showRound() {
  const r = play.list[play.i];
  const text = GAME_TEXT[r.game];
  play.picks = new Set();
  play.locked = false;
  $("#round-title").textContent = `${r.game[0].toUpperCase() + r.game.slice(1)} · ${r.difficulty}`;
  $("#round-progress").textContent = `Round ${play.i + 1} of ${play.list.length}`;
  $("#round-prompt").textContent = text.prompt;
  $("#stage-still").src = r.still;
  $("#stage-still").alt = `Level with ${text.noun}s labeled by number`;
  const layer = $("#stage-targets");
  layer.replaceChildren();
  for (const t of r.targets) {
    const b = el("button", { class: "target", type: "button", "aria-pressed": "false", "aria-label": `${text.noun} ${t.id}` });
    b.style.left = `${(t.x / W) * 100}%`;
    b.style.top = `${(t.y / H) * 100}%`;
    b.style.width = `${(t.w / W) * 100}%`;
    b.style.height = `${(t.h / H) * 100}%`;
    b.dataset.id = t.id;
    b.addEventListener("click", () => {
      if (play.locked) return;
      if (play.picks.has(t.id)) play.picks.delete(t.id); else play.picks.add(t.id);
      b.setAttribute("aria-pressed", String(play.picks.has(t.id)));
    });
    layer.append(b);
  }
  $("#round-result").hidden = true;
  $("#btn-submit").hidden = false;
  $("#btn-next").hidden = true;
  $("#btn-replay").hidden = false;
  playClip(r.context, 0.35, false);
}

function playClip(src, rate, keepLastFrame) {
  const stage = $("#stage"), v = $("#stage-video");
  v.onended = null;
  v.src = src;
  stage.classList.add("playing");
  v.onloadeddata = () => { v.playbackRate = rate; v.play().catch(() => {}); };
  v.onended = () => { if (!keepLastFrame) stage.classList.remove("playing"); };
}

function submit() {
  const r = play.list[play.i];
  play.locked = true;
  const picked = [...play.picks].sort((a, b) => a - b);
  const n = r.targets.length;
  const you = roundScore(picked, r.truth, r.targets);
  play.tally.you[0] += you; play.tally.you[1] += n;
  for (const b of $("#stage-targets").children) {
    const id = Number(b.dataset.id);
    b.classList.add(play.picks.has(id) === r.truth.includes(id) ? "right" : "wrong");
  }
  const rows = [["You", picked, you]];
  for (const [name, ids] of Object.entries(r.models)) {
    const s = roundScore(ids, r.truth, r.targets);
    const t = (play.tally.models[name] ||= [0, 0]);
    t[0] += s; t[1] += n;
    rows.push([name, ids, s]);
  }
  const ev = GAME_TEXT[r.game].event;
  const table = el("table", {}, el("thead", {}, el("tr", {}, el("th", { text: "" }), el("th", { text: `${GAME_TEXT[r.game].noun}s that ${ev}` }), el("th", { class: "num", text: "correct" }))));
  const body = el("tbody", {}, el("tr", {}, el("td", {}, el("b", { text: "What happened" })), el("td", {}, chips(r.truth, null)), el("td", {})));
  for (const [name, ids, s] of rows) body.append(el("tr", {}, el("td", { text: name }), el("td", {}, chips(ids, r.truth)), el("td", { class: "num", text: `${s} / ${n}` })));
  table.append(body);
  $("#round-result").replaceChildren(el("div", { class: "table-scroll" }, table));
  $("#round-result").hidden = false;
  $("#btn-submit").hidden = true;
  $("#btn-replay").hidden = true;
  $("#btn-next").hidden = false;
  $("#btn-next").textContent = play.i + 1 < play.list.length ? "Next round" : "See your score";
  playClip(r.outcome, 1, true);
}

function next() {
  play.i += 1;
  if (play.i < play.list.length) return showRound();
  $("#round").hidden = true;
  const acc = ([a, b]) => (b ? a / b : 0);
  const you = acc(play.tally.you);
  const key = `pa-best-${play.game}`;
  const best = store.get(key);
  if (best == null || you > best) store.set(key, you);
  const versus = el("div", { class: "versus" }, el("div", {}, el("b", { text: pct(you) }), "You"));
  for (const [name, t] of Object.entries(play.tally.models)) versus.append(el("div", {}, el("b", { text: pct(acc(t)) }), name));
  const beat = Object.values(play.tally.models).filter((t) => you > acc(t)).length;
  const again = el("button", { class: "btn primary", type: "button", text: "Play again" });
  again.addEventListener("click", () => startGame(play.game));
  const other = el("a", { class: "btn", href: "#play", text: "Try another game" });
  other.addEventListener("click", () => { $("#summary").hidden = true; renderPicker(play.rounds); });
  $("#summary").replaceChildren(
    el("p", { class: "muted", text: "Targets called correctly over 8 rounds" }),
    el("p", { class: "big", text: beat === Object.keys(play.tally.models).length ? "You beat every model." : beat ? "You beat one of the models." : "The models win this time." }),
    versus, el("div", { class: "round-actions" }, again, other));
  $("#summary").hidden = false;
  renderPicker(play.rounds);
}

/* ---------------- Gallery ---------------- */

function renderGallery(items) {
  const grid = $("#gallery-grid");
  grid.replaceChildren();
  for (const it of items) {
    const text = GAME_TEXT[it.game];
    const video = el("video", { src: it.outcome, muted: "", loop: "", autoplay: "", playsinline: "", preload: "metadata", "aria-label": `What actually happened (${it.game})` });
    video.muted = true;
    const body = el("div", { class: "body" },
      el("div", { class: "meta", text: `${it.game[0].toUpperCase() + it.game.slice(1)} · ${it.difficulty} · ${it.model}` }),
      el("div", { class: "rows" },
        el("span", { class: "muted", text: `Model: ${text.noun}s that ${text.event}` }), chips(it.answer, it.truth),
        el("span", { class: "muted", text: "What happened" }), chips(it.truth, null)),
      it.reasoning ? el("blockquote", { text: it.reasoning }) : el("blockquote", { text: "Answered with the JSON only: no reasoning at all." }));
    grid.append(el("article", { class: "card" }, video, body));
  }
}

/* ---------------- Boot ---------------- */

(async function main() {
  try {
    const [board, challenge, gallery] = await Promise.all([load("leaderboard"), load("challenge"), load("gallery")]);
    renderLeaderboard(board, { game: board.games[0].name, style: "flat" });
    play.rounds = challenge.rounds;
    renderPicker(play.rounds);
    renderGallery(gallery.items);
  } catch (err) {
    $("main").prepend(el("p", { class: "wrap muted", text: `Could not load the benchmark data (${err.message}).` }));
  }
  $("#btn-submit").addEventListener("click", submit);
  $("#btn-next").addEventListener("click", next);
  $("#btn-replay").addEventListener("click", () => playClip(play.list[play.i].context, 0.35, false));
})();
