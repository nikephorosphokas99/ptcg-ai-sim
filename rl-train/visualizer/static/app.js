// Front end for the player-vs-agent simulator.
//
// The client is a renderer, not a source of truth. Every payload it receives is
// the human seat's own observation as the engine produced it, so there is no
// opponent hand or prize list on this side to leak: the opponent's half of the
// mat is drawn from counts and face-down backs alone.
//
// The board is the physical playmat: Active Spots facing each other across the
// centre line, five Bench slots behind each Active, Prizes on the outer left,
// deck and discard on the outer right, your hand along the bottom. Empty slots
// are drawn, not omitted, so a bench of two reads as "two of five".

const $ = (id) => document.getElementById(id);
let view = null;
let picked = [];        // option indices chosen so far this decision
let targets = new Map();  // "side:area:index" -> [option index, ...]

// Which game on the server is ours. The server runs one engine process per
// session — the engine can only hold one battle per process — and this id is how
// a request says which one it belongs to. It is kept in localStorage and sent as
// a header rather than a cookie, because the page is normally viewed in an
// iframe on huggingface.co, where the Space's own cookies are third-party and
// silently dropped; a header behaves the same in an iframe as on a direct URL,
// and surviving a reload is what lets you return to a game in progress.
const SESSION_KEY = "ptcg.session";

function sessionId() {
  let id = null;
  try { id = localStorage.getItem(SESSION_KEY); } catch (e) { id = null; }
  if (!id) {
    id = (crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2) + Date.now());
    try { localStorage.setItem(SESSION_KEY, id); } catch (e) { /* private window */ }
  }
  return id;
}

async function api(path, body) {
  const headers = { "X-Session": sessionId() };
  // In a two-player room the server tells the seats apart by this token alone.
  if (pvp) headers["X-Player"] = pvp.token;
  // Served through ngrok's free tier, a browser request can be answered with
  // ngrok's "you are about to visit" page instead of the API; this header opts
  // the page's own requests out of it. Anywhere else it is ignored.
  headers["ngrok-skip-browser-warning"] = "1";
  if (body) headers["Content-Type"] = "application/json";
  const res = await fetch(path, {
    method: body ? "POST" : "GET",
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

// A closed tab should not hold a game process open until the idle reaper runs.
// keepalive lets the request outlive the page; a failure here costs nothing.
window.addEventListener("pagehide", () => {
  try {
    fetch("/api/leave", { method: "POST", keepalive: true,
                          headers: { "X-Session": sessionId() } });
  } catch (e) { /* nothing to do on the way out */ }
});

function el(tag, cls, html) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html !== undefined) n.innerHTML = html;
  return n;
}

function fill(select, values) {
  select.innerHTML = "";
  for (const v of values) select.appendChild(el("option", null, v)).value = v;
}

let catalogue = null;      // every card, for the builder's search
let prebuilt = [];         // the prebuilt decklists, expanded into rows
let chosenDeck = null;     // {name} for a prebuilt list, or null while building
let customDeck = [];       // card ids, when building your own
let selectedCard = null;   // the search result "Add to Deck" acts on
let agents = [];           // registered opponents, with archetype and architecture
let chosenAgent = null;    // the one that will pilot the other seat
let showing = "deck";      // which side of the dialog the right pane is showing

async function loadConfig() {
  const decks = await api("/api/decks");
  prebuilt = decks.decks;
  fill($("startFrom"), ["start from a prebuilt deck…", ...prebuilt.map((d) => d.name)]);
  agents = (await api("/api/agents")).agents;
  if (agents.length && !chosenAgent) chosenAgent = agents[0].id;
  renderDeckList();
}

// -- new-game dialog --------------------------------------------------------

function openSetup() {
  $("setupView").hidden = false;
  // ``#build`` opens straight into the builder, so a link can point at it.
  if (location.hash === "#build") chooseCustom();
  else if (!chosenDeck && !customDeck.length && prebuilt.length) chooseDeck(prebuilt[0].name);
  else renderDeckList();
}

function renderDeckList() {
  renderAgentList();
  const list = $("deckList");
  list.innerHTML = "";
  for (const deck of prebuilt) {
    const li = el("li", chosenDeck === deck.name && showing === "deck" ? "on" : "",
      `<span>${deck.name}</span><span class="n">${deck.total}</span>`);
    li.addEventListener("click", () => chooseDeck(deck.name));
    list.appendChild(li);
  }
  const own = el("li", `custom ${chosenDeck === null && showing === "deck" ? "on" : ""}`,
    `<span>Create your own…</span><span class="n">${customDeck.length || ""}</span>`);
  own.addEventListener("click", () => chooseCustom());
  list.appendChild(own);
}

/** Opponents in the same list style as the decks: name, and what it plays. */
function renderAgentList() {
  const list = $("agentList");
  list.innerHTML = "";
  for (const agent of agents) {
    const archetype = (agent.archetype || {}).name || "";
    const li = el("li", `agentRow ${chosenAgent === agent.id && showing === "agent" ? "on" : ""}`,
      `<span class="agentId">${agent.id}</span>` +
      `<span class="agentSub">${archetype.replace(/^.*'s /, "")}` +
      `${agent.arch && agent.arch.parameters ? ` · ${(agent.arch.parameters / 1e6).toFixed(1)}M` : ""}</span>`);
    li.addEventListener("click", () => chooseAgent(agent.id));
    list.appendChild(li);
  }
  $("pickedAgent").textContent = chosenAgent ? `vs ${chosenAgent}` : "no opponent registered";
}

// What each line of the architecture sheet means, in the terms of *this*
// network. Shown from the (?) beside each label — the numbers are meaningless to
// a reader who does not already know the vocabulary, and the point of the sheet
// is that they should not have to.
const ARCH_HELP = {
  "family":
    "A pointer network: instead of choosing from a fixed list of actions, it scores every legal " +
    "option the engine offers against the current state and points at one. That is what lets a " +
    "choice name a specific card on the board — a fixed action vocabulary could not.",
  "Parameters":
    "The learned numbers in the checkpoint. Roughly, capacity: more can represent more, and needs " +
    "more data to fit without memorising.",
  "Embedding dim":
    "The width of every vector inside the network. A card, a board and an option are each " +
    "described by this many numbers.",
  "Attention heads":
    "Attention runs this many times in parallel in each layer, each head free to look at something " +
    "different, and the results are concatenated.",
  "Layers":
    "Transformer blocks in total, summed over the towers listed below. Depth is how many rounds of " +
    "\u201clook at everything else, then update\u201d each vector goes through.",
  "· decision chain":
    "Reads the decisions already made this turn, causally — step three sees steps one and two, not " +
    "what comes after. A turn is a sequence of choices, not one choice.",
  "· decision context":
    "Reads the decision in front of it right now: the board, the prompt, and the legal options.",
  "· opponent history":
    "Reads what the opponent has been seen to do. It is the only model of the other seat the " +
    "network has — their hand and deck are not in its input.",
  "· option cross":
    "Cross-attention: each legal option queries the state and the turn so far, so the score of an " +
    "option depends on the position rather than on the option alone.",
  "Feed-forward":
    "The width of the small MLP inside each transformer block, as a multiple of the embedding dim.",
  "Dropout":
    "The fraction of activations randomly zeroed while training, to stop it leaning on any one " +
    "feature. Disabled when playing.",
  "Decision chain":
    "How many of this turn's earlier decisions the network is shown.",
  "Opponent history":
    "How many past observations of the opponent's board the network is shown.",
  "Training Method":
    "Behavioural cloning is supervised imitation of recorded games — it learns to play like the " +
    "games it was shown. PPO is reinforcement learning, improving against opponents by playing.",
  "Checkpoint":
    "The weights file inside the bundle that these numbers were read from.",
};
// The row has been labelled both ways; keep either spelling working so renaming
// it in the sheet cannot silently drop its explanation.
ARCH_HELP["Trained by"] = ARCH_HELP["Training Method"];

/** Show one opponent: the deck it pilots, and what its policy actually is. */
function chooseAgent(id) {
  chosenAgent = id;
  showing = "agent";
  $("builder").hidden = true;
  const agent = agents.find((a) => a.id === id);
  const arch = agent.arch || {};
  const archetype = agent.archetype || {};
  $("deckTitle").textContent = agent.id;
  setStatus(`${agent.deckTotal} cards · ${agent.deck.length} different`, "good");

  const grid = $("deckGrid");
  grid.innerHTML = "";
  const sheet = el("div", "agentSheet");
  if (archetype.image) {
    const art = el("div", "agentArt");
    const img = el("img");
    img.src = archetype.image;
    img.alt = archetype.name;
    art.appendChild(img);
    art.appendChild(el("div", "agentDeckName", archetype.name || ""));
    sheet.appendChild(art);
  }
  const spec = el("div", "agentSpec");
  const family = el("div", "agentFamily", arch.family || "policy network");
  family.appendChild(helpMark(ARCH_HELP.family));
  spec.appendChild(family);
  const rows = [
    ["Parameters", arch.parameters && arch.parameters.toLocaleString()],
    ["Embedding dim", arch.dim],
    ["Attention heads", arch.heads],
    ["Layers", arch.layers && `${arch.layers} across ${Object.keys(arch.towers || {}).length} towers`],
    ...Object.entries(arch.towers || {}).map(([name, n]) => [`· ${name.replace(/_/g, " ")}`, n]),
    ["Feed-forward", arch.ffMult && `${arch.ffMult}× dim`],
    ["Dropout", arch.dropout],
    ["Decision chain", arch.decisionChain && `${arch.decisionChain} frames`],
    ["Opponent history", arch.opponentHistory && `${arch.opponentHistory} frames`],
    ["Training Method", arch.trainedBy],
    ["Checkpoint", arch.checkpoint],
  ].filter(([, value]) => value !== undefined && value !== null && value !== "");
  const table = el("dl", "specTable");
  for (const [label, value] of rows) {
    const term = el("dt", null, label);
    if (ARCH_HELP[label]) term.appendChild(helpMark(ARCH_HELP[label]));
    table.appendChild(term);
    table.appendChild(el("dd", null, String(value)));
  }
  spec.appendChild(table);
  if (!rows.length) {
    spec.appendChild(el("div", "hint",
      "No ARCH.json in this bundle — run `python describe_agent.py agents/<name>`."));
  }
  sheet.appendChild(spec);
  grid.appendChild(sheet);

  grid.appendChild(el("h2", "listHead", "The deck it's trained"));
  const deck = el("div", "deckGrid inner");
  for (const row of agent.deck) {
    const wrap = el("div", "entry");
    wrap.appendChild(cardEl(row, { extra: "mini-card" }));
    wrap.appendChild(el("span", "xn", `x${row.count}`));
    deck.appendChild(wrap);
  }
  grid.appendChild(deck);
  renderDeckList();
}

function chooseDeck(name) {
  chosenDeck = name;
  showing = "deck";
  $("builder").hidden = true;
  const deck = prebuilt.find((d) => d.name === name);
  $("deckTitle").textContent = name;
  setStatus(`${deck.total} cards · ${deck.cards.length} different`, "good");
  renderDeckRows(deck.cards, false);
  renderDeckList();
}

async function chooseCustom() {
  chosenDeck = null;
  showing = "deck";
  $("builder").hidden = false;
  $("deckTitle").textContent = "Your deck";
  if (!catalogue) {
    setStatus("loading the card pool…");
    catalogue = (await api("/api/cards")).cards;
    runSearch();
  }
  await refreshCustom();
  renderDeckList();
}

/** Ask the server what the built deck still gets wrong, and show the rows. */
async function refreshCustom() {
  const res = await api("/api/deck/check", customDeck);
  renderDeckRows(res.listing, true);
  const kinds = res.listing.length;
  // The size is already on screen, so only the *other* rule breaks are listed.
  const rest = res.problems.filter((p) => !/^\d+\/60 cards$/.test(p));
  const size = `${customDeck.length}/60`;
  if (res.problems.length) setStatus(`${size}${rest.length ? " · " + rest.join(" · ") : ""}`, "bad");
  else setStatus(`${size} · ${kinds} different · legal`, "good");
  renderDeckList();
}

function setStatus(text, cls = "") {
  const node = $("deckStatus");
  node.className = `deckStatus ${cls}`;
  node.textContent = text;
}

/** ``[image] x N`` — one tile per distinct card, the count in the corner. */
function renderDeckRows(rows, editable) {
  const grid = $("deckGrid");
  grid.innerHTML = "";
  for (const row of rows) {
    const wrap = el("div", `entry ${editable ? "editable" : ""}`);
    wrap.appendChild(cardEl(row, { extra: "mini-card" }));
    wrap.appendChild(el("span", "xn", `x${row.count}`));
    if (editable) {
      const step = el("div", "step");
      step.appendChild(stepButton("minus", `Remove one ${row.name}`,
                                  () => removeCard(row.id)));
      step.appendChild(stepButton("plus", `Add one ${row.name}`,
                                  () => addCard(row.id, 1)));
      wrap.appendChild(step);
    }
    grid.appendChild(wrap);
  }
  if (!rows.length) grid.appendChild(el("div", "hint", "No cards yet — search below and add some."));
}

/** One of the deck tile's stepper buttons, with its glyph drawn inline.
 *
 *  Drawn rather than taken from an icon font. The page fetches nothing from the
 *  network -- index.html loads its own stylesheet and script and that is all --
 *  so a webfont for two glyphs would be the app's only external request, and
 *  the first thing to fail offline, inside the Docker image, or across the
 *  metered tunnel the README warns about. Two <path>s cost nothing and cannot
 *  fail to arrive.
 *
 *  The label is spelled out for screen readers and for the hover title: a bare
 *  + on a tile does not say what it adds.
 */
function stepButton(kind, label, onClick) {
  const b = el("button", `stepBtn ${kind}`);
  b.type = "button";
  b.title = label;
  b.setAttribute("aria-label", label);
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", "0 0 16 16");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("focusable", "false");
  for (const d of kind === "plus" ? ["M2.75 8h10.5", "M8 2.75v10.5"] : ["M2.75 8h10.5"]) {
    const path = document.createElementNS(NS, "path");
    path.setAttribute("d", d);
    svg.appendChild(path);
  }
  b.appendChild(svg);
  b.addEventListener("click", (ev) => { ev.stopPropagation(); onClick(); });
  return b;
}

function addCard(id, n = 1) {
  for (let i = 0; i < n && customDeck.length < 60; i++) customDeck.push(id);
  refreshCustom();
}

function removeCard(id) {
  const at = customDeck.lastIndexOf(id);
  if (at >= 0) customDeck.splice(at, 1);
  refreshCustom();
}

function runSearch() {
  if (!catalogue) return;
  const q = $("search").value.trim().toLowerCase();
  const kind = $("kindFilter").value;
  const hits = catalogue.filter((c) => {
    if (kind && c.kind !== kind) return false;
    if (!q) return true;
    if (c.name.toLowerCase().includes(q)) return true;
    if ((c.skills || []).some((s) => `${s.name} ${s.text}`.toLowerCase().includes(q))) return true;
    return (c.attacks || []).some((a) => `${a.name} ${a.text}`.toLowerCase().includes(q));
  }).slice(0, 120);

  const box = $("results");
  box.innerHTML = "";
  for (const card of hits) {
    const node = cardEl(card, { extra: `mini-card ${selectedCard === card.id ? "on" : ""}` });
    node.addEventListener("click", () => { selectedCard = card.id; runSearch(); });
    node.addEventListener("dblclick", () => addCard(card.id, 1));
    box.appendChild(node);
  }
  $("builderHint").textContent = hits.length >= 120
    ? "showing the first 120 matches — narrow the search"
    : `${hits.length} match${hits.length === 1 ? "" : "es"} · click to select, double-click to add`;
}

async function startGame() {
  if (buildingForRoom) { useDeckForRoom(); return; }
  if (pvp) leaveRoomLocally();
  const body = {
    agent: chosenAgent,
    seat: Number($("seat").value),
    humanDeck: chosenDeck,
    humanCards: chosenDeck === null ? customDeck : null,
    // For the saved replay's file name; the last name given in a 2-player lobby.
    playerName: (() => { try { return localStorage.getItem("ptcg.name") || "Player"; } catch (e) { return "Player"; } })(),
  };
  $("status").textContent = "starting…";
  try {
    view = await api("/api/new", body);
  } catch (err) {
    setStatus(err.message, "bad");
    $("status").textContent = `error: ${err.message}`;
    return;
  }
  $("setupView").hidden = true;
  $("loadout").textContent = `${chosenDeck || "your own deck"} vs ${chosenAgent}`;
  picked = [];
  render();
}

// -- cards ------------------------------------------------------------------

function cardEl(card, opts = {}) {
  // Art first: the picture is the card, and everything the picture cannot keep
  // current — damage taken, energy attached, conditions, tools — is overlaid on
  // top of it. Cards with no file on disk fall back to the text layout, which is
  // also what the face-down back uses.
  const node = el("div", `card ${card.facedown ? "facedown" : card.kind || ""} ${opts.extra || ""}`);
  if (card.serial !== undefined) node.dataset.serial = card.serial;
  if (card.image && !card.facedown) {
    node.classList.add("art");
    const img = el("img");
    img.src = card.image;
    img.alt = card.name;
    img.draggable = false;
    node.appendChild(img);
    node.appendChild(overlay(card, opts));
  } else {
    node.appendChild(textFace(card, opts));
  }
  if (!card.facedown) {
    node.addEventListener("mousemove", (ev) => showTip(card, ev));
    node.addEventListener("mouseleave", hideTip);
  }
  return node;
}

/** The strip drawn over the art: current HP, attachments, conditions. */
function overlay(card, opts) {
  const box = el("div", "overlay");
  const badges = [];
  for (const c of opts.conditions || []) badges.push(`<span class="badge cond">${c}</span>`);
  for (const t of card.tools || []) badges.push(`<span class="badge tool">${t}</span>`);
  if (card.appearThisTurn) badges.push(`<span class="badge new">new</span>`);
  if (badges.length) box.appendChild(el("div", "badges top", badges.join("")));

  if (card.curHp !== undefined) {
    const pct = card.maxHp ? Math.max(0, (card.curHp / card.maxHp) * 100) : 0;
    const damage = card.maxHp - card.curHp;
    const foot = el("div", "foot");
    foot.appendChild(el("div", "hpline",
      `<span class="hpnum ${pct < 34 ? "low" : pct < 67 ? "hurt" : ""}">${card.curHp}/${card.maxHp}</span>` +
      (damage ? `<span class="dmg">-${damage}</span>` : "")));
    foot.appendChild(el("div", `hpbar ${pct < 34 ? "low" : pct < 67 ? "hurt" : ""}`, `<i style="width:${pct}%"></i>`));
    if (card.energies && card.energies.length) {
      foot.appendChild(el("div", "energies", card.energies.map((e) => `<span class="pip">${e}</span>`).join("")));
    }
    box.appendChild(foot);
  }
  return box;
}

/** The pre-art card face, still used for face-down cards and missing files. */
function textFace(card, opts) {
  const box = el("div", "face");
  const rows = [];
  if (card.facedown) {
    rows.push(`<div class="name">Face-down</div>`);
  } else {
    rows.push(`<div class="name">${card.name}</div>`);
    const line = [card.stage, card.ex && !card.megaEx ? "ex" : "", card.megaEx ? "Mega ex" : "",
                  card.tera ? "Tera" : "", card.aceSpec ? "ACE SPEC" : ""].filter(Boolean).join(" ");
    if (line) rows.push(`<div class="meta">${line}</div>`);
    else if (card.kind && card.kind !== "pokemon") rows.push(`<div class="meta">${card.kind}</div>`);
  }
  if (card.curHp !== undefined) {
    const pct = card.maxHp ? Math.max(0, (card.curHp / card.maxHp) * 100) : 0;
    const badges = [];
    for (const c of opts.conditions || []) badges.push(`<span class="badge cond">${c}</span>`);
    for (const t of card.tools || []) badges.push(`<span class="badge">${t}</span>`);
    if (card.appearThisTurn) badges.push(`<span class="badge new">new</span>`);
    if (badges.length) rows.push(`<div class="badges">${badges.join("")}</div>`);
    if (card.energies && card.energies.length) rows.push(`<div class="energies">${card.energies.join(" ")}</div>`);
    rows.push(`<div class="hp">${card.curHp}/${card.maxHp}</div>`);
    rows.push(`<div class="hpbar ${pct < 34 ? "low" : pct < 67 ? "hurt" : ""}"><i style="width:${pct}%"></i></div>`);
  }
  box.innerHTML = rows.join("");
  return box;
}

function showTip(card, ev) {
  const tip = $("tooltip");
  const rows = [];
  if (card.image) rows.push(`<img class="tipart" src="${card.image}" alt="${card.name}">`);
  rows.push(`<b>${card.name}</b>`);
  if (card.stage) rows.push(`<div>${card.stage}${card.evolvesFrom ? ` — evolves from ${card.evolvesFrom}` : ""}</div>`);
  if (card.hp) {
    rows.push(`<div>HP ${card.hp} · ${card.energy || "-"} · retreat ${card.retreat}` +
      `${card.weakness ? ` · weakness ${card.weakness}` : ""}${card.resistance ? ` · resistance ${card.resistance}` : ""}</div>`);
  }
  for (const s of card.skills || []) rows.push(`<div class="atk"><b>${s.name}</b><br>${s.text}</div>`);
  for (const a of card.attacks || []) {
    rows.push(`<div class="atk"><b>${(a.cost || []).join("")} ${a.name}${a.damage ? ` — ${a.damage}` : ""}</b>` +
      `${a.text ? `<br>${a.text}` : ""}</div>`);
  }
  tip.innerHTML = rows.join("");
  tip.hidden = false;
  placeTip(tip, ev);
}

/** Placed after measuring, and flipped rather than clamped: a card hovered near
 *  the bottom of the dialog was having its art cut off by the window edge. */
function placeTip(tip, ev) {
  tip.style.left = "0px";
  tip.style.top = "0px";
  const box = tip.getBoundingClientRect();
  const pad = 10;
  let x = ev.clientX + 16;
  if (x + box.width > window.innerWidth - pad) x = ev.clientX - box.width - 16;
  let y = ev.clientY + 16;
  if (y + box.height > window.innerHeight - pad) y = window.innerHeight - box.height - pad;
  tip.style.left = `${Math.max(pad, x)}px`;
  tip.style.top = `${Math.max(pad, y)}px`;
}
function hideTip() { $("tooltip").hidden = true; }

/** A (?) that explains a term on hover, in the same tooltip the cards use. */
function helpMark(text) {
  const mark = el("span", "help", "?");
  mark.addEventListener("mousemove", (ev) => showHelp(text, ev));
  mark.addEventListener("mouseleave", hideTip);
  return mark;
}

function showHelp(text, ev) {
  const tip = $("tooltip");
  tip.innerHTML = `<div class="helpText">${text}</div>`;
  tip.hidden = false;
  placeTip(tip, ev);
}

// -- slots ------------------------------------------------------------------

function key(side, area, index) { return `${side}:${area}:${index}`; }

function slot(side, area, index, card, opts = {}) {
  const k = key(side, area, index);
  const hits = targets.get(k) || [];
  const holder = el("div", `slot ${opts.cls || ""} ${hits.length ? "targetable" : ""}`);
  if (hits.length) holder.dataset.key = k;
  if (card) holder.appendChild(cardEl(card, opts));
  else holder.textContent = opts.empty || "empty";
  if (hits.length) {
    holder.addEventListener("click", () => hitOptions(hits));
    holder.addEventListener("mouseenter", () => markHot(hits, true));
    holder.addEventListener("mouseleave", () => markHot(hits, false));
    if (hits.some((i) => picked.includes(i))) holder.classList.add("picked");
  }
  return holder;
}

/** Clicking a board slot.
 *
 *  One option there: take it. Several during a multi-select (two Energy on the
 *  same Pokémon, say): take the next one not yet picked, and unpick on the way
 *  back — clicking the slot has to make progress, or Confirm never enables and
 *  the slot feels dead. Several during a single choice really is ambiguous, so
 *  that one only highlights the rows it could mean.
 */
function hitOptions(hits) {
  // A drop ends with a click on whatever was under the pointer. The drop has
  // already chosen; letting that click through would choose a second time.
  if (performance.now() - dropEndedAt < 250) return;
  chooseAmong(hits);
}

function chooseAmong(hits) {
  if (hits.length === 1) { pickOption(hits[0]); return; }
  const sel = view.select;
  const many = sel && (sel.maxCount > 1 || sel.minCount === 0);
  if (many) {
    const next = hits.find((i) => !picked.includes(i));
    if (next !== undefined) { pickOption(next); return; }
    pickOption(hits[hits.length - 1]);  // all taken: give one back
    return;
  }
  const first = document.querySelector(`.option[data-index="${hits[0]}"]`);
  if (first) first.scrollIntoView({ block: "nearest" });
  markHot(hits, true);
  setTimeout(() => markHot(hits, false), 1200);
}

function markHot(hits, on) {
  for (const i of hits) {
    const node = document.querySelector(`.option[data-index="${i}"]`);
    if (node) node.classList.toggle("hot", on);
  }
}

// -- dragging ---------------------------------------------------------------
//
// Dragging is a second way to say what a click already says, and deliberately
// nothing more. A drop never invents a move: it looks up the option the
// *server* sent for that exact (card, destination) pair and picks it, so the
// wire carries the same `{"options": [i]}` a click would have sent and the
// engine validates it identically. A gesture the option list does not contain
// is not a drop target, which is what keeps the rules in the engine rather
// than duplicated here -- there is no second, weaker copy of "can I attach
// this?" in the browser that could disagree with the real one.
//
// Only two engine option types name both a card and where it lands: ATTACH
// (Energy, Pokémon Tool) and EVOLVE, which are exactly the gestures the rules
// describe as putting a card from your hand onto a Pokémon. Playing a Basic
// names only the card, because the engine picks the Bench slot itself, so a
// Basic drops onto the Bench as a whole and never onto one numbered space --
// claiming otherwise would invent a choice the engine never offered. Anything
// else stays a click: a Supporter has nowhere on the board to land.

const DRAG_SLOP = 6;      // px of travel before a press is a drag and not a click
const ZONE_FOR_GROUP = { "Pokémon": "bench" };  // single-ref plays, by option group

let drags = new Map();    // source key -> Map(destination key -> [option index, ...])
let held = null;          // {node, ghost, plan, dest, lit} while a card is off the table
let dropEndedAt = 0;      // when the last drop finished, to swallow its trailing click

function zoneKey(name) { return `zone:${name}`; }

/** Index every option that reads as "put this card there".
 *
 *  Built from the same ``refs`` the highlight map uses and from nothing else:
 *  refs[0] is the card leaving the hand, refs[1] the slot it lands on.
 */
function buildDrags() {
  drags = new Map();
  if (!view || !view.yourTurn) return;
  for (const opt of view.options) {
    const refs = opt.refs || [];
    const src = refs[0];
    if (!src || src.area !== "hand") continue;   // you pick up from your hand, nowhere else
    let dest;
    if (refs.length > 1) {
      dest = key(refs[1].side || "me", refs[1].area, refs[1].index);
    } else if (ZONE_FOR_GROUP[opt.group]) {
      dest = zoneKey(ZONE_FOR_GROUP[opt.group]);
    } else {
      continue;
    }
    const from = key(src.side || "me", src.area, src.index);
    if (!drags.has(from)) drags.set(from, new Map());
    const byDest = drags.get(from);
    if (!byDest.has(dest)) byDest.set(dest, []);
    byDest.get(dest).push(opt.index);
  }
}

/** The element standing for a destination: a board slot, or a whole zone. */
function dropNode(dest) {
  if (dest.startsWith("zone:")) {
    return document.querySelector(`[data-zone="${dest.slice(5)}"]`);
  }
  return document.querySelector(`.slot[data-key="${dest}"], .card[data-key="${dest}"]`);
}

/** Let this node be picked up, if the option list gives it somewhere to go. */
function armDrag(node, from) {
  if (!drags.has(from)) return;
  node.classList.add("draggable");
  node.addEventListener("pointerdown", (ev) => pressToDrag(ev, node, from));
}

/** A press becomes a drag only after DRAG_SLOP px, so clicking still works.
 *
 *  Pointer events rather than HTML5 drag-and-drop: the card art sets
 *  ``draggable = false`` to stop the browser's own image drag, the board is
 *  played on touchscreens as well as with a mouse, and a native drag image
 *  cannot be styled to look like the card leaving the hand.
 */
function pressToDrag(ev, node, from) {
  if (ev.pointerType === "mouse" && ev.button !== 0) return;
  if (busy || replaying || animating || watch) return;
  if (!view || !view.yourTurn) return;
  const plan = drags.get(from);
  if (!plan || !plan.size) return;

  const ox = ev.clientX, oy = ev.clientY;

  const move = (m) => {
    if (!held && Math.hypot(m.clientX - ox, m.clientY - oy) < DRAG_SLOP) return;
    if (!held) liftCard(node, plan);
    trackCard(m.clientX, m.clientY);
  };
  const finish = (u) => {
    unwire();
    if (!held) return;          // never travelled: it was a click, leave it alone
    dropCard(u.clientX, u.clientY);
  };
  const abandon = () => { unwire(); releaseCard(); };
  const onKey = (k) => { if (k.key === "Escape") { k.preventDefault(); abandon(); } };
  function unwire() {
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", finish);
    window.removeEventListener("pointercancel", abandon);
    window.removeEventListener("keydown", onKey, true);
  }

  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", finish);
  window.addEventListener("pointercancel", abandon);
  window.addEventListener("keydown", onKey, true);
}

/** Take the card off the table: a ghost follows the pointer, the places it may
 *  legally land light up, and the original stays behind greyed out. */
function liftCard(node, plan) {
  hideTip();   // the card is moving; a tooltip pinned to where it was is noise
  const box = node.getBoundingClientRect();
  const ghost = node.cloneNode(true);
  ghost.classList.add("ghost");
  ghost.classList.remove("targetable", "picked", "draggable");
  // The clone must not answer to the destination lookups below, or a card could
  // be dropped on its own shadow.
  delete ghost.dataset.key;
  for (const inner of ghost.querySelectorAll("[data-key]")) delete inner.dataset.key;
  ghost.style.width = `${box.width}px`;
  node.classList.add("lifted");
  document.body.appendChild(ghost);
  document.body.classList.add("dragging");
  held = { node, ghost, plan, dest: null, lit: [], dx: box.width / 2, dy: box.height / 2 };
  for (const dest of plan.keys()) dropNode(dest)?.classList.add("dropOk");
}

function trackCard(x, y) {
  held.ghost.style.left = `${x - held.dx}px`;
  held.ghost.style.top = `${y - held.dy}px`;
  const dest = destUnder(x, y);
  if (dest === held.dest) return;
  if (held.dest) dropNode(held.dest)?.classList.remove("dropHot");
  if (held.lit.length) { markHot(held.lit, false); held.lit = []; }
  held.dest = dest;
  if (!dest) return;
  dropNode(dest)?.classList.add("dropHot");
  // Name the move in the side list too, so the gesture and the wording the
  // player would otherwise have clicked are visibly the same thing.
  held.lit = held.plan.get(dest) || [];
  markHot(held.lit, true);
}

/** Which legal destination, if any, is under the pointer. A slot the plan knows
 *  wins over the zone containing it: dropping an Energy on a benched Pokémon
 *  means that Pokémon, not "somewhere on the Bench". */
function destUnder(x, y) {
  const at = document.elementFromPoint(x, y);
  if (!at) return null;
  const slot = at.closest("[data-key]");
  if (slot && held.plan.has(slot.dataset.key)) return slot.dataset.key;
  const zone = at.closest("[data-zone]");
  if (zone && held.plan.has(zoneKey(zone.dataset.zone))) return zoneKey(zone.dataset.zone);
  return null;
}

function dropCard(x, y) {
  const dest = destUnder(x, y);
  const hits = dest ? held.plan.get(dest) : null;
  releaseCard();
  dropEndedAt = performance.now();
  if (!hits || !hits.length) return;      // dropped on nothing: no move, nothing sent
  if (hits.length === 1) { pickOption(hits[0]); return; }
  // One card, one destination, two different options -- two Energy of the same
  // type on the same Pokémon during a multi-select. The gesture cannot say
  // which, so hand it to the same code a click uses.
  chooseAmong(hits);
}

/** Put everything back the way it was drawn. Safe to call when nothing is held. */
function releaseCard() {
  if (!held) return;
  held.ghost.remove();
  held.node.classList.remove("lifted");
  if (held.lit.length) markHot(held.lit, false);
  for (const dest of held.plan.keys()) {
    dropNode(dest)?.classList.remove("dropOk", "dropHot");
  }
  document.body.classList.remove("dragging");
  held = null;
}

function stackPrizes(side, count) {
  const box = el("div", "stack prizes");
  box.appendChild(el("div", "label", `Prizes ${count}`));
  const grid = el("div", "grid");
  for (let i = 0; i < 6; i++) grid.appendChild(el("div", `mini ${i < count ? "" : "taken"}`));
  box.appendChild(grid);
  return box;
}

function stackPiles(sideName, sideData, isMe) {
  const box = el("div", "stack piles");
  const deck = el("div", "pile deck", `${sideData.deckCount}`);
  deck.title = "Deck (face-down)";
  const discard = el("div", "pile discard", `${sideData.discardCount}`);
  discard.title = "Discard pile — click to browse";
  discard.addEventListener("click", () => showPile(`${sideName} discard pile`, sideData.discard));
  box.appendChild(el("div", "label", "Deck"));
  box.appendChild(deck);
  box.appendChild(el("div", "label", "Discard"));
  box.appendChild(discard);
  return box;
}

function showPile(title, cards) {
  $("pileTitle").textContent = `${title} (${cards.length})`;
  const box = $("pileCards");
  box.innerHTML = "";
  for (const c of cards) box.appendChild(cardEl(c));
  if (!cards.length) box.appendChild(el("div", null, "empty"));
  $("pileView").hidden = false;
}

function chips(side, isMe) {
  const out = [];
  // The once-per-turn flags belong to a live decision; a replay has none to show.
  const flags = !watch && view ? view.flags : null;
  if (isMe && flags) {
    out.push(`<span class="chip ${flags.energy ? "spent" : ""}">energy attach</span>`);
    out.push(`<span class="chip ${flags.supporter ? "spent" : ""}">supporter</span>`);
    out.push(`<span class="chip ${flags.stadium ? "spent" : ""}">stadium</span>`);
    out.push(`<span class="chip ${flags.retreat ? "spent" : ""}">retreat</span>`);
  }
  if (isMe || watch) out.push(`<span class="chip">hand ${side.handCount}</span>`);
  else out.push(`<span class="chip">hand &amp; prizes hidden</span>`);
  for (const c of side.conditions) out.push(`<span class="chip warn">${c}</span>`);
  return out.join("");
}

// -- animation --------------------------------------------------------------
//
// A decision the opponent made is otherwise invisible: by the time you get the
// board back, the Poké Pad has been played, the attack has resolved and all you
// have is a changed number. So the events that arrive with a view are acted out
// in order — the card that was played is held up, the attack is announced, and
// the HP change floats off the Pokémon it happened to. The board itself is
// already up to date; these are overlays on top of it.

let animating = null;
let replaying = false;   // a turn is being played back
let replayingOpponent = false;  // ...and it contains the opponent's own actions
let shownHand = [];      // the hand the player last held, kept for mid-turn boards
// Clicks are ignored briefly after a selection is sent. Hilda searches twice in
// a row with identical prompts, so a second click meant for the first prompt
// landed on the second one — and because those searches allow taking nothing,
// it silently skipped the Energy search instead of doing nothing.
let busy = false;

function cardNode(ev) {
  if (ev.serial === undefined || ev.serial === null) return null;
  return document.querySelector(`.card[data-serial="${ev.serial}"]`);
}

/** A number floating up from a card (or from the centre if it has left play). */
function floatNumber(ev, ms = 1050) {
  const node = cardNode(ev);
  const text = ev.value > 0 ? `+${ev.value}` : `${ev.value}`;
  const tag = el("div", `floater ${ev.value > 0 ? "heal" : "damage"}`, text);
  tag.style.animationDuration = `${ms}ms`;
  const box = node ? node.getBoundingClientRect() : null;
  tag.style.left = box ? `${box.left + box.width / 2}px` : "50%";
  tag.style.top = box ? `${box.top + box.height / 3}px` : "45%";
  document.body.appendChild(tag);
  if (node) {
    node.classList.add(ev.value > 0 ? "flash-heal" : "flash-hit");
    setTimeout(() => node.classList.remove("flash-heal", "flash-hit"), ms);
  }
  setTimeout(() => tag.remove(), ms + 60);
}

/** Move the HP a card shows now, ahead of the snapshot that confirms it.
 *
 *  A decision is snapshotted whole, so the board only catches up once the whole
 *  action has resolved — an attack's damage and the Knock Out it caused would
 *  otherwise land in the same frame, with the HP never seen to fall. Applying
 *  the number as it floats lets the bar drop on its own beat and the Pokémon
 *  leave on the next one, the order the cards move in a real game. This is only
 *  ever a preview: the snapshot that follows still has the last word.
 */
function nudgeHp(ev) {
  const node = cardNode(ev);
  if (!node) return;
  const label = node.querySelector(".hp");
  const bar = node.querySelector(".hpbar");
  const fill = bar && bar.querySelector("i");
  if (!label || !fill) return;
  const [cur, max] = label.textContent.split("/").map((n) => parseInt(n, 10));
  if (!Number.isFinite(cur) || !Number.isFinite(max) || !max) return;
  const now = Math.min(max, Math.max(0, cur + ev.value));
  const pct = (now / max) * 100;
  label.textContent = `${now}/${max}`;
  fill.style.width = `${pct}%`;
  bar.classList.toggle("low", pct < 34);
  bar.classList.toggle("hurt", pct >= 34 && pct < 67);
}

/** Where a zone lives on screen, so a card can be seen travelling to it. */
function zoneAnchor(side, area, serial) {
  const half = side === "opp" ? $("oppActive").parentElement : $("meActive").parentElement;
  switch (area) {
    case "deck": return half.querySelector(".piles .pile.deck");
    case "discard": return half.querySelector(".piles .pile.discard");
    case "prize": return half.querySelector(".prizes");
    case "hand": return side === "opp" ? $("oppHand") : $("meHand");
    case "active": return side === "opp" ? $("oppActive") : $("meActive");
    case "bench": {
      const known = serial != null && document.querySelector(`.card[data-serial="${serial}"]`);
      return known || (side === "opp" ? $("oppBench") : $("meBench"));
    }
    case "stadium": return $("stadium");
    case "revealed": return $("looking");
    default: return null;
  }
}

/** Send a card across the board from one zone to another.
 *
 *  The board itself only ever swaps between positions, so without this a card
 *  drawn from the deck simply appears in hand and a Knocked Out Pokémon simply
 *  stops existing. The travelling card is a copy: face-up when the move is
 *  public and the card is known, face-down otherwise — a card going into an
 *  opponent's hand is not ours to show.
 */
function flyCard(fromArea, toArea, side, card, ms, extra = "") {
  const from = zoneAnchor(side, fromArea, card && card.serial);
  const to = zoneAnchor(side, toArea, card && card.serial);
  if (!from || !to) return;
  const a = from.getBoundingClientRect();
  const b = to.getBoundingClientRect();
  const known = card && card.image && toArea !== "hand" ? card : (card && card.image && side === "me" ? card : null);

  const ghost = el("div", `flycard ${extra}`);
  if (known) {
    const img = el("img");
    img.src = known.image;
    ghost.appendChild(img);
  } else {
    ghost.classList.add("back");
  }
  ghost.style.left = `${a.left + a.width / 2 - 33}px`;
  ghost.style.top = `${a.top + a.height / 2 - 46}px`;
  document.body.appendChild(ghost);
  const dx = (b.left + b.width / 2) - (a.left + a.width / 2);
  const dy = (b.top + b.height / 2) - (a.top + a.height / 2);
  const travel = Math.max(320, Math.round(ms * 0.7));
  requestAnimationFrame(() => {
    ghost.style.transition = `transform ${travel}ms cubic-bezier(.2,.7,.3,1), opacity ${travel}ms ease-in`;
    ghost.style.transform = `translate(${dx}px, ${dy}px) scale(${extra === "ko" ? 0.6 : 0.9})`;
    if (extra === "ko" || toArea === "deck" || toArea === "prize") ghost.style.opacity = "0.15";
  });
  setTimeout(() => ghost.remove(), travel + 80);
}

/** Several cards at once, fanned out slightly so a draw of six reads as six. */
function flyMany(fromArea, toArea, side, card, ms, count) {
  const many = Math.min(count || 1, 4);
  for (let i = 0; i < many; i++) {
    setTimeout(() => flyCard(fromArea, toArea, side, card, ms), i * 90);
  }
}

/** The banner used for a played card, an attack, an ability, a coin. */
function banner(ev, kindClass, title, subtitle = "", life = 1400) {
  const box = el("div", `banner ${kindClass} ${ev.side === "me" ? "mine" : "theirs"}`);
  if (ev.card && ev.card.image) {
    const img = el("img");
    img.src = ev.card.image;
    img.alt = ev.card.name;
    box.appendChild(img);
  }
  const text = el("div", "bannerText");
  text.appendChild(el("div", "who", ev.side === "me" ? "You" : "Opponent"));
  text.appendChild(el("div", "title", title));
  if (subtitle) text.appendChild(el("div", "sub", subtitle));
  box.appendChild(text);
  // The banner has to outlive the beat that follows it, or a slow replay would
  // show an empty stage between steps.
  box.style.animation = `bannerIn .18s ease-out, bannerOut .3s ease-in ${Math.max(200, life - 300)}ms forwards`;
  const stage = $("stage");
  stage.appendChild(box);
  // Two at a time: a taller stack starts covering the board it is describing.
  while (stage.children.length > 2) stage.firstChild.remove();
  setTimeout(() => box.remove(), life + 100);
}

function highlight(ev, cls, ms) {
  const node = cardNode(ev);
  if (!node) return;
  node.classList.add(cls);
  setTimeout(() => node.classList.remove(cls), ms);
}

// How long each kind of event is held. The opponent's whole turn arrives in one
// response, so this pacing is the only thing standing between you and a board
// that changes all at once. Every action gets at least MIN_BEAT — one second at
// normal speed — and the ones worth dwelling on get more; the speed control
// multiplies the lot.
const MIN_BEAT = 1000;
const BEATS = {
  result: 2200, attack: 1800, play: 1400, evolve: 1400, mulligan: 1400, condition: 1200,
  coin: 1200, damage: 1200, heal: 1200, switch: 1200, move: 1100,
  attach: 1000, recover: 1000, draw: 1000, shuffle: 1000,
};
const SPEEDS = { slow: 1.6, normal: 1.0, fast: 0.55 };

function speedFactor() {
  return SPEEDS[localStorage.getItem("animSpeed") || "normal"] ?? 1;
}

/** Play one event; resolve after the beat it is worth. */
function playEvent(ev, scale) {
  const ms = Math.round(Math.max(MIN_BEAT, BEATS[ev.kind] ?? MIN_BEAT) * scale);
  const beat = () => new Promise((done) => setTimeout(done, ms));
  switch (ev.kind) {
    case "damage":
    case "heal":
      floatNumber(ev, ms);
      nudgeHp(ev);
      return beat();
    case "attack":
      highlight(ev, "attacking", ms);
      banner(ev, "attack", ev.name, ev.value ? `${ev.value} damage` : ev.text || "", ms);
      return beat();
    case "play":
      // No flight here: the engine logs the card's actual movement separately,
      // and animating both would send the same card twice, once to the wrong
      // place (a Stadium does not go to the discard, a Pokémon may go active).
      banner(ev, "play", ev.name || "a card", "played", ms);
      return beat();
    case "evolve":
      highlight(ev, "evolving", ms);
      banner(ev, "play", ev.name || "Evolution", "evolved", ms);
      return beat();
    case "attach":
      highlight(ev, "attaching", ms);
      return beat();
    case "condition":
      highlight(ev, "flash-hit", ms);
      banner(ev, "condition", ev.name, "", ms);
      return beat();
    case "coin":
      banner(ev, "coin", ev.name, "coin flip", ms);
      return beat();
    case "draw": {
      const what = ev.count > 1 ? `drew ${ev.count} cards` : (ev.name ? "drew" : "drew a card");
      banner(ev, "quiet", ev.name || "Draw", what, ms);
      flyMany("deck", "hand", ev.side, ev.card, ms, ev.count);
      return beat();
    }
    case "move": {
      const what = ev.count > 1 ? `${ev.count} cards · ${ev.from} → ${ev.to}` : `${ev.from} → ${ev.to}`;
      banner(ev, "quiet", ev.name || "Card moved", what, ms);
      // A Pokémon leaving the board for the discard pile is a Knock Out: it
      // shrinks and fades on the way out instead of blinking out of existence.
      const ko = (ev.from === "active" || ev.from === "bench") && ev.to === "discard";
      if (ko) flyCard(ev.from, ev.to, ev.side, ev.card, ms, "ko");
      else flyMany(ev.from, ev.to, ev.side, ev.card, ms, ev.count);
      return beat();
    }
    case "shuffle":
      banner(ev, "quiet", "Shuffled the deck", "", ms);
      return beat();
    case "switch":
      highlight(ev, "evolving", ms);
      banner(ev, "play", ev.name || "Switch", "to the Active Spot", ms);
      return beat();
    case "mulligan":
      banner(ev, "condition", "Mulligan", "no Basic Pokémon — redrawing", ms);
      return beat();
    case "result":
      // The verdict is the closing beat, not an announcement over a turn still
      // being played: the Pokémon has to reach the discard pile first.
      banner(ev, "result", ev.name, ev.text, ms);
      return beat();
    default:
      return beat();
  }
}

/** Collapse runs of identical bookkeeping events into one counted beat.
 *
 *  A Supporter that draws six cards logs six draws. Six separate beats is a
 *  minute of watching a counter change; one beat that says "drew 6 cards" is
 *  the decision. Only draws, moves and shuffles merge — the choices a player
 *  actually made each keep their own beat.
 */
function mergeEvents(events) {
  const RUNS = new Set(["draw", "move", "shuffle"]);   // counted
  const SUMS = new Set(["damage", "heal"]);            // added up
  const out = [];
  for (const ev of events) {
    const last = out[out.length - 1];
    if (last && last.kind === ev.kind && last.side === ev.side) {
      if (RUNS.has(ev.kind) && last.from === ev.from && last.to === ev.to) {
        last.count = (last.count || 1) + 1;
        if (last.name !== ev.name) {
          // A run of different cards: drop the identity, or four different
          // cards would all fly wearing the first one's face.
          last.name = null;
          last.card = null;
        }
        continue;
      }
      // Damage counters land one log entry at a time; three of them on the same
      // Pokémon are one hit, and should read as -30, not as -10 three times.
      if (SUMS.has(ev.kind) && last.serial === ev.serial) {
        last.value += ev.value;
        continue;
      }
    }
    out.push({ ...ev, count: 1 });
  }
  return out;
}

/** Run a view's events in order, narrating the progress in the status bar. */
/** How long a finished move is left standing before the next one begins. */
const HOLD = 700;
const pause = (ms) => new Promise((done) => setTimeout(done, ms));

async function playEvents(events, steps = []) {
  const token = {};
  animating = token;
  // No length-based compression: a turn with twelve actions in it takes twelve
  // beats to watch, which is the point. The speed control is how you shorten it.
  const scale = speedFactor();
  const total = mergeEvents(events).length;
  // Your options stay locked until the replay finishes. Before this, acting
  // immediately cancelled the rest of the opponent's turn — the very thing the
  // replay exists to show — and it was easy to do by accident.
  replaying = true;
  replayingOpponent = events.some((ev) => ev.side === "opp");
  renderChoice();
  // A step is one move: the events it was made of, then the board it produced.
  // Replaying it that way needs no alignment between narration and position —
  // the engine paired them when it recorded the step.
  let done = 0;
  for (const step of steps) {
    const beats = mergeEvents(step.events || []);
    for (const ev of beats) {
      if (animating !== token) break;
      done += 1;
      // Whose action it is shows on the banner; a batch usually contains the
      // tail of your own turn as well as the opponent's, so this stays neutral.
      $("status").textContent = `replaying — ${done}/${total} · press S to skip`;
      await playEvent(ev, scale);
    }
    if (animating !== token) break;
    drawBoard(step.board, shownHand);          // what the move left behind
    if (beats.length) await pause(Math.round(HOLD * scale));
  }
  if (animating === token) animating = null;
  if (!animating) {
    replaying = false;
    drawFinal();          // land on the real position, hand and all
    $("status").textContent = statusLine();
  showTurn();
    renderChoice();
  }
}

/** End the replay now and hand the board back. */
/** Cut the replay short. Bound to the S key only — deliberately not to anything
 *  clickable, so it cannot happen by accident. */
function skipReplay() {
  if (!replaying) return;
  animating = null;
  replaying = false;
  $("stage").innerHTML = "";
  drawFinal();
  $("status").textContent = statusLine();
  showTurn();
  renderChoice();
}

// -- main render ------------------------------------------------------------

function buildTargets() {
  targets = new Map();
  if (!view || !view.yourTurn) return;
  for (const opt of view.options) {
    // An option can touch two slots at once (the card leaving your hand and the
    // Pokémon it lands on); both light up, and either one can be clicked.
    for (const r of opt.refs || []) {
      const k = key(r.side || "me", r.area, r.index);
      if (!targets.has(k)) targets.set(k, []);
      if (!targets.get(k).includes(opt.index)) targets.get(k).push(opt.index);
    }
  }
  buildDrags();
}

/** The header line, recomputed rather than remembered — a replay finishing must
 *  not restore a status from before the click that started it. */
/** Say whose turn it is, loudly.
 *
 *  The status line already carried this, but as one clause of a sentence in
 *  12px grey -- you had to read "waiting on grimmsnarl_ex_xattn" to notice the
 *  board was not yours to touch. This is the same fact as a badge you cannot
 *  miss, and it also marks the half of the mat that is currently acting, so
 *  the answer is where you are already looking.
 */
function showTurn() {
  const badge = $("turnBadge");
  if (!badge) return;
  let state = "wait", text = "Waiting";
  if (!view || !view.ready) {
    state = "wait"; text = "Setting up";
  } else if (view.finished && !replaying) {
    state = "over"; text = "Game over";
  } else if (replaying) {
    // Mid-replay the board is showing the other side's turn being played out,
    // whatever the view says about whose decision comes next.
    state = "opp"; text = replayingOpponent ? "Opponent's turn" : "Your turn";
  } else if (view.yourTurn) {
    state = "me"; text = "Your turn";
  } else {
    state = "opp";
    text = view.pvp ? `${view.names.opponent}'s turn` : "Opponent's turn";
  }
  badge.textContent = text;
  badge.dataset.turn = state;
  // The mat reads the same attribute, so the acting half lights up with it.
  document.body.dataset.turn = state;
}

function statusLine() {
  if (!view) return "";
  if (!view.ready) return `waiting on ${view.waitingOn}`;
  if (view.finished && !replaying) return `Game over — ${view.verdict}`;
  const who = view.pvp ? view.names.you : `player ${view.seat}`;
  return `Turn ${view.turn} · you are ${who} · ` +
    (view.yourTurn ? "your move" : `waiting on ${view.waitingOn}`);
}

function render() {
  if (watch) return;   // a replay owns the board; the live game redraws on close
  hideTip();  // a card can be replaced while hovered, and then no mouseleave fires
  releaseCard();  // ...and the same is true of a card being dragged
  if (!view || !view.ready) {
    $("status").textContent = statusLine();
  showTurn();
    renderLog(view ? view.log : []);
    return;
  }
  if (view.agentError) console.warn("agent fell back to a random legal move:", view.agentError);
  buildTargets();

  $("agentName").textContent = view.agent ? `· ${view.agent}` : "";
  // Keep the picker on the opponent actually in play (page reloads, /api/state).
  if (view.agent && !view.pvp) chosenAgent = view.agent;
  // Two players: the nameplates carry the players' names instead of You/Opponent.
  document.querySelector("#oppHalf .who").textContent = view.pvp ? view.names.opponent : "Opponent";
  document.querySelector("#meHalf .who").textContent = view.pvp ? `${view.names.you} (you)` : "You";
  if (view.pvp) $("agentName").textContent = "";
  $("status").textContent = statusLine();
  showTurn();

  const events = view.events || [];
  const steps = view.steps || [];
  view.events = [];   // consumed: a re-render must not replay the turn again

  if (events.length && steps.length) {
    // Rewind the board to where it stood before anything in this batch
    // happened — including the human's own action — and let the replay walk it
    // forward. Drawing a later position first and narrating over it is what
    // made the pop-ups describe moves the board had already made: the cards sat
    // in their end-of-turn places from the first beat, and an attack showed its
    // damage and its Knock Out before the attack was ever animated.
    drawBoard(steps[0].board, shownHand);
    renderChoice();
    renderLog(view.log);
    playEvents(events, steps);
  } else {
    drawFinal();
    renderChoice();
    renderLog(view.log);
  }
}

/** Draw the position the current view describes, with the real hand. */
function drawFinal() {
  shownHand = view.me.hand || [];
  drawBoard(view, shownHand);
}

/** Draw one board: either a view or a mid-turn snapshot.
 *
 *  A snapshot carries no hands — the agent's is private and the human's is not
 *  in the observation it was taken from — so the hand on screen is the one the
 *  player last held, trimmed or padded to the count the snapshot reports.
 */
function drawBoard(board, handCards) {
  const oppBench = $("oppBench"); oppBench.innerHTML = "";
  for (let i = 0; i < board.opp.benchMax; i++) {
    oppBench.appendChild(slot("opp", "bench", i, board.opp.bench[i] || null, { empty: "bench" }));
  }
  $("oppActive").innerHTML = "";
  $("oppActive").appendChild(slot("opp", "active", 0, board.opp.active, {
    cls: "active-spot", extra: "active-card", empty: "active", conditions: board.opp.conditions,
  }));
  const oppRow = $("oppActive").parentElement;
  oppRow.querySelector(".prizes")?.replaceWith(stackPrizes("opp", board.opp.prizeCount));
  oppRow.querySelector(".piles")?.replaceWith(stackPiles("Opponent's", board.opp, false));
  $("oppChips").innerHTML = chips(board.opp, false);

  // The opponent's hand: their cards, face-down and counted the way they are
  // across a table — a row of backs, not the number 6.
  const oppHand = $("oppHand");
  oppHand.innerHTML = "";
  oppHand.appendChild(el("span", "label", `Hand ${board.opp.handCount}`));
  if (Array.isArray(board.opp.hand) && board.opp.hand.length) {
    // Only a replay sends the other hand: the game is over, nothing is hidden.
    for (const c of board.opp.hand) oppHand.appendChild(cardEl(c, { extra: "mini-card" }));
  } else {
    for (let i = 0; i < board.opp.handCount; i++) oppHand.appendChild(el("div", "handback"));
  }
  if (!board.opp.handCount) oppHand.appendChild(el("span", "hint", "empty"));

  $("meActive").innerHTML = "";
  $("meActive").appendChild(slot("me", "active", 0, board.me.active, {
    cls: "active-spot", extra: "active-card", empty: "active", conditions: board.me.conditions,
  }));
  const meBench = $("meBench"); meBench.innerHTML = "";
  meBench.dataset.zone = "bench";
  for (let i = 0; i < board.me.benchMax; i++) {
    meBench.appendChild(slot("me", "bench", i, board.me.bench[i] || null, { empty: "bench" }));
  }
  const meRow = $("meActive").parentElement;
  meRow.querySelector(".prizes")?.replaceWith(stackPrizes("me", board.me.prizeCount));
  meRow.querySelector(".piles")?.replaceWith(stackPiles("Your", board.me, true));
  $("meChips").innerHTML = chips(board.me, true);

  const hand = $("meHand"); hand.innerHTML = "";
  const cards = (board.me.hand || handCards || []).slice(0, board.me.handCount);
  $("handCount").textContent = `(${board.me.handCount})`;
  cards.forEach((c, i) => {
    const k = key("me", "hand", i);
    const hits = targets.get(k) || [];
    const node = cardEl(c, { extra: hits.length ? "targetable" : "" });
    if (hits.length) {
      node.dataset.key = k;
      armDrag(node, k);
      node.addEventListener("click", () => hitOptions(hits));
      node.addEventListener("mouseenter", () => markHot(hits, true));
      node.addEventListener("mouseleave", () => markHot(hits, false));
      if (hits.some((x) => picked.includes(x))) node.classList.add("picked");
    }
    hand.appendChild(node);
  });
  for (let i = cards.length; i < board.me.handCount; i++) {
    hand.appendChild(el("div", "card facedown", '<div class="face"><div class="name">Face-down</div></div>'));
  }

  const stadium = $("stadium");
  stadium.innerHTML = "";
  stadium.appendChild(el("span", "label", "Stadium"));
  if (board.stadium) stadium.appendChild(cardEl(board.stadium, { extra: "mini-card" }));
  else stadium.appendChild(el("div", "slot mini-slot", "empty"));
  const actions = board.actions === undefined ? "" : ` · ${board.actions} action${board.actions === 1 ? "" : "s"} this turn`;
  $("turnbar").textContent = `Turn ${board.turn}${actions}`;
  const looking = $("looking");
  looking.innerHTML = "";
  if ((board.looking || []).length) {
    looking.appendChild(el("span", "label", "Revealed"));
    for (const c of board.looking) looking.appendChild(cardEl(c, { extra: "mini-card" }));
  }
}

function renderChoice() {
  const box = $("options");
  box.innerHTML = "";
  $("confirm").hidden = true;
  $("choiceHint").classList.remove("bad");

  if (replaying) {
    // The options exist, but they belong to a board the player has not seen
    // arrive yet. Hold them until the turn has finished playing out.
    // A batch often ends with your own last action, so say whose turn is
    // actually being shown rather than always blaming the opponent.
    const them = view && view.pvp ? view.names.opponent : "Opponent";
    $("prompt").textContent = replayingOpponent ? `${them} is playing…` : "Replaying your move…";
    // No skip control on screen: a button here, or a click anywhere over the
    // board, is too easy to hit by accident — and losing the turn you were
    // waiting to watch is the one thing this must not do. The S key remains as
    // a deliberate way out.
    $("choiceHint").textContent = "watch their turn — press S to cut it short";
    return;
  }
  if (view.finished) {
    $("prompt").textContent = `Game over — ${view.verdict}`;
    $("choiceHint").textContent = "Start a new game above.";
    return;
  }
  if (!view.yourTurn) {
    $("prompt").textContent = view && view.pvp ? `${view.names.opponent} is thinking…` : "Opponent is thinking…";
    $("choiceHint").textContent = "";
    return;
  }

  const sel = view.select;
  $("prompt").textContent = sel.prompt || sel.context;
  const many = sel.maxCount > 1 || sel.minCount === 0;
  const bits = [`${sel.context}`];
  if (sel.effect) bits.push(`from ${sel.effect.name}`);
  if (sel.contextCard) bits.push(`about ${sel.contextCard.name}`);
  bits.push(many ? `choose ${sel.minCount}–${sel.maxCount}, then confirm` : "choose one");
  $("choiceHint").textContent = bits.join(" · ");

  // Grouped under headings — Pokémon, Evolve, Energy, Trainer, Ability, Attack,
  // Misc — because a main-phase list is otherwise thirty rows in the order the
  // rules happened to walk, and a player looking for "can I evolve?" has to
  // read all of it. A prompt whose options all fall in one group (choosing a
  // card from the deck, say) keeps its flat list: a lone heading says nothing.
  const groups = new Map();
  for (const opt of view.options) {
    const name = opt.group || "Misc";
    if (!groups.has(name)) groups.set(name, []);
    groups.get(name).push(opt);
  }
  const order = view.groupOrder || [...groups.keys()];
  const named = [...groups.keys()].sort((a, b) => order.indexOf(a) - order.indexOf(b));

  for (const name of named) {
    if (named.length > 1) box.appendChild(el("h3", "optionGroup", name));
    for (const opt of groups.get(name)) {
      const b = el("button", "option", `<span class="kind">${opt.type}</span>${opt.label}`);
      b.dataset.index = opt.index;
      if (picked.includes(opt.index)) b.classList.add("picked");
      b.addEventListener("click", () => pickOption(opt.index));
      box.appendChild(b);
    }
  }
  if (many) {
    const confirm = $("confirm");
    confirm.hidden = false;
    confirm.textContent = confirmLabel(sel);
    confirm.disabled = picked.length < sel.minCount;
    // Wired here, next to the state it sends, rather than once at startup: the
    // startup wiring was lost in an edit and left every multi-select confirm
    // (Ultra Ball's discard, a bench search) enabled but dead.
    confirm.onclick = () => { if (!busy) submit(picked.slice()); };
  }
}

/** Taking nothing is a legal answer to a search, so say so on the button rather
 *  than showing an enabled "Confirm (0/1)" that looks like it needs a pick. */
function confirmLabel(sel) {
  if (sel.minCount === 0 && picked.length === 0) return "Take nothing";
  return `Confirm (${picked.length}/${sel.maxCount})`;
}

function pickOption(index) {
  if (busy) return;
  const sel = view.select;
  const many = sel.maxCount > 1 || sel.minCount === 0;
  if (!many) { submit([index]); return; }
  const at = picked.indexOf(index);
  if (at >= 0) picked.splice(at, 1);
  else if (picked.length < sel.maxCount) picked.push(index);
  refreshPicks();
}

/** Show what is currently picked, in place.
 *
 *  A full re-render on every click rebuilt the option list, which threw away
 *  the scroll position mid-way down a deck search and replaced the very buttons
 *  being clicked. Only the marks change here: the option rows, the board slots
 *  and hand cards they point at, and Confirm's count.
 */
function refreshPicks() {
  for (const b of document.querySelectorAll("#options .option")) {
    b.classList.toggle("picked", picked.includes(Number(b.dataset.index)));
  }
  for (const node of document.querySelectorAll("[data-key]")) {
    const hits = targets.get(node.dataset.key) || [];
    node.classList.toggle("picked", hits.some((i) => picked.includes(i)));
  }
  const sel = view.select;
  const confirm = $("confirm");
  if (sel && !confirm.hidden) {
    confirm.textContent = confirmLabel(sel);
    confirm.disabled = picked.length < sel.minCount;
  }
}

async function submit(options) {
  if (busy) return;
  busy = true;
  announceOwnAbility(options);
  $("status").textContent = pvp ? "sending…" : "opponent thinking…";
  try {
    view = pvp
      ? await api("/api/room/select", { room: pvp.room, options })
      : await api("/api/select", { options });
  } catch (err) {
    busy = false;
    // The engine refuses some selections that pass the count check — an Energy
    // set that does not pay the cost, say. Say so next to the prompt, keep what
    // was picked, and leave Confirm live so it can be adjusted and retried;
    // reporting this only in the corner made the button look broken.
    showChoiceError(err.message || "the engine refused that selection");
    return;
  }
  picked = [];
  render();
  // A held-over click from the previous prompt lands in this window and is
  // dropped, instead of answering a question the player has not read yet.
  setTimeout(() => { busy = false; }, 400);
}

function showChoiceError(message) {
  const hint = $("choiceHint");
  hint.textContent = message;
  hint.classList.add("bad");
  $("status").textContent = `rejected: ${message}`;
  $("confirm").disabled = false;
}

/** Abilities leave no log entry, so your own use of one is announced here.
 *  (The opponent's is genuinely not in the observation — nothing to show.) */
function announceOwnAbility(options) {
  for (const i of options) {
    const opt = view.options.find((o) => o.index === i);
    if (opt && opt.type === "Ability") {
      banner({ side: "me", card: null }, "ability", opt.label, "ability",
             Math.round(BEATS.play * speedFactor()));
    }
  }
}

function renderLog(lines) {
  const node = $("log");
  node.innerHTML = "";
  for (const line of (lines || []).slice().reverse()) {
    const d = el("div", line.startsWith("---") ? "turn" : line.startsWith("===") ? "result" : null);
    d.textContent = line;
    node.appendChild(d);
  }
}

// -- two players --------------------------------------------------------------
//
// A room is an address, /?room=<id>, shared by both players. Who you are in it is
// the token the server issued when you gave your name, kept per tab in
// sessionStorage: a reload returns you to the same seat, and two tabs of one
// browser can be the two players. The lobby and the game are both polled, since
// the other player's actions reach this page only by asking.

let pvp = null;              // {room, token} while this tab is in a room
let joiningRoom = null;      // the room the name step will join (null: create one)
let lobbyState = null;
let lobbyTimer = null;
let pvpTimer = null;
let buildingForRoom = false; // the deck dialog is open on behalf of the lobby
let lastSignature = "";
const CUSTOM_DECK = "__custom__";

const roomKey = (room) => `ptcg.room.${room}`;
function roomToken(room) { try { return sessionStorage.getItem(roomKey(room)); } catch (e) { return null; } }
function saveRoomToken(room, token) { try { sessionStorage.setItem(roomKey(room), token); } catch (e) { /* private window */ } }
function roomFromUrl() { return new URLSearchParams(location.search).get("room"); }
// Where the other player can reach this server. The page's own origin is wrong
// when the host opened it on localhost and shares it through a tunnel, so the
// server is asked first (PTCG_PUBLIC_URL, or the running ngrok agent's address).
let publicBase = null;
async function loadPublicBase() {
  try { publicBase = (await api("/api/public-url")).url; } catch (e) { publicBase = null; }
}
function inviteUrl(room) {
  const base = publicBase || `${location.origin}${location.pathname.replace(/\/$/, "")}`;
  return `${base}/?room=${encodeURIComponent(room)}`;
}

function openMode() {
  exitWatch();   // starting anything new leaves the replay
  if (view) render();
  $("modeView").hidden = false;
}

function openPvpName(room = null, host = "") {
  joiningRoom = room;
  $("modeView").hidden = true;
  $("pvpView").hidden = false;
  $("nameStep").hidden = false;
  $("lobbyStep").hidden = true;
  $("nameTitle").textContent = room ? `Join ${host}'s room` : "2 players";
  $("nameHint").textContent = room
    ? `${host} invited you to a game. Enter your name to join their lobby.`
    : "Enter your name. You'll get an invitation link to send to the other player.";
  $("nameGo").textContent = room ? "Join lobby" : "Create invitation link";
  $("nameError").textContent = "";
  let saved = "";
  try { saved = localStorage.getItem("ptcg.name") || ""; } catch (e) { /* ignore */ }
  $("playerName").value = saved;
  $("playerName").focus();
}

async function submitName() {
  const name = $("playerName").value.trim();
  if (!name) { $("nameError").textContent = "Please enter a name."; return; }
  try { localStorage.setItem("ptcg.name", name); } catch (e) { /* ignore */ }
  $("nameGo").disabled = true;
  let res;
  try {
    res = joiningRoom
      ? await api("/api/room/join", { room: joiningRoom, name })
      : await api("/api/room/create", { name });
  } catch (err) {
    $("nameError").textContent = err.message;
    $("nameGo").disabled = false;
    return;
  }
  $("nameGo").disabled = false;
  saveRoomToken(res.room, res.token);
  pvp = { room: res.room, token: res.token };
  // Both players end up on the same address: the lobby is the room's own URL.
  history.replaceState(null, "", `?room=${encodeURIComponent(res.room)}`);
  showLobby(res.lobby);
}

function fillLobbyDecks(keep) {
  const sel = $("lobbyDeck");
  const current = keep ?? sel.value;
  sel.innerHTML = "";
  for (const deck of prebuilt) sel.appendChild(el("option", null, deck.name)).value = deck.name;
  if (customDeck.length) {
    sel.appendChild(el("option", null, `My deck (${customDeck.length} cards, from the deck builder)`)).value = CUSTOM_DECK;
  }
  if ([...sel.options].some((o) => o.value === current)) sel.value = current;
}

function showLobby(lobby) {
  lobbyState = lobby;
  if (lobby.status === "playing") { enterRoomGame(); return; }
  $("pvpView").hidden = false;
  $("nameStep").hidden = true;
  $("lobbyStep").hidden = false;
  $("inviteLink").value = inviteUrl(lobby.room);
  if (!publicBase) loadPublicBase().then(() => { $("inviteLink").value = inviteUrl(lobby.room); });

  const list = $("lobbyPlayers");
  list.innerHTML = "";
  for (let i = 0; i < 2; i++) {
    const p = lobby.players[i];
    if (!p) { list.appendChild(el("li", "empty", "<span>Waiting for the other player…</span>")); continue; }
    const li = el("li");
    const who = el("span");
    who.textContent = `${p.name}${i === lobby.you ? " (you)" : ""}${p.host ? " · host" : ""}`;
    const state = el("span", `state ${p.ready ? "ready" : ""}`);
    state.textContent = p.ready ? `ready · ${p.deck}` : "choosing a deck";
    li.append(who, state);
    list.appendChild(li);
  }
  const me = lobby.players[lobby.you];
  $("lobbyWait").textContent = !lobby.full
    ? "You're in the lobby. Send the invitation link — the other player joins this same address."
    : "Both players are here. Pick a deck and press Ready; the game starts when you're both ready.";
  if (!$("lobbyDeck").options.length) fillLobbyDecks();
  $("lobbyDeck").disabled = me.ready;
  $("lobbyBuild").disabled = me.ready;
  $("lobbyReady").disabled = false;
  $("lobbyReady").textContent = me.ready ? "Not ready" : "Ready";
  $("lobbyReady").classList.toggle("on", me.ready);
  $("lobbyError").textContent = lobby.error || "";
  if (!lobbyTimer) lobbyTimer = setInterval(pollLobby, 1500);
}

// One lobby request in flight at a time, each held open by the server until the
// lobby changes: a long poll, so waiting costs a request per change, not per second.
let lobbyInFlight = false;
async function pollLobby() {
  if (!pvp || $("lobbyStep").hidden || buildingForRoom || lobbyInFlight) return;
  lobbyInFlight = true;
  const room = pvp.room;
  const since = lobbyState && lobbyState.tag ? `&since=${lobbyState.tag}` : "";
  try {
    const next = await api(`/api/room/lobby?room=${encodeURIComponent(room)}${since}`);
    if (pvp && pvp.room === room && !buildingForRoom) showLobby(next);
  } catch (err) {
    if (pvp && pvp.room === room) roomClosed(err.message);
  } finally {
    lobbyInFlight = false;
  }
}

async function toggleReady() {
  const me = lobbyState.players[lobbyState.you];
  const body = { room: pvp.room, ready: !me.ready };
  if (!me.ready) {
    const choice = $("lobbyDeck").value;
    if (choice === CUSTOM_DECK) body.cards = customDeck;
    else body.deck = choice;
  }
  $("lobbyReady").disabled = true;
  try {
    showLobby(await api("/api/room/ready", body));
  } catch (err) {
    $("lobbyError").textContent = err.message;
    $("lobbyReady").disabled = false;
  }
}

function buildForRoom() {
  buildingForRoom = true;
  $("pvpView").hidden = true;
  $("startGame").textContent = "Use this deck";
  openSetup();
}

/** The deck dialog was opened from the lobby: take its deck back there. */
function useDeckForRoom() {
  if (chosenDeck === null && customDeck.length !== 60) {
    setStatus(`a deck needs 60 cards (${customDeck.length} so far)`, "bad");
    return;
  }
  closeBuilderForRoom();
  fillLobbyDecks(chosenDeck === null ? CUSTOM_DECK : chosenDeck);
}

function closeBuilderForRoom() {
  buildingForRoom = false;
  $("startGame").textContent = "Start game";
  $("setupView").hidden = true;
  $("pvpView").hidden = false;
  if (lobbyState) showLobby(lobbyState);
}

function enterRoomGame() {
  clearInterval(lobbyTimer); lobbyTimer = null;
  $("pvpView").hidden = true;
  $("modeView").hidden = true;
  $("setupView").hidden = true;
  const names = (lobbyState && lobbyState.players.map((p) => p.name)) || [];
  $("loadout").textContent = names.length === 2 ? `${names[0]} vs ${names[1]}` : "2 players";
  lastSignature = "";
  if (!pvpTimer) pvpTimer = setInterval(pollRoomGame, 1200);
  pollRoomGame(true);
}

/** What the board looks like, so an unchanged poll does not redraw the page. */
function signature(v) {
  return JSON.stringify([v.turn, v.actions, v.yourTurn, v.finished, v.me, v.opp, v.stadium, v.options.length]);
}

let gameInFlight = false;
async function pollRoomGame(force = false) {
  if (!pvp || busy || replaying || gameInFlight || watch) return;
  if (!force && view && view.pvp && (view.yourTurn || view.finished)) return;
  // While the other player is deciding, ask the server to hold the request until
  // the game moves past the version on screen (a long poll).
  const since = !force && view && view.pvp ? `&since=${view.version}` : "";
  const room = pvp.room;
  let next;
  gameInFlight = true;
  try {
    next = await api(`/api/room/state?room=${encodeURIComponent(room)}${since}`);
  } catch (err) {
    if (pvp && pvp.room === room) roomClosed(err.message);
    return;
  } finally {
    gameInFlight = false;
  }
  if (!pvp || pvp.room !== room || busy) return;
  const sig = signature(next);
  if (!force && sig === lastSignature && !(next.events || []).length) return;
  lastSignature = sig;
  view = next;
  picked = [];
  render();
}

function leaveRoomLocally() {
  clearInterval(lobbyTimer); lobbyTimer = null;
  clearInterval(pvpTimer); pvpTimer = null;
  if (pvp) { try { sessionStorage.removeItem(roomKey(pvp.room)); } catch (e) { /* ignore */ } }
  pvp = null;
  lobbyState = null;
  $("lobbyDeck").innerHTML = "";
  history.replaceState(null, "", location.pathname);
}

async function leaveRoom() {
  if (pvp) {
    try { await api("/api/room/leave", { room: pvp.room }); } catch (e) { /* already gone */ }
  }
  leaveRoomLocally();
  $("pvpView").hidden = true;
  view = null;
  $("prompt").textContent = "No game in progress";
  $("options").innerHTML = "";
  $("status").textContent = "";
  $("loadout").textContent = "";
  openMode();
}

function roomClosed(message) {
  leaveRoomLocally();
  $("pvpView").hidden = true;
  $("status").textContent = `room closed: ${message}`;
  openMode();
}

/** On load: a room link either returns you to your seat or asks your name. */
async function resumeRoom(room) {
  const token = roomToken(room);
  if (token) {
    pvp = { room, token };
    try {
      showLobby(await api(`/api/room/lobby?room=${encodeURIComponent(room)}`));
      return;
    } catch (err) {
      pvp = null;   // a stale token: fall through and ask for a name again
    }
  }
  let info;
  try {
    info = await api(`/api/room/info?room=${encodeURIComponent(room)}`);
  } catch (err) {
    history.replaceState(null, "", location.pathname);
    $("status").textContent = `that room is closed: ${err.message}`;
    openMode();
    return;
  }
  if (info.full) {
    history.replaceState(null, "", location.pathname);
    $("status").textContent = `${info.host}'s room already has two players`;
    openMode();
    return;
  }
  openPvpName(room, info.host);
}

// -- watch replays -------------------------------------------------------------
//
// A saved game (replays/, written by replay_log.py) comes back as one board per
// decision, seen from the seat chosen under "View as", with both hands face up.
// Left/Right step one decision; Play walks forward on its own, at the replay
// speed picked in the header, animating each move the way a live turn is.

// The raw JSON is only stringified while its panel is open; opening it fills it in.
document.addEventListener("DOMContentLoaded", () => {
  $("rpRaw").addEventListener("toggle", () => { if (watch && $("rpRaw").open) updateReplayPanel(); });
});

let watch = null;   // {file, seat, data, i, playing, lines} while a replay is shown
let watchToken = 0; // bumped to cancel an animation or an autoplay in progress

async function openReplayList() {
  $("modeView").hidden = true;
  $("replayListView").hidden = false;
  const box = $("replayList");
  box.innerHTML = "";
  $("replayListHint").textContent = "loading…";
  let list;
  try {
    list = (await api("/api/replays")).replays;
  } catch (err) {
    $("replayListHint").textContent = `could not list replays: ${err.message}`;
    return;
  }
  $("replayListHint").textContent = list.length
    ? `${list.length} saved game${list.length === 1 ? "" : "s"} — click one to watch it`
    : "No saved games yet. Every game you finish (or leave) here is saved to replays/.";
  for (const r of list) {
    const row = el("button", "replayRow");
    const when = new Date(r.started * 1000);
    const who = r.names.map((n) => (n === r.winner ? `<span class="won">${esc(n)} ★</span>` : esc(n))).join(" vs ");
    const mode = { pvp: "2 players", vs_agent: "vs AI", kaggle: "Kaggle" }[r.mode] || r.mode;
    row.innerHTML = `<span class="when">${when.toLocaleString()}</span><span class="who">${who}</span>` +
      `<span class="meta">${mode}${r.finished ? "" : " · unfinished"}</span>`;
    row.addEventListener("click", () => startWatching(r.file, 0));
    box.appendChild(row);
  }
}

function esc(text) {
  return String(text).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
}

async function startWatching(file, seat) {
  stopAutoplay();
  $("replayListView").hidden = true;
  $("status").textContent = "loading replay…";
  let data;
  try {
    data = await api(`/api/replays/view?file=${encodeURIComponent(file)}&seat=${seat}`);
  } catch (err) {
    $("status").textContent = `could not open replay: ${err.message}`;
    return;
  }
  // Keep the position when only the point of view changed.
  const keep = watch && watch.file === file ? watch.i : 0;
  // The log is cumulative: where each frame's lines end in the whole game's log.
  const lines = [];
  const ends = data.frames.map((f) => { lines.push(...f.lines); return lines.length; });
  watch = { file, seat, data, lines, ends, i: Math.min(keep, data.frames.length - 1), playing: false };
  $("choice").hidden = true;
  $("replayPanel").hidden = false;
  $("rpTitle").textContent = `Replay · ${data.names[0]} vs ${data.names[1]}`;
  const sel = $("rpSeat");
  sel.innerHTML = "";
  data.names.forEach((n, i) => { sel.appendChild(el("option", null, esc(n))).value = String(i); });
  sel.value = String(seat);
  $("rpSeek").max = String(Math.max(0, data.frames.length - 1));
  $("loadout").textContent = `replay: ${data.names[0]} vs ${data.names[1]}`;
  showFrame(watch.i, false);
}

/** Draw decision ``i``; with ``animate``, play its events first, as a live turn does. */
async function showFrame(i, animate) {
  if (!watch) return;
  const token = ++watchToken;
  const frames = watch.data.frames;
  watch.i = Math.max(0, Math.min(i, frames.length - 1));
  const frame = frames[watch.i];
  const names = watch.data.names;
  document.querySelector("#oppHalf .who").textContent = names[1 - watch.seat];
  document.querySelector("#meHalf .who").textContent = names[watch.seat];
  $("agentName").textContent = "";
  targets = new Map();
  picked = [];
  updateReplayPanel();
  renderLog(watch.lines.slice(0, watch.ends[watch.i]).slice(-400));
  if (animate && frame.events.length) {
    const scale = speedFactor();
    for (const ev of mergeEvents(frame.events)) {
      if (token !== watchToken || !watch) return;
      await playEvent(ev, scale);
    }
    if (token !== watchToken || !watch) return;
  }
  $("stage").innerHTML = "";
  drawBoard(frame.board, frame.board.me.hand);
}

function updateReplayPanel() {
  const frames = watch.data.frames;
  const frame = frames[watch.i];
  const n = frames.length;
  $("rpSeek").value = String(watch.i);
  $("rpPos").textContent = `Decision ${watch.i + 1} / ${n} · turn ${frame.board.turn}`;
  const who = frame.mover ? `<b>${esc(frame.mover)}</b>: ` : "";
  let verdict = "";
  if (frame.result !== -1) {
    verdict = frame.result === 2 ? "<br>Game over — draw"
      : `<br>Game over — <b>${esc(watch.data.names[frame.result])}</b> wins`;
  }
  $("rpDecision").innerHTML = who + esc(frame.decision) + verdict;
  if ($("rpRaw").open) $("rpRawText").textContent = JSON.stringify(frame.raw, null, 2);
  $("rpPrev").disabled = watch.i === 0;
  $("rpNext").disabled = watch.i >= n - 1;
  $("rpPlay").disabled = watch.playing || watch.i >= n - 1;
  $("rpPause").disabled = !watch.playing;
  $("status").textContent = watch.playing ? "replay playing" : "replay paused";
}

function stepReplay(delta) {
  if (!watch) return;
  stopAutoplay();
  const next = watch.i + delta;
  if (next < 0 || next >= watch.data.frames.length) return;
  // Forward animates the move; backward just shows the earlier position.
  showFrame(next, delta === 1);
}

async function playReplay() {
  if (!watch || watch.playing) return;
  watch.playing = true;
  updateReplayPanel();
  while (watch && watch.playing && watch.i < watch.data.frames.length - 1) {
    await showFrame(watch.i + 1, true);
    if (!watch || !watch.playing) break;
    await pause(Math.round(900 * speedFactor()));
  }
  if (watch) { watch.playing = false; updateReplayPanel(); }
}

function stopAutoplay() {
  if (watch && watch.playing) {
    watch.playing = false;
    watchToken++;          // cut the move being animated short
    $("stage").innerHTML = "";
    showFrame(watch.i, false);
  }
}

function pauseReplay() { stopAutoplay(); if (watch) updateReplayPanel(); }

/** Leave the replay and give the board back to the live game, if there is one. */
function exitWatch() {
  if (!watch) return;
  stopAutoplay();
  watch = null;
  watchToken++;
  $("stage").innerHTML = "";
  $("replayPanel").hidden = true;
  $("choice").hidden = false;
  $("loadout").textContent = "";
}

function closeReplay() {
  exitWatch();
  if (view) { render(); return; }
  $("prompt").textContent = "No game in progress";
  $("status").textContent = "";
  renderLog([]);
  openMode();
}

// -- wiring -----------------------------------------------------------------

$("speed").value = localStorage.getItem("animSpeed") || "normal";
$("speed").addEventListener("change", (ev) => localStorage.setItem("animSpeed", ev.target.value));
$("newGame").addEventListener("click", openMode);
$("modeAgent").addEventListener("click", () => { $("modeView").hidden = true; openSetup(); });
$("modeFriend").addEventListener("click", () => openPvpName(null));
$("modeReplays").addEventListener("click", openReplayList);
$("replayListClose").addEventListener("click", () => { $("replayListView").hidden = true; if (!watch && !view) openMode(); });
$("rpPrev").addEventListener("click", () => stepReplay(-1));
$("rpNext").addEventListener("click", () => stepReplay(1));
$("rpPlay").addEventListener("click", playReplay);
$("rpPause").addEventListener("click", pauseReplay);
$("rpClose").addEventListener("click", closeReplay);
$("rpSeat").addEventListener("change", (ev) => { if (watch) startWatching(watch.file, Number(ev.target.value)); });
$("rpSeek").addEventListener("input", (ev) => { if (watch) { stopAutoplay(); showFrame(Number(ev.target.value), false); } });
$("modeClose").addEventListener("click", () => { $("modeView").hidden = true; });
$("nameGo").addEventListener("click", submitName);
$("playerName").addEventListener("keydown", (ev) => { if (ev.key === "Enter") submitName(); });
$("nameCancel").addEventListener("click", () => {
  $("pvpView").hidden = true;
  if (joiningRoom) history.replaceState(null, "", location.pathname);
  openMode();
});
$("copyInvite").addEventListener("click", () => {
  const link = $("inviteLink");
  link.select();
  navigator.clipboard?.writeText(link.value);
  $("copyInvite").textContent = "Copied";
  setTimeout(() => { $("copyInvite").textContent = "Copy"; }, 1500);
});
$("lobbyReady").addEventListener("click", toggleReady);
$("lobbyBuild").addEventListener("click", buildForRoom);
$("lobbyLeave").addEventListener("click", leaveRoom);
$("setupClose").addEventListener("click", () => {
  if (buildingForRoom) { closeBuilderForRoom(); return; }
  $("setupView").hidden = true;
});
$("startGame").addEventListener("click", startGame);
$("search").addEventListener("input", runSearch);
$("kindFilter").addEventListener("change", runSearch);
$("addToDeck").addEventListener("click", () => { if (selectedCard !== null) addCard(selectedCard, 1); });
$("addFour").addEventListener("click", () => { if (selectedCard !== null) addCard(selectedCard, 4); });
$("clearDeck").addEventListener("click", () => { customDeck = []; refreshCustom(); });
$("startFrom").addEventListener("change", (ev) => {
  // A prebuilt list as a starting point: the fastest way to build a deck is to
  // edit one that is already legal.
  const deck = prebuilt.find((d) => d.name === ev.target.value);
  if (!deck) return;
  customDeck = [];
  for (const row of deck.cards) for (let i = 0; i < row.count; i++) customDeck.push(row.id);
  ev.target.selectedIndex = 0;
  refreshCustom();
});
$("exportDeck").addEventListener("click", () => {
  // deck.csv is one card id per line — the same file the bundles ship.
  const text = customDeck.join("\n");
  navigator.clipboard?.writeText(text);
  setStatus(`${customDeck.length} card ids copied to the clipboard`, "good");
});
$("importDeck").addEventListener("click", () => {
  const text = prompt("Paste a deck.csv — one card id per line");
  if (!text) return;
  const ids = text.split(/[\s,]+/).map((t) => parseInt(t, 10)).filter((n) => Number.isInteger(n));
  customDeck = ids.slice(0, 60);
  refreshCustom();
});

document.addEventListener("keydown", (ev) => {
  // Replay keys: Left/Right step a decision, Space plays or pauses.
  if (watch && !ev.metaKey && !ev.ctrlKey && !ev.altKey && !/INPUT|SELECT|TEXTAREA/.test(ev.target.tagName)) {
    if (ev.key === "ArrowLeft") { ev.preventDefault(); stepReplay(-1); return; }
    if (ev.key === "ArrowRight") { ev.preventDefault(); stepReplay(1); return; }
    if (ev.key === " ") { ev.preventDefault(); if (watch.playing) pauseReplay(); else playReplay(); return; }
  }
  if ((ev.key === "s" || ev.key === "S") && replaying && !ev.metaKey && !ev.ctrlKey) skipReplay();
});
$("pileClose").addEventListener("click", () => { $("pileView").hidden = true; });
$("pileView").addEventListener("click", (ev) => { if (ev.target.id === "pileView") $("pileView").hidden = true; });

// Say which build this is. If this number does not change after a restart, the
// page is running from cache and nothing you were told was fixed is loaded.
api("/api/version").then((v) => {
  console.info(`simulator build ${v.build}`);
  const tag = $("build");
  if (tag) tag.textContent = `build ${v.build}`;
}).catch(() => {});

loadConfig().then(async () => {
  await loadPublicBase();
  const room = roomFromUrl();
  if (room) { await resumeRoom(room); return; }
  try {
    view = await api("/api/state");
    render();
  } catch {
    openMode();  // nothing in progress: start by choosing who to play
  }
});
