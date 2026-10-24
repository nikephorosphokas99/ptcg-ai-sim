"""Turn an engine observation into fixed-shape arrays and an action mask.

Everything is from the deciding player's point of view (``current.yourIndex``),
so the same encoder serves the learner and a frozen copy of it playing the
other seat.

Actions are ``Discrete(MAX_OPTIONS + 1)``: index ``i`` picks option ``i`` and
``STOP`` ends a multi-select. The engine asks for between ``minCount`` and
``maxCount`` distinct options at once; ``Decision`` breaks that into one pick per
step so the policy only ever chooses a single index.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .engine import N_ENERGY_TYPES

if TYPE_CHECKING:  # gymnasium is imported where needed, so the submission bundle runs without it
    from gymnasium import spaces

MAX_OPTIONS = 64
STOP = MAX_OPTIONS
N_ACTIONS = MAX_OPTIONS + 1

BENCH_SLOTS = 8
SLOTS = 2 * (1 + BENCH_SLOTS)  # active + bench, mine then opponent's
HAND_SLOTS = 40
PILE_SLOTS = 60
MISC_IDS = 3  # stadium, select.contextCard, select.effect
DECK_SLOTS = 60  # my_unseen_ids / opp_seen_ids
ATTACH_SLOTS = 8  # tools then energy cards attached to one Pokémon
RECENT_SLOTS = 16  # opponent cards from the latest log events
TACTICAL_FEATURES = 7  # per attack option, see _tactical

# AreaType
DECK, HAND, DISCARD, ACTIVE, BENCH, PRIZE, STADIUM = 1, 2, 3, 4, 5, 6, 7
LOOKING = 12
N_AREAS = 13
# OptionType
PLAY, ATTACH, EVOLVE, ABILITY, DISCARD_OPT = 7, 8, 9, 10, 11
TOOL_CARD, ENERGY_CARD, ENERGY = 4, 5, 6
RETREAT = 12
ATTACK, END, SKILL = 13, 14, 15
# SelectContext
MAIN_CONTEXT, DAMAGE_COUNTER, DAMAGE_COUNTER_ANY = 0, 13, 14
N_OPTION_TYPES = 17
N_SELECT_TYPES = 11
N_CONTEXTS = 49
N_CONDITIONS = 5

SLOT_FEATURES = 9 + N_ENERGY_TYPES
OPTION_FEATURES = N_OPTION_TYPES + N_AREAS * 2 + 3 + 6 + N_CONDITIONS + 1
GLOBAL_FEATURES = 9 + 2 * 13 + N_SELECT_TYPES + N_CONTEXTS + 7

_CONDITIONS = ("poisoned", "burned", "asleep", "paralyzed", "confused")


def observation_space(max_card_id: int) -> spaces.Dict:
    from gymnasium import spaces

    ids = lambda n: spaces.Box(0, max_card_id, shape=(n,), dtype=np.int32)  # noqa: E731
    feats = lambda *s: spaces.Box(-np.inf, np.inf, shape=s, dtype=np.float32)  # noqa: E731
    return spaces.Dict({
        "global": feats(GLOBAL_FEATURES),
        "slot_ids": ids(SLOTS),
        "slot_feats": feats(SLOTS, SLOT_FEATURES),
        "hand_ids": ids(HAND_SLOTS),
        "my_discard_ids": ids(PILE_SLOTS),
        "opp_discard_ids": ids(PILE_SLOTS),
        "view_ids": ids(PILE_SLOTS),
        "misc_ids": ids(MISC_IDS),
        "opt_ids": ids(MAX_OPTIONS),
        "opt_target_ids": ids(MAX_OPTIONS),
        "opt_feats": feats(MAX_OPTIONS, OPTION_FEATURES),
        # Memory, added for the pointer policy; older policies read only the keys above.
        "my_unseen_ids": ids(DECK_SLOTS),  # my decklist minus every card of mine in view
        "opp_seen_ids": ids(DECK_SLOTS),   # distinct opponent cards seen face-up this game
        # Added for the entity policy (v3); older policies ignore them.
        "slot_attach_ids": spaces.Box(0, max_card_id, shape=(SLOTS, ATTACH_SLOTS), dtype=np.int32),
        "opt_src_slot": spaces.Box(-1, SLOTS - 1, shape=(MAX_OPTIONS,), dtype=np.int32),  # board slot acting
        "opt_tgt_slot": spaces.Box(-1, SLOTS - 1, shape=(MAX_OPTIONS,), dtype=np.int32),  # board slot targeted
        "opt_tactical": feats(MAX_OPTIONS, TACTICAL_FEATURES),
        "opp_recent_ids": ids(RECENT_SLOTS),
    })


def recent_opponent_cards(obs: dict, me: int) -> list[int]:
    """Opponent card ids named in this observation's log events, oldest first.

    The caller keeps the last ``RECENT_SLOTS`` across observations: the engine
    hands each seat the events since that seat's previous observation.
    """
    out = []
    for log in obs.get("logs") or []:
        if log.get("playerIndex") == 1 - me:
            out += [log[k] for k in ("cardId", "cardIdAfter") if log.get(k)]
    return out


def visible_cards(state: dict, player: int) -> list[int]:
    """Ids of every face-up card ``player`` owns: hand, discard, prizes, in play."""
    ps = state["players"][player]
    out = [_cid(c) for c in (ps["hand"] or []) + ps["discard"] + ps["prize"] if c]
    for mon in ps["active"] + ps["bench"]:
        if mon:
            out.append(mon["id"])
            out += [_cid(c) for c in mon["energyCards"] + mon["tools"] + mon["preEvolution"] if c]
    out += [c["id"] for c in state["stadium"] if c and c.get("playerIndex") == player]
    return out


def unseen_own(deck: list[int], state: dict, player: int) -> list[int]:
    """Cards of ``deck`` not in view: what the deck and face-down prizes hold."""
    from collections import Counter

    return sorted((Counter(deck) - Counter(visible_cards(state, player))).elements())


class Decision:
    """One engine selection, resolved one pick at a time.

    With ``safe_attacks`` (attack ids without a drawback, ``engine.safe_attack_ids``)
    a main-phase choice that offers one of them cannot end the turn instead: the
    v3 policy at 216M steps passed on 7% of the turns it could attack, often with
    a knockout or Phantom Dive's six free damage counters on the table.
    """

    def __init__(self, obs: dict, safe_attacks: frozenset[int] | None = None):
        sel = obs["select"]
        self.obs = obs
        self.n = min(len(sel["option"]), MAX_OPTIONS)
        self.max = min(sel["maxCount"], self.n)
        self.min = min(sel["minCount"], self.max)
        self.picked: list[int] = []
        self.blocked: list[int] = []
        if safe_attacks and sel["context"] == MAIN_CONTEXT:
            options = sel["option"][: self.n]
            if any(o["type"] == ATTACK and o.get("attackId") in safe_attacks for o in options):
                self.blocked = [i for i, o in enumerate(options) if o["type"] == END]

    def mask(self) -> np.ndarray:
        m = np.zeros(N_ACTIONS, dtype=bool)
        m[: self.n] = True
        m[self.picked] = False
        m[self.blocked] = False
        m[STOP] = len(self.picked) >= self.min
        return m

    @property
    def complete(self) -> bool:
        return len(self.picked) >= self.max or (
            len(self.picked) >= self.min and len(self.picked) == self.n
        )

    def apply(self, action: int) -> bool:
        """Record a pick; True once the selection is ready to submit."""
        if action == STOP:
            if len(self.picked) < self.min:
                raise ValueError("STOP before minCount picks")
            return True
        if not (0 <= action < self.n) or action in self.picked:
            raise ValueError(f"illegal pick {action}")
        self.picked.append(action)
        return self.complete


class CounterPlanner:
    """Places damage counters on the opponent's Pokémon for the most Prize cards.

    "Put 6 damage counters on your opponent's Benched Pokémon in any way you like"
    (Phantom Dive) reaches us as six one-counter picks. The observation shows how
    many counters are left, not where the earlier ones went (and the engine does
    not always lower HP between picks), so a policy cannot split them: at 216M
    steps 93% of Phantom Dives put all six on one Pokémon, leaving ~0.46 Prize
    cards a game on the table. ``DAMAGE_COUNTER_ANY`` picks are therefore planned
    in full at the first pick (the most Prize cards from knockouts, then the
    fewest counters; the rest go on the most valuable Pokémon left) and played
    out by card serial. A ``DAMAGE_COUNTER`` pick, where one Pokémon takes every
    counter (Cursed Blast, Adrena-Brain), takes a knockout when one is on offer.
    Anything else, and any pick that could land on our own Pokémon, stays with
    the policy (``pick`` returns None).

    One planner per seat; ``reset`` it between games.
    """

    def __init__(self, card_meta: np.ndarray | None = None):
        self.card_meta = card_meta
        self.reset()

    def reset(self) -> None:
        self.plan: list[int] = []  # target serial for each counter still to place
        self.turn = -1

    def pick(self, obs: dict) -> int | None:
        sel, state = obs.get("select"), obs.get("current")
        if (not sel or not state or sel["context"] not in (DAMAGE_COUNTER, DAMAGE_COUNTER_ANY)
                or sel["minCount"] != 1 or sel["maxCount"] != 1):
            return None
        counters = sel.get("remainDamageCounter") or 0
        targets = self._targets(state, sel)
        if counters <= 0 or not targets:
            return None
        if sel["context"] == DAMAGE_COUNTER:
            kos = [t for t in targets if 0 < t[2] <= 10 * counters]
            return max(kos, key=lambda t: (t[3], t[2]))[0] if kos else None
        serials = {t[1] for t in targets}
        if not (self.turn == state["turn"] and len(self.plan) == counters and set(self.plan) <= serials):
            self.plan, self.turn = self._allocate(targets, counters), state["turn"]
        serial = self.plan.pop(0)
        return next(t[0] for t in targets if t[1] == serial)

    def _targets(self, state: dict, sel: dict) -> list[tuple[int, int, int, int]] | None:
        """(option index, serial, HP, Prize cards) per option; None unless all are the opponent's Pokémon."""
        opp = 1 - state["yourIndex"]
        out = []
        for i, opt in enumerate(sel["option"][:MAX_OPTIONS]):
            if opt.get("playerIndex") != opp or opt.get("area") not in (ACTIVE, BENCH):
                return None
            mon = _at(_area_list(state, sel, opp, opt["area"]), opt.get("index"))
            if mon is None:
                return None
            prizes = int(self.card_meta[mon["id"], 3]) if self.card_meta is not None else 1
            out.append((i, mon["serial"], mon["hp"], max(prizes, 1)))
        return out

    @staticmethod
    def _allocate(targets: list[tuple[int, int, int, int]], counters: int) -> list[int]:
        alive = [t for t in targets if t[2] > 0][:12]
        cost = [-(-t[2] // 10) for t in alive]
        best, best_key = (), (0, 0)
        for subset in range(1, 1 << len(alive)):
            chosen = [j for j in range(len(alive)) if subset >> j & 1]
            need = sum(cost[j] for j in chosen)
            if need <= counters:
                key = (sum(alive[j][3] for j in chosen), -need)
                if key > best_key:
                    best, best_key = tuple(chosen), key
        plan = [alive[j][1] for j in best for _ in range(cost[j])]
        rest = [alive[j] for j in range(len(alive)) if j not in best] or alive or targets
        # Leftovers: the most Prize cards, then the closest to a knockout next turn.
        sink = max(rest, key=lambda t: (t[3], -t[2]))
        return plan + [sink[1]] * (counters - len(plan))


def _one_hot(vec: np.ndarray, offset: int, value: int | None, size: int) -> int:
    if value is not None and 0 <= value < size:
        vec[offset + value] = 1.0
    return offset + size


def _cid(card: dict | None) -> int:
    return card["id"] if card else 0


def _area_list(state: dict, sel: dict, player: int, area: int | None) -> list:
    ps = state["players"][player]
    if area == ACTIVE:
        return ps["active"]
    if area == BENCH:
        return ps["bench"]
    if area == HAND:
        return ps["hand"] or []
    if area == DISCARD:
        return ps["discard"]
    if area == PRIZE:
        return ps["prize"]
    if area == STADIUM:
        return state["stadium"]
    if area == LOOKING:
        return state["looking"] or []
    if area == DECK:
        return sel.get("deck") or []
    return []


def _at(lst: list, index: int | None):
    return lst[index] if index is not None and 0 <= index < len(lst) else None


def _resolve(state: dict, sel: dict, opt: dict, me: int) -> tuple[int, int]:
    """(card the option refers to, Pokémon it targets) as card ids, 0 if none.

    Options name cards by position ("hand slot 5"), not by id, so without this
    join the policy would be choosing between options blind to which card each
    one is.
    """
    t = opt["type"]
    owner = opt.get("playerIndex", me)
    area, index = opt.get("area"), opt.get("index")
    if t == PLAY:
        return _cid(_at(_area_list(state, sel, me, HAND), index)), 0
    if t in (ATTACH, EVOLVE):
        card = _at(_area_list(state, sel, me, area), index)
        target = _at(_area_list(state, sel, me, opt.get("inPlayArea")), opt.get("inPlayIndex"))
        return _cid(card), _cid(target)
    if t in (TOOL_CARD, ENERGY_CARD, ENERGY):
        mon = _at(_area_list(state, sel, owner, area), index)
        if mon is None:
            return 0, 0
        attached = mon["tools"] if t == TOOL_CARD else mon["energyCards"]
        sub = opt.get("toolIndex") if t == TOOL_CARD else opt.get("energyIndex")
        return _cid(_at(attached, sub)), _cid(mon)
    if t == ATTACK:
        return 0, _cid(_at(state["players"][me]["active"], 0))
    if t == SKILL:
        return opt.get("cardId") or 0, 0
    if area is not None:
        return _cid(_at(_area_list(state, sel, owner, area), index)), 0
    return 0, 0


def _ids(cards: list | None, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.int32)
    if cards:
        vals = [_cid(c) for c in cards[:n]]
        out[: len(vals)] = vals
    return out


def _slot(feats: np.ndarray, mon: dict, is_active: bool) -> None:
    feats[0] = 1.0
    feats[1] = mon["hp"] / 300.0
    feats[2] = mon["maxHp"] / 300.0
    feats[3] = 1.0 - mon["hp"] / max(mon["maxHp"], 1)
    feats[4] = len(mon["energies"]) / 6.0
    feats[5] = len(mon["tools"])
    feats[6] = len(mon["preEvolution"]) / 2.0
    feats[7] = float(mon["appearThisTurn"])
    feats[8] = float(is_active)
    for e in mon["energies"]:
        if 0 <= e < N_ENERGY_TYPES:
            feats[9 + e] += 0.25


def _slot_index(side: int, area: int | None, index: int | None) -> int:
    """Board slot (0-8 mine, 9-17 the opponent's; active first) or -1."""
    base = side * (1 + BENCH_SLOTS)
    if area == ACTIVE:
        return base
    if area == BENCH and index is not None and 0 <= index < BENCH_SLOTS:
        return base + 1 + index
    return -1


def _option_slots(opt: dict, me: int) -> tuple[int, int]:
    """(slot the option acts from, slot it targets), -1 where none."""
    t = opt["type"]
    side = 0 if opt.get("playerIndex", me) == me else 1
    area, index = opt.get("area"), opt.get("index")
    if t == ATTACK:
        return 0, 1 + BENCH_SLOTS
    if t == RETREAT:
        return 0, -1
    if t in (ATTACH, EVOLVE):
        return -1, _slot_index(0, opt.get("inPlayArea"), opt.get("inPlayIndex"))
    if t in (ABILITY, DISCARD_OPT):
        return _slot_index(0, area, index), -1
    if t in (TOOL_CARD, ENERGY_CARD, ENERGY) or area in (ACTIVE, BENCH):
        return -1, _slot_index(side, area, index)
    return -1, -1


def _tactical(f: np.ndarray, opt: dict, state: dict, me: int, attack_table: np.ndarray,
              card_meta: np.ndarray) -> None:
    """What an attack would do to the opponent's Active Pokémon, before effects.

    [base damage, damage after Weakness (x2) and Resistance (-30), defender HP,
    knocks out, Prize cards for that knockout, Weakness applies, Resistance applies]
    """
    aid = opt.get("attackId")
    attacker = _at(state["players"][me]["active"], 0)
    defender = _at(state["players"][1 - me]["active"], 0)
    if aid is None or not (0 <= aid < len(attack_table)) or attacker is None or defender is None:
        return
    base = float(attack_table[aid, 0]) * 300.0
    a_type = card_meta[attacker["id"], 0]
    weak = card_meta[defender["id"], 1] == a_type and a_type > 0
    resist = card_meta[defender["id"], 2] == a_type and a_type > 0
    damage = base * (2.0 if weak else 1.0)
    damage = max(0.0, damage - 30.0) if resist else damage
    hp = float(defender["hp"])
    f[:] = [base / 300.0, damage / 300.0, hp / 300.0, float(damage >= hp > 0),
            card_meta[defender["id"], 3] / 3.0, float(weak), float(resist)]


def encode(decision: Decision, attack_table: np.ndarray, my_deck: list[int] | None = None,
           opp_seen: set[int] | None = None, opp_recent: list[int] | None = None,
           card_meta: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Encode a decision for the player to move.

    ``my_deck`` (their 60-card list) and ``opp_seen`` (opponent card ids seen
    so far this game, kept by the caller across decisions) fill the memory
    keys; ``opp_recent`` (the last opponent cards from the log) and
    ``card_meta`` (``engine.card_meta_table``, for attack tactics) fill the
    entity policy's keys. Anything not given stays empty.
    """
    obs = decision.obs
    state, sel = obs["current"], obs["select"]
    me = state["yourIndex"]
    seats = (me, 1 - me)

    g = np.zeros(GLOBAL_FEATURES, dtype=np.float32)
    g[0] = state["turn"] / 30.0
    g[1] = state["turnActionCount"] / 20.0
    g[2] = float(state["firstPlayer"] == me)
    g[3] = float(state["firstPlayer"] == 1 - me)
    g[4:8] = [float(state[k]) for k in ("supporterPlayed", "stadiumPlayed", "energyAttached", "retreated")]
    g[8] = float(state["looking"] is not None)
    o = 9
    for p in seats:
        ps = state["players"][p]
        g[o : o + 8] = [
            ps["deckCount"] / 60.0, ps["handCount"] / 20.0, len(ps["prize"]) / 6.0,
            len(ps["discard"]) / 60.0, ps["benchMax"] / 8.0, len(ps["bench"]) / 8.0,
            float(len(ps["active"]) > 0), sum(c is not None for c in ps["prize"]) / 6.0,
        ]
        g[o + 8 : o + 13] = [float(ps[k]) for k in _CONDITIONS]
        o += 13
    o = _one_hot(g, o, sel["type"], N_SELECT_TYPES)
    o = _one_hot(g, o, sel["context"], N_CONTEXTS)
    g[o : o + 7] = [
        sel["minCount"] / 5.0, sel["maxCount"] / 5.0, sel["remainDamageCounter"] / 20.0,
        sel["remainEnergyCost"] / 5.0, len(decision.picked) / 5.0,
        float(len(decision.picked) >= decision.min), decision.n / MAX_OPTIONS,
    ]

    slot_ids = np.zeros(SLOTS, dtype=np.int32)
    slot_feats = np.zeros((SLOTS, SLOT_FEATURES), dtype=np.float32)
    slot_attach = np.zeros((SLOTS, ATTACH_SLOTS), dtype=np.int32)
    for side, p in enumerate(seats):
        ps = state["players"][p]
        base = side * (1 + BENCH_SLOTS)
        mons = [(0, m, True) for m in ps["active"][:1]] + [
            (1 + i, m, False) for i, m in enumerate(ps["bench"][:BENCH_SLOTS])
        ]
        for i, mon, is_active in mons:
            if mon is not None:
                slot_ids[base + i] = mon["id"]
                _slot(slot_feats[base + i], mon, is_active)
                attached = [_cid(c) for c in mon["tools"] + mon["energyCards"] if c][:ATTACH_SLOTS]
                slot_attach[base + i, : len(attached)] = attached

    opt_ids = np.zeros(MAX_OPTIONS, dtype=np.int32)
    opt_target_ids = np.zeros(MAX_OPTIONS, dtype=np.int32)
    opt_feats = np.zeros((MAX_OPTIONS, OPTION_FEATURES), dtype=np.float32)
    opt_src = np.full(MAX_OPTIONS, -1, dtype=np.int32)
    opt_tgt = np.full(MAX_OPTIONS, -1, dtype=np.int32)
    opt_tactical = np.zeros((MAX_OPTIONS, TACTICAL_FEATURES), dtype=np.float32)
    picked = set(decision.picked)
    for i, opt in enumerate(sel["option"][: decision.n]):
        opt_ids[i], opt_target_ids[i] = _resolve(state, sel, opt, me)
        opt_src[i], opt_tgt[i] = _option_slots(opt, me)
        if opt["type"] == ATTACK and card_meta is not None:
            _tactical(opt_tactical[i], opt, state, me, attack_table, card_meta)
        f = opt_feats[i]
        k = _one_hot(f, 0, opt["type"], N_OPTION_TYPES)
        k = _one_hot(f, k, opt.get("area"), N_AREAS)
        k = _one_hot(f, k, opt.get("inPlayArea"), N_AREAS)
        owner = opt.get("playerIndex")
        f[k + (2 if owner is None else int(owner != me))] = 1.0
        k += 3
        aid = opt.get("attackId")
        if aid is not None and 0 <= aid < len(attack_table):
            f[k : k + 2] = attack_table[aid]
        f[k + 2] = (opt.get("index") or 0) / 10.0
        f[k + 3] = (opt.get("inPlayIndex") or 0) / 8.0
        f[k + 4] = (opt.get("number") or 0) / 10.0
        f[k + 5] = (opt.get("count") or 0) / 5.0
        k += 6
        k = _one_hot(f, k, opt.get("specialConditionType"), N_CONDITIONS)
        f[k] = float(i in picked)

    my, opp = state["players"][me], state["players"][1 - me]
    return {
        "global": g,
        "slot_ids": slot_ids,
        "slot_feats": slot_feats,
        "hand_ids": _ids(my["hand"], HAND_SLOTS),
        "my_discard_ids": _ids(my["discard"], PILE_SLOTS),
        "opp_discard_ids": _ids(opp["discard"], PILE_SLOTS),
        "view_ids": _ids(sel.get("deck") or state["looking"], PILE_SLOTS),
        "misc_ids": np.array(
            [_cid(_at(state["stadium"], 0)), _cid(sel.get("contextCard")), _cid(sel.get("effect"))],
            dtype=np.int32,
        ),
        "opt_ids": opt_ids,
        "opt_target_ids": opt_target_ids,
        "opt_feats": opt_feats,
        "my_unseen_ids": _id_array(unseen_own(my_deck, state, me) if my_deck else [], DECK_SLOTS),
        "opp_seen_ids": _id_array(sorted(opp_seen) if opp_seen else [], DECK_SLOTS),
        "slot_attach_ids": slot_attach,
        "opt_src_slot": opt_src,
        "opt_tgt_slot": opt_tgt,
        "opt_tactical": opt_tactical,
        "opp_recent_ids": _id_array(list(opp_recent)[-RECENT_SLOTS:] if opp_recent else [], RECENT_SLOTS),
    }


def _id_array(ids: list[int], n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.int32)
    ids = ids[:n]
    out[: len(ids)] = ids
    return out


# -- shared-memory transport ----------------------------------------------------
#
# Workers write each game's pending observation and mask straight into one
# shared block, one row per game, so the pipes only carry rewards and flags.
# Pickling ~19 KB observations through pipes cost the trainer half its time.

OPTION_KEYS = ("opt_ids", "opt_target_ids", "opt_feats")


class SharedObs:
    """Row-per-game observation and mask arrays in shared memory."""

    def __init__(self, space: spaces.Dict, n: int, name: str | None = None):
        from multiprocessing import resource_tracker, shared_memory

        layout = [(k, (n, *s.shape), s.dtype) for k, s in space.spaces.items()]
        layout.append(("mask", (n, N_ACTIONS), np.dtype(bool)))
        size = sum(int(np.prod(shape)) * np.dtype(dt).itemsize for _, shape, dt in layout)
        self.owner = name is None
        self.shm = shared_memory.SharedMemory(name=name, create=self.owner, size=size)
        if not self.owner:
            # Python < 3.13 registers attached blocks too, and would unlink the
            # trainer's block when this worker exits.
            resource_tracker.unregister(self.shm._name, "shared_memory")
        self.name = self.shm.name
        self.arrays: dict[str, np.ndarray] = {}
        offset = 0
        for k, shape, dt in layout:
            self.arrays[k] = np.ndarray(shape, dt, buffer=self.shm.buf, offset=offset)
            offset += int(np.prod(shape)) * np.dtype(dt).itemsize

    def put(self, i: int, obs: dict[str, np.ndarray], mask: np.ndarray) -> None:
        for k, v in obs.items():
            self.arrays[k][i] = v
        self.arrays["mask"][i] = mask

    def close(self) -> None:
        self.arrays.clear()
        self.shm.close()
        if self.owner:
            self.shm.unlink()
