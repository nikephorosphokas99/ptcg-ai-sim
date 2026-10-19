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

import numpy as np
from gymnasium import spaces

from .engine import N_ENERGY_TYPES

MAX_OPTIONS = 64
STOP = MAX_OPTIONS
N_ACTIONS = MAX_OPTIONS + 1

BENCH_SLOTS = 8
SLOTS = 2 * (1 + BENCH_SLOTS)  # active + bench, mine then opponent's
HAND_SLOTS = 40
PILE_SLOTS = 60
MISC_IDS = 3  # stadium, select.contextCard, select.effect

# AreaType
DECK, HAND, DISCARD, ACTIVE, BENCH, PRIZE, STADIUM = 1, 2, 3, 4, 5, 6, 7
LOOKING = 12
N_AREAS = 13
# OptionType
PLAY, ATTACH, EVOLVE, ABILITY, DISCARD_OPT = 7, 8, 9, 10, 11
TOOL_CARD, ENERGY_CARD, ENERGY = 4, 5, 6
ATTACK, SKILL = 13, 15
N_OPTION_TYPES = 17
N_SELECT_TYPES = 11
N_CONTEXTS = 49
N_CONDITIONS = 5

SLOT_FEATURES = 9 + N_ENERGY_TYPES
OPTION_FEATURES = N_OPTION_TYPES + N_AREAS * 2 + 3 + 6 + N_CONDITIONS + 1
GLOBAL_FEATURES = 9 + 2 * 13 + N_SELECT_TYPES + N_CONTEXTS + 7

_CONDITIONS = ("poisoned", "burned", "asleep", "paralyzed", "confused")


def observation_space(max_card_id: int) -> spaces.Dict:
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
    })


class Decision:
    """One engine selection, resolved one pick at a time."""

    def __init__(self, obs: dict):
        sel = obs["select"]
        self.obs = obs
        self.n = min(len(sel["option"]), MAX_OPTIONS)
        self.max = min(sel["maxCount"], self.n)
        self.min = min(sel["minCount"], self.max)
        self.picked: list[int] = []

    def mask(self) -> np.ndarray:
        m = np.zeros(N_ACTIONS, dtype=bool)
        m[: self.n] = True
        m[self.picked] = False
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


def encode(decision: Decision, attack_table: np.ndarray) -> dict[str, np.ndarray]:
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

    opt_ids = np.zeros(MAX_OPTIONS, dtype=np.int32)
    opt_target_ids = np.zeros(MAX_OPTIONS, dtype=np.int32)
    opt_feats = np.zeros((MAX_OPTIONS, OPTION_FEATURES), dtype=np.float32)
    picked = set(decision.picked)
    for i, opt in enumerate(sel["option"][: decision.n]):
        opt_ids[i], opt_target_ids[i] = _resolve(state, sel, opt, me)
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
    }
