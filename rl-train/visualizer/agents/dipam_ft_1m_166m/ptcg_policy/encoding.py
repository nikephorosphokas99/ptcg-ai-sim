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

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

B_COIN = 21
B_DEF_BURNED = 14
B_DEF_DAMAGED = 4
B_DEF_EVOLUTION = 12
B_DEF_EX = 2
B_DEF_POISONED = 13
B_EXTRA_ENERGY = 7
B_OPP_PRIZES_AT_MOST = 18
B_PER_BENCH_BOTH = 8
B_PER_DEF_COUNTER = 5
B_PER_DEF_ENERGY = 3
B_PER_ENERGY_BOTH_ACTIVE = 20
B_PER_MY_BENCH = 10
B_PER_OPP_BENCH = 9
B_PER_OPP_PRIZE_TAKEN = 11
B_PER_SELF_COUNTER = 6
B_PER_SELF_TYPE_ENERGY = 15
B_SELF_DAMAGED = 19
B_SELF_SPECIAL_ENERGY = 17
B_SELF_TOOL = 16
B_STADIUM = 1
BONUS_AMOUNT = 3
BONUS_KIND = 2
BONUS_PARAM = 4
BENCH_EACH = 6
BENCH_ONE = 5
COUNTERS_EACH = 8
COUNTERS_ONE = 7
G_ATTACHED = 5
G_FROM_BENCH = 3
G_IN_PLAY = 2
G_SELF_BENCHED = 1
G_STADIUM = 4
GUARD_COUNTERS = 2
GUARD_DAMAGE = 1
GUARD_FIELDS = 4
META_GUARD = 16
ATT_ABILITY = 3
ATT_BASIC_EX = 2
ATT_ENERGY_AT_MOST = 6
ATT_EX = 1
ATT_SPECIAL_ENERGY = 5
ATT_TERA = 4
ATT_TOOL = 8
ATT_TYPE = 7
DEF_ENERGY_TYPE = 2
DEF_NO_RULE_BOX = 3
DEF_STEVENS = 4
DEF_TYPE = 1
FLAG_ABILITY = 4
FLAG_BASIC = 2
FLAG_EX = 1
FLAG_RULE_BOX = 32
FLAG_SPECIAL_ENERGY = 64
FLAG_STEVENS = 16
FLAG_TERA = 8
LESS_AFTER_WR = 3
LESS_BEFORE_WR = 4
META_FLAGS = 4
META_RULE = 5
META_ATTACKS = 14
META_NAME = 12
META_PREV = 13
N_CARD_ATTACKS = 2
N_ENERGY_TYPES = 12
NO_DAMAGE = 1
NO_DAMAGE_AT_LEAST = 2
ON_BENCH = 4
ON_IN_PLAY = 3
ON_SELF = 1
ON_STADIUM = 5
ON_TOOL = 2

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
TACTICAL_FEATURES = 10  # per attack option: 7 on the Active (_tactical), 3 on the Bench (v5, _bench_tactical)
LEDGER_SLOTS = 40  # distinct cards in my decklist
LEDGER_FEATURES = 14  # per distinct card, see _ledger
LOG_MOVE_CARD = 6  # LogType.MoveCard

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

THREAT_FEATURES = 5  # per board slot, see _threat
OFFENSE_FEATURES = 3  # per board slot (v5), see _offense
BENCH_THREAT_FEATURES = 3  # per board slot (v5), see _bench_threat
THREAT_AT = 9 + N_ENERGY_TYPES
OFFENSE_AT = THREAT_AT + THREAT_FEATURES
BENCH_THREAT_AT = OFFENSE_AT + OFFENSE_FEATURES
SLOT_FEATURES = BENCH_THREAT_AT + BENCH_THREAT_FEATURES
OPTION_FEATURES = N_OPTION_TYPES + N_AREAS * 2 + 3 + 6 + N_CONDITIONS + 1
GLOBAL_FEATURES = 9 + 2 * 13 + N_SELECT_TYPES + N_CONTEXTS + 7 + 2  # last two: whose Stadium is in play

OPT_OPPONENT = N_OPTION_TYPES + N_AREAS * 2 + 1  # opt_feats column: the option is the opponent's card

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
        # Deck ledger (v4): where each card of my list is, and whether the deck still holds it.
        "ledger_ids": ids(LEDGER_SLOTS),
        "ledger_feats": feats(LEDGER_SLOTS, LEDGER_FEATURES),
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
    return sorted((Counter(deck) - Counter(visible_cards(state, player))).elements())


def own_hidden(deck: list[int], state: dict, sel: dict, player: int) -> Counter:
    """``player``'s cards in their deck and face-down Prize cards.

    ``unseen_own`` also counts the cards out of every zone for the moment: those
    being looked at and the card being played (an Ultra Ball while it resolves).
    """
    hidden = Counter(unseen_own(deck, state, player))
    hidden -= Counter(c["id"] for c in state["looking"] or [] if c and c.get("playerIndex", player) == player)
    effect = sel.get("effect")
    if effect and effect.get("playerIndex", player) == player:
        hidden -= Counter([effect["id"]])
    return hidden


def face_down_prizes(state: dict, player: int) -> int:
    return sum(c is None for c in state["players"][player]["prize"])


class PrizeMemory:
    """One player's face-down Prize cards, once known.

    Searching the deck shows all of it, so what of the player's list is neither
    in the deck nor in view is prized: the prize check a player makes on their
    first search. Prize cards taken later are named in the player's own log, so
    the knowledge lasts until a card goes into the Prizes face down. Feed it
    every observation the player receives, forced decisions included: each one
    carries the log since the player's previous observation.
    """

    def __init__(self, deck: list[int]):
        self.deck = deck
        self.prized: Counter | None = None

    def observe(self, obs: dict) -> None:
        state, sel = obs.get("current"), obs.get("select") or {}
        if state is None:
            return
        me = state["yourIndex"]
        for log in obs.get("logs") or []:
            if log.get("playerIndex") != me or log.get("type") != LOG_MOVE_CARD:
                continue
            if log.get("toArea") == PRIZE:
                self.prized = None
            elif log.get("fromArea") == PRIZE and self.prized is not None:
                self.prized[log.get("cardId")] -= 1
                self.prized = +self.prized
        shown = sel.get("deck")
        if shown and len(shown) == state["players"][me]["deckCount"]:
            hidden = own_hidden(self.deck, state, sel, me)
            view = Counter(_cid(c) for c in shown)
            rest = hidden - view
            if not view - hidden and sum(rest.values()) == face_down_prizes(state, me):
                self.prized = rest


def known_prizes(prized: Counter | None, hidden: Counter, face_down: int) -> Counter | None:
    """The face-down Prize cards if ``prized`` (``PrizeMemory.prized``) still adds up, else None."""
    if face_down == 0:
        return Counter()
    if prized is None:
        return None
    known = prized & hidden
    return known if sum(known.values()) == face_down else None


def _all_prized(copies: int, face_down: int, pool: int) -> float:
    """Chance that all ``copies`` of a card among ``pool`` hidden cards are among the ``face_down`` Prizes."""
    if copies > face_down:
        return 0.0
    p = 1.0
    for i in range(copies):
        p *= (face_down - i) / (pool - i)
    return p


def _ledger(deck: list[int], state: dict, sel: dict, me: int, prized: Counter | None,
            card_meta: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """One row per distinct card of ``deck``: where its copies are, and what the deck still holds.

    [copies in the list, in hand, in play, in discard, hidden (deck + face-down Prizes),
    in the deck, prized, chance the deck holds at least one, Prize cards known,
    the next Stage in the deck and the chance of at least one, the Stage after it likewise,
    the Stage it evolves from in play]. Counts are / 4. In the deck and prized are exact
    once ``prized`` is known (``PrizeMemory``), else expected values; Stages come from
    ``card_meta`` (name and evolves-from ids), zero without it.
    """
    ids = np.zeros(LEDGER_SLOTS, dtype=np.int32)
    f = np.zeros((LEDGER_SLOTS, LEDGER_FEATURES), dtype=np.float32)
    if not deck:
        return ids, f
    ps = state["players"][me]
    total = Counter(deck)
    hand = Counter(_cid(c) for c in ps["hand"] or [] if c)
    discard = Counter(_cid(c) for c in ps["discard"] if c)
    in_play = Counter(visible_cards(state, me)) - hand - discard - Counter(_cid(c) for c in ps["prize"] if c)
    hidden = own_hidden(deck, state, sel, me)
    face_down = face_down_prizes(state, me)
    known = known_prizes(prized, hidden, face_down)
    pool = sum(hidden.values())

    def in_deck(cids: list[int]) -> tuple[float, float]:
        """(copies of ``cids`` in the deck, chance of at least one there)."""
        h = sum(hidden[c] for c in cids)
        if known is not None:
            n = h - sum(known[c] for c in cids)
            return float(n), float(n > 0)
        if h == 0:
            return 0.0, 0.0
        return h * (pool - face_down) / pool, 1.0 - _all_prized(h, face_down, pool)

    lines = card_meta is not None and card_meta.shape[1] > META_PREV
    name = (lambda c: int(card_meta[c, META_NAME])) if lines else (lambda c: c)  # noqa: E731
    prev = (lambda c: int(card_meta[c, META_PREV])) if lines else (lambda c: 0)  # noqa: E731
    distinct = sorted(total)[:LEDGER_SLOTS]
    evolves_to = lambda names: [c for c in distinct if prev(c) and prev(c) in names]  # noqa: E731
    on_board = Counter(name(m["id"]) for m in ps["active"] + ps["bench"] if m)
    for row, cid in enumerate(distinct):
        ids[row] = cid
        deck_n, deck_p = in_deck([cid])
        nxt = evolves_to({name(cid)}) if lines else []
        after = evolves_to({name(c) for c in nxt}) if nxt else []
        nxt_n, nxt_p = in_deck(nxt) if nxt else (0.0, 0.0)
        after_n, after_p = in_deck(after) if after else (0.0, 0.0)
        f[row] = [total[cid] / 4.0, hand[cid] / 4.0, in_play[cid] / 4.0, discard[cid] / 4.0, hidden[cid] / 4.0,
                  deck_n / 4.0, (hidden[cid] - deck_n) / 4.0, deck_p, float(known is not None),
                  nxt_n / 4.0, nxt_p, after_n / 4.0, after_p, on_board[prev(cid)] / 4.0 if prev(cid) else 0.0]
    return ids, f


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
    """What an attack would do to the opponent's Active Pokémon, before the attack's own text.

    [base damage, damage after Weakness (x2), Resistance (-30) and the Abilities, Pokémon
    Tools and Stadium that prevent or reduce it (``_damage_rules``), defender HP, knocks out,
    Prize cards for that knockout, Weakness applies, Resistance applies]
    """
    aid = opt.get("attackId")
    attacker = _at(state["players"][me]["active"], 0)
    defender = _at(state["players"][1 - me]["active"], 0)
    if aid is None or not (0 <= aid < len(attack_table)) or attacker is None or defender is None:
        return
    base, damage, weak, resist = _hit(state, me, attacker, defender, aid, attack_table, card_meta)
    hp = float(defender["hp"])
    f[:] = [base / 300.0, damage / 300.0, hp / 300.0, float(damage >= hp > 0),
            card_meta[defender["id"], 3] / 3.0, float(weak), float(resist)]


def _hit(state: dict, me: int, attacker: dict, defender: dict, aid: int, attack_table: np.ndarray,
         card_meta: np.ndarray) -> tuple[float, float, bool, bool]:
    """(base damage, damage done, Weakness applies, Resistance applies) when player ``me``'s Active
    ``attacker`` uses attack ``aid`` on the other player's Active ``defender``, before the attack's text."""
    base = float(attack_table[aid, 0]) * 300.0
    if attack_table.shape[1] > BONUS_KIND and attack_table[aid, BONUS_KIND]:
        base += _bonus(attack_table[aid], state, me, attacker, defender, card_meta)
    a_type = card_meta[attacker["id"], 0]
    weak = bool(card_meta[defender["id"], 1] == a_type and a_type > 0)
    resist = bool(card_meta[defender["id"], 2] == a_type and a_type > 0)
    rules = _damage_rules(state, me, attacker, defender, card_meta) if card_meta.shape[1] > META_RULE else []
    before = sum(amount for effect, amount in rules if effect == LESS_BEFORE_WR)
    damage = max(0.0, base - before) * (2.0 if weak else 1.0) if base > 0 else 0.0
    damage = max(0.0, damage - 30.0) if resist else damage
    damage = max(0.0, damage - sum(amount for effect, amount in rules if effect == LESS_AFTER_WR))
    if any(effect == NO_DAMAGE or (effect == NO_DAMAGE_AT_LEAST and damage >= amount) for effect, amount in rules):
        damage = 0.0
    return base, damage, weak, resist


def _threat(f: np.ndarray, mon: dict, owner: int, state: dict, attack_table: np.ndarray,
            card_meta: np.ndarray) -> None:
    """What losing ``mon`` (``owner``'s Pokémon in play) would cost, and whether the other
    player's Active can take it.

    [Prize cards the other player takes for knocking it out, that knockout wins them the game,
    the most damage the other player's Active can do to it with one more Energy attached,
    that knocks it out, it is knocked out with the Energy already attached]. The damage
    columns are for an Active ``mon`` only (attacks hit the Active; Bench damage is left out),
    and count Energy without its type. Zero without ``card_meta``'s attack columns.
    """
    prizes = int(card_meta[mon["id"], 3]) if 0 < mon["id"] < len(card_meta) else 1
    taker = state["players"][1 - owner]
    f[0] = prizes / 3.0
    f[1] = float(prizes >= len(taker["prize"]) > 0)
    attacker = _at(taker["active"], 0)
    if (attacker is None or _at(state["players"][owner]["active"], 0) is not mon
            or card_meta.shape[1] < META_ATTACKS + N_CARD_ATTACKS or not 0 < attacker["id"] < len(card_meta)):
        return
    energy, hp = len(attacker["energies"]), float(mon["hp"])
    best = best_now = 0.0
    for aid in card_meta[attacker["id"], META_ATTACKS:META_ATTACKS + N_CARD_ATTACKS]:
        aid = int(aid)
        if not 0 < aid < len(attack_table):
            continue
        cost = round(float(attack_table[aid, 1]) * 5)
        if cost > energy + 1:
            continue
        _, damage, _, _ = _hit(state, 1 - owner, attacker, mon, aid, attack_table, card_meta)
        best = max(best, damage)
        if cost <= energy:
            best_now = max(best_now, damage)
    f[2:5] = [best / 300.0, float(best >= hp > 0), float(best_now >= hp > 0)]


def _usable_attacks(mon: dict, card_meta: np.ndarray, attack_table: np.ndarray):
    """(attack id, usable with the Energy attached now) for each of ``mon``'s attacks it could use
    with one more Energy attached (Energy counted without its type)."""
    if not 0 < mon["id"] < len(card_meta):
        return
    energy = len(mon["energies"])
    for aid in card_meta[mon["id"], META_ATTACKS:META_ATTACKS + N_CARD_ATTACKS]:
        aid = int(aid)
        if 0 < aid < len(attack_table):
            cost = round(float(attack_table[aid, 1]) * 5)
            if cost <= energy + 1:
                yield aid, cost <= energy


def _offense(f: np.ndarray, mon: dict, owner: int, state: dict, attack_table: np.ndarray,
             card_meta: np.ndarray) -> None:
    """What ``mon`` (``owner``'s Pokémon in play, Active or Benched) could do to the other player's
    Active Pokémon if it were attacking: [the most damage with one more Energy attached, after
    Weakness, Resistance and the damage rules (``_damage_rules``: Crustle's Mysterious Rock Inn,
    Cornerstone Mask Ogerpon ex, ...), that knocks it out, it knocks it out with the Energy already
    attached]. Says which of our Pokémon can hurt a wall, so switching and setting up can aim for it."""
    defender = _at(state["players"][1 - owner]["active"], 0)
    if defender is None:
        return
    hp, best, ko_now = float(defender["hp"]), 0.0, False
    for aid, now in _usable_attacks(mon, card_meta, attack_table):
        _, damage, _, _ = _hit(state, owner, mon, defender, aid, attack_table, card_meta)
        best = max(best, damage)
        ko_now |= now and damage >= hp > 0
    f[:] = [best / 300.0, float(best >= hp > 0), float(ko_now)]


def _bench_guarded(state: dict, side: int, target: dict, attacker: dict, card_meta: np.ndarray) -> int:
    """``GUARD_*`` bits of what protects ``side``'s Benched ``target`` from attacks by ``attacker``."""
    ps = state["players"][side]
    t_flags = int(card_meta[target["id"], META_FLAGS]) if 0 < target["id"] < len(card_meta) else 0
    a_energy = len(attacker["energies"])
    sources = [(target["id"], G_SELF_BENCHED)] + [(_cid(e), G_ATTACHED) for e in target["energyCards"] if e]
    sources += [(m["id"], G_IN_PLAY) for m in ps["active"] + ps["bench"] if m]
    sources += [(m["id"], G_FROM_BENCH) for m in ps["bench"] if m]
    sources += [(_cid(s), G_STADIUM) for s in state["stadium"] or [] if s]
    stops = 0
    for cid, where in sources:
        if not 0 < cid < len(card_meta):
            continue
        g_where, g_stops, no_rule_box, energy_at_most = (int(x) for x in card_meta[cid, META_GUARD:META_GUARD + GUARD_FIELDS])
        if (g_where == where and not (no_rule_box and t_flags & FLAG_RULE_BOX)
                and not (energy_at_most and a_energy > energy_at_most)):
            stops |= g_stops
    return stops


def _bench_hit(state: dict, owner: int, attacker: dict, target: dict, aid: int, attack_table: np.ndarray,
               card_meta: np.ndarray) -> tuple[float, float, bool]:
    """(to one Benched Pokémon, to each, guarded) when ``owner``'s ``attacker`` uses attack ``aid`` and
    ``target`` is one of the other player's Benched Pokémon: damage and damage counters (x10) that land
    after the Bench guards (``_bench_guarded``: Shaymin's Flower Curtain, ...); guarded is whether a
    guard stopped some of it."""
    row = attack_table[aid]
    dmg_one, dmg_each, ctr_one, ctr_each = (float(row[c]) for c in (BENCH_ONE, BENCH_EACH, COUNTERS_ONE, COUNTERS_EACH))
    if not (dmg_one or dmg_each or ctr_one or ctr_each):
        return 0.0, 0.0, False
    stops = _bench_guarded(state, 1 - owner, target, attacker, card_meta)
    if stops & GUARD_DAMAGE:
        dmg_one = dmg_each = 0.0
    if stops & GUARD_COUNTERS:
        ctr_one = ctr_each = 0.0
    guarded = bool(stops & GUARD_DAMAGE and (row[BENCH_ONE] or row[BENCH_EACH])
                   or stops & GUARD_COUNTERS and (row[COUNTERS_ONE] or row[COUNTERS_EACH]))
    each = max(dmg_each, ctr_each)
    return max(dmg_one, ctr_one, each), each, guarded


def _bench_threat(f: np.ndarray, mon: dict, owner: int, state: dict, attack_table: np.ndarray,
                  card_meta: np.ndarray) -> None:
    """What the other player's Active Pokémon could do to ``mon`` (``owner``'s Benched Pokémon) with one
    more Energy attached: [the most damage landing on it (snipes, spread, damage counters), that knocks
    it out, a Bench guard is stopping some of it]. Says when a Bench needs Shaymin."""
    attacker = _at(state["players"][1 - owner]["active"], 0)
    if attacker is None:
        return
    hp, best, guarded = float(mon["hp"]), 0.0, False
    for aid, _ in _usable_attacks(attacker, card_meta, attack_table):
        one, _, g = _bench_hit(state, 1 - owner, attacker, mon, aid, attack_table, card_meta)
        best, guarded = max(best, one), guarded or g
    f[:] = [best / 300.0, float(best >= hp > 0), float(guarded)]


def _bench_tactical(f: np.ndarray, opt: dict, state: dict, me: int, attack_table: np.ndarray,
                    card_meta: np.ndarray) -> None:
    """What an attack would do to the opponent's Bench: [the most damage landing on one Benched
    Pokémon, Benched Pokémon it knocks out (/ 3), a Bench guard stops some of it]."""
    aid = opt.get("attackId")
    attacker = _at(state["players"][me]["active"], 0)
    if aid is None or not (0 < aid < len(attack_table)) or attacker is None:
        return
    best, each_kos, one_ko, guarded = 0.0, 0, False, False
    for mon in state["players"][1 - me]["bench"]:
        if mon is None:
            continue
        one, each, g = _bench_hit(state, me, attacker, mon, aid, attack_table, card_meta)
        best, guarded = max(best, one), guarded or g
        each_kos += int(each >= mon["hp"] > 0)
        one_ko |= one >= mon["hp"] > 0
    f[:] = [best / 300.0, max(each_kos, int(one_ko)) / 3.0, float(guarded)]


def _bonus(row: np.ndarray, state: dict, me: int, attacker: dict, defender: dict, card_meta: np.ndarray) -> float:
    """The attack's conditional extra damage (``engine._BONUS_RULES``) on this board; a coin flip counts half."""
    kind, amount, param = int(row[BONUS_KIND]), float(row[BONUS_AMOUNT]), int(row[BONUS_PARAM])
    mine, theirs = state["players"][me], state["players"][1 - me]
    counters = lambda mon: max(0, mon["maxHp"] - mon["hp"]) // 10  # noqa: E731
    special = lambda mon: any(int(card_meta[_cid(e), META_FLAGS]) & FLAG_SPECIAL_ENERGY  # noqa: E731
                              for e in mon["energyCards"] if e)
    times = {
        B_STADIUM: any(state["stadium"] or []),
        B_DEF_EX: int(card_meta[defender["id"], META_FLAGS]) & FLAG_EX,
        B_PER_DEF_ENERGY: len(defender["energies"]),
        B_DEF_DAMAGED: counters(defender) > 0,
        B_PER_DEF_COUNTER: counters(defender),
        B_PER_SELF_COUNTER: counters(attacker),
        B_EXTRA_ENERGY: len(attacker["energies"]) - round(float(row[1]) * 5) >= param,
        B_PER_BENCH_BOTH: sum(1 for m in mine["bench"] + theirs["bench"] if m),
        B_PER_OPP_BENCH: sum(1 for m in theirs["bench"] if m),
        B_PER_MY_BENCH: sum(1 for m in mine["bench"] if m),
        B_PER_OPP_PRIZE_TAKEN: 6 - len(theirs["prize"]),
        B_DEF_EVOLUTION: not int(card_meta[defender["id"], META_FLAGS]) & FLAG_BASIC,
        B_DEF_POISONED: theirs["poisoned"],
        B_DEF_BURNED: theirs["burned"],
        B_PER_SELF_TYPE_ENERGY: sum(1 for e in attacker["energies"] if param >> int(e) & 1),
        B_SELF_TOOL: any(attacker["tools"]),
        B_SELF_SPECIAL_ENERGY: special(attacker),
        B_OPP_PRIZES_AT_MOST: len(theirs["prize"]) <= param,
        B_SELF_DAMAGED: counters(attacker) > 0,
        B_PER_ENERGY_BOTH_ACTIVE: len(attacker["energies"]) + len(defender["energies"]),
        B_COIN: 0.5,
    }.get(kind, 0)
    return amount * float(times)


def _damage_rules(state: dict, me: int, attacker: dict, defender: dict,
                  card_meta: np.ndarray) -> list[tuple[int, int]]:
    """(effect, amount) of each damage rule (``engine._DAMAGE_RULES``) that protects the
    opponent's Active ``defender`` from an attack by our Active ``attacker``."""
    side = state["players"][1 - me]
    sources = [(defender["id"], ON_SELF)] + [(_cid(t), ON_TOOL) for t in defender["tools"]]
    sources += [(m["id"], ON_IN_PLAY) for m in side["active"] + side["bench"] if m]
    sources += [(m["id"], ON_BENCH) for m in side["bench"] if m]
    sources += [(_cid(s), ON_STADIUM) for s in state["stadium"] or [] if s]

    a_flags, d_flags = int(card_meta[attacker["id"], META_FLAGS]), int(card_meta[defender["id"], META_FLAGS])
    a_type, d_type = int(card_meta[attacker["id"], 0]), int(card_meta[defender["id"], 0])
    a_energy = [_cid(e) for e in attacker["energyCards"] if e]
    attacker_ok = {
        ATT_EX: lambda _: a_flags & FLAG_EX,
        ATT_BASIC_EX: lambda _: a_flags & FLAG_EX and a_flags & FLAG_BASIC,
        ATT_ABILITY: lambda _: a_flags & FLAG_ABILITY,
        ATT_TERA: lambda _: a_flags & FLAG_TERA,
        ATT_SPECIAL_ENERGY: lambda _: any(card_meta[e, META_FLAGS] & FLAG_SPECIAL_ENERGY for e in a_energy),
        ATT_ENERGY_AT_MOST: lambda n: len(a_energy) <= n,
        ATT_TYPE: lambda mask: mask >> a_type & 1,
        ATT_TOOL: lambda _: any(attacker["tools"]),
    }
    defender_ok = {
        DEF_TYPE: lambda mask: mask >> d_type & 1,
        DEF_ENERGY_TYPE: lambda mask: any(mask >> int(e) & 1 for e in defender["energies"]),
        DEF_NO_RULE_BOX: lambda _: not d_flags & FLAG_RULE_BOX,
        DEF_STEVENS: lambda _: d_flags & FLAG_STEVENS,
    }
    out = []
    for cid, where in sources:
        if not 0 < cid < len(card_meta):
            continue
        rule = card_meta[cid, META_RULE:META_RULE + 7]
        rule_where, att, att_param, dfn, dfn_param, effect, amount = (int(x) for x in rule)
        if (rule_where == where and attacker_ok.get(att, lambda _: True)(att_param)
                and defender_ok.get(dfn, lambda _: True)(dfn_param)):
            out.append((effect, amount))
    return out


def encode(decision: Decision, attack_table: np.ndarray, my_deck: list[int] | None = None,
           opp_seen: set[int] | None = None, opp_recent: list[int] | None = None,
           card_meta: np.ndarray | None = None, prized: Counter | None = None) -> dict[str, np.ndarray]:
    """Encode a decision for the player to move.

    ``my_deck`` (their 60-card list) and ``opp_seen`` (opponent card ids seen
    so far this game, kept by the caller across decisions) fill the memory
    keys; ``opp_recent`` (the last opponent cards from the log) and
    ``card_meta`` (``engine.card_meta_table``, for attack tactics) fill the
    entity policy's keys; ``prized`` (the mover's ``PrizeMemory.prized``)
    makes the deck ledger exact. Anything not given stays empty.
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
    stadium = _at(state["stadium"], 0)
    if stadium:  # mine / theirs: the Stadium's card alone (misc_ids) does not say who played it
        g[o + 7 + (stadium.get("playerIndex") != me)] = 1.0

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
                if card_meta is not None:
                    row = slot_feats[base + i]
                    _threat(row[THREAT_AT:OFFENSE_AT], mon, p, state, attack_table, card_meta)
                    if card_meta.shape[1] >= META_GUARD + GUARD_FIELDS and attack_table.shape[1] > COUNTERS_EACH:
                        _offense(row[OFFENSE_AT:BENCH_THREAT_AT], mon, p, state, attack_table, card_meta)
                        if not is_active:
                            _bench_threat(row[BENCH_THREAT_AT:SLOT_FEATURES], mon, p, state, attack_table,
                                          card_meta)
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
            _tactical(opt_tactical[i, :7], opt, state, me, attack_table, card_meta)
            if card_meta.shape[1] >= META_GUARD + GUARD_FIELDS and attack_table.shape[1] > COUNTERS_EACH:
                _bench_tactical(opt_tactical[i, 7:], opt, state, me, attack_table, card_meta)
        f = opt_feats[i]
        k = _one_hot(f, 0, opt["type"], N_OPTION_TYPES)
        k = _one_hot(f, k, opt.get("area"), N_AREAS)
        k = _one_hot(f, k, opt.get("inPlayArea"), N_AREAS)
        owner = opt.get("playerIndex")
        f[k + (2 if owner is None else int(owner != me))] = 1.0
        k += 3
        aid = opt.get("attackId")
        if aid is not None and 0 <= aid < len(attack_table):
            f[k : k + 2] = attack_table[aid, :2]
        f[k + 2] = (opt.get("index") or 0) / 10.0
        f[k + 3] = (opt.get("inPlayIndex") or 0) / 8.0
        f[k + 4] = (opt.get("number") or 0) / 10.0
        f[k + 5] = (opt.get("count") or 0) / 5.0
        k += 6
        k = _one_hot(f, k, opt.get("specialConditionType"), N_CONDITIONS)
        f[k] = float(i in picked)

    my, opp = state["players"][me], state["players"][1 - me]
    ledger_ids, ledger_feats = _ledger(my_deck, state, sel, me, prized, card_meta)
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
        "ledger_ids": ledger_ids,
        "ledger_feats": ledger_feats,
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
