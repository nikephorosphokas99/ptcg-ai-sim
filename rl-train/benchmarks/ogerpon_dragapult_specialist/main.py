"""Rule-based Teal Mask Ogerpon ex / Hydrapple ex / Meganium agent, tuned to beat Dragapult ex.

The deck is the leaderboard Ogerpon list (Kaggle episodes 125180090, 125181129,
125184513: bai fang, EkremKutukculer, Jarno Rantala). The rules follow what the
strong Ogerpon players did in those replays -- Teal Dance every turn, Grass
energy onto Ogerpon, Bug Catching Set / Poké Pad early, Forest of Vitality,
build Meganium (Wild Growth) and Hydrapple ex, attack with Myriad Leaf Shower or
Syrup Storm -- plus counters to Dragapult ex:

* Boss's Orders onto a Dreepy / Drakloak (70 / 90 HP) that the Active Pokémon
  can Knock Out now, before it evolves into a 320 HP Dragapult ex.
* A small, sturdy Bench against Phantom Dive's 6 damage counters.
* Forest of Vitality over Risky Ruins (which damages Basics as they are benched).
* Items early, before Budew's Itchy Pollen locks them.
* Hydrapple ex as the tank and finisher: Ripening Charge heals, Syrup Storm
  scales with every Grass energy in play (doubled by Meganium's Wild Growth).

Self-contained: works on the raw observation dict, imports nothing but the
standard library, and always returns a legal selection.
"""

import os

# -- card ids (official engine) ----------------------------------------------------
OGERPON, APPLIN, DIPPLIN, HYDRAPPLE = 96, 92, 93, 150
CHIKORITA, BAYLEEF, MEGANIUM = 917, 709, 710
MEOWTH, FEZANDIPITI, TAPU_BULU = 1071, 140, 920
ULTRA_BALL, BUG_SET, POKE_PAD, UNFAIR_STAMP, NIGHT_STRETCHER = 1121, 1094, 1152, 1080, 1097
LILLIE, DAWN, BOSS, LANA, JUDGE = 1227, 1231, 1182, 1184, 1213
FOREST = 1261
GRASS = 1
# opponent (Dragapult) pieces
DREEPY, DRAKLOAK, DRAGAPULT = 119, 120, 121
RISKY_RUINS = 1260

# attack id -> (base damage, energy needed)
ATTACKS = {120: (30, 3), 114: (10, 1), 115: (0, 1), 195: (30, 2), 1322: (0, 1), 1323: (30, 2),
           1027: (50, 2), 1028: (140, 4), 1546: (60, 3), 183: (0, 3), 1326: (220, 4)}
MYRIAD_LEAF_SHOWER, SYRUP_STORM = 120, 195

# OptionType
NUMBER, YES, NO, CARD, TOOL_CARD, ENERGY_CARD, ENERGY = 0, 1, 2, 3, 4, 5, 6
PLAY, ATTACH, EVOLVE, ABILITY, DISCARD_OPT, RETREAT, ATTACK, END = 7, 8, 9, 10, 11, 12, 13, 14
# AreaType
DECK, HAND, DISCARD_AREA, ACTIVE, BENCH, PRIZE, STADIUM, LOOKING = 1, 2, 3, 4, 5, 6, 7, 12
# SelectContext
(MAIN, SETUP_ACTIVE, SETUP_BENCH, SWITCH, TO_ACTIVE, TO_BENCH, TO_FIELD, TO_HAND, DISCARD_CTX,
 TO_DECK) = range(10)
DAMAGE_COUNTER, DAMAGE_COUNTER_ANY, DAMAGE_CTX, HEAL = 13, 14, 15, 17
ATTACH_FROM, ATTACH_TO = 21, 22
IS_FIRST, MULLIGAN, ACTIVATE = 41, 42, 43
GUSTED = {SWITCH, TO_ACTIVE}

EX_PRIZES = {DRAGAPULT: 2, 140: 2, 1071: 2}  # opponent Pokémon ex worth 2
RETREAT_COST = {OGERPON: 1, APPLIN: 1, DIPPLIN: 2, HYDRAPPLE: 3, CHIKORITA: 1, BAYLEEF: 2, MEGANIUM: 2,
                MEOWTH: 1, FEZANDIPITI: 1, TAPU_BULU: 3}
ATTACKERS = (OGERPON, HYDRAPPLE)
# every Pokémon in the engine's card pool that is Weak to {G} (x2 damage), e.g. Marnie's Grimmsnarl ex
GRASS_WEAK = frozenset((
    21, 22, 23, 41, 57, 61, 81, 82, 83, 117, 135, 136, 137, 187, 188, 189, 225, 227, 228, 229,
    248, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 312, 334, 378, 379, 380, 381,
    382, 383, 384, 385, 386, 387, 388, 389, 440, 441, 442, 443, 444, 447, 464, 526, 527, 532, 533, 534,
    538, 539, 540, 599, 600, 601, 603, 604, 607, 608, 609, 610, 611, 614, 615, 616, 617, 618, 642, 643,
    644, 645, 646, 647, 648, 649, 669, 670, 671, 675, 676, 682, 683, 684, 685, 686, 687, 688, 690, 691,
    775, 776, 820, 821, 822, 823, 824, 825, 826, 827, 828, 830, 831, 832, 881, 882, 887, 888, 889, 892,
    893, 894, 895, 896, 897, 973, 975, 976, 985, 986, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1056,
    1063, 1315, 1316, 1317, 1318, 1319, 1385, 1386, 1387, 1389, 1392, 1393, 1394, 1395, 1396, 1397, 1398, 1399, 1400,
))


def _read_deck() -> list:
    for path in ("deck.csv", os.path.join("/kaggle_simulations/agent", "deck.csv"),
                 os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".", "deck.csv")):
        if os.path.exists(path):
            with open(path) as f:
                return [int(x) for x in f.read().split()][:60]
    raise FileNotFoundError("deck.csv")


# -- board helpers -------------------------------------------------------------------

def _area(state, sel, player, area):
    ps = state["players"][player]
    return {ACTIVE: ps["active"], BENCH: ps["bench"], HAND: ps["hand"] or [], DISCARD_AREA: ps["discard"],
            PRIZE: ps["prize"], STADIUM: state["stadium"], LOOKING: state.get("looking") or [],
            DECK: sel.get("deck") or []}.get(area, [])


def _at(lst, i):
    return lst[i] if i is not None and 0 <= i < len(lst) else None


def _card_of(state, sel, opt, me):
    """(card dict, owner) the option refers to, or (None, None)."""
    t = opt["type"]
    owner = opt.get("playerIndex", me)
    if t == PLAY:
        return _at(_area(state, sel, me, HAND), opt.get("index")), me
    if t in (ATTACH, EVOLVE):
        return _at(_area(state, sel, me, opt.get("area")), opt.get("index")), me
    if opt.get("area") is not None:
        return _at(_area(state, sel, owner, opt.get("area")), opt.get("index")), owner
    return None, None


def _target_of(state, sel, opt, me):
    if opt["type"] in (ATTACH, EVOLVE):
        return _at(_area(state, sel, me, opt.get("inPlayArea")), opt.get("inPlayIndex"))
    return None


def _id(card):
    return card["id"] if card else 0


def _energy(mon):
    return len(mon["energies"]) if mon else 0


def _grass_in_play(ps):
    return sum(sum(1 for e in m["energies"] if e == 1) for m in ps["active"] + ps["bench"] if m)


class Ctx:
    def __init__(self, obs):
        self.obs, self.state, self.sel = obs, obs["current"], obs["select"]
        self.me = self.state["yourIndex"]
        self.my = self.state["players"][self.me]
        self.op = self.state["players"][1 - self.me]
        self.hand = [_id(c) for c in (self.my["hand"] or [])]
        self.mine_in_play = [_id(m) for m in self.my["active"] + self.my["bench"] if m]
        self.active = _at(self.my["active"], 0)
        self.opp_active = _at(self.op["active"], 0)
        self.stadium = _id(_at(self.state["stadium"], 0))
        self.stadium_ours = bool(self.state["stadium"]) and self.state["stadium"][0].get("playerIndex") == self.me

    def count(self, cid):
        return self.mine_in_play.count(cid)

    def best_damage(self, mon, target=None):
        """What ``mon`` could do this turn to ``target`` (default: their Active) if it could attack."""
        target = self.opp_active if target is None else target
        if not mon:
            return 0
        if mon["id"] == OGERPON and _energy(mon) >= 3:
            dmg = 30 + 30 * (_energy(mon) + _energy(target))
        elif mon["id"] == HYDRAPPLE and _energy(mon) >= 2:
            dmg = 30 + 30 * _grass_in_play(self.my)
        elif mon["id"] == MEGANIUM and _energy(mon) >= 4:
            dmg = 140
        elif mon["id"] == TAPU_BULU and _energy(mon) >= 4:
            dmg = 220
        else:
            return 0
        return dmg * 2 if target and target["id"] in GRASS_WEAK else dmg

    def needs(self):
        """Evolution pieces this board is missing, most urgent first."""
        out = []
        if self.count(BAYLEEF) and not self.count(MEGANIUM) and MEGANIUM not in self.hand:
            out.append(MEGANIUM)
        if self.count(DIPPLIN) and not self.count(HYDRAPPLE) and HYDRAPPLE not in self.hand:
            out.append(HYDRAPPLE)
        if self.count(CHIKORITA) and BAYLEEF not in self.hand:
            out.append(BAYLEEF)
        if self.count(APPLIN) and DIPPLIN not in self.hand:
            out.append(DIPPLIN)
        if self.count(OGERPON) + self.hand.count(OGERPON) < 2:
            out.append(OGERPON)
        if not (self.count(CHIKORITA) + self.count(BAYLEEF) + self.count(MEGANIUM)) and CHIKORITA not in self.hand:
            out.append(CHIKORITA)
        if not (self.count(APPLIN) + self.count(DIPPLIN) + self.count(HYDRAPPLE)) and APPLIN not in self.hand:
            out.append(APPLIN)
        return out

    def ko_targets(self):
        """Opponent benched Pokémon our Active could Knock Out if gusted in, best first."""
        out = []
        for i, m in enumerate(self.op["bench"]):
            if m and self.best_damage(self.active, m) >= m["hp"]:
                value = EX_PRIZES.get(m["id"], 1) * 10 + (8 if m["id"] in (DRAKLOAK, DREEPY) else 0)
                out.append((value, i))
        return [i for _, i in sorted(out, reverse=True)]

    def potential(self, mon):
        """How good ``mon`` is as the Active: damage now, else energy already on an attacker."""
        if not mon:
            return -1
        dmg = self.best_damage(mon)
        if dmg:
            return dmg
        return 5 * _energy(mon) + 1 if mon["id"] in ATTACKERS else 0

    def pivot_target(self):
        """The benched Pokémon to bring up, if the Active should make way for it."""
        best = max((m for m in self.my["bench"] if m), key=self.potential, default=None)
        if best is None or not self.active:
            return None
        cur = self.potential(self.active)
        if self.active["id"] not in ATTACKERS or cur == 0:
            return best if self.potential(best) > cur else None
        # an attacker that cannot KO gives way to one that can
        opp_hp = self.opp_active["hp"] if self.opp_active else 10 ** 6
        if self.best_damage(self.active) < opp_hp <= self.best_damage(best):
            return best
        return None

    def can_retreat(self):
        return self.active is not None and _energy(self.active) >= RETREAT_COST.get(self.active["id"], 2)

    def bench_room(self):
        return self.my["benchMax"] - len(self.my["bench"])


# -- scoring ---------------------------------------------------------------------------

def score_main(ctx, opt):
    t = opt["type"]
    card, _ = _card_of(ctx.state, ctx.sel, opt, ctx.me)
    cid = _id(card)
    if t == ABILITY:
        if cid == OGERPON:  # Teal Dance only powers itself: concentrate on the attacker
            if GRASS not in ctx.hand:
                return 1
            return 100 + (6 if card is ctx.active else 0) + min(_energy(card), 3)
        if cid == HYDRAPPLE:
            return 96 if GRASS in ctx.hand else 1
        if cid == FEZANDIPITI:
            return 92
        return 20
    if t == EVOLVE:
        target = _target_of(ctx.state, ctx.sel, opt, ctx.me)
        return {MEGANIUM: 98, HYDRAPPLE: 97, BAYLEEF: 93, DIPPLIN: 92}.get(cid, 90) - (5 if target is ctx.active and cid in (BAYLEEF, DIPPLIN) else 0)
    if t == PLAY:
        if cid == FOREST:
            return 0 if ctx.stadium == FOREST else (94 if ctx.stadium == RISKY_RUINS else 88)
        if cid == UNFAIR_STAMP:
            return 89
        if cid == BUG_SET:
            return 86
        if cid == POKE_PAD:
            return 84 if ctx.needs() else 20
        if cid == ULTRA_BALL:
            return 78 if ctx.needs() and len(ctx.hand) >= 3 else 5
        if cid == NIGHT_STRETCHER:
            dead = [_id(c) for c in ctx.my["discard"]]
            return 70 if any(x in dead for x in (HYDRAPPLE, MEGANIUM, OGERPON, DIPPLIN, BAYLEEF, GRASS)) else 10
        if cid == OGERPON:
            return 82 if ctx.count(OGERPON) < 3 and ctx.bench_room() > 1 else 25
        if cid == CHIKORITA:
            return 80 if not (ctx.count(CHIKORITA) + ctx.count(BAYLEEF) + ctx.count(MEGANIUM)) else -3
        if cid == APPLIN:
            return 79 if not (ctx.count(APPLIN) + ctx.count(DIPPLIN) + ctx.count(HYDRAPPLE)) else -3
        if cid == MEOWTH:
            has_supporter = any(x in ctx.hand for x in (LILLIE, DAWN, BOSS, JUDGE, LANA))
            return 74 if not has_supporter and ctx.bench_room() > 1 and not ctx.state["supporterPlayed"] else 12
        if cid in (FEZANDIPITI, TAPU_BULU):
            return 15 if ctx.bench_room() > 2 and not ctx.mine_in_play.count(OGERPON) else -3
        if cid == BOSS:
            return 96 if ctx.ko_targets() else -5
        if cid == DAWN:
            return 72 if len(ctx.needs()) >= 2 else 40
        if cid == LILLIE:
            if len(ctx.my["prize"]) == 6 and len(ctx.hand) <= 7:
                return 75
            return 68 if len(ctx.hand) <= 5 else 35
        if cid == JUDGE:
            return 66 if ctx.op["handCount"] >= 6 and len(ctx.hand) <= 4 else 20
        if cid == LANA:
            grass_in_discard = sum(1 for c in ctx.my["discard"] if _id(c) == GRASS)
            if grass_in_discard >= 2 and ctx.hand.count(GRASS) < 2:
                return 70
            return 45 if len(ctx.hand) <= 3 else 15
        return 30
    if t == ATTACH:
        target = _target_of(ctx.state, ctx.sel, opt, ctx.me)
        tid = _id(target)
        base = 64
        if target is ctx.active and ctx.pivot_target() is not None and not ctx.can_retreat():
            return 77  # pay the retreat cost so the powered attacker can come up
        if target is ctx.active and tid in (OGERPON, HYDRAPPLE):
            return 85 if _energy(target) < 3 else base + 10
        if tid == OGERPON:
            return base + 6 - _energy(target)
        if tid == HYDRAPPLE:
            return base + 5
        if tid in (DIPPLIN, APPLIN):
            return base + 2
        return base - 4
    if t == RETREAT:
        if ctx.pivot_target() is not None:
            return 76
        return -5
    if t == ATTACK:
        base, _ = ATTACKS.get(opt.get("attackId"), (0, 0))
        dmg = base
        if opt.get("attackId") in (MYRIAD_LEAF_SHOWER, SYRUP_STORM):
            dmg = ctx.best_damage(ctx.active)
        ko = ctx.opp_active is not None and dmg >= ctx.opp_active["hp"]
        # attacking ends the turn: after every useful play, but always over passing
        return 40 + min(dmg, 400) / 20 + (20 if ko else 0)
    if t == DISCARD_OPT:
        return -10
    if t == END:
        return 0
    return 10


def score_card(ctx, opt):
    """Card-choice contexts (search, discard, switch, gust, attach targets...)."""
    ctxt = ctx.sel["context"]
    card, owner = _card_of(ctx.state, ctx.sel, opt, ctx.me)
    cid = _id(card)
    if ctxt == SETUP_ACTIVE:
        return {OGERPON: 10, CHIKORITA: 7, APPLIN: 6, MEOWTH: 4, FEZANDIPITI: 3, TAPU_BULU: 2}.get(cid, 1)
    if ctxt in (SETUP_BENCH, TO_BENCH, TO_FIELD):
        return {OGERPON: 9, CHIKORITA: 8, APPLIN: 7, MEOWTH: 3, FEZANDIPITI: 1, TAPU_BULU: 0}.get(cid, 2)
    if ctxt == TO_HAND:
        needs = ctx.needs()
        if cid in needs:
            return 50 - needs.index(cid)
        if cid == GRASS:  # the deck's engine: Teal Dance, Ripening Charge and the manual attach
            return {0: 60, 1: 52}.get(ctx.hand.count(GRASS), 12)
        if cid == BOSS:
            return 45 if ctx.ko_targets() else 18
        if cid == LILLIE:
            return 35 if len(ctx.hand) <= 4 else 20
        if cid == DAWN:
            return 33
        if cid in (UNFAIR_STAMP, JUDGE):
            return 22
        return {HYDRAPPLE: 26, MEGANIUM: 25, OGERPON: 24, FOREST: 16}.get(cid, 8)
    if ctxt == DISCARD_CTX:  # lowest-value first: Ultra Ball costs, hand limits
        keep = {GRASS: 40 - 8 * max(0, ctx.hand.count(GRASS) - 2), OGERPON: 60, HYDRAPPLE: 70, MEGANIUM: 70,
                DIPPLIN: 55, BAYLEEF: 55, BOSS: 65, LILLIE: 50, DAWN: 45, FOREST: 30 if ctx.stadium != FOREST else 5,
                UNFAIR_STAMP: 35, JUDGE: 20, LANA: 15, TAPU_BULU: 10, FEZANDIPITI: 18, MEOWTH: 25}
        return -keep.get(cid, 30)
    if ctxt in GUSTED:
        if owner is not None and owner != ctx.me:  # Boss's Orders target
            targets = ctx.ko_targets()
            idx = opt.get("index") if opt.get("area") == BENCH else None
            if idx in targets:
                return 100 - targets.index(idx)
            return {DREEPY: 40, DRAKLOAK: 35}.get(cid, 10) - (card["hp"] / 100 if card else 0)
        return ctx.potential(card) / 10 + (5 if cid in ATTACKERS else 0) + (card["hp"] / 100 if card else 0)
    if ctxt == ATTACH_TO:  # which card to attach: Grass energy first
        return 10 if cid == GRASS else 1
    if ctxt in (ATTACH_FROM, HEAL):  # which Pokémon (Ripening Charge, effects): our attackers first
        if owner == ctx.me and card and "hp" in card:
            damaged = card["maxHp"] - card["hp"]
            return (10 if card is ctx.active else 0) + {HYDRAPPLE: 6, OGERPON: 5}.get(cid, 0) + damaged / 30
        return 0
    if ctxt in (DAMAGE_COUNTER, DAMAGE_COUNTER_ANY, DAMAGE_CTX):  # e.g. Cruel Arrow: finish something
        if owner is not None and owner != ctx.me and card:
            return 50 - card["hp"] / 10 + (20 if card["id"] in (DREEPY, DRAKLOAK) else 0)
        return -10
    return 0


def choose(obs):
    sel = obs["select"]
    opts = sel["option"]
    n = len(opts)
    lo, hi = sel["minCount"], min(sel["maxCount"], n)
    if hi == 0:
        return []
    ctxt = sel["context"]
    if ctxt == IS_FIRST:   # go second: Ogerpon can attack (and Teal Dance) on its first turn
        return [next((i for i, o in enumerate(opts) if o["type"] == NO), 0)]
    if ctxt in (MULLIGAN, ACTIVATE) or all(o["type"] in (YES, NO) for o in opts):
        return [next((i for i, o in enumerate(opts) if o["type"] == YES), 0)]
    if all(o["type"] == NUMBER for o in opts):  # draw counts etc: as many as allowed
        return [max(range(n), key=lambda i: opts[i].get("number") or 0)]
    ctx = Ctx(obs)
    scorer = score_main if ctxt == MAIN else score_card
    scores = [scorer(ctx, o) for o in opts]
    order = sorted(range(n), key=lambda i: -scores[i])
    if ctxt == MAIN:
        return [order[0]]
    if lo == hi:
        return order[:hi]
    # optional extra picks: take the ones that score above zero
    picks = order[:lo] + [i for i in order[lo:hi] if scores[i] > 0]
    return picks[:hi]


_MY_DECK = None  # not DECK: that name is the AreaType constant used by _area


def agent(obs_dict):
    global _MY_DECK
    if obs_dict.get("select") is None:
        if _MY_DECK is None:
            _MY_DECK = _read_deck()
        return list(_MY_DECK)
    return choose(obs_dict)
