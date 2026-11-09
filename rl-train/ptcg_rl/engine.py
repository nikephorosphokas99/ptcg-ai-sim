"""Per-game ctypes bindings to the C++ engine.

The engine itself keeps every battle in its own heap object (``ApiData*``), so
any number of games can run side by side in one process. The competition's
``cg/game.py`` wrapper hides that by storing the pointer on a class attribute
(``Battle.battle_ptr``), which makes it a process-wide singleton. ``Game`` here
holds its own pointer instead, which is all a vectorised env needs.

The library is the official competition binary shipped in
``rl-train/engine/sample_submission/cg``, so card ids match
``EN_Card_Data_R2_full.csv`` and what the competition runs. Set ``PTCG_LIBCG``
to load a different build.
"""

from __future__ import annotations

import ctypes
import json
import threading
import os
import platform
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

TRAIN_ROOT = Path(__file__).resolve().parents[1]
_CG_DIR = TRAIN_ROOT / "engine" / "sample_submission" / "cg"
if platform.system() == "Windows":
    DEFAULT_LIB = _CG_DIR / "cg.dll"
elif platform.system() == "Darwin":
    DEFAULT_LIB = _CG_DIR / "libcg.dylib"
elif platform.machine() in ("arm64", "aarch64"):
    DEFAULT_LIB = _CG_DIR / "libcg-arm64.so"
else:
    DEFAULT_LIB = _CG_DIR / "libcg.so"

AGENT_DECK = TRAIN_ROOT / "decks" / "main_agent.csv"
OPPONENT_DECK_DIR = TRAIN_ROOT / "decks" / "opponents"


def default_opponent_decks() -> list[str]:
    """Every filled-in deck under ``decks/opponents``; empty placeholders are skipped."""
    return [str(p) for p in sorted(OPPONENT_DECK_DIR.glob("*.csv")) if p.read_text().strip()]


class StartData(ctypes.Structure):
    _fields_ = [
        ("battlePtr", ctypes.c_void_p),
        ("errorPlayer", ctypes.c_int),
        ("errorType", ctypes.c_int),
    ]


class SerialData(ctypes.Structure):
    _fields_ = [
        ("json", ctypes.c_char_p),
        ("data", ctypes.POINTER(ctypes.c_ubyte)),
        ("count", ctypes.c_int),
        ("selectPlayer", ctypes.c_int),
    ]


@lru_cache(maxsize=1)
def lib() -> ctypes.CDLL:
    """Load and initialise the engine once per process."""
    path = Path(os.environ.get("PTCG_LIBCG", DEFAULT_LIB))
    if not path.exists():
        raise FileNotFoundError(
            f"engine library not found at {path}; it ships in rl-train/engine/sample_submission/cg, "
            "or point PTCG_LIBCG at another build"
        )
    cdll = ctypes.cdll.LoadLibrary(str(path))
    cdll.GameInitialize()
    cdll.BattleStart.restype = StartData
    cdll.BattleStart.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cdll.BattleFinish.argtypes = [ctypes.c_void_p]
    cdll.GetBattleData.restype = SerialData
    cdll.GetBattleData.argtypes = [ctypes.c_void_p]
    cdll.Select.restype = ctypes.c_int
    cdll.Select.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cdll.AllCard.restype = ctypes.c_char_p
    cdll.AllAttack.restype = ctypes.c_char_p
    return cdll


# The engine is not thread-safe, and ctypes releases the GIL during a call: threads
# sharing one process (the benchmark's parallel games) corrupted its buffers
# ("buffer full. capacity:7", then a segfault). Every battle call holds this lock.
_ENGINE_LOCK = threading.Lock()


class Game:
    """One battle. ``obs`` is always the engine's current observation dict."""

    def __init__(self, deck0: list[int], deck1: list[int], search_input: bool = False):
        """``search_input`` adds ``obs["search_begin_input"]`` as Kaggle's cabt does (for agents that search)."""
        self.search_input = search_input
        if len(deck0) != 60 or len(deck1) != 60:
            raise ValueError("decks must contain 60 cards")
        cards = (ctypes.c_int * 120)(*(deck0 + deck1))
        with _ENGINE_LOCK:
            start = lib().BattleStart(cards)
        if not start.battlePtr:
            raise ValueError(
                f"engine rejected deck of player {start.errorPlayer} (errorType {start.errorType})"
            )
        self._ptr = start.battlePtr
        self.obs = self._fetch()

    def _fetch(self) -> dict:
        with _ENGINE_LOCK:
            sd = lib().GetBattleData(self._ptr)
            raw = sd.json.decode()
            sbi = ctypes.string_at(sd.data, sd.count).decode("ascii") if self.search_input else None
        obs = json.loads(raw)
        if sbi is not None:
            obs["search_begin_input"] = sbi
        return obs

    @property
    def result(self) -> int:
        """Winning player index, 2 for a draw, -1 while the game is running."""
        current = self.obs.get("current")
        return -1 if current is None else current["result"]

    @property
    def done(self) -> bool:
        return self.result >= 0

    @property
    def to_move(self) -> int:
        return self.obs["current"]["yourIndex"]

    def select(self, picks: list[int]) -> dict:
        arg = (ctypes.c_int * max(1, len(picks)))(*picks)
        with _ENGINE_LOCK:
            err = lib().Select(self._ptr, arg, len(picks))
        if err != 0:
            raise ValueError(f"engine rejected selection {picks} (error {err})")
        self.obs = self._fetch()
        return self.obs

    def close(self) -> None:
        if self._ptr:
            with _ENGINE_LOCK:
                lib().BattleFinish(self._ptr)
            self._ptr = None

    def __del__(self):
        self.close()


def load_deck(path: str | Path) -> list[int]:
    """Read a 60-line deck CSV and check every card exists in this engine build."""
    deck = [int(x) for x in Path(path).read_text().split() if x.strip()]
    if len(deck) != 60:
        raise ValueError(f"{path}: expected 60 cards, found {len(deck)}")
    unknown = sorted(set(deck) - set(all_cards()))
    if unknown:
        raise ValueError(f"{path}: card ids not implemented in this engine build: {unknown}")
    return deck


@lru_cache(maxsize=1)
def all_cards() -> dict[int, dict]:
    return {c["cardId"]: c for c in json.loads(lib().AllCard().decode())}


@lru_cache(maxsize=1)
def all_attacks() -> dict[int, dict]:
    return {a["attackId"]: a for a in json.loads(lib().AllAttack().decode())}


N_ENERGY_TYPES = 12
N_CARD_TYPES = 7
CARD_FEATURES = N_CARD_TYPES + N_ENERGY_TYPES * 2 + 7 + 6


@lru_cache(maxsize=1)
def card_table() -> np.ndarray:
    """Static features for every card id; row 0 (padding / face-down) is zeros.

    Identity is learned through an embedding in the policy; these rows give it
    what the rules text says about a card so unseen or rare cards are not blank.
    """
    cards, attacks = all_cards(), all_attacks()
    table = np.zeros((max(cards) + 1, CARD_FEATURES), dtype=np.float32)
    for cid, c in cards.items():
        row = table[cid]
        row[c["cardType"]] = 1.0
        o = N_CARD_TYPES
        row[o + c["energyType"]] = 1.0
        o += N_ENERGY_TYPES
        if c["weakness"] is not None:
            row[o + c["weakness"]] = 1.0
        o += N_ENERGY_TYPES
        for flag in ("basic", "stage1", "stage2", "ex", "megaEx", "tera", "aceSpec"):
            row[o] = float(c[flag])
            o += 1
        atk = [attacks[a] for a in c["attacks"] if a in attacks]
        row[o] = c["hp"] / 300.0
        row[o + 1] = c["retreatCost"] / 4.0
        row[o + 2] = len(atk) / 3.0
        row[o + 3] = max((a["damage"] for a in atk), default=0) / 300.0
        row[o + 4] = min((len(a["energies"]) for a in atk), default=0) / 5.0
        row[o + 5] = float(len(c["skills"]) > 0)
    return table


# Card flags, column META_FLAGS of card_meta_table.
FLAG_EX, FLAG_BASIC, FLAG_ABILITY, FLAG_TERA, FLAG_STEVENS, FLAG_RULE_BOX, FLAG_SPECIAL_ENERGY = (
    1, 2, 4, 8, 16, 32, 64)
# A damage rule: where its card has to be for it to protect the defending (Active) Pokémon ...
ON_SELF, ON_TOOL, ON_IN_PLAY, ON_BENCH, ON_STADIUM = 1, 2, 3, 4, 5
# ... what the attacking Pokémon must be or have (param: a count or a type bit mask) ...
ATT_ANY, ATT_EX, ATT_BASIC_EX, ATT_ABILITY, ATT_TERA, ATT_SPECIAL_ENERGY, ATT_ENERGY_AT_MOST, ATT_TYPE, \
    ATT_TOOL = range(9)
# ... what the defending Pokémon must be or have (param: a type bit mask) ...
DEF_ANY, DEF_TYPE, DEF_ENERGY_TYPE, DEF_NO_RULE_BOX, DEF_STEVENS = range(5)
# ... and what it does to the damage.
NO_DAMAGE, NO_DAMAGE_AT_LEAST, LESS_AFTER_WR, LESS_BEFORE_WR = 1, 2, 3, 4
META_FLAGS, META_RULE = 4, 5  # card_meta columns: flags, then the 7 rule fields
_RULE_FIELDS = 7  # where, attacker condition, its param, defender condition, its param, effect, amount
META_NAME = META_RULE + _RULE_FIELDS  # card_meta: name id, then evolves-from id
META_PREV = META_NAME + 1
META_ATTACKS, N_CARD_ATTACKS = META_PREV + 1, 2  # card_meta: the card's attack ids, 0 for none
# card_meta: what protects Benched Pokémon from attacks (``_BENCH_GUARDS``; zeros for none):
# where the card has to be, what it stops (GUARD_* bits), only Pokémon without a Rule Box, only
# from attackers with at most this much Energy attached (0: any).
META_GUARD = META_ATTACKS + N_CARD_ATTACKS
GUARD_FIELDS = 4
# Where: the Pokémon itself while Benched; any of the owner's Pokémon in play (its Ability covers
# the whole Bench); one of the owner's Benched Pokémon (covers all of the owner's Pokémon); the
# Stadium (both players' Benches); an Energy card attached to the Benched Pokémon.
G_SELF_BENCHED, G_IN_PLAY, G_FROM_BENCH, G_STADIUM, G_ATTACHED = 1, 2, 3, 4, 5
GUARD_DAMAGE, GUARD_COUNTERS = 1, 2  # attack damage; damage counters put on by attack effects

_TYPE_SYMBOLS = {"C": 0, "G": 1, "R": 2, "W": 3, "L": 4, "P": 5, "F": 6, "D": 7, "M": 8, "N": 9}
_TYPES = r"((?:\{\w\}(?:, | or |, or )?)+)"  # "{G}, {R}, {W}, or {L}"


def _type_mask(symbols: str) -> int:
    return sum(1 << _TYPE_SYMBOLS[s.upper()] for s in re.findall(r"\{(\w)\}", symbols))


# Ability, Pokémon Tool and Stadium texts that stop or reduce the damage an attack does to the
# Active Pokémon, as (pattern, where, attacker condition, defender condition, effect). A
# callable condition or effect takes the match and returns (kind, param) / (effect, amount).
# Effects for Benched Pokémon only (Tera, Shaymin, Battle Cage), effects of attacks and
# Supporters for the next turn (not visible in the state) and Bouffalant's Curly Wall (two
# copies needed) are left out.
_DAMAGE_RULES = [
    (r"prevent all damage done to this pokémon by attacks from your opponent's basic pokémon \{ex\}",
     ON_SELF, (ATT_BASIC_EX, 0), (DEF_ANY, 0), (NO_DAMAGE, 0)),
    (r"prevent all damage done to this pokémon by attacks from your opponent's pokémon \{ex\}",
     ON_SELF, (ATT_EX, 0), (DEF_ANY, 0), (NO_DAMAGE, 0)),
    (r"prevent all damage from attacks done to this pokémon by your opponent's pokémon that have an ability",
     ON_SELF, (ATT_ABILITY, 0), (DEF_ANY, 0), (NO_DAMAGE, 0)),
    (r"prevent all damage from and effects of attacks from your opponent's tera pokémon done to this pokémon",
     ON_SELF, (ATT_TERA, 0), (DEF_ANY, 0), (NO_DAMAGE, 0)),
    (r"done to this pokémon by your opponent's pokémon that have any special energy attached",
     ON_SELF, (ATT_SPECIAL_ENERGY, 0), (DEF_ANY, 0), (NO_DAMAGE, 0)),
    (r"prevent all damage done to this pokémon by attacks from your opponent's pokémon if that damage is (\d+) or more",
     ON_SELF, (ATT_ANY, 0), (DEF_ANY, 0), lambda m: (NO_DAMAGE_AT_LEAST, int(m[1]))),
    (r"^this pokémon takes (\d+) less damage from attacks \(after",
     ON_SELF, (ATT_ANY, 0), (DEF_ANY, 0), lambda m: (LESS_AFTER_WR, int(m[1]))),
    (rf"^this pokémon takes (\d+) less damage from attacks from your opponent's {_TYPES} pokémon",
     ON_SELF, lambda m: (ATT_TYPE, _type_mask(m[2])), (DEF_ANY, 0), lambda m: (LESS_AFTER_WR, int(m[1]))),
    (r"as long as this pokémon is in the active spot, attacks used by your opponent's active pokémon do (\d+) less damage",
     ON_SELF, (ATT_ANY, 0), (DEF_ANY, 0), lambda m: (LESS_BEFORE_WR, int(m[1]))),
    (r"as long as this pokémon is in the active spot, all of your pokémon take (\d+) less damage",
     ON_SELF, (ATT_ANY, 0), (DEF_ANY, 0), lambda m: (LESS_AFTER_WR, int(m[1]))),
    (r"^attacks used by your opponent's active pokémon that has a pokémon tool attached do (\d+) less damage",
     ON_IN_PLAY, (ATT_TOOL, 0), (DEF_ANY, 0), lambda m: (LESS_BEFORE_WR, int(m[1]))),
    (r"^all of your pokémon that have any (\{\w\}) energy attached take (\d+) less damage",
     ON_IN_PLAY, (ATT_ANY, 0), lambda m: (DEF_ENERGY_TYPE, _type_mask(m[1])), lambda m: (LESS_AFTER_WR, int(m[2]))),
    (r"on your bench, prevent all damage done to each of your pokémon by attacks from your opponent's pokémon that have (\d+) or less energy",
     ON_BENCH, lambda m: (ATT_ENERGY_AT_MOST, int(m[1])), (DEF_ANY, 0), (NO_DAMAGE, 0)),
    (r"on your bench, all of your steven's pokémon take (\d+) less damage",
     ON_BENCH, (ATT_ANY, 0), (DEF_STEVENS, 0), lambda m: (LESS_AFTER_WR, int(m[1]))),
    (r"^if the pokémon this card is attached to is damaged by an attack from your opponent's (\{\w\}) pokémon, it takes (\d+) less",
     ON_TOOL, lambda m: (ATT_TYPE, _type_mask(m[1])), (DEF_ANY, 0), lambda m: (LESS_AFTER_WR, int(m[2]))),
    (r"^the pokémon this card is attached to takes (\d+) less damage from attacks from your opponent's pokémon that have an ability",
     ON_TOOL, (ATT_ABILITY, 0), (DEF_ANY, 0), lambda m: (LESS_AFTER_WR, int(m[1]))),
    (rf"^the (\{{\w\}}) pokémon this card is attached to takes (\d+) less damage from attacks from your opponent's {_TYPES} pokémon",
     ON_TOOL, lambda m: (ATT_TYPE, _type_mask(m[3])), lambda m: (DEF_TYPE, _type_mask(m[1])),
     lambda m: (LESS_AFTER_WR, int(m[2]))),
    (r"^(\{\w\}) pokémon \(both yours and your opponent's\) take (\d+) less damage",
     ON_STADIUM, (ATT_ANY, 0), lambda m: (DEF_TYPE, _type_mask(m[1])), lambda m: (LESS_AFTER_WR, int(m[2]))),
    (r"^steven's pokémon \(both yours and your opponent's\) take (\d+) less damage",
     ON_STADIUM, (ATT_ANY, 0), (DEF_STEVENS, 0), lambda m: (LESS_AFTER_WR, int(m[1]))),
    (r"^prevent all damage done to pokémon that don't have a rule box \(both yours and your opponent's\) by attacks from the opponent's pokémon \{ex\}",
     ON_STADIUM, (ATT_EX, 0), (DEF_NO_RULE_BOX, 0), (NO_DAMAGE, 0)),
]


# Ability, Stadium and Energy texts that protect Benched Pokémon from attacks (Shaymin's Flower
# Curtain, Rabsca, Battle Cage, ...), as (pattern, where, what it stops, Rule Box-less only,
# attacker Energy at most: an int or the match group holding it).
_BENCH_GUARDS = [
    (r"prevent all damage done to your benched pokémon that don't have a rule box by attacks from your opponent's",
     G_IN_PLAY, GUARD_DAMAGE, 1, 0),
    (r"prevent all damage from and effects of attacks from your opponent's pokémon done to your benched pokémon",
     G_IN_PLAY, GUARD_DAMAGE | GUARD_COUNTERS, 0, 0),
    (r"as long as this pokémon is on your bench, prevent all damage from and effects of attacks",
     G_SELF_BENCHED, GUARD_DAMAGE | GUARD_COUNTERS, 0, 0),
    (r"as long as this pokémon is on your bench, prevent all damage done to this pokémon by attacks",
     G_SELF_BENCHED, GUARD_DAMAGE, 0, 0),
    (r"on your bench, prevent all damage done to each of your pokémon by attacks from your opponent's pokémon "
     r"that have (\d+) or less energy", G_FROM_BENCH, GUARD_DAMAGE, 0, 1),
    (r"prevent all damage counters from being placed on benched pokémon \(both yours and your opponent's\) "
     r"by effects of attacks", G_STADIUM, GUARD_COUNTERS, 0, 0),
    (r"is on your bench, prevent all damage done to it by attacks", G_ATTACHED, GUARD_DAMAGE, 0, 0),
]


def _bench_guard(card: dict) -> tuple[int, ...] | None:
    """The first ``_BENCH_GUARDS`` entry one of the card's texts matches, as the ``GUARD_FIELDS`` fields."""
    for skill in card["skills"]:
        text = re.sub(r"\s+", " ", skill["text"].replace("’", "'")).lower()
        for pattern, where, stops, no_rule_box, energy in _BENCH_GUARDS:
            m = re.search(pattern, text)
            if m:
                return where, stops, no_rule_box, int(m[energy]) if energy else 0
    return None


def _damage_rule(card: dict) -> tuple[int, ...] | None:
    """The first ``_DAMAGE_RULES`` entry one of the card's texts matches, as the 7 rule fields."""
    for skill in card["skills"]:
        text = re.sub(r"\s+", " ", skill["text"].replace("’", "'")).lower()
        for pattern, where, att, dfn, effect in _DAMAGE_RULES:
            m = re.search(pattern, text)
            if m:
                att, dfn, effect = (f(m) if callable(f) else f for f in (att, dfn, effect))
                return (where, *att, *dfn, *effect)
    return None


@lru_cache(maxsize=1)
def card_meta_table() -> np.ndarray:
    """Per card id: [type, Weakness type, Resistance type, Prize cards if Knocked Out, flags,
    damage rule (7 fields), name id, evolves-from id, attack ids (2), Bench guard (4 fields)].

    Types are ``EnergyType`` values, -1 for no Weakness / Resistance; Prize cards
    are 1, 2 for Pokémon ex, 3 for Mega Evolution Pokémon ex. Flags are ``FLAG_*`` bits. The
    damage rule (``_DAMAGE_RULES``; zeros for none) is an Ability, Pokémon Tool or Stadium
    that prevents or reduces attack damage. Used for the attack features in
    ``encoding._tactical``. The name id is the lowest card id with the card's name, so
    reprints compare equal; evolves-from is the name id of the Pokémon it evolves from (0 for
    none). Used for evolution lines in ``encoding.encode``'s deck ledger. Attack ids (0 for
    none) let ``encoding._threat`` work out what each Pokémon in play can hit for. The Bench
    guard (``_BENCH_GUARDS``; zeros for none) protects Benched Pokémon (``encoding._bench_hit``).
    """
    cards = all_cards()
    table = np.full((max(cards) + 1, META_GUARD + GUARD_FIELDS), -1, dtype=np.int16)
    table[:, META_FLAGS:] = 0
    for cid, c in cards.items():
        prizes = 3 if c["megaEx"] else 2 if c["ex"] else 1
        table[cid, :META_FLAGS] = (c["energyType"], -1 if c["weakness"] is None else c["weakness"],
                                   -1 if c["resistance"] is None else c["resistance"], prizes)
        ex = c["ex"] or c["megaEx"]  # a Mega Evolution Pokémon ex is a Pokémon ex
        table[cid, META_FLAGS] = (
            FLAG_EX * ex | FLAG_BASIC * c["basic"] | FLAG_TERA * c["tera"]
            | FLAG_ABILITY * (c["cardType"] == 0 and len(c["skills"]) > 0)
            | FLAG_STEVENS * c["name"].replace("’", "'").startswith("Steven's") | FLAG_RULE_BOX * ex
            | FLAG_SPECIAL_ENERGY * (c["cardType"] == 6))
        rule = _damage_rule(c)
        if rule:
            table[cid, META_RULE:META_NAME] = rule
        guard = _bench_guard(c)
        if guard:
            table[cid, META_GUARD:META_GUARD + GUARD_FIELDS] = guard
    name_id: dict[str, int] = {}
    for cid in sorted(cards):
        name_id.setdefault(cards[cid]["name"], cid)
    for cid, c in cards.items():
        table[cid, META_NAME] = name_id[c["name"]]
        table[cid, META_PREV] = name_id.get(c["evolvesFrom"] or "", 0)
        attacks = list(c["attacks"])[:N_CARD_ATTACKS]
        table[cid, META_ATTACKS:META_ATTACKS + N_CARD_ATTACKS] = attacks + [0] * (N_CARD_ATTACKS - len(attacks))
    return table


def _expected_damage(attack: dict) -> float:
    """The attack's printed damage, or for "Flip 2 coins. This attack does 40 damage for each
    heads." (printed as 0) the expected damage: half a hit per coin, one hit for "until you get
    tails". Coins counted from the board (one per Energy attached, ...) stay 0."""
    if attack["damage"]:
        return float(attack["damage"])
    text = re.sub(r"\s+", " ", attack.get("text") or "").lower()
    m = re.match(r"flip (\d+) coins\. this attack does (\d+) damage for each heads", text)
    if m:
        return int(m[1]) * int(m[2]) / 2
    m = re.match(r"flip a coin until you get tails\. this attack does (\d+) damage for each heads", text)
    return float(m[1]) if m else 0.0


# Conditional extra damage ("If a Stadium is in play, this attack does 60 more damage."), as
# attack_table columns BONUS_KIND, BONUS_AMOUNT (raw damage), BONUS_PARAM; encoding._bonus checks
# the condition on the board. Only conditions the board shows are covered.
BONUS_KIND, BONUS_AMOUNT, BONUS_PARAM = 2, 3, 4
(B_STADIUM, B_DEF_EX, B_PER_DEF_ENERGY, B_DEF_DAMAGED, B_PER_DEF_COUNTER, B_PER_SELF_COUNTER, B_EXTRA_ENERGY,
 B_PER_BENCH_BOTH, B_PER_OPP_BENCH, B_PER_MY_BENCH, B_PER_OPP_PRIZE_TAKEN, B_DEF_EVOLUTION, B_DEF_POISONED,
 B_DEF_BURNED, B_PER_SELF_TYPE_ENERGY, B_SELF_TOOL, B_SELF_SPECIAL_ENERGY, B_OPP_PRIZES_AT_MOST, B_SELF_DAMAGED,
 B_PER_ENERGY_BOTH_ACTIVE, B_COIN) = range(1, 22)
_BONUS_RULES = [
    (r"if a stadium is in play, this attack does (\d+) more damage", B_STADIUM, None),
    (r"if your opponent's active pokémon is a pokémon \{ex\}, this attack does (\d+) more damage", B_DEF_EX, None),
    (r"this attack does (\d+) more damage for each energy attached to your opponent's active pokémon",
     B_PER_DEF_ENERGY, None),
    (r"if your opponent's active pokémon already has any damage counters on it, this attack does (\d+) more damage",
     B_DEF_DAMAGED, None),
    (r"this attack does (\d+) more damage for each damage counter on your opponent's active pokémon",
     B_PER_DEF_COUNTER, None),
    (r"this attack does (\d+) more damage for each damage counter on this pokémon", B_PER_SELF_COUNTER, None),
    (r"if this pokémon has at least (\d+) extra energy attached \(in addition to this attack's cost\), "
     r"this attack does (\d+) more damage", B_EXTRA_ENERGY, "first"),
    (r"this attack does (\d+) more damage for each benched pokémon \(both yours and your opponent's\)",
     B_PER_BENCH_BOTH, None),
    (r"this attack does (\d+) more damage for each of your opponent's benched pokémon", B_PER_OPP_BENCH, None),
    (r"this attack does (\d+) more damage for each of your benched pokémon", B_PER_MY_BENCH, None),
    (r"this attack does (\d+) more damage for each prize card your opponent has taken", B_PER_OPP_PRIZE_TAKEN, None),
    (r"if your opponent's active pokémon is an evolution pokémon, this attack does (\d+) more damage",
     B_DEF_EVOLUTION, None),
    (r"if your opponent's active pokémon is poisoned, this attack does (\d+) more damage", B_DEF_POISONED, None),
    (r"if your opponent's active pokémon is burned, this attack does (\d+) more damage", B_DEF_BURNED, None),
    (r"this attack does (\d+) more damage for each (\{\w\}) energy attached to this pokémon",
     B_PER_SELF_TYPE_ENERGY, "type"),
    (r"if this pokémon has a pokémon tool attached, this attack does (\d+) more damage", B_SELF_TOOL, None),
    (r"if this pokémon has any special energy attached, this attack does (\d+) more damage",
     B_SELF_SPECIAL_ENERGY, None),
    (r"if your opponent has (\d+) or fewer prize cards remaining, this attack does (\d+) more damage",
     B_OPP_PRIZES_AT_MOST, "first"),
    (r"if this pokémon has any damage counters on it, this attack does (\d+) more damage", B_SELF_DAMAGED, None),
    (r"this attack does (\d+) more damage for each energy attached to both active pokémon",
     B_PER_ENERGY_BOTH_ACTIVE, None),
    (r"flip a coin\. if heads, this attack does (\d+) more damage", B_COIN, None),
]


def _damage_bonus(attack: dict) -> tuple[int, int, int]:
    """(kind, amount, param) of the attack's conditional extra damage; zeros for none."""
    text = re.sub(r"\s+", " ", (attack.get("text") or "").replace("’", "'")).lower()
    for pattern, kind, extra in _BONUS_RULES:
        m = re.search(pattern, text)
        if m:
            if extra == "first":  # "at least 2 extra Energy ... 120 more": the param comes first
                return kind, int(m[2]), int(m[1])
            if extra == "type":
                return kind, int(m[1]), _type_mask(m[2])
            return kind, int(m[1]), 0
    return 0, 0, 0


# What an attack does to the opponent's Benched Pokémon, as attack_table columns (raw damage;
# damage counters x10): damage to one of them (a "1 of your opponent's Pokémon" snipe counts),
# damage to each, damage counters on one (or "in any way you like"), damage counters on each.
# Weakness and Resistance never apply to the Bench. Conditions in the text are not checked.
BENCH_ONE, BENCH_EACH, COUNTERS_ONE, COUNTERS_EACH = 5, 6, 7, 8
_N = r"(?:1|2|3|one|two)"
_BENCH_REACH = [
    (r"does (\d+) damage to each of your opponent's benched pokémon(?! for)", BENCH_EACH, 1),
    (r"does (\d+) damage to each benched pokémon \(both yours and your opponent's\)", BENCH_EACH, 1),
    (r"does (\d+) damage to each of your opponent's pokémon(?! for)", BENCH_EACH, 1),
    (rf"does (\d+) damage to {_N} of your opponent's (?:benched )?pokémon(?! for| that)", BENCH_ONE, 1),
    (rf"(?:put|place) (\d+) damage counters? on {_N} of your opponent's (?:benched )?pokémon(?! for)", COUNTERS_ONE, 10),
    (r"(?:put|place) (\d+) damage counters? on your opponent's (?:benched )?pokémon in any way", COUNTERS_ONE, 10),
    (r"(?:put|place) (\d+) damage counters? on each of your opponent's (?:benched )?pokémon", COUNTERS_EACH, 10),
]


def _bench_reach(attack: dict) -> tuple[float, float, float, float]:
    """The attack's (``BENCH_ONE``, ``BENCH_EACH``, ``COUNTERS_ONE``, ``COUNTERS_EACH``) in raw damage."""
    text = re.sub(r"\s+", " ", (attack.get("text") or "").replace("’", "'")).lower()
    out = [0.0] * 4
    for pattern, column, scale in _BENCH_REACH:
        for m in re.finditer(pattern, text):
            out[column - BENCH_ONE] = max(out[column - BENCH_ONE], float(m[1]) * scale)
    return tuple(out)


@lru_cache(maxsize=1)
def attack_table() -> np.ndarray:
    """Per attack id: (damage, energy cost) scaled, damage being the expected damage for coin flips,
    then its conditional extra damage (``BONUS_KIND``, ``BONUS_AMOUNT``, ``BONUS_PARAM``), then what it
    does to the opponent's Bench (``BENCH_ONE`` .. ``COUNTERS_EACH``, raw damage)."""
    attacks = all_attacks()
    table = np.zeros((max(attacks) + 1, 9), dtype=np.float32)
    for aid, a in attacks.items():
        table[aid] = (_expected_damage(a) / 300.0, len(a["energies"]) / 5.0, *_damage_bonus(a), *_bench_reach(a))
    return table


# An attack has a drawback when its text costs us something: damage to itself,
# discarding from itself, our hand or our deck, knocking itself out.
_DRAWBACK = re.compile(r"to itself|from this pok[eé]mon|from your hand|of your deck|"
                       r"this pok[eé]mon is knocked out|shuffle this pok[eé]mon", re.IGNORECASE)


@lru_cache(maxsize=1)
def safe_attack_ids() -> frozenset[int]:
    """Attack ids without a drawback: a turn that offers one is not ended without attacking
    (``encoding.Decision``)."""
    return frozenset(aid for aid, a in all_attacks().items() if not _DRAWBACK.search(a.get("text") or ""))
