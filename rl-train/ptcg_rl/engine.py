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
    damage rule (7 fields)].

    Types are ``EnergyType`` values, -1 for no Weakness / Resistance; Prize cards
    are 1, 2 for Pokémon ex, 3 for Mega Evolution Pokémon ex. Flags are ``FLAG_*`` bits. The
    damage rule (``_DAMAGE_RULES``; zeros for none) is an Ability, Pokémon Tool or Stadium
    that prevents or reduces attack damage. Used for the attack features in
    ``encoding._tactical``.
    """
    cards = all_cards()
    table = np.full((max(cards) + 1, META_RULE + _RULE_FIELDS), -1, dtype=np.int16)
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
            table[cid, META_RULE:] = rule
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


@lru_cache(maxsize=1)
def attack_table() -> np.ndarray:
    """(damage, energy cost) per attack id, scaled; damage is the expected damage for coin flips."""
    attacks = all_attacks()
    table = np.zeros((max(attacks) + 1, 2), dtype=np.float32)
    for aid, a in attacks.items():
        table[aid] = (_expected_damage(a) / 300.0, len(a["energies"]) / 5.0)
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
