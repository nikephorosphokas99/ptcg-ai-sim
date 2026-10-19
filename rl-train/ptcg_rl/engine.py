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
import os
import platform
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


class Game:
    """One battle. ``obs`` is always the engine's current observation dict."""

    def __init__(self, deck0: list[int], deck1: list[int]):
        if len(deck0) != 60 or len(deck1) != 60:
            raise ValueError("decks must contain 60 cards")
        cards = (ctypes.c_int * 120)(*(deck0 + deck1))
        start = lib().BattleStart(cards)
        if not start.battlePtr:
            raise ValueError(
                f"engine rejected deck of player {start.errorPlayer} (errorType {start.errorType})"
            )
        self._ptr = start.battlePtr
        self.obs = self._fetch()

    def _fetch(self) -> dict:
        sd = lib().GetBattleData(self._ptr)
        return json.loads(sd.json.decode())

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
        err = lib().Select(self._ptr, arg, len(picks))
        if err != 0:
            raise ValueError(f"engine rejected selection {picks} (error {err})")
        self.obs = self._fetch()
        return self.obs

    def close(self) -> None:
        if self._ptr:
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


@lru_cache(maxsize=1)
def attack_table() -> np.ndarray:
    """(damage, energy cost) per attack id, scaled."""
    attacks = all_attacks()
    table = np.zeros((max(attacks) + 1, 2), dtype=np.float32)
    for aid, a in attacks.items():
        table[aid] = (a["damage"] / 300.0, len(a["energies"]) / 5.0)
    return table
