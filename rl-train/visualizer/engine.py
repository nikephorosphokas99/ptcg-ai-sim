from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_RL_ENGINE = _HERE.parent / "engine" / "sample_submission"  # the official cg bindings, as in training
if str(_RL_ENGINE) not in sys.path:
    sys.path.insert(0, str(_RL_ENGINE))

import ctypes  # noqa: E402

from cg.game import _get_battle_data, battle_finish, battle_start  # noqa: E402
from cg.api import LogType  # noqa: E402
from cg.sim import Battle as _CgBattle, lib as _lib  # noqa: E402

import render  # noqa: E402
import replay_log  # noqa: E402
from cg.game import visualize_data  # noqa: E402
SELECT_ERRORS = {
    3: "the match is already over",
    4: "wrong number of options chosen for this selection",
    5: "an option index is out of range",
    6: "the same option was chosen twice",
    30: "the battle pointer is broken",
}


def raw_select(indices: list[int]) -> dict:
    arg = (ctypes.c_int * len(indices))(*indices)
    error = _lib.Select(_CgBattle.battle_ptr, arg, len(indices))
    if error:
        raise ValueError(SELECT_ERRORS.get(error, f"the engine refused that selection (code {error})"))
    return _get_battle_data()

DECK_SIZE = 60

DECK_DIR = _HERE / "decks"
DECKS = {
    "Crustle": DECK_DIR / "crustle.csv",
    "Grimmsnarl ex": DECK_DIR / "grimmsnarl_ex.csv",
    "Marnie's Grimmsnarl ex": DECK_DIR / "marnie_grimmsnarl_ex.csv",
    "Alakazam": DECK_DIR / "alakazam.csv",
    "Mega Lucario ex": DECK_DIR / "mega_lucario_ex.csv",
    "Dragapult": DECK_DIR / "dragapult.csv",
    "Mega Excadrill ex": DECK_DIR / "mega_excadrill_ex.csv",
    "Bastiodon / Rampardos ex": DECK_DIR / "bastiodon_rampardos_ex.csv",
    "Iron Thorns ex / Bastiodon": DECK_DIR / "iron_thorns_bastiodon.csv",
    "Mega Greninja ex": DECK_DIR / "mega_greninja_ex.csv",
    "Greninja ex": DECK_DIR / "greninja_ex.csv",
    "Dhelmise Hide 'n' Sneak": DECK_DIR / "dhelmise.csv",
    "Mega Chandelure ex": DECK_DIR / "mega_chandelure_ex.csv",
}

DECK_SAME_CARD_MAX = 4
ACE_SPEC_MAX = 1


def deck_names() -> list[str]:
    return [name for name, path in DECKS.items() if path.is_file()]


def deck_check(cards: list[int], card_db: dict[int, dict]) -> list[str]:
    problems = []
    if len(cards) != DECK_SIZE:
        problems.append(f"{len(cards)}/60 cards")

    by_name: dict[str, int] = {}
    ace_specs = 0
    basics = 0
    for card_id in cards:
        data = card_db.get(card_id)
        if data is None:
            problems.append(f"unknown card id {card_id}")
            continue
        by_name[data["name"]] = by_name.get(data["name"], 0) + 1
        if data.get("aceSpec"):
            ace_specs += 1
        if data.get("basic") and data.get("cardType") == 0:
            basics += 1

    for name, count in sorted(by_name.items()):
        data = next((c for c in card_db.values() if c["name"] == name), None)
        if count > DECK_SAME_CARD_MAX and (data or {}).get("cardType") != 5:
            problems.append(f"{count} copies of {name} (max {DECK_SAME_CARD_MAX})")
    if ace_specs > ACE_SPEC_MAX:
        problems.append(f"{ace_specs} ACE SPEC cards (max {ACE_SPEC_MAX})")
    if not basics:
        problems.append("no Basic Pokémon")
    return problems


def read_deck(path: Path) -> list[int]:
    lines = [line for line in Path(path).read_text().split("\n") if line.strip()]
    if len(lines) != DECK_SIZE:
        raise ValueError(f"{path} has {len(lines)} cards, expected {DECK_SIZE}")
    return [int(line) for line in lines]


def deck_by_name(name: str) -> list[int]:
    if name not in DECKS:
        raise KeyError(name)
    return read_deck(DECKS[name])


START_ERRORS = {
    1: "unknown card id",
    2: "more than four copies of a card",
    3: "no Basic Pokémon in the deck",
    4: "more than one ACE SPEC card",
}


_PRIVATE_AREAS = {1, 2, 6, 12}

_PUBLIC_TYPES = {
    LogType.SHUFFLE, LogType.HAS_BASIC_POKEMON, LogType.TURN_START, LogType.TURN_END,
    LogType.DRAW_REVERSE, LogType.MOVE_CARD_REVERSE, LogType.SWITCH, LogType.CHANGE,
    LogType.PLAY, LogType.ATTACH, LogType.EVOLVE, LogType.DEVOLVE, LogType.MOVE_ATTACHED,
    LogType.ATTACK, LogType.HP_CHANGE, LogType.POISONED, LogType.BURNED, LogType.ASLEEP,
    LogType.PARALYZED, LogType.CONFUSED, LogType.COIN,
}

_REDACT = {LogType.DRAW: LogType.DRAW_REVERSE, LogType.MOVE_CARD: LogType.MOVE_CARD_REVERSE}


def public_logs(logs: list[dict], hidden_seat: int) -> list[dict]:
    keep = []
    for entry in logs:
        kind = entry.get("type")
        if kind == LogType.RESULT or entry.get("playerIndex") != hidden_seat:
            keep.append(entry)
            continue
        if kind == LogType.DRAW:
            keep.append({"type": LogType.DRAW_REVERSE, "playerIndex": entry.get("playerIndex")})
        elif kind == LogType.MOVE_CARD:
            public_destination = entry.get("toArea") not in _PRIVATE_AREAS
            if public_destination:
                keep.append(entry)
            else:
                keep.append({
                    "type": LogType.MOVE_CARD_REVERSE,
                    "playerIndex": entry.get("playerIndex"),
                    "fromArea": entry.get("fromArea"),
                    "toArea": entry.get("toArea"),
                })
        elif kind in _PUBLIC_TYPES:
            keep.append(entry)
    return keep


def _record(recorder, seat: int, choice: list[int], obs: dict) -> None:
    if recorder is None:
        return
    recorder.selection(seat, choice, obs)
    result = obs["current"]["result"]
    if result != -1:
        _save(recorder, result)


def _save(recorder, result: int) -> None:
    if recorder is None or recorder.saved is not None:
        return
    try:
        visualize = json.loads(visualize_data())
    except Exception:  # noqa: BLE001 -- the full-information frames are a bonus
        visualize = None
    try:
        path = recorder.save(result, visualize)
        if path is not None:
            print(f"replay saved: {path}", file=sys.stderr, flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"replay not saved: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)


class Battle:
    def __init__(self, human_deck: list[int], agent_deck: list[int], human_seat: int, agent,
                 names: tuple[str, str] = ("Player", "agent")) -> None:
        self.human_seat = human_seat
        self.agent = agent
        self.agent_seat = 1 - human_seat
        decks = [None, None]
        decks[human_seat] = human_deck
        decks[self.agent_seat] = agent_deck

        obs, start = battle_start(decks[0], decks[1])
        if obs is None:
            reason = START_ERRORS.get(start.errorType, f"error {start.errorType}")
            raise ValueError(f"player {start.errorPlayer}'s deck is illegal: {reason}")

        seat_names = [None, None]
        seat_names[human_seat], seat_names[1 - human_seat] = names
        self.recorder = replay_log.Recorder(seat_names, decks, "vs_agent", obs)
        self.obs = obs
        self.view_obs: dict | None = None
        self.pending_logs: list[dict] = []
        self.pending_steps: list[dict] = []
        self._agent_trail: list[dict] = []
        self._trail_open = False
        self._delivered = 0
        self.finished = False
        self.result = -1
        if hasattr(agent, "reset"):
            agent.reset(0)


    @property
    def acting_seat(self) -> int:
        return self.obs["current"]["yourIndex"]

    def _absorb(self) -> None:
        state = self.obs["current"]
        if state["result"] != -1:
            self.finished = True
            self.result = state["result"]
        logs = self.obs.get("logs") or []

        if self.acting_seat == self.human_seat:
            self.view_obs = self.obs
            self.pending_logs.extend(logs)
            self._agent_trail.clear()
            self._trail_open = False
            return

        if not self._trail_open:
            logs = logs[self._delivered:]
            self._trail_open = True
        self._agent_trail.extend(public_logs(logs, self.agent_seat))
        if self.finished:
            self.pending_logs.extend(self._agent_trail)
            self._agent_trail.clear()
            self._trail_open = False

    def advance(self) -> None:
        shown = self._delivered
        self._absorb()
        first = self.obs.get("logs") or []
        self._snapshot(wrote=max(0, len(first) - shown))
        while not self.finished and self.acting_seat == self.agent_seat:
            select = self.obs["select"]
            if select is None:
                raise RuntimeError("engine asked for a deck mid-battle")
            choice = self.agent.act(self.obs)
            self.obs = raw_select(list(choice))
            _record(self.recorder, self.agent_seat, list(choice), self.obs)
            self._absorb()
            wrote = len(self.obs.get("logs") or []) if self.acting_seat == self.agent_seat else None
            self._snapshot(wrote=wrote)
            self._delivered = 0

    def _snapshot(self, wrote: int | None = None) -> None:
        self.pending_steps.append({
            "board": render.board_snapshot(self.obs, self.human_seat),
            "_wrote": wrote,
        })

    def select(self, choice: list[int]) -> None:
        if self.finished:
            raise RuntimeError("the match is over")
        if self.acting_seat != self.human_seat:
            raise RuntimeError("not your decision")
        select = self.obs["select"]
        low, high = select["minCount"], select["maxCount"]
        if not low <= len(choice) <= high:
            raise ValueError(f"choose between {low} and {high} options")
        if len(set(choice)) != len(choice):
            raise ValueError("duplicate selections")
        if any(not 0 <= i < len(select["option"]) for i in choice):
            raise ValueError("option index out of range")
        self._snapshot(wrote=0)
        self.obs = raw_select(list(choice))
        _record(self.recorder, self.human_seat, list(choice), self.obs)
        self.advance()

    def take_logs(self) -> list[dict]:
        logs, self.pending_logs = self.pending_logs, []
        self._delivered += len(logs)
        return logs

    def take_steps(self, logs: list[dict]) -> list[dict]:
        steps, self.pending_steps = self.pending_steps, []
        if not steps:
            return steps

        counts = [step.pop("_wrote") for step in steps]
        known = sum(n for n in counts if n is not None)
        unknown = [i for i, n in enumerate(counts) if n is None]
        spare = max(0, len(logs) - known)
        for i in unknown:
            counts[i] = spare if i == unknown[-1] else 0

        at = 0
        for step, count in zip(steps, counts):
            slice_ = logs[at:at + count]
            at += count
            step["events"] = render.log_events(slice_, self.human_seat)
        if at < len(logs):
            steps[-1]["events"].extend(render.log_events(logs[at:], self.human_seat))
        return steps

    def close(self) -> None:
        _save(self.recorder, self.result)
        battle_finish()
        if hasattr(self.agent, "close"):
            self.agent.close()


class PvpBattle:
    def __init__(self, decks: list[list[int]], names: list[str] | None = None) -> None:
        obs, start = battle_start(decks[0], decks[1])
        if obs is None:
            reason = START_ERRORS.get(start.errorType, f"error {start.errorType}")
            raise ValueError(f"player {start.errorPlayer}'s deck is illegal: {reason}")
        self.recorder = replay_log.Recorder(names or ["Player 1", "Player 2"], decks, "pvp", obs)
        self.obs = obs
        self.finished = False
        self.result = -1
        self._count = 0
        self._seen = [0, 0]
        self.own_obs: list[dict | None] = [None, None]
        self._logs: list[list[dict]] = [[], []]
        self._steps: list[list[dict]] = [[], []]
        self.version = 0
        self._absorb()

    @property
    def acting_seat(self) -> int:
        return self.obs["current"]["yourIndex"]

    def _absorb(self) -> None:
        self.version += 1
        state = self.obs["current"]
        if state["result"] != -1:
            self.finished = True
            self.result = state["result"]
        acting = self.acting_seat
        logs = self.obs.get("logs") or []
        fresh = logs[max(0, self._count - self._seen[acting]):]
        self._count += len(fresh)
        self._seen[acting] = self._count
        self.own_obs[acting] = self.obs
        for seat in (0, 1):
            mine = fresh if seat == acting else public_logs(fresh, acting)
            self._logs[seat].extend(mine)
            self._steps[seat].append({
                "board": render.board_snapshot(self.obs, seat),
                "events": render.log_events(mine, seat),
            })

    def select(self, seat: int, choice: list[int]) -> None:
        if self.finished:
            raise RuntimeError("the match is over")
        if self.acting_seat != seat:
            raise RuntimeError("not your decision")
        select = self.obs["select"]
        low, high = select["minCount"], select["maxCount"]
        if not low <= len(choice) <= high:
            raise ValueError(f"choose between {low} and {high} options")
        if len(set(choice)) != len(choice):
            raise ValueError("duplicate selections")
        if any(not 0 <= i < len(select["option"]) for i in choice):
            raise ValueError("option index out of range")
        for s in (0, 1):
            self._steps[s].append({"board": render.board_snapshot(self.obs, s), "events": []})
        self.obs = raw_select(list(choice))
        _record(self.recorder, seat, list(choice), self.obs)
        self._absorb()

    def take(self, seat: int) -> tuple[list[dict], list[dict]]:
        logs, self._logs[seat] = self._logs[seat], []
        steps, self._steps[seat] = self._steps[seat], []
        if not any(step["events"] for step in steps):
            steps = []
        return logs, steps

    def close(self) -> None:
        _save(self.recorder, self.result)
        battle_finish()
