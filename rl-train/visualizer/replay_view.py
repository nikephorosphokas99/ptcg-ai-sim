from __future__ import annotations

import json
from pathlib import Path

import render
import replay_log


def _replay_files() -> list[Path]:
    root = replay_log.REPLAY_DIR
    if not root.is_dir():
        return []
    files = [p for p in root.glob("*/*") if p.suffix in (".parquet", ".json")]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def resolve(name: str) -> Path:
    root = replay_log.REPLAY_DIR.resolve()
    path = (root / name).resolve()
    if root not in path.parents or not path.is_file() or path.suffix not in (".parquet", ".json"):
        raise FileNotFoundError(name)
    return path


def _header(path: Path) -> dict:
    if path.suffix == ".parquet":
        return replay_log.summary(path)
    episode = json.loads(path.read_text(encoding="utf-8"))
    episode.pop("steps", None)
    return episode


def listing() -> list[dict]:
    out = []
    root = replay_log.REPLAY_DIR.resolve()
    for path in _replay_files():
        try:
            head = _header(path)
        except (OSError, ValueError, KeyError):
            continue
        sim = head.get("simulator", {})
        names = head.get("info", {}).get("TeamNames") or ["?", "?"]
        rewards = head.get("rewards") or [None, None]
        winner = sim.get("winner")
        if winner is None and rewards and rewards[0] is not None and rewards[0] != rewards[1]:
            winner = names[0] if rewards[0] > rewards[1] else names[1]
        out.append({
            "file": str(path.resolve().relative_to(root)),
            "names": names,
            "winner": winner,
            "finished": sim.get("finished", head.get("statuses", [None])[0] == "DONE"),
            "mode": sim.get("mode", "kaggle"),
            "started": sim.get("started") or int(path.stat().st_mtime),
            "steps": head.get("configuration", {}).get("episodeSteps"),
        })
    return out


def _load(path: Path) -> dict:
    if path.suffix == ".parquet":
        return replay_log.load(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _hand(frame: dict | None, seat: int) -> list[dict] | None:
    if not frame:
        return None
    try:
        cards = frame["current"]["players"][seat].get("hand")
    except (KeyError, IndexError, TypeError):
        return None
    if cards is None:
        return None
    return [render.card_info(c["id"]) for c in cards if c]


def _decision(select: dict, state: dict, seat: int, action: list[int]) -> str:
    labels = []
    for i in action:
        if 0 <= i < len(select.get("option") or []):
            labels.append(render.option_label(select["option"][i], state, seat, select))
    chosen = ", ".join(labels) or "(nothing)"
    prompt = render.PROMPT.get(select.get("context"), "")
    return f"{prompt} → {chosen}" if prompt else chosen


def frames(path: Path, seat: int) -> dict:
    episode = _load(path)
    steps = episode["steps"]
    names = episode.get("info", {}).get("TeamNames") or ["Player 1", "Player 2"]
    visualize = (steps[0][0].get("visualize") if steps and steps[0] else None) or []
    other = 1 - seat

    count = 0
    seen = [0, 0]
    own: list[dict | None] = [None, None]
    previous: list[str | None] = [None, None]
    history: list[str] = []
    out = []
    for k in range(1, len(steps)):
        frame = steps[k]
        actions = [r.get("action") or [] for r in frame]
        mover = None
        decision = ""
        if k == 1:
            decision = "Both players submitted their decks"
        else:
            for s in (0, 1):
                if actions[s] and own[s] is not None and own[s].get("select"):
                    mover = s
                    decision = _decision(own[s]["select"], own[s]["current"], s, actions[s])
        fresh_logs: list[dict] = []
        board_obs = None
        for s in (0, 1):
            obs = frame[s].get("observation") or {}
            if obs.get("current") is None:
                continue
            text = json.dumps({k: v for k, v in obs.items() if k not in ("step", "remainingOverageTime")},
                              sort_keys=True)
            if text == previous[s]:
                continue
            previous[s] = text
            logs = obs.get("logs") or []
            new = logs[max(0, count - seen[s]):]
            count += len(new)
            seen[s] = count
            fresh_logs.extend(new)
            own[s] = obs
            board_obs = obs
        if board_obs is None:
            continue
        result = board_obs["current"].get("result", -1)
        final = frame[0].get("status") == "DONE"
        if final and result == -1:
            rewards = [frame[s].get("reward") for s in (0, 1)]
            if rewards[0] is not None and rewards[1] is not None:
                result = 2 if rewards[0] == rewards[1] else (0 if rewards[0] > rewards[1] else 1)
        view = render.build_view(board_obs, seat, names[other], [], your_turn=False, result=result)
        vis = visualize[k - 1] if 0 <= k - 1 < len(visualize) else None
        mine = _hand(vis, seat)
        theirs = _hand(vis, other)
        if mine is None and own[seat] is not None:
            mine = render.player_view(own[seat]["current"]["players"][seat], True)["hand"]
        if theirs is None and own[other] is not None:
            theirs = render.player_view(own[other]["current"]["players"][other], True)["hand"]
        view["me"]["hand"] = mine or []
        view["opp"]["hand"] = theirs or []
        lines = render.log_lines(fresh_logs, seat)
        history.extend(lines)
        out.append({
            "board": {key: view[key] for key in ("me", "opp", "stadium", "looking", "turn", "actions", "flags")},
            "events": render.log_events(fresh_logs, seat),
            "lines": lines,
            "mover": names[mover] if mover is not None else None,
            "moverSeat": mover,
            "decision": decision,
            "result": result,
            "raw": {"step": k, "action": actions, "observation": board_obs},
        })
    last = steps[-1] if steps else []
    if out and out[-1]["result"] == -1 and last and all(r.get("status") == "DONE" for r in last):
        rewards = [r.get("reward") for r in last]
        if None not in rewards:
            out[-1]["result"] = 2 if rewards[0] == rewards[1] else (0 if rewards[0] > rewards[1] else 1)
    return {
        "names": names,
        "seat": seat,
        "frames": out,
        "winner": next((f["result"] for f in reversed(out) if f["result"] != -1), -1),
    }

