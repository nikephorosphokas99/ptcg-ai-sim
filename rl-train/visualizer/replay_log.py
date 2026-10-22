from __future__ import annotations

import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_HERE = Path(__file__).resolve().parent
REPLAY_DIR = Path(os.environ.get("PTCG_REPLAY_DIR", _HERE / "replays"))
ENABLED = os.environ.get("PTCG_SAVE_REPLAYS", "1") not in ("0", "false", "")
COMPRESSION_LEVEL = 19

SCHEMA = pa.schema([
    ("step", pa.int16()),
    ("seat", pa.int8()),
    ("status", pa.string()),
    ("reward", pa.float32()),
    ("action", pa.list_(pa.int16())),
    ("observation", pa.string()),
    ("visualize", pa.string()),
])

_EMPTY_OBS = {"current": None, "logs": [], "search_begin_input": None, "select": None}


def _compact(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-.")
    return slug[:24] or "player"


class Recorder:
    def __init__(self, names: list[str], decks: list[list[int]], mode: str, first_obs: dict) -> None:
        self.names = list(names)
        self.decks = [list(d) for d in decks]
        self.mode = mode
        self.started = time.time()
        self.episode_id = str(uuid.uuid4())
        self._last: list[dict] = [dict(_EMPTY_OBS), dict(_EMPTY_OBS)]
        self.frames: list[list[dict]] = []
        self.saved: Path | None = None
        self.frames.append([self._record([], dict(_EMPTY_OBS, step=0), "ACTIVE") for _ in (0, 1)])
        self._frame([self.decks[0], self.decks[1]], first_obs)

    def _record(self, action: list[int], obs: dict, status: str) -> dict:
        return {"action": list(action), "info": {}, "observation": obs, "reward": 0, "status": status}

    def _frame(self, actions: list[list[int]], obs: dict) -> None:
        step = len(self.frames)
        acting = obs["current"]["yourIndex"]
        fresh = dict(obs, step=step)
        self._last[acting] = fresh
        self.frames.append([
            self._record(actions[seat], self._last[seat],
                         "ACTIVE" if seat == acting else "INACTIVE")
            for seat in (0, 1)
        ])

    def selection(self, seat: int, choice: list[int], obs: dict) -> None:
        actions: list[list[int]] = [[], []]
        actions[seat] = list(choice)
        self._frame(actions, obs)

    def save(self, result: int, visualize: list | None) -> Path | None:
        if self.saved is not None or not ENABLED:
            return self.saved
        finished = result in (0, 1, 2)
        rewards = [0, 0] if result == 2 else [1 if result == s else -1 for s in (0, 1)] if finished else [None, None]
        last = self.frames[-1]
        for seat in (0, 1):
            last[seat]["status"] = "DONE" if finished else "INCOMPLETE"
            last[seat]["reward"] = rewards[seat]

        columns = {name: [] for name in SCHEMA.names}
        seen: list[str | None] = [None, None]
        for step, frame in enumerate(self.frames):
            for seat, record in enumerate(frame):
                obs = _compact(record["observation"])
                columns["step"].append(step)
                columns["seat"].append(seat)
                columns["status"].append(record["status"])
                columns["reward"].append(record["reward"])
                columns["action"].append(record["action"])
                columns["observation"].append(None if obs == seen[seat] else obs)
                seen[seat] = obs
                v = visualize[step - 1] if visualize and seat == 0 and 1 <= step <= len(visualize) else None
                columns["visualize"].append(_compact(v) if v is not None else None)

        stamp = int(self.started)
        day = datetime.fromtimestamp(stamp, tz=timezone.utc).strftime("%Y-%m-%d")
        header = {
            "id": self.episode_id,
            "name": "cabt",
            "title": "Card Battle",
            "version": "1.0.0",
            "module_version": None,
            "schema_version": 1,
            "description": "Limited Card Battle.",
            "info": {"Agents": [{"Name": n, "ThumbnailUrl": None} for n in self.names],
                     "TeamNames": self.names, "EpisodeId": int(self.started * 1000),
                     "LiveVideoPath": None},
            "rewards": rewards,
            "statuses": [last[0]["status"], last[1]["status"]],
            "configuration": {"episodeSteps": len(self.frames)},
            "simulator": {"mode": self.mode, "decks": self.decks, "started": stamp,
                          "ended": int(time.time()), "finished": finished,
                          "winner": self.names[result] if result in (0, 1) else None},
        }
        table = pa.table(columns, schema=SCHEMA).replace_schema_metadata({"episode": _compact(header)})
        folder = REPLAY_DIR / day
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{_slug(self.names[0])}_vs_{_slug(self.names[1])}_{day}_{stamp}.parquet"
        n = 1
        while path.exists():
            path = path.with_name(f"{path.stem}_{n}.parquet")
            n += 1
        pq.write_table(table, path, compression="zstd", compression_level=COMPRESSION_LEVEL)
        self.saved = path
        return path


def load(path: str | Path) -> dict:
    table = pq.read_table(path)
    header = json.loads(table.schema.metadata[b"episode"])
    rows = table.to_pylist()
    steps: list[list[dict]] = []
    last: list[dict | None] = [None, None]
    visualize = []
    for row in rows:
        seat, step = row["seat"], row["step"]
        if row["observation"] is not None:
            last[seat] = json.loads(row["observation"])
        while len(steps) <= step:
            steps.append([])
        steps[step].append({"action": row["action"], "info": {}, "observation": last[seat],
                            "reward": row["reward"], "status": row["status"]})
        if row["visualize"] is not None:
            frame = json.loads(row["visualize"])
            frame["action"] = [None, None]
            visualize.append((step, frame))
    for step, frame in visualize:
        frame["action"] = [record["action"] for record in steps[step]]
    if visualize:
        steps[0][0]["visualize"] = [frame for _, frame in visualize]
    return dict(header, steps=steps)


def summary(path: str | Path) -> dict:
    return json.loads(pq.read_schema(path).metadata[b"episode"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Show a saved game, or export it as Kaggle replay JSON.")
    parser.add_argument("replay", help="a .parquet file written by the simulator")
    parser.add_argument("--json", metavar="OUT", help="write the Kaggle-format episode JSON here")
    args = parser.parse_args()
    head = summary(args.replay)
    sim = head.get("simulator", {})
    print(f"{' vs '.join(head['info']['TeamNames'])}  ({sim.get('mode')})  "
          f"winner: {sim.get('winner') or ('none, abandoned' if not sim.get('finished') else 'draw')}  "
          f"steps: {head['configuration']['episodeSteps']}")
    if args.json:
        Path(args.json).write_text(json.dumps(load(args.replay)), encoding="utf-8")
        print(f"wrote {args.json}")
