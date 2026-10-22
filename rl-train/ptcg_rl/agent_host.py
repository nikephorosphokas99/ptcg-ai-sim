"""Run one competition agent bundle in its own process, speaking JSON lines.

    python -m ptcg_rl.agent_host <bundle dir>

A bundle is a submission folder: ``main.py`` with ``agent(obs) -> list[int]``
and its ``deck.csv``. Agents keep game state in module globals and import ``cg``
by bare name, so each gets a process of its own. The official ``cg`` bindings
are put on the import path for bundles that do not ship them.

Protocol: one JSON observation per stdin line, one reply per stdout line,
``{"action": [...]}`` plus ``"error"`` when the agent raised (the action is then
a random legal one) and ``"seconds"`` taken. The agent's own prints go to stderr.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

OFFICIAL_CG_PARENT = Path(__file__).resolve().parents[1] / "engine" / "sample_submission"
# benchmark.py points this at a compatibility copy of cg (see benchmark.compat_cg).
CG_PARENT = Path(os.environ.get("PTCG_CG_PARENT", OFFICIAL_CG_PARENT))


def load_agent(bundle: Path):
    os.chdir(bundle)  # agents open "deck.csv" relative to the working directory
    sys.path.insert(0, str(bundle))
    if not (bundle / "cg").is_dir():
        sys.path.insert(1, str(CG_PARENT))
    spec = importlib.util.spec_from_file_location("submission_main", bundle / "main.py")
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(sys.stderr):
        spec.loader.exec_module(module)
    return module.agent


def random_legal(obs: dict) -> list[int]:
    select = obs["select"]
    n = len(select["option"])
    return random.sample(range(n), random.randint(select["minCount"], min(select["maxCount"], n)))


def main() -> None:
    out = sys.stdout
    try:
        agent = load_agent(Path(sys.argv[1]).resolve())
    except Exception as exc:  # noqa: BLE001
        out.write(json.dumps({"ready": False, "error": f"{type(exc).__name__}: {exc}"}) + "\n")
        out.flush()
        raise
    out.write(json.dumps({"ready": True}) + "\n")
    out.flush()
    for line in sys.stdin:
        if not line.strip():
            continue
        obs = json.loads(line)
        start = time.perf_counter()
        try:
            with contextlib.redirect_stdout(sys.stderr):
                action = [int(i) for i in agent(obs)]
            reply = {"action": action}
        except Exception as exc:  # noqa: BLE001 - report, keep the game going
            reply = {"action": random_legal(obs) if obs.get("select") else [],
                     "error": f"{type(exc).__name__}: {exc}"}
        reply["seconds"] = time.perf_counter() - start
        out.write(json.dumps(reply) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
