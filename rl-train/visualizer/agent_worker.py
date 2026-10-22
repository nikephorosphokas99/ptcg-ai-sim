from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path


def load_agent(bundle: Path):
    for entry in (bundle, bundle / "policy_network"):
        if entry.is_dir():
            sys.path.insert(0, str(entry))
    os.chdir(bundle)
    spec = importlib.util.spec_from_file_location("submission_main", bundle / "main.py")
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(sys.stderr):
        spec.loader.exec_module(module)
    return module.agent


def fallback(obs: dict) -> list[int]:
    select = obs["select"]
    count = min(random.randint(select["minCount"], select["maxCount"]), len(select["option"]))
    return random.sample(range(len(select["option"])), count)


def main() -> None:
    bundle = Path(sys.argv[1]).resolve()
    out = sys.stdout
    try:
        agent = load_agent(bundle)
    except Exception as exc:  # noqa: BLE001
        out.write(json.dumps({"ready": False, "error": f"{type(exc).__name__}: {exc}"}) + "\n")
        out.flush()
        raise
    out.write(json.dumps({"ready": True}) + "\n")
    out.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        obs = json.loads(line)
        try:
            with contextlib.redirect_stdout(sys.stderr):
                action = [int(i) for i in agent(obs)]
            reply = {"action": action}
        except Exception as exc:  # noqa: BLE001 - report and keep the game alive
            reply = {"action": fallback(obs), "error": f"{type(exc).__name__}: {exc}"}
        out.write(json.dumps(reply) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
