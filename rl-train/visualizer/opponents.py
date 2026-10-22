from __future__ import annotations

import json
import select
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
AGENT_DIR = _HERE / "agents"

START_TIMEOUT = 180.0
MOVE_TIMEOUT = 60.0


def _deck_ok(path: Path) -> bool:
    try:
        lines = [line for line in path.read_text().split("\n") if line.strip()]
    except OSError:
        return False
    return len(lines) == 60


def discover_bundles() -> dict[str, Path]:
    found: dict[str, Path] = {}
    if not AGENT_DIR.is_dir():
        return found
    for bundle in sorted(p for p in AGENT_DIR.iterdir() if p.is_dir()):
        if not (bundle / "main.py").is_file():
            continue
        if not _deck_ok(bundle / "deck.csv"):
            continue
        if not any(bundle.glob("*.pt")) and not any(bundle.glob("*.npz")):
            continue
        found[bundle.name] = bundle
    return found


def available_agents() -> list[str]:
    return list(discover_bundles())


def agent_arch(bundle: Path) -> dict:
    path = bundle / "ARCH.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def agent_deck(spec: str) -> Path:
    bundles = discover_bundles()
    if spec not in bundles:
        raise KeyError(spec)
    return bundles[spec] / "deck.csv"


class BundleAgent:
    def __init__(self, spec: str, bundle: Path) -> None:
        self.name = spec
        self.bundle = bundle
        self.last_error: str | None = None
        self.process = subprocess.Popen(
            [sys.executable, str(_HERE / "agent_worker.py"), str(bundle)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1, cwd=str(bundle),
        )
        line = self._read(START_TIMEOUT, "loading")
        hello = json.loads(line)
        if not hello.get("ready"):
            self.close()
            raise RuntimeError(f"{spec} failed to load: {hello.get('error')}")

    def _read(self, timeout: float, what: str) -> str:
        ready, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not ready:
            self.close()
            raise RuntimeError(f"{self.name} stopped responding while {what} "
                               f"(no answer in {timeout:.0f}s)")
        line = self.process.stdout.readline()
        if not line:
            code = self.process.poll()
            self.close()
            raise RuntimeError(f"{self.name} exited while {what} (code {code}) — "
                               f"its own error is on the server's stderr")
        return line

    def reset(self, episode: int) -> None:
        pass

    def act(self, obs: dict) -> list[int]:
        if self.process.poll() is not None:
            raise RuntimeError(f"{self.name} exited (code {self.process.returncode})")
        self.process.stdin.write(json.dumps(obs) + "\n")
        self.process.stdin.flush()
        reply = json.loads(self._read(MOVE_TIMEOUT, "choosing a move"))
        self.last_error = reply.get("error")
        return reply["action"]

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.process.stdin.close()
            except OSError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


def build_agent(spec: str) -> BundleAgent:
    bundles = discover_bundles()
    if spec not in bundles:
        raise KeyError(f"unknown opponent {spec}")
    return BundleAgent(spec, bundles[spec])
