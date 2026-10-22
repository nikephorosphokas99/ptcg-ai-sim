"""Benchmark a submission against the rule-based opponents in ``benchmarks/``.

    cd rl-train
    python -m ptcg_rl.benchmark submissions/first_final.zip
    python -m ptcg_rl.benchmark submissions/first_final.zip --games 200 --opponents crustle_bot venusaur_v1

The agent under test is a submission zip or bundle folder, run exactly as the
competition runs it: its own process, asked for its deck at the start of each
game, then for each of its decisions. Each opponent likewise
(``agent_host.py``). Seats alternate; games run ``--parallel`` at a time, each
pair of agent processes playing its share of the games in turn.

Results go to stdout and ``runs/benchmarks/<agent>_<time>.json``. An agent
error or illegal selection is replaced by a random legal move and counted. A game
still running after ``--max-decisions`` is reported as stalled and left out of
the win rate: so far that has always been an opponent looping in its own turn
(``venusaur_v1`` re-uses an energy-moving Ability forever).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import select
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from .agent_host import random_legal
from .engine import TRAIN_ROOT, Game, all_cards

BENCHMARKS = TRAIN_ROOT / "benchmarks"
OUT_DIR = TRAIN_ROOT / "runs" / "benchmarks"
DECK_REQUEST = {"select": None, "logs": [], "current": None}
OFFICIAL_CG = TRAIN_ROOT / "engine" / "sample_submission" / "cg"


@lru_cache(maxsize=1)
def compat_cg() -> Path:
    """The official cg with ``State.round`` and ``State.draw`` declared; returns its parent.

    The engine emits both fields (``ToJson.h``) but the ``api.py`` shipped beside
    it does not declare them, so ``to_observation_class`` drops them. Agents
    written against the competition's later ``api.py`` read ``obs.current.round``
    (rounds of a best-of-three) and fail on every move without it. Locally the
    engine always sends ``round`` 1 and ``draw`` 0.
    """
    parent = OUT_DIR / "cg_compat"
    target = parent / "cg"
    if not target.exists():
        shutil.copytree(OFFICIAL_CG, target, ignore=shutil.ignore_patterns("__pycache__"))
        api = target / "api.py"
        source = api.read_text(encoding="utf-8-sig")
        anchor = "    players: list[PlayerState]  # An array of player states. The number of elements is 2.\n"
        if source.count(anchor) != 1:
            raise RuntimeError("cg/api.py changed; update benchmark.compat_cg")
        api.write_text(source.replace(anchor, anchor + "    draw: int = 0  # added by benchmark.compat_cg\n"
                                                 "    round: int = 1  # added by benchmark.compat_cg\n"),
                       encoding="utf-8")
    return parent


class AgentProcess:
    def __init__(self, bundle: Path, timeout: float):
        env = {**os.environ, "PTCG_CG_PARENT": str(compat_cg()),
               "PYTHONPATH": os.pathsep.join([str(TRAIN_ROOT), os.environ.get("PYTHONPATH", "")])}
        self.bundle, self.timeout = bundle, timeout
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "ptcg_rl.agent_host", str(bundle)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env,
        )
        hello = json.loads(self._readline(120.0))
        if not hello.get("ready"):
            raise RuntimeError(f"{bundle.name} failed to load: {hello.get('error')}")

    def _readline(self, timeout: float) -> str:
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
        if not ready:
            raise TimeoutError(f"{self.bundle.name} gave no answer within {timeout:.0f}s")
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"{self.bundle.name} exited (code {self.proc.poll()})")
        return line

    def ask(self, obs: dict) -> dict:
        self.proc.stdin.write(json.dumps(obs) + "\n")
        self.proc.stdin.flush()
        return json.loads(self._readline(self.timeout))

    def deck(self) -> list[int]:
        reply = self.ask(DECK_REQUEST)
        deck = reply.get("action") or []
        if len(deck) != 60:  # an agent that cannot answer the request still has its deck.csv
            deck = [int(x) for x in (self.bundle / "deck.csv").read_text().split()[:60]]
        return deck

    def close(self) -> None:
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            self.proc.kill()


@dataclass
class Tally:
    games: int = 0
    wins: float = 0.0
    draws: int = 0
    unfinished: int = 0
    decisions: int = 0
    our_errors: int = 0
    their_errors: int = 0
    our_illegal: int = 0
    their_illegal: int = 0
    our_slowest_s: float = 0.0
    first_errors: list[str] = field(default_factory=list)

    def add(self, other: "Tally") -> None:
        for k in ("games", "wins", "draws", "unfinished", "decisions", "our_errors", "their_errors",
                  "our_illegal", "their_illegal"):
            setattr(self, k, getattr(self, k) + getattr(other, k))
        self.our_slowest_s = max(self.our_slowest_s, other.our_slowest_s)
        self.first_errors = (self.first_errors + other.first_errors)[:5]


def play_game(ours: AgentProcess, theirs: AgentProcess, our_seat: int, max_decisions: int) -> Tally:
    t = Tally(games=1)
    decks = [ours.deck(), theirs.deck()]
    if our_seat == 1:
        decks.reverse()
    game = Game(*decks, search_input=True)  # as Kaggle's cabt observations
    try:
        while not game.done and t.decisions < max_decisions:
            mine = game.to_move == our_seat
            reply = (ours if mine else theirs).ask(game.obs)
            if reply.get("error"):
                setattr(t, "our_errors" if mine else "their_errors", getattr(t, "our_errors" if mine else "their_errors") + 1)
                t.first_errors.append(("ours: " if mine else "theirs: ") + reply["error"])
            if mine:
                t.our_slowest_s = max(t.our_slowest_s, reply.get("seconds", 0.0))
            try:
                game.select(reply["action"])
            except ValueError:
                setattr(t, "our_illegal" if mine else "their_illegal", getattr(t, "our_illegal" if mine else "their_illegal") + 1)
                game.select(random_legal(game.obs))
            t.decisions += 1
        if not game.done:
            t.unfinished = 1
        elif game.result == 2:
            t.draws, t.wins = 1, 0.5
        else:
            t.wins = float(game.result == our_seat)
    finally:
        game.close()
    return t


def run_pair(agent: Path, opponent: Path, games: int, first_seat: int, timeout: float, max_decisions: int) -> Tally:
    total = Tally()
    ours, theirs = AgentProcess(agent, timeout), AgentProcess(opponent, timeout)
    try:
        for g in range(games):
            total.add(play_game(ours, theirs, (first_seat + g) % 2, max_decisions))
    finally:
        ours.close()
        theirs.close()
    return total


def resolve_agent(path: Path) -> Path:
    if path.is_dir():
        return path.resolve()
    target = OUT_DIR / "bundles" / path.stem
    if target.exists() and target.stat().st_mtime >= path.stat().st_mtime:
        return target
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(target)
    return target


def deck_name(bundle: Path) -> str:
    """The deck's headline Pokémon: its most evolved line, then the most copies."""
    cards = all_cards()
    ids = [int(x) for x in (bundle / "deck.csv").read_text().split()[:60]]
    mons = [cards[i] for i in set(ids) if i in cards and cards[i]["cardType"] == 0]
    if not mons:
        return "?"
    stage = lambda c: 2 * c["stage2"] + c["stage1"] + c["megaEx"]  # noqa: E731
    return max(mons, key=lambda c: (stage(c), c["ex"] or c["megaEx"], ids.count(c["cardId"]), c["hp"]))["name"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("agent", type=Path, help="submission .zip or bundle folder")
    p.add_argument("--opponents", nargs="*", help="benchmark folder names (default: all with a deck.csv)")
    p.add_argument("--games", type=int, default=100, help="games per opponent")
    p.add_argument("--parallel", type=int, default=8, help="games in flight per opponent")
    p.add_argument("--timeout", type=float, default=30.0, help="seconds an agent may take per decision")
    p.add_argument("--max-decisions", type=int, default=3000,
                   help="a game this long is reported as stalled and not scored")
    args = p.parse_args()

    agent = resolve_agent(args.agent)
    compat_cg()  # built once, before the agent threads start
    opponents = [d for d in sorted(BENCHMARKS.iterdir()) if (d / "main.py").is_file() and (d / "deck.csv").is_file()]
    if args.opponents:
        opponents = [d for d in opponents if d.name in args.opponents]
    print(f"{args.agent} ({deck_name(agent)}) vs {len(opponents)} rule-based agents, {args.games} games each\n")
    print(f"{'opponent':<20} {'deck':<24} {'win rate':>9} {'±95%':>6} {'scored':>7} {'stalled':>8} "
          f"{'errors us/them':>15} {'slowest move':>13}")

    results, lock = {}, threading.Lock()
    for opp in opponents:
        start = time.time()
        shares = [args.games // args.parallel + (i < args.games % args.parallel) for i in range(args.parallel)]
        with ThreadPoolExecutor(args.parallel) as pool:
            futures = [pool.submit(run_pair, agent, opp, n, i % 2, args.timeout, args.max_decisions)
                       for i, n in enumerate(shares) if n]
            total = Tally()
            for f in futures:
                total.add(f.result())
        scored = total.games - total.unfinished
        rate = total.wins / max(scored, 1)
        ci = 1.96 * math.sqrt(max(rate * (1 - rate), 1e-9) / max(scored, 1))
        with lock:
            results[opp.name] = {**asdict(total), "deck": deck_name(opp), "win_rate": rate, "scored": scored,
                                 "seconds": time.time() - start}
        print(f"{opp.name:<20} {deck_name(opp)[:24]:<24} {rate:>8.1%} {ci:>6.1%} {scored:>7} {total.unfinished:>8} "
              f"{f'{total.our_errors + total.our_illegal}/{total.their_errors + total.their_illegal}':>15} "
              f"{total.our_slowest_s:>12.2f}s", flush=True)
        for e in total.first_errors[:2]:
            print(f"    e.g. {e}")

    overall = (sum(r["wins"] for r in results.values())
               / max(sum(r["games"] - r["unfinished"] for r in results.values()), 1))
    print(f"\n{'overall':<45} {overall:>8.1%}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.agent.stem}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"agent": str(args.agent), "overall": overall, "opponents": results}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
