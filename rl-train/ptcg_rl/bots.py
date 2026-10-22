"""Rule-based competition agents as in-process training opponents.

The bundles in ``benchmarks/`` are competitors' rule-based agents: ``main.py``
with ``agent(obs)`` and the ``deck.csv`` it plays. Training workers run them
directly on the engine they already host, with three constraints:

* ``cg.sim`` calls ``GameInitialize()`` on import, and a second initialisation
  in a process that already runs battles crashes the engine (``buffer full``).
  So the bots' ``cg`` gets a stand-in ``cg.sim`` that reuses this process's
  engine (``engine.lib()``).
* Bots keep per-game state in module globals, so every game loads a fresh copy
  of the module.
* Their ``cg.api`` must declare ``State.round`` / ``State.draw`` (see
  ``benchmark.compat_cg``), passed in as ``cg_parent``.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import itertools
import os
import random
import sys
import types
from pathlib import Path
from typing import Callable

from . import engine

_counter = itertools.count()


def _install_cg(cg_parent: Path) -> None:
    if "cg.sim" in sys.modules and getattr(sys.modules["cg.sim"], "_ptcg_stub", False):
        return
    if str(cg_parent) not in sys.path:
        sys.path.insert(0, str(cg_parent))
    stub = types.ModuleType("cg.sim")
    stub._ptcg_stub = True
    stub.lib = engine.lib()  # already initialised; never call GameInitialize again
    stub.StartData = engine.StartData
    stub.SerialData = engine.SerialData

    class Battle:  # cg.game's singleton; bots read only the module, not a live battle
        battle_ptr = None
        obs = None

    stub.Battle = Battle
    sys.modules["cg.sim"] = stub
    import cg  # noqa: F401  (the package from cg_parent, so cg.api resolves there)
    import cg.api  # noqa: F401


def load_bot(bundle: Path, cg_parent: Path) -> Callable[[dict], list[int]]:
    """A fresh instance of the bundle's ``agent`` (new module globals)."""
    _install_cg(cg_parent)
    name = f"_ptcg_bot_{bundle.name}_{next(_counter)}"
    spec = importlib.util.spec_from_file_location(name, bundle / "main.py")
    module = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    try:
        os.chdir(bundle)  # bots open "deck.csv" relative to the working directory
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(module)
    finally:
        os.chdir(cwd)
    return module.agent


def bot_deck(bundle: Path) -> list[int]:
    return engine.load_deck(bundle / "deck.csv")


def random_legal(obs: dict) -> list[int]:
    select = obs["select"]
    n = len(select["option"])
    return random.sample(range(n), random.randint(select["minCount"], min(select["maxCount"], n)))


def bot_move(agent: Callable[[dict], list[int]], obs: dict) -> tuple[list[int], bool]:
    """(selection, whether the bot failed and a random legal move stands in)."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            picks = [int(i) for i in agent(obs)]
        n, select = len(obs["select"]["option"]), obs["select"]
        if (len(set(picks)) == len(picks) and all(0 <= i < n for i in picks)
                and select["minCount"] <= len(picks) <= select["maxCount"]):
            return picks, False
    except Exception:  # noqa: BLE001 - a broken bot move must not end training
        pass
    return random_legal(obs), True
