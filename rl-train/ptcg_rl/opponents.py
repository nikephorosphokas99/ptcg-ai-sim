"""Policies that play the env's other seat.

An opponent maps a full engine observation to the list of option indices the
engine expects (``minCount <= len <= maxCount``, distinct).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Protocol

from .encoding import Decision, encode
from .engine import attack_table


class Opponent(Protocol):
    name: str

    def act(self, obs: dict) -> list[int]: ...


class RandomOpponent:
    name = "random"

    def act(self, obs: dict) -> list[int]:
        sel = obs["select"]
        n = len(sel["option"])
        k = random.randint(sel["minCount"], min(sel["maxCount"], n))
        return random.sample(range(n), k)


class PolicyOpponent:
    """A frozen MaskablePPO checkpoint, sampled stochastically."""

    def __init__(self, path: str | Path, deterministic: bool = False):
        import torch
        from sb3_contrib import MaskablePPO

        # Many of these live in parallel worker processes; one thread each keeps
        # them from oversubscribing the CPU.
        torch.set_num_threads(1)
        self.model = MaskablePPO.load(str(path), device="cpu")
        self.name = Path(path).stem
        self.deterministic = deterministic
        self._attacks = attack_table()

    def act(self, obs: dict) -> list[int]:
        d = Decision(obs)
        while not d.complete:
            action, _ = self.model.predict(
                encode(d, self._attacks), action_masks=d.mask(), deterministic=self.deterministic
            )
            if d.apply(int(action)):
                break
        return d.picked
