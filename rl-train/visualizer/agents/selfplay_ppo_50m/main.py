"""Submission entry point: ``agent(obs) -> list[int]``.

The self-play MaskablePPO policy from ``rl-train``, decoded greedily one pick at
a time exactly as it was trained (``Decision``: pick option indices, then STOP
once ``minCount`` is met). Built by ``python -m ptcg_rl.export``.

Two properties of the harness shape this file:

* kaggle_environments may exec this source against a bare globals dict, with no
  ``__file__``; the bundle directory then falls back to /kaggle_simulations/agent
  or the working directory.
* A crash forfeits the match, so any failure falls back to a random legal
  selection: a weak move scores far better than no move.
"""

import os
import random
import sys
from pathlib import Path

try:
    _ROOT = Path(__file__).resolve().parent
except NameError:
    _ROOT = Path("/kaggle_simulations/agent")
    if not _ROOT.is_dir():
        _ROOT = Path.cwd()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _resolve(filename: str) -> Path:
    for base in (_ROOT, Path("/kaggle_simulations/agent"), Path.cwd()):
        if (base / filename).exists():
            return base / filename
    return _ROOT / filename


def read_deck_csv() -> list[int]:
    lines = _resolve("deck.csv").read_text().split()
    return [int(lines[i]) for i in range(60)]


class _Policy:
    """Loaded on the first real decision: the first call only asks for a deck."""

    def __init__(self) -> None:
        self.net = None

    def _load(self) -> None:
        import torch

        from ptcg_policy.net import PolicyNet

        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        ckpt = torch.load(_resolve("policy.pt"), map_location="cpu", weights_only=True)
        net = PolicyNet(ckpt["card_static"], ckpt["arch"])
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        self.attack_table = ckpt["attack_table"].numpy()
        self.net = net

    def act(self, obs: dict) -> list[int]:
        import torch

        from ptcg_policy.encoding import Decision, encode

        if self.net is None:
            self._load()
        decision = Decision(obs)
        with torch.inference_mode():
            while not decision.complete:
                features = encode(decision, self.attack_table)
                batch = {k: torch.as_tensor(v).unsqueeze(0).float() for k, v in features.items()}
                logits = self.net(batch)[0]
                mask = torch.as_tensor(decision.mask())
                action = int(logits.masked_fill(~mask, float("-inf")).argmax())
                if decision.apply(action):
                    break
        return decision.picked


_POLICY = _Policy()


def _random_selection(obs: dict) -> list[int]:
    select = obs["select"]
    n = len(select["option"])
    return random.sample(range(n), min(select["maxCount"], n))


def agent(obs_dict: dict) -> list[int]:
    """Return the deck on the first call, then option indices for each decision.

    Each returned list holds distinct indices, ``0 <= i < len(options)``, and
    its length is within ``[minCount, maxCount]``.
    """
    if obs_dict.get("select") is None:
        return read_deck_csv()
    try:
        return _POLICY.act(obs_dict)
    except Exception as error:  # never forfeit over a bug
        print(f"agent fallback to random: {error!r}", file=sys.stderr)
        return _random_selection(obs_dict)
