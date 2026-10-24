"""Submission entry point: ``agent(obs) -> list[int]``.

The self-play MaskablePPO policy from ``rl-train``, decoded greedily one pick at
a time exactly as it was trained (``Decision``: pick option indices, then STOP
once ``minCount`` is met). Built by ``python -m ptcg_rl.export``. When the
bundle has a ``rules.json``, the decisions training plays by rule are played
the same way here: damage counters "in any way you like" by
``CounterPlanner``, and no ending a turn that offers a drawback-free attack.

How kaggle_environments loads this file (``agent.get_last_callable``): it
appends the bundle directory to ``sys.path``, execs the source in a bare dict
(no ``__file__``), then pops that directory off ``sys.path`` again, and calls
the last callable defined. So everything from the bundle is imported here, at
load time, from a path entry of our own; a lazy import at the first decision
found the directory gone (``No module named 'ptcg_policy'``). And ``agent``
must stay the last thing defined.

There is deliberately no fallback move: if the policy cannot run, the error
surfaces (the harness marks the agent ERROR) instead of being hidden behind
random play.
"""

import sys
from collections import deque
from pathlib import Path

try:
    _ROOT = Path(__file__).resolve().parent
except NameError:
    # The loader's own sys.path entry (appended last) is still present while this runs.
    _candidates = [Path(p) for p in reversed(sys.path) if p]
    _candidates += [Path("/kaggle_simulations/agent"), Path.cwd()]
    _ROOT = next((c for c in _candidates if (c / "ptcg_policy").is_dir()), Path("/kaggle_simulations/agent"))
sys.path.insert(0, str(_ROOT))  # unconditionally: the loader removes its entry after this file runs

import json  # noqa: E402

import numpy as np  # noqa: E402

from ptcg_policy.encoding import (  # noqa: E402
    RECENT_SLOTS, CounterPlanner, Decision, encode, recent_opponent_cards, visible_cards)
from ptcg_policy.net import load  # noqa: E402

# ISMCTS bundles (``export --ismcts``) ship ismcts.json and search with the engine of
# the bundled ``cg``, imported now while the bundle directory is still on sys.path.
_ISMCTS_CONFIG = next((b / "ismcts.json" for b in (_ROOT, Path("/kaggle_simulations/agent"), Path.cwd())
                       if (b / "ismcts.json").exists()), None)
if _ISMCTS_CONFIG is not None:
    import cg.sim  # noqa: E402  (loads and initialises the engine for SearchBegin / SearchStep)
    from ptcg_policy.ismcts import ISMCTS, EngineSearch  # noqa: E402


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
        self.planner = None
        self.new_game()

    def new_game(self) -> None:
        """Forget the last game: opponent cards seen so far feed the policy's memory."""
        self.opp_seen: set[int] = set()
        self.opp_recent: deque[int] = deque(maxlen=RECENT_SLOTS)
        self.last_turn = -1
        if self.planner is not None:
            self.planner.reset()

    def _load(self) -> None:
        self.net, self.attack_table, self.card_meta = load(_resolve("policy.npz").parent)
        self.deck = read_deck_csv()
        # rules.json (written by export): the decisions training plays by rule rather
        # than by policy, so the agent plays them here exactly as it trained.
        rules_path = _resolve("rules.json")
        rules = json.loads(rules_path.read_text()) if rules_path.exists() else {}
        self.safe_attacks = frozenset(rules["safe_attacks"]) if "safe_attacks" in rules else None
        self.planner = CounterPlanner(self.card_meta) if rules.get("counter_planner") else None
        self.search = None
        if _ISMCTS_CONFIG is not None:
            self.search = ISMCTS(self.net, self.attack_table, self.card_meta, self.deck,
                                 json.loads(_resolve("opp_decks.json").read_text()),
                                 EngineSearch(cg.sim.lib), json.loads(_ISMCTS_CONFIG.read_text()))

    def act(self, obs: dict) -> list[int]:
        if self.net is None:
            self._load()
        state = obs["current"]
        if state["turn"] < self.last_turn:  # a new game without a deck request in between
            self.new_game()
        self.last_turn = state["turn"]
        # Every observation feeds the recent-events memory, forced decisions included,
        # exactly as training's Match does.
        self.opp_recent.extend(recent_opponent_cards(obs, state["yourIndex"]))
        planned = self.planner.pick(obs) if self.planner is not None else None
        if planned is not None:
            return [planned]
        decision = Decision(obs, self.safe_attacks)
        while not decision.complete:
            mask = decision.mask()
            if mask.sum() == 1:  # forced: training plays these without a look
                if decision.apply(int(mask.argmax())):
                    break
                continue
            # Training updates the memory only when a seat faces a real choice.
            self.opp_seen.update(visible_cards(state, 1 - state["yourIndex"]))
            features = encode(decision, self.attack_table, my_deck=self.deck, opp_seen=self.opp_seen,
                              opp_recent=self.opp_recent, card_meta=self.card_meta)
            logits = self.net({k: v[None].astype(np.float32) for k, v in features.items()})[0]
            action = int(np.where(mask, logits, -np.inf).argmax())
            if self.search is not None:
                searched = self.search.choose(obs, decision, self.opp_seen, self.opp_recent)
                action = action if searched is None else searched
            if decision.apply(action):
                break
        return decision.picked


_POLICY = _Policy()


def agent(obs_dict: dict) -> list[int]:
    """Return the deck on the first call, then option indices for each decision.

    Each returned list holds distinct indices, ``0 <= i < len(options)``, and
    its length is within ``[minCount, maxCount]``.
    """
    if obs_dict.get("select") is None:
        _POLICY.new_game()
        return read_deck_csv()
    return _POLICY.act(obs_dict)
