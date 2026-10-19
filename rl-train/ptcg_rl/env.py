"""Single-agent Gymnasium env: the learner plays one seat, an opponent the other.

Two-player self-play as a single-agent env is what SB3 expects: the opponent
(random, or a frozen snapshot of the learner) is stepped inside ``step`` until
the engine is waiting on the learner again. Vectorise it with ``SubprocVecEnv``
for parallel games.

Each episode draws a seat (0 or 1), an opponent deck and an opponent from the
pool, so the learner sees both sides of the coin flip and every matchup.

The engine seeds its RNG from ``std::random_device``, so games are not
reproducible from ``reset(seed=...)``; the seed only fixes this env's own
choices (seat, deck, opponent).
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import gymnasium as gym
import numpy as np

from .encoding import N_ACTIONS, Decision, encode, observation_space
from .engine import Game, attack_table, card_table, load_deck
from .opponents import Opponent, PolicyOpponent, RandomOpponent


class PTCGEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        agent_deck: str | Path,
        opponent_decks: list[str | Path],
        opponent_pool: list[str] | None = None,
        prize_reward: float = 0.05,
        skip_forced: bool = True,
        max_steps: int = 1000,
        max_cached_policies: int = 4,
    ):
        """
        Args:
            agent_deck: the learner's 60-card CSV.
            opponent_decks: CSVs the opponent's deck is drawn from each episode.
            opponent_pool: ``"random"`` and/or paths to MaskablePPO ``.zip``
                checkpoints; one is drawn uniformly per episode.
            prize_reward: shaping per Prize card taken (and minus per Prize card
                given up), on top of +1 / -1 for a win / loss.
            skip_forced: play decisions with exactly one legal action
                automatically instead of returning them to the learner.
            max_steps: learner decisions before the episode is truncated.
        """
        super().__init__()
        self.agent_deck = load_deck(agent_deck)
        self.opponent_decks = {Path(p).stem: load_deck(p) for p in opponent_decks}
        self.opponent_pool = list(opponent_pool or ["random"])
        self.prize_reward = prize_reward
        self.skip_forced = skip_forced
        self.max_steps = max_steps
        self.max_cached_policies = max_cached_policies

        self.observation_space = observation_space(len(card_table()) - 1)
        self.action_space = gym.spaces.Discrete(N_ACTIONS)

        self._attacks = attack_table()
        self._random = RandomOpponent()
        self._policies: OrderedDict[str, PolicyOpponent] = OrderedDict()
        self._game: Game | None = None
        self._decision: Decision | None = None

    # -- opponent pool, driven from the trainer via VecEnv.env_method ---------

    def add_opponent(self, spec: str, max_pool: int = 10) -> None:
        """Add a checkpoint to the pool, keeping "random" and the newest snapshots."""
        self.opponent_pool.append(spec)
        snapshots = [s for s in self.opponent_pool if s != "random"]
        if len(snapshots) > max_pool:
            self.opponent_pool.remove(snapshots[0])

    def _load_opponent(self, spec: str) -> Opponent:
        if spec == "random":
            return self._random
        if spec in self._policies:
            self._policies.move_to_end(spec)
        else:
            self._policies[spec] = PolicyOpponent(spec)
            if len(self._policies) > self.max_cached_policies:
                self._policies.popitem(last=False)
        return self._policies[spec]

    # -- gym API --------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """Legal actions for the pending decision (read by MaskablePPO)."""
        return self._decision.mask()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self._game is not None:
            self._game.close()
        rng = self.np_random
        self.seat = int(rng.integers(2))
        self.opp_deck_name = list(self.opponent_decks)[rng.integers(len(self.opponent_decks))]
        self.opponent = self._load_opponent(self.opponent_pool[rng.integers(len(self.opponent_pool))])
        decks = [self.agent_deck, self.opponent_decks[self.opp_deck_name]]
        if self.seat == 1:
            decks.reverse()
        self._game = Game(*decks)
        self._decision = None
        self._steps = 0
        self._advance()
        if self._game.done:  # decided before the learner ever had a choice
            return self.reset()
        return self._obs(), {}

    def step(self, action: int):
        prizes_before = self._prizes()
        if self._decision.apply(int(action)):
            self._submit()
        self._advance()
        self._steps += 1

        # Only decreases count: during setup the prize piles grow from 0 to 6.
        mine, theirs = self._prizes()
        taken = max(0, prizes_before[0] - mine)
        lost = max(0, prizes_before[1] - theirs)
        reward = self.prize_reward * (taken - lost)
        info: dict = {}
        terminated = self._game.done
        if terminated:
            result = self._game.result
            outcome = 0.5 if result == 2 else float(result == self.seat)
            reward += 2.0 * outcome - 1.0
            info = {"outcome": outcome, "opponent": self.opponent.name, "opp_deck": self.opp_deck_name}
            obs = {k: np.zeros(s.shape, s.dtype) for k, s in self.observation_space.items()}
        else:
            obs = self._obs()
        truncated = not terminated and self._steps >= self.max_steps
        return obs, reward, terminated, truncated, info

    def close(self):
        if self._game is not None:
            self._game.close()
            self._game = None

    # -- internals ------------------------------------------------------------

    def _prizes(self) -> tuple[int, int]:
        """Prize cards left (learner, opponent). Fewer left means more taken."""
        players = self._game.obs["current"]["players"]
        return len(players[self.seat]["prize"]), len(players[1 - self.seat]["prize"])

    def _obs(self) -> dict[str, np.ndarray]:
        return encode(self._decision, self._attacks)

    def _submit(self) -> None:
        self._game.select(self._decision.picked)
        self._decision = None

    def _advance(self) -> None:
        """Play the opponent, and forced learner moves, until the learner must choose."""
        game = self._game
        while not game.done:
            if self._decision is None:
                self._decision = Decision(game.obs)
            d = self._decision
            if d.complete:  # e.g. maxCount == 0: nothing to choose
                self._submit()
            elif game.to_move != self.seat:
                game.select(self.opponent.act(game.obs))
                self._decision = None
            elif self.skip_forced and d.mask().sum() == 1:
                if d.apply(int(np.flatnonzero(d.mask())[0])):
                    self._submit()
            else:
                return
