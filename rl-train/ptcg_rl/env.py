"""One game between the learner and an opponent, stepped one decision at a time.

``Match`` knows nothing about policies: ``step`` applies an action for
whichever seat the engine is waiting on and reports whether the learner moves
next. ``SelfPlayCollector`` decides for both seats, batched on the GPU, so the
worker processes that host matches never import torch.

Each game draws a seat (0 or 1) and an opponent deck, so the learner sees both
sides of the coin flip. With no opponent decks the opponent pilots the learner's
own deck (mirror match).

The engine seeds its RNG from ``std::random_device``, so games are not
reproducible; ``seed`` only fixes this match's own choices (seat, deck).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .bots import bot_deck, bot_move, load_bot, random_legal
from .encoding import RECENT_SLOTS, Decision, encode, recent_opponent_cards, visible_cards
from .engine import Game, attack_table, card_meta_table, load_deck


@dataclass
class Turn:
    """The pending decision: its encoding, legal actions, and whose it is.

    ``learner`` is True when the match's learner seat (``Match.seat``) is to
    move; the other seat is the opponent, whatever policy plays it.
    """

    obs: dict[str, np.ndarray]
    mask: np.ndarray
    learner: bool


@dataclass
class StepResult:
    reward: float  # always from the learner's point of view
    done: bool = False
    truncated: bool = False
    turn: Turn | None = None  # None once the game is over
    info: dict = field(default_factory=dict)


STALL_DECISIONS = 2000  # decisions without the learner choosing: a looping bot, end the game


class Match:
    def __init__(
        self,
        agent_deck: str | Path,
        opponent_decks: list[str | Path] | None = None,
        prize_reward: float = 0.05,
        skip_forced: bool = True,
        max_steps: int = 1000,
        bots: list[str] | None = None,
        bot_frac: float = 0.0,
        bot_cg: str | None = None,
        seed: int | None = None,
    ):
        """
        Args:
            agent_deck: the learner's 60-card CSV.
            opponent_decks: CSVs the opponent's deck is drawn from each game;
                empty or None means a mirror match with ``agent_deck``.
            prize_reward: shaping per Prize card taken (and minus per Prize card
                given up), on top of +1 / -1 for a win / loss.
            skip_forced: play decisions with exactly one legal action
                automatically, for either seat.
            max_steps: learner decisions before the game is truncated.
            bots: rule-based agent bundles (``benchmarks/<name>``); with
                probability ``bot_frac`` a game's opponent is one of them,
                playing its own deck, moved here without involving the trainer.
            bot_cg: parent folder of the ``cg`` the bots import (``benchmark.compat_cg``).
        """
        self.agent_deck = load_deck(agent_deck)
        self.opponent_decks = {Path(p).stem: load_deck(p) for p in opponent_decks or []}
        if not self.opponent_decks:
            self.opponent_decks = {"mirror": self.agent_deck}
        self.prize_reward = prize_reward
        self.skip_forced = skip_forced
        self.max_steps = max_steps
        self.bots = [Path(b) for b in bots or []]
        self.bot_frac = bot_frac if self.bots else 0.0
        self.bot_cg = Path(bot_cg) if bot_cg else None
        self._bot_decks = {b.name: bot_deck(b) for b in self.bots}
        self._rng = np.random.default_rng(seed)
        self._attacks = attack_table()
        self._meta = card_meta_table()
        self._game: Game | None = None
        self._decision: Decision | None = None

    def reset(self) -> Turn:
        self.close()
        self.seat = int(self._rng.integers(2))
        self.bot, self.bot_name, self.bot_failures = None, None, 0
        if self.bot_frac and self._rng.random() < self.bot_frac:
            bundle = self.bots[self._rng.integers(len(self.bots))]
            self.bot, self.bot_name = load_bot(bundle, self.bot_cg), bundle.name
            self.opp_deck_name, opp_deck = f"bot:{bundle.name}", self._bot_decks[bundle.name]
        else:
            self.opp_deck_name = list(self.opponent_decks)[self._rng.integers(len(self.opponent_decks))]
            opp_deck = self.opponent_decks[self.opp_deck_name]
        decks = [self.agent_deck, opp_deck]
        if self.seat == 1:
            decks.reverse()
        self._decks = decks
        self._seen: list[set[int]] = [set(), set()]  # opponent cards each seat has seen
        self._recent = [deque(maxlen=RECENT_SLOTS), deque(maxlen=RECENT_SLOTS)]
        self._game = Game(*decks)
        self._observe()
        self._decision = None
        self._steps = 0
        self._since_learner = 0
        self._advance()
        if self._game.done or self._stalled:  # decided before the learner had a choice
            return self.reset()
        return self._turn()

    def step(self, action: int) -> StepResult:
        learner_moved = self._game.to_move == self.seat
        prizes_before = self._prizes()
        if self._decision.apply(int(action)):
            self._select(self._decision.picked)
            self._decision = None
        self._since_learner = 0
        self._advance()
        self._steps += learner_moved

        # Only decreases count: during setup the prize piles grow from 0 to 6.
        mine, theirs = self._prizes()
        taken = max(0, prizes_before[0] - mine)
        lost = max(0, prizes_before[1] - theirs)
        reward = self.prize_reward * (taken - lost)

        info = {"opp_deck": self.opp_deck_name, "bot": self.bot_name, "bot_failures": self.bot_failures}
        if self._stalled:  # a bot looping in its own turn: end the game, score nothing
            return StepResult(reward, done=True, truncated=True, info={**info, "stalled": True})
        if self._game.done:
            result = self._game.result
            outcome = 0.5 if result == 2 else float(result == self.seat)
            return StepResult(reward + 2.0 * outcome - 1.0, done=True, info={**info, "outcome": outcome})
        turn = self._turn()
        if self._steps >= self.max_steps:
            return StepResult(reward, done=True, truncated=True, turn=turn, info=info)
        return StepResult(reward, turn=turn)

    def close(self) -> None:
        if self._game is not None:
            self._game.close()
            self._game = None

    def _turn(self) -> Turn:
        d = self._decision
        mover = self._game.to_move
        self._seen[mover].update(visible_cards(self._game.obs["current"], 1 - mover))
        obs = encode(d, self._attacks, my_deck=self._decks[mover], opp_seen=self._seen[mover],
                     opp_recent=self._recent[mover], card_meta=self._meta)
        return Turn(obs, d.mask(), mover == self.seat)

    def _select(self, picks: list[int]) -> None:
        self._game.select(picks)
        self._observe()

    def _observe(self) -> None:
        """Every observation a seat receives feeds that seat's recent-events memory,
        forced decisions included: the submission sees those too."""
        obs = self._game.obs
        if obs.get("current") is not None and not self._game.done:
            me = obs["current"]["yourIndex"]
            self._recent[me].extend(recent_opponent_cards(obs, me))

    def _prizes(self) -> tuple[int, int]:
        """Prize cards left (learner, opponent). Fewer left means more taken."""
        players = self._game.obs["current"]["players"]
        return len(players[self.seat]["prize"]), len(players[1 - self.seat]["prize"])

    @property
    def _stalled(self) -> bool:
        return self._since_learner > STALL_DECISIONS

    def _advance(self) -> None:
        """Resolve empty and forced decisions, and bot moves, until a policy has a real choice."""
        game = self._game
        while not game.done and not self._stalled:
            if self.bot is not None and game.to_move != self.seat:
                obs = game.obs
                picks, failed = bot_move(self.bot, obs)
                self._decision = None
                self._since_learner += 1
                try:
                    self._select(picks)
                except ValueError:  # the engine refused it after all
                    failed = True
                    self._select(random_legal(obs))
                self.bot_failures += failed
                continue
            if self._decision is None:
                self._decision = Decision(game.obs)
            d = self._decision
            mask = d.mask()
            if d.complete:  # e.g. maxCount == 0: nothing to choose
                pass
            elif self.skip_forced and mask.sum() == 1:
                if not d.apply(int(np.flatnonzero(mask)[0])):
                    continue
            else:
                return
            self._since_learner += 1
            self._select(d.picked)
            self._decision = None

