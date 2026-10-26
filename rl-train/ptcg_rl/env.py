"""One game between the learner and an opponent, stepped one decision at a time.

``Match`` knows nothing about policies: ``step`` applies an action for
whichever seat the engine is waiting on and reports whether the learner moves
next. ``SelfPlayCollector`` decides for both seats, batched on the GPU, so the
worker processes that host matches never import torch.

Each game draws a seat (0 or 1) and an opponent deck, so the learner sees both
sides of the coin flip. With no opponent decks the opponent pilots the learner's
own deck (mirror match); with ``opponent_weights`` decks are drawn in proportion
to them (``decks/training_pool/weights.json``: Kaggle meta share, more weight
on the matchups the agent loses).

With ``rules`` (the default) both seats play two decisions by rule instead of by
policy, exactly as the submission does: damage counters placed "in any way you
like" go where they take the most Prize cards (``encoding.CounterPlanner``), and
a turn that offers an attack without a drawback is not ended without attacking
(``encoding.Decision``).

The engine seeds its RNG from ``std::random_device``, so games are not
reproducible; ``seed`` only fixes this match's own choices (seat, deck).
"""

from __future__ import annotations

import re
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .bots import bot_deck, bot_move, load_bot, random_legal
from .encoding import RECENT_SLOTS, CounterPlanner, Decision, encode, recent_opponent_cards, visible_cards
from .engine import Game, all_attacks, all_cards, attack_table, card_meta_table, load_deck, safe_attack_ids


BASIC_ENERGY = range(1, 9)  # basic energy card ids, equal to the energy type they provide
MAIN_CONTEXT, ATTACH_FROM_CONTEXT, PLAY_OPTION, ATTACH_OPTION, ABILITY_OPTION = 0, 21, 7, 8, 10
ACTIVE_AREA, BENCH_AREA = 4, 5
LOG_PLAY, LOG_EVOLVE, LOG_ATTACK = 10, 12, 15
# Abilities whose cost is the Pokémon itself, e.g. Cursed Blast: "... this Pokémon is Knocked Out."
SELF_KO = re.compile(r"this pok[eé]mon is knocked out", re.IGNORECASE)


def _missing(energies: list[int], cost: list[int]) -> int:
    """Energy still missing to pay ``cost`` (type ids, 0 = any type) from ``energies``."""
    need: dict[int, int] = {}
    for t in cost:
        need[t] = need.get(t, 0) + 1
    have: dict[int, int] = {}
    for t in energies:
        have[t] = have.get(t, 0) + 1
    typed = {t: min(n, have.get(t, 0)) for t, n in need.items() if t != 0}
    missing = sum(need[t] - typed[t] for t in typed)
    spare = len(energies) - sum(typed.values())
    return missing + max(0, need.get(0, 0) - spare)


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
        opponent_weights: dict[str, float] | None = None,
        prize_reward: float = 0.05,
        energy_penalty: float = 0.0,
        skip_forced: bool = True,
        max_steps: int = 1000,
        bots: list[str] | None = None,
        bot_frac: float = 0.0,
        bot_cg: str | None = None,
        rules: bool = True,
        sacrifice_refund: bool = False,
        seed: int | None = None,
    ):
        """
        Args:
            agent_deck: the learner's 60-card CSV.
            opponent_decks: CSVs the opponent's deck is drawn from each game;
                empty or None means a mirror match with ``agent_deck``.
            opponent_weights: relative draw weight per deck (file stem); decks
                not named get 1.0. None draws every deck equally often.
            prize_reward: shaping per Prize card taken (and minus per Prize card
                given up), on top of +1 / -1 for a win / loss.
            energy_penalty: subtracted when the learner attaches a basic Energy (from
                hand, or by an effect such as Crispin) to a Pokémon it does not help
                towards any attack, while another of its Pokémon in play could have
                used it (e.g. a third Energy on an already powered Dragapult ex).
            skip_forced: play decisions with exactly one legal action
                automatically, for either seat.
            max_steps: learner decisions before the game is truncated.
            bots: rule-based agent bundles (``benchmarks/<name>``); with
                probability ``bot_frac`` a game's opponent is one of them,
                playing its own deck, moved here without involving the trainer.
            bot_cg: parent folder of the ``cg`` the bots import (``benchmark.compat_cg``).
            rules: place "any way you like" damage counters by plan and never end
                a turn that offers a drawback-free attack, for both seats.
            sacrifice_refund: when the learner uses an Ability that Knocks Out its
                own Pokémon (Cursed Blast), the Prize card the opponent takes for it
                is not counted against the learner in ``prize_reward``. Otherwise
                every such play costs a Prize card's shaping at once while its
                payoff (a knockout from the counters) often comes later, which
                taught the policy to leave Dusclops / Dusknoir unused.

        Finished games report ``info["usage"]``: per game, the learner seat's cards
        played, evolutions, attacks and Abilities by card name (from the engine log
        and the learner's own picks; the game's very last events can be missed).
        """
        self.agent_deck = load_deck(agent_deck)
        self.opponent_decks = {Path(p).stem: load_deck(p) for p in opponent_decks or []}
        if not self.opponent_decks:
            self.opponent_decks = {"mirror": self.agent_deck}
        self._deck_names = list(self.opponent_decks)
        self.set_opponent_weights(opponent_weights)
        self.prize_reward = prize_reward
        self.energy_penalty = energy_penalty
        if energy_penalty:
            attacks = all_attacks()
            cards = all_cards()
            self._costs = {cid: [attacks[a]["energies"] for a in c["attacks"] if a in attacks]
                           for cid, c in cards.items() if c["cardType"] == 0}
            self._stage2 = {cid for cid, c in cards.items() if c["cardType"] == 0 and c["stage2"]}
            # Supporters whose whole effect is attaching Energy to a Stage 2 Pokémon
            self._stage2_chargers = {cid for cid, c in cards.items() if c["name"] == "Rosa's Encouragement"}
        self.skip_forced = skip_forced
        self.max_steps = max_steps
        self.bots = [Path(b) for b in bots or []]
        self.bot_frac = bot_frac if self.bots else 0.0
        self.bot_cg = Path(bot_cg) if bot_cg else None
        self._bot_decks = {b.name: bot_deck(b) for b in self.bots}
        self._rng = np.random.default_rng(seed)
        self._attacks = attack_table()
        self._meta = card_meta_table()
        self._safe_attacks = safe_attack_ids() if rules else None
        self._planners = [CounterPlanner(self._meta), CounterPlanner(self._meta)] if rules else None
        cards = all_cards()
        self._card_names = {cid: c["name"] for cid, c in cards.items()}
        self._attack_names = {aid: a["name"] for aid, a in all_attacks().items()}
        self.sacrifice_refund = sacrifice_refund
        self._self_ko_abilities = {
            cid for cid, c in cards.items()
            if any(SELF_KO.search(s.get("text") or "") for s in c.get("skills") or [])}
        self._usage: Counter = Counter()
        self._sacrifice_turn: int | None = None
        self._game: Game | None = None
        self._decision: Decision | None = None

    def set_opponent_weights(self, opponent_weights: dict[str, float] | None) -> None:
        """Relative draw weight per opponent deck (file stem) from the next game on; 1 if not named."""
        weights = np.array([(opponent_weights or {}).get(n, 1.0) for n in self._deck_names], dtype=float)
        if (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("opponent_weights must be non-negative and not all zero")
        self._deck_p = weights / weights.sum()

    def reset(self) -> Turn:
        self.close()
        self.seat = int(self._rng.integers(2))
        self.bot, self.bot_name, self.bot_failures = None, None, 0
        if self.bot_frac and self._rng.random() < self.bot_frac:
            bundle = self.bots[self._rng.integers(len(self.bots))]
            self.bot, self.bot_name = load_bot(bundle, self.bot_cg), bundle.name
            self.opp_deck_name, opp_deck = f"bot:{bundle.name}", self._bot_decks[bundle.name]
        else:
            self.opp_deck_name = self._deck_names[self._rng.choice(len(self._deck_names), p=self._deck_p)]
            opp_deck = self.opponent_decks[self.opp_deck_name]
        decks = [self.agent_deck, opp_deck]
        if self.seat == 1:
            decks.reverse()
        self._decks = decks
        self._seen: list[set[int]] = [set(), set()]  # opponent cards each seat has seen
        self._recent = [deque(maxlen=RECENT_SLOTS), deque(maxlen=RECENT_SLOTS)]
        for planner in self._planners or ():
            planner.reset()
        self._usage = Counter()
        self._sacrifice_turn = None
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
        penalty = self._wasted_energy(int(action)) if learner_moved and self.energy_penalty else 0.0
        if learner_moved:
            self._note_ability(int(action))
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
        current = (self._game.obs or {}).get("current")
        if self._sacrifice_turn is not None and current is not None:
            turn_now = current["turn"]
            if lost and turn_now - self._sacrifice_turn <= 1:
                # The Prize card the opponent took for our own Cursed Blast-style knockout.
                lost -= 1
                self._usage["refunded sacrifice"] += 1
                self._sacrifice_turn = None
            elif turn_now - self._sacrifice_turn > 1:
                self._sacrifice_turn = None
        reward = self.prize_reward * (taken - lost) - penalty

        info = {"opp_deck": self.opp_deck_name, "bot": self.bot_name, "bot_failures": self.bot_failures}
        if self._stalled:  # a bot looping in its own turn: end the game, score nothing
            return StepResult(reward, done=True, truncated=True, info={**info, "stalled": True})
        if self._game.done:
            result = self._game.result
            outcome = 0.5 if result == 2 else float(result == self.seat)
            return StepResult(reward + 2.0 * outcome - 1.0, done=True,
                              info={**info, "outcome": outcome, "usage": dict(self._usage)})
        turn = self._turn()
        if self._steps >= self.max_steps:
            return StepResult(reward, done=True, truncated=True, turn=turn, info={**info, "usage": dict(self._usage)})
        return StepResult(reward, turn=turn)

    def _note_ability(self, action: int) -> None:
        """Count the learner's Ability uses by card, and mark a self-Knock-Out one for the refund."""
        obs = self._game.obs
        sel = obs.get("select")
        if not sel or action >= len(sel["option"]):
            return
        opt = sel["option"][action]
        if opt.get("type") != ABILITY_OPTION:
            return
        me = obs["current"]["players"][self.seat]
        area, index = opt.get("area"), opt.get("index")
        mons = me["active"] if area == ACTIVE_AREA else me["bench"] if area == BENCH_AREA else []
        mon = mons[index] if index is not None and 0 <= index < len(mons) else None
        if mon is None:
            return
        self._usage[f"ability {self._card_names.get(mon['id'], mon['id'])}"] += 1
        if self.sacrifice_refund and mon["id"] in self._self_ko_abilities:
            self._sacrifice_turn = obs["current"]["turn"]

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
            if me == self.seat:  # each seat's logs hold the events since its own last observation
                self._count_usage(obs.get("logs") or [])

    def _count_usage(self, logs: list[dict]) -> None:
        for log in logs:
            if log.get("playerIndex") != self.seat:
                continue
            kind = log.get("type")
            if kind == LOG_PLAY:
                self._usage[f"play {self._card_names.get(log.get('cardId'), log.get('cardId'))}"] += 1
            elif kind == LOG_EVOLVE:
                self._usage[f"evolve {self._card_names.get(log.get('cardId'), log.get('cardId'))}"] += 1
            elif kind == LOG_ATTACK:
                self._usage[f"attack {self._attack_names.get(log.get('attackId'), log.get('attackId'))}"] += 1

    def _helps(self, mon: dict, energy: int) -> bool:
        """Whether one more ``energy`` brings ``mon`` closer to paying any of its attacks."""
        have = list(mon["energies"])
        return any(_missing(have + [energy], cost) < _missing(have, cost) for cost in self._costs.get(mon["id"], []))

    def _wasted_energy(self, action: int) -> float:
        """``energy_penalty`` if this pick attaches a basic Energy where it helps no attack
        while another of our Pokémon in play could use it; else 0."""
        obs = self._game.obs
        sel, d = obs["select"], self._decision
        if action >= d.n:
            return 0.0
        opt, me = sel["option"][action], obs["current"]["players"][self.seat]
        if sel["context"] == MAIN_CONTEXT and opt["type"] == PLAY_OPTION:
            card = (me["hand"] or [None] * (opt["index"] + 1))[opt["index"]]
            if card and card["id"] in self._stage2_chargers:  # e.g. Rosa onto Dragapult ex that is already powered
                stage2 = [m for m in me["active"] + me["bench"] if m and m["id"] in self._stage2]
                if stage2 and all(self._powered(m) for m in stage2):
                    return self.energy_penalty
            return 0.0
        if sel["context"] == MAIN_CONTEXT and opt["type"] == ATTACH_OPTION:
            card = (me["hand"] or [None] * (opt["index"] + 1))[opt["index"]]
            area, index = opt.get("inPlayArea"), opt.get("inPlayIndex")
        elif sel["context"] == ATTACH_FROM_CONTEXT and opt.get("playerIndex", self.seat) == self.seat:
            card = sel.get("contextCard")
            area, index = opt.get("area"), opt.get("index")
            if card is None:  # e.g. Rosa's Encouragement: the engine picks the Energy after the target
                return self._wasted_target(sel["option"], me, area, index)
        else:
            return 0.0
        if not card or card["id"] not in BASIC_ENERGY or area not in (ACTIVE_AREA, BENCH_AREA):
            return 0.0
        mons = me["active"] if area == ACTIVE_AREA else me["bench"]
        target = mons[index] if index is not None and index < len(mons) else None
        if target is None or self._helps(target, card["id"]):
            return 0.0
        others = [m for m in me["active"] + me["bench"] if m and m is not target]
        return self.energy_penalty if any(self._helps(m, card["id"]) for m in others) else 0.0

    def _powered(self, mon: dict) -> bool:
        return all(_missing(mon["energies"], cost) == 0 for cost in self._costs.get(mon["id"], []))

    def _wasted_target(self, options: list[dict], me: dict, area: int | None, index: int | None) -> float:
        """Energy of unknown type going to ``area``/``index``: wasted if that Pokémon can already
        pay every attack while another offered Pokémon cannot."""
        def mon(a, i):
            mons = me["active"] if a == ACTIVE_AREA else me["bench"] if a == BENCH_AREA else []
            return mons[i] if i is not None and i < len(mons) else None
        target = mon(area, index)
        if target is None or not self._powered(target):
            return 0.0
        others = [mon(o.get("area"), o.get("index")) for o in options]
        return self.energy_penalty if any(m and m is not target and not self._powered(m) for m in others) else 0.0

    def _prizes(self) -> tuple[int, int]:
        """Prize cards left (learner, opponent). Fewer left means more taken."""
        players = self._game.obs["current"]["players"]
        return len(players[self.seat]["prize"]), len(players[1 - self.seat]["prize"])

    @property
    def _stalled(self) -> bool:
        return self._since_learner > STALL_DECISIONS

    def _advance(self) -> None:
        """Resolve empty, forced and rule-played decisions, and bot moves, until a policy has a real choice."""
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
                self._decision = Decision(game.obs, self._safe_attacks)
            d = self._decision
            mask = d.mask()
            planned = self._planners[game.to_move].pick(game.obs) if self._planners and not d.picked else None
            if d.complete:  # e.g. maxCount == 0: nothing to choose
                pass
            elif planned is not None:
                d.apply(planned)
            elif self.skip_forced and mask.sum() == 1:
                if not d.apply(int(np.flatnonzero(mask)[0])):
                    continue
            else:
                return
            self._since_learner += 1
            self._select(d.picked)
            self._decision = None

