"""Determinized lookahead (ISMCTS-style) over the engine's own search API.

At a decision where the policy is unsure, each of its top candidate moves is
played forward in the engine's search copy of the game (``SearchBegin`` /
``SearchStep``, the official ``cg`` API) under several *determinizations*: a
guess of every hidden card -- our deck order and prizes, the opponent's deck,
prizes and hand -- consistent with what has been seen. The policy network plays
both sides (the opponent with the deck we guess it holds) through the rest of
our turn and the opponent's whole turn; the value head then scores the position
at our next decision (a finished game scores +1 / -1). The candidate with the
best mean value is played, unless it beats the policy's own choice by less than
``margin``, so noise from a few samples does not overrule the network.

The opponent's deck is guessed by matching the cards they have shown against a
list of known decks (``opp_decks.json``, the training pool), sampling the cards
not yet seen from the best match.
"""

from __future__ import annotations

import ctypes
import json
import random
import time
from collections import Counter

import numpy as np

from .encoding import STOP, Decision, encode, unseen_own, visible_cards


class EngineSearch:
    """Thin ctypes wrapper of ``SearchBegin`` / ``SearchStep`` returning plain dicts."""

    def __init__(self, lib):
        self.lib = lib
        self.ptr = lib.AgentStart()

    @staticmethod
    def _arr(xs):
        return (ctypes.c_int * len(xs))(*xs)

    def begin(self, obs, your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active=()) -> dict:
        sbi = obs["search_begin_input"].encode("ascii")
        if obs["select"].get("deck") is not None:
            your_deck = []
        out = json.loads(self.lib.SearchBegin(
            self.ptr, sbi, len(sbi), self._arr(your_deck), self._arr(your_prize), self._arr(opp_deck),
            self._arr(opp_prize), self._arr(opp_hand), self._arr(list(opp_active)), 0))
        if out.get("error"):
            raise RuntimeError(f"SearchBegin error {out['error']}")
        return out["state"]

    def step(self, search_id: int, picks: list[int]) -> dict:
        out = json.loads(self.lib.SearchStep(self.ptr, search_id, self._arr(picks), len(picks)))
        if out.get("error"):
            raise RuntimeError(f"SearchStep error {out['error']}")
        return out["state"]

    def end(self) -> None:
        self.lib.SearchEnd(self.ptr)


class Searcher:
    def __init__(self, net, attack_table, card_meta, my_deck: list[int], opp_decks: list[list[int]],
                 engine: EngineSearch, config: dict):
        self.net, self.attacks, self.meta = net, attack_table, card_meta
        self.my_deck, self.opp_decks, self.engine = my_deck, opp_decks, engine
        self.cfg = {"max_seconds": 1.0, "budget_frac": 0.004, "min_remaining": 60.0, "top_k": 4,
                    "min_prior": 0.03, "confident": 0.9, "margin": 0.03, "max_steps": 120, **config}
        self.rng = random.Random(0)
        self.stats = Counter()

    # -- hidden information -------------------------------------------------------------

    def guess_opp_deck(self, state: dict, opp: int, opp_seen: set[int]) -> list[int]:
        shown = Counter(visible_cards(state, opp))
        for c in opp_seen:
            shown[c] = max(shown[c], 1)
        best, best_score = None, -1.0
        for deck in self.opp_decks:
            have = Counter(deck)
            score = sum(min(n, have[c]) for c, n in shown.items()) + self.rng.random() * 0.1
            if score > best_score:
                best, best_score = deck, score
        return best

    def determinize(self, state: dict, opp_deck: list[int]):
        me = state["yourIndex"]
        mine, theirs = state["players"][me], state["players"][1 - me]
        own = unseen_own(self.my_deck, state, me)
        self.rng.shuffle(own)
        n_deck, n_prize = mine["deckCount"], len(mine["prize"])
        while len(own) < n_deck + n_prize:
            own.append(self.rng.choice(self.my_deck))
        your_prize, your_deck = own[:n_prize], own[n_prize:n_prize + n_deck]

        pool = list((Counter(opp_deck) - Counter(visible_cards(state, 1 - me))).elements())
        self.rng.shuffle(pool)
        n_hand, n_odeck, n_oprize = theirs["handCount"], theirs["deckCount"], len(theirs["prize"])
        while len(pool) < n_hand + n_odeck + n_oprize:
            pool.append(self.rng.choice(opp_deck))
        opp_hand, pool = pool[:n_hand], pool[n_hand:]
        opp_prize, opp_rest = pool[:n_oprize], pool[n_oprize:n_oprize + n_odeck]
        return your_deck, your_prize, opp_rest, opp_prize, opp_hand

    # -- network ------------------------------------------------------------------------

    def _features(self, d: Decision, me_view: bool, ctx: dict):
        if me_view:
            return encode(d, self.attacks, my_deck=self.my_deck, opp_seen=ctx["opp_seen"],
                          opp_recent=ctx["opp_recent"], card_meta=self.meta)
        return encode(d, self.attacks, my_deck=ctx["opp_deck"], opp_seen=ctx["my_shown"],
                      opp_recent=(), card_meta=self.meta)

    def _play_greedy(self, obs: dict, me_view: bool, ctx: dict) -> tuple[list[int], float | None]:
        """The policy's picks for this observation, and its value at the first real choice."""
        d, value = Decision(obs), None
        while not d.complete:
            mask = d.mask()
            if mask.sum() == 1:
                if d.apply(int(mask.argmax())):
                    break
                continue
            logits, v = self.net.evaluate(self._features(d, me_view, ctx))
            value = v if value is None else value
            if d.apply(int(np.where(mask, logits, -np.inf).argmax())):
                break
        return d.picked, value

    def rollout(self, sstate: dict, me: int, root_turn: int, ctx: dict, hard_stop: float) -> float | None:
        """Play on until our first decision after the opponent has moved; our value there.

        None when ``hard_stop`` passes first (the sample is then dropped)."""
        saw_opponent, last_value = False, None
        for _ in range(self.cfg["max_steps"]):
            if time.perf_counter() > hard_stop:
                return None
            obs = sstate["observation"]
            cur = obs["current"]
            if cur["result"] >= 0:
                return 0.0 if cur["result"] == 2 else (1.0 if cur["result"] == me else -1.0)
            mover = cur["yourIndex"]
            if mover != me:
                saw_opponent = True
            elif saw_opponent:
                d = Decision(obs)
                _, v = self.net.evaluate(self._features(d, True, ctx))
                return v
            picks, v = self._play_greedy(obs, mover == me, ctx)
            if mover == me and v is not None:
                last_value = v
            sstate = self.engine.step(sstate["searchId"], picks)
            self.stats["steps"] += 1
        return last_value if last_value is not None else 0.0

    # -- root -----------------------------------------------------------------------------

    def budget(self, obs: dict) -> float:
        remaining = obs.get("remainingOverageTime")
        if remaining is None:
            return self.cfg["max_seconds"]
        if remaining < self.cfg["min_remaining"]:
            return 0.0
        return min(self.cfg["max_seconds"], remaining * self.cfg["budget_frac"])

    def choose(self, obs: dict, decision: Decision, logits: np.ndarray, opp_seen: set[int], opp_recent) -> list[int] | None:
        """Picks for a single-pick decision, or None to let the policy decide."""
        if not obs.get("search_begin_input") or decision.max != 1 or decision.picked:
            return None
        state = obs["current"]
        me = state["yourIndex"]
        opp_active = state["players"][1 - me]["active"]
        if state.get("turn", 0) < 1 or (opp_active and opp_active[0] is None):
            return None
        mask = decision.mask()
        prior = np.where(mask, logits, -np.inf)
        prior = np.exp(prior - prior.max())
        prior /= prior.sum()
        order = [int(i) for i in np.argsort(-prior) if mask[i] and prior[i] >= self.cfg["min_prior"]]
        order = order[: self.cfg["top_k"]]
        if len(order) < 2 or prior[order[0]] >= self.cfg["confident"]:
            return None
        seconds = self.budget(obs)
        if seconds <= 0:
            return None

        deadline = time.perf_counter() + seconds
        ctx = {"opp_seen": opp_seen, "opp_recent": opp_recent,
               "my_shown": set(visible_cards(state, me)),
               "opp_deck": self.guess_opp_deck(state, 1 - me, opp_seen)}
        totals, counts = Counter(), Counter()
        hard_stop = deadline + 0.25 * seconds  # a determinization still running then is dropped
        try:
            while time.perf_counter() < deadline:
                root = self.engine.begin(obs, *self.determinize(state, ctx["opp_deck"]))
                values = {}
                for a in order:
                    if time.perf_counter() > hard_stop:
                        break
                    picks = [] if a == STOP else [a]
                    child = self.engine.step(root["searchId"], picks)
                    values[a] = self.rollout(child, me, state.get("turn", 0), ctx, hard_stop)
                self.engine.end()
                if len(values) < len(order) or None in values.values():
                    break
                for a, v in values.items():  # only complete determinizations count, all candidates alike
                    totals[a] += v
                    counts[a] += 1
                self.stats["determinizations"] += 1
        except Exception:  # noqa: BLE001 - the search is advisory: on any engine error, the policy plays
            self.engine.end()
            self.stats["errors"] += 1
            return None
        if not counts or min(counts[a] for a in order) == 0:
            return None
        means = {a: totals[a] / counts[a] for a in order}
        best = max(order, key=means.get)
        self.stats["searched"] += 1
        if best != order[0] and means[best] - means[order[0]] < self.cfg["margin"]:
            best = order[0]
        if best != order[0]:
            self.stats["overruled"] += 1
        return [] if best == STOP else [best]
