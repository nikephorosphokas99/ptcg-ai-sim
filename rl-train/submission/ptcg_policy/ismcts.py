"""Single-Observer Information Set MCTS (Cowling, Powley & Whitehouse, 2012).

One tree per decision, searched from the agent's (the observer's) point of view.
Every iteration:

1. **Determinize**: sample one complete game state consistent with what the
   agent knows, uniformly ("classic" determinization):

   * our deck order and Prize cards: our unseen cards (deck list minus every
     card in view) shuffled, then split into Prizes and deck;
   * the opponent's deck list: one of the known lists that contain every card
     they have shown, chosen uniformly (if none contains them all, the lists
     that contain the most);
   * the opponent's hand, Prize cards and deck: that list's unseen cards,
     shuffled and split.

   The engine's own search API (``SearchBegin``) turns the guess into a full
   game state; ``SearchStep`` plays moves in it.
2. **Select**: walk down the tree considering only the moves legal in this
   determinization, with *availability counts* (exploration counts the visits
   in which a move was legal, not all visits to the parent). The opponent's
   nodes maximise the opponent's result. Two rules (``selection``):

   * ``"ucb1"`` -- classic ISMCTS: every move equally plausible until tried;
   * ``"puct"`` (default) -- the policy network's probability of each move is
     its prior, as in AlphaZero: visits follow the policy unless the values
     say otherwise. With ~100 iterations per pick, UCB1 spreads them evenly
     over 5-10 moves whose values differ by less than their noise, and so
     overruled moves the policy was 99% sure of (4.8% vs 64.3% won against
     the old Grimmsnarl agent); PUCT only overrules on value evidence.
3. **Expand**: add one legal move not in the tree yet -- uniformly chosen with
   UCB1; with PUCT, the untried move whose prior makes it the best choice.
4. **Simulate**: instead of a random playout to the end of the game, the trained
   network evaluates the new position: the value head when it is our move;
   when it is the opponent's, their moves are played with the policy until we
   are to move again, then the value head. A finished game scores +1 / -1 / 0.
5. **Backpropagate** that result (from our point of view) up the path.

After the time budget the most visited move at the root is played.

Moves are tree edges one pick at a time, exactly like ``Decision`` (multi-card
selections are a sequence of picks, then STOP), and are identified across
determinizations by what they do -- move type, card, target, attack, board
position -- not by their option index, which shifts when hidden cards differ.
"""

from __future__ import annotations

import ctypes
import json
import math
import random
import time
from collections import Counter

import numpy as np

from .encoding import ACTIVE, BENCH, STOP, Decision, _resolve, encode, unseen_own, visible_cards


class _OutOfTime(Exception):
    """An iteration still running at the hard stop: it is dropped, nothing is backpropagated."""


# -- the engine's search API ---------------------------------------------------------------


class EngineSearch:
    """``SearchBegin`` / ``SearchStep`` / ``SearchEnd`` of the official ``cg`` library, on plain dicts."""

    def __init__(self, lib):
        self.lib = lib
        self.ptr = lib.AgentStart()

    @staticmethod
    def _ints(values):
        return (ctypes.c_int * len(values))(*values)

    def begin(self, obs: dict, your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active=()) -> dict:
        """Root state ``{"observation": ..., "searchId": ...}`` of one determinization."""
        sbi = obs["search_begin_input"].encode("ascii")
        if obs["select"].get("deck") is not None:  # our deck is in view: the engine ignores the guess
            your_deck = []
        out = json.loads(self.lib.SearchBegin(
            self.ptr, sbi, len(sbi), self._ints(your_deck), self._ints(your_prize), self._ints(opp_deck),
            self._ints(opp_prize), self._ints(opp_hand), self._ints(list(opp_active)), 0))
        if out.get("error"):
            raise RuntimeError(f"SearchBegin error {out['error']}")
        return out["state"]

    def step(self, search_id: int, picks: list[int]) -> dict:
        out = json.loads(self.lib.SearchStep(self.ptr, search_id, self._ints(picks), len(picks)))
        if out.get("error"):
            raise RuntimeError(f"SearchStep error {out['error']}")
        return out["state"]

    def end(self) -> None:
        """Free every state of this search (the memory is reused by the next one)."""
        self.lib.SearchEnd(self.ptr)


# -- tree ------------------------------------------------------------------------------------


def action_key(obs: dict, action: int) -> tuple:
    """What a pick does, comparable across determinizations (option indices are not)."""
    if action == STOP:
        return ("stop",)
    state, sel = obs["current"], obs["select"]
    opt = sel["option"][action]
    card, target = _resolve(state, sel, opt, state["yourIndex"])
    area = opt.get("area")
    # Board positions are public and the same in every determinization; hand and deck
    # positions are not, so there the card id alone names the move.
    index = opt.get("index") if area in (ACTIVE, BENCH) else None
    return (opt["type"], card, target, opt.get("attackId"), area, index,
            opt.get("inPlayArea"), opt.get("inPlayIndex"), opt.get("number"), opt.get("playerIndex"))


class ISMCTSNode:
    """A node of the single-observer tree: the information set reached by a sequence of moves.

    ``player`` made the move leading here; ``total`` sums results from the root
    player's point of view; ``avail`` counts the visits to the parent in which
    this move was legal (the availability count of ISMCTS).
    """

    __slots__ = ("parent", "key", "player", "children", "visits", "total", "avail", "priors")

    def __init__(self, parent: "ISMCTSNode | None" = None, key: tuple | None = None, player: int = -1) -> None:
        self.parent = parent
        self.key = key
        self.player = player
        self.children: dict[tuple, ISMCTSNode] = {}
        self.visits = 0
        self.total = 0.0
        self.avail = 0
        self.priors: dict[tuple, float] = {}  # PUCT: the mover's policy probability of each move seen here

    def mean(self, player: int, root_player: int) -> float:
        """Average result from ``player``'s point of view (totals are kept from the root player's)."""
        m = self.total / self.visits
        return m if player == root_player else -m

    def untried(self, legal: dict[tuple, int]) -> list[tuple]:
        return [k for k in legal if k not in self.children]

    def add_child(self, key: tuple, player: int) -> "ISMCTSNode":
        child = ISMCTSNode(self, key, player)
        self.children[key] = child
        return child

    def select(self, legal: dict[tuple, int], root_player: int, c: float) -> "ISMCTSNode":
        """UCB1 over the children legal in this determinization, from the mover's point of view."""
        best, best_score = None, -math.inf
        for key in legal:
            child = self.children[key]
            if child.visits == 0:  # added by an iteration an engine error cut short
                return child
            mean = child.total / child.visits
            if child.player != root_player:
                mean = -mean
            score = mean + c * math.sqrt(math.log(max(child.avail, 1)) / child.visits)
            if score > best_score:
                best, best_score = child, score
        assert best is not None  # ``legal`` is never empty here
        return best

    def select_puct(self, legal: dict[tuple, int], mover: int, root_player: int, c: float) -> tuple:
        """PUCT over the legal moves (tried or not); returns the chosen move's key."""
        tried = [self.children[k] for k in legal if k in self.children]
        n_parent = 1 + sum(ch.visits for ch in tried)  # visits in which these moves were available
        # Untried moves start from the parent's value for the mover ("first play urgency").
        fpu = self.mean(mover, root_player) if self.visits else 0.0
        best, best_score = None, -math.inf
        for key in legal:
            child = self.children.get(key)
            n = child.visits if child is not None else 0
            q = child.mean(mover, root_player) if n else fpu
            score = q + c * self.priors.get(key, 0.0) * math.sqrt(n_parent) / (1 + n)
            if score > best_score:
                best, best_score = key, score
        assert best is not None
        return best


# -- search ----------------------------------------------------------------------------------


class ISMCTS:
    def __init__(self, net, attack_table, card_meta, my_deck: list[int], opp_decks: list[list[int]],
                 engine: EngineSearch, config: dict | None = None):
        self.net, self.attacks, self.meta = net, attack_table, card_meta
        self.my_deck, self.opp_decks, self.engine = my_deck, opp_decks, engine
        self.cfg = {
            "max_seconds": 1.0,      # most thinking time per searched pick
            "max_iterations": 2000,
            "budget_frac": 0.004,    # of Kaggle's remainingOverageTime, when it is given
            "min_remaining": 60.0,   # seconds of overage below which the policy plays alone
            "selection": "puct",     # "puct" (policy priors) or "ucb1" (classic ISMCTS)
            "exploration": 0.7,      # UCB1 constant; results lie in [-1, 1]
            "c_puct": 1.5,
            "playout_steps": 200,    # opponent picks played by the policy before giving up on a leaf
            **(config or {}),
        }
        self.rng = random.Random()
        self.stats = Counter()
        self._hard_stop = math.inf

    # -- determinization ---------------------------------------------------------------------

    def _consistent_decks(self, state: dict, opp: int, opp_seen: set[int]) -> list[list[int]]:
        """Known deck lists containing every card the opponent has shown (else the closest ones)."""
        shown = Counter(visible_cards(state, opp))
        for cid in opp_seen:
            shown[cid] = max(shown[cid], 1)
        overlap = [sum(min(n, Counter(d)[c]) for c, n in shown.items()) for d in self.opp_decks]
        best = max(overlap)
        return [d for d, o in zip(self.opp_decks, overlap) if o == best]

    def determinize(self, state: dict, candidates: list[list[int]]):
        """One uniform guess of every hidden card: (our deck, our prizes, their deck, prizes, hand, list)."""
        me = state["yourIndex"]
        mine, theirs = state["players"][me], state["players"][1 - me]

        own = unseen_own(self.my_deck, state, me)
        self.rng.shuffle(own)
        n_prize, n_deck = len(mine["prize"]), mine["deckCount"]
        while len(own) < n_prize + n_deck:  # cards we cannot account for (e.g. lost zone): any of ours
            own.append(self.rng.choice(self.my_deck))
        your_prize, your_deck = own[:n_prize], own[n_prize:n_prize + n_deck]

        deck_list = self.rng.choice(candidates)
        pool = list((Counter(deck_list) - Counter(visible_cards(state, 1 - me))).elements())
        self.rng.shuffle(pool)
        n_hand, n_oprize, n_odeck = theirs["handCount"], len(theirs["prize"]), theirs["deckCount"]
        while len(pool) < n_hand + n_oprize + n_odeck:
            pool.append(self.rng.choice(deck_list))
        opp_hand = pool[:n_hand]
        opp_prize = pool[n_hand:n_hand + n_oprize]
        opp_deck = pool[n_hand + n_oprize:n_hand + n_oprize + n_odeck]
        return your_deck, your_prize, opp_deck, opp_prize, opp_hand, deck_list

    # -- network -------------------------------------------------------------------------------

    def _evaluate(self, decision: Decision, root_player: int, ctx: dict) -> tuple[np.ndarray, float]:
        mover = decision.obs["current"]["yourIndex"]
        if mover == root_player:
            features = encode(decision, self.attacks, my_deck=self.my_deck, opp_seen=ctx["opp_seen"],
                              opp_recent=ctx["opp_recent"], card_meta=self.meta)
        else:
            features = encode(decision, self.attacks, my_deck=ctx["opp_list"], opp_seen=ctx["our_shown"],
                              opp_recent=(), card_meta=self.meta)
        return self.net.evaluate(features)

    def _leaf_value(self, sstate: dict, decision: Decision | None, root_player: int, ctx: dict) -> float:
        """Simulation step: the value head at our next decision (opponent moves played by the policy)."""
        for _ in range(self.cfg["playout_steps"]):
            if time.perf_counter() > self._hard_stop:
                raise _OutOfTime
            obs = sstate["observation"]
            result = obs["current"]["result"]
            if result >= 0:
                return 0.0 if result == 2 else (1.0 if result == root_player else -1.0)
            d = decision if decision is not None else Decision(obs)
            decision = d
            mover = obs["current"]["yourIndex"]
            mask = d.mask()
            if mover == root_player and mask.sum() > 1:
                return float(np.clip(self._evaluate(d, root_player, ctx)[1], -1.0, 1.0))
            if mask.sum() == 1:
                action = int(mask.argmax())
            else:
                logits, _ = self._evaluate(d, root_player, ctx)
                action = int(np.where(mask, logits, -np.inf).argmax())
            if d.apply(action):
                sstate = self.engine.step(sstate["searchId"], d.picked)
                decision = None
        self.stats["playout_cut"] += 1
        return 0.0

    # -- one iteration ---------------------------------------------------------------------------

    def _iterate(self, root: ISMCTSNode, obs: dict, prefix: list[int], candidates, ctx: dict) -> None:
        state = obs["current"]
        root_player = state["yourIndex"]
        your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_list = self.determinize(state, candidates)
        ctx["opp_list"] = opp_list
        sstate = self.engine.begin(obs, your_deck, your_prize, opp_deck, opp_prize, opp_hand)
        try:
            first = Decision(sstate["observation"])
            for pick in prefix:  # earlier picks of this same multi-card selection
                first.apply(pick)
            decision: Decision | None = first
            node, path = root, [root]
            value = None
            while True:
                if time.perf_counter() > self._hard_stop:
                    raise _OutOfTime
                cur = sstate["observation"]
                result = cur["current"]["result"]
                if result >= 0:
                    value = 0.0 if result == 2 else (1.0 if result == root_player else -1.0)
                    break
                d = decision if decision is not None else Decision(cur)
                decision = d
                mask = d.mask()
                if mask.sum() == 1:  # forced: no tree node, just play it
                    if d.apply(int(mask.argmax())):
                        sstate = self.engine.step(sstate["searchId"], d.picked)
                        decision = None
                    continue
                legal: dict[tuple, int] = {}
                for a in np.flatnonzero(mask):
                    legal.setdefault(action_key(cur, int(a)), int(a))
                mover = cur["current"]["yourIndex"]
                for key in legal:
                    if key in node.children:
                        node.children[key].avail += 1
                if self.cfg["selection"] == "puct":
                    if any(k not in node.priors for k in legal):  # first time these moves are seen here
                        logits, _ = self._evaluate(d, root_player, ctx)
                        probs = np.where(mask, logits, -np.inf)
                        probs = np.exp(probs - probs.max())
                        probs /= probs.sum()
                        seen: dict[tuple, float] = {}
                        for a in np.flatnonzero(mask):  # identical moves (e.g. two copies in hand) share one key
                            k = action_key(cur, int(a))
                            seen[k] = seen.get(k, 0.0) + float(probs[a])
                        for k, prob in seen.items():
                            node.priors.setdefault(k, prob)
                    key = node.select_puct(legal, mover, root_player, self.cfg["c_puct"])
                    expanded = key not in node.children
                    if expanded:
                        node = node.add_child(key, mover)
                        node.avail += 1
                    else:
                        node = node.children[key]
                else:
                    untried = node.untried(legal)
                    if untried:  # expansion: one new move, then simulate from there
                        key = self.rng.choice(untried)
                        node = node.add_child(key, mover)
                        node.avail += 1
                        expanded = True
                    else:
                        node = node.select(legal, root_player, self.cfg["exploration"])
                        key, expanded = node.key, False
                        assert key is not None  # only the root has no key
                path.append(node)
                if d.apply(legal[key]):
                    sstate = self.engine.step(sstate["searchId"], d.picked)
                    decision = None
                if expanded:
                    break
            if value is None:
                value = self._leaf_value(sstate, decision, root_player, ctx)
        finally:
            self.engine.end()
        for n in path:  # backpropagation, from the root player's point of view
            n.visits += 1
            n.total += value

    # -- entry point ---------------------------------------------------------------------------

    def budget(self, obs: dict) -> float:
        remaining = obs.get("remainingOverageTime")
        if remaining is None:
            return self.cfg["max_seconds"]
        if remaining < self.cfg["min_remaining"]:
            return 0.0
        return min(self.cfg["max_seconds"], remaining * self.cfg["budget_frac"])

    def choose(self, obs: dict, decision: Decision, opp_seen: set[int], opp_recent) -> int | None:
        """The next pick for ``decision`` (an index into ``decision.mask()``), or None to let the policy play."""
        if not obs.get("search_begin_input"):
            return None
        state = obs["current"]
        me = state["yourIndex"]
        opp_active = state["players"][1 - me]["active"]
        if opp_active and opp_active[0] is None:  # set-up: the opponent's Active is face down
            return None
        mask = decision.mask()
        legal: dict[tuple, int] = {}
        for a in np.flatnonzero(mask):
            legal.setdefault(action_key(obs, int(a)), int(a))
        if len(legal) < 2:
            return None
        seconds = self.budget(obs)
        if seconds <= 0:
            return None

        ctx = {"opp_seen": opp_seen, "opp_recent": opp_recent, "our_shown": set(visible_cards(state, me))}
        candidates = self._consistent_decks(state, 1 - me, opp_seen)
        root = ISMCTSNode(player=me)
        deadline = time.perf_counter() + seconds
        self._hard_stop = deadline + 0.5 * seconds  # an iteration still running then is dropped
        iterations = 0
        try:
            while iterations < self.cfg["max_iterations"] and time.perf_counter() < deadline:
                self._iterate(root, obs, list(decision.picked), candidates, ctx)
                iterations += 1
        except _OutOfTime:
            self.stats["dropped_iterations"] += 1
        except Exception:  # noqa: BLE001 - an engine error ends the search; the policy plays this pick
            self.stats["errors"] += 1
            if iterations == 0:
                return None
        self.stats["searches"] += 1
        self.stats["iterations"] += iterations
        visited = [(child.visits, key) for key, child in root.children.items() if key in legal and child.visits]
        if not visited:
            return None
        return legal[max(visited, key=lambda vk: vk[0])[1]]  # the most visited move
