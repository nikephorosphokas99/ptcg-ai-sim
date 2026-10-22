"""Self-play data collection: every game advances one decision per tick.

SB3's ``VecEnv`` contract makes every env wait for the learner's next turn, so a
game halfway through a long opponent turn holds up the whole batch; measured on
288 games that was ~20 synchronous rounds per learner step with most games idle.
Here nothing waits: each tick, every game asks whoever is to move, the learner's
decisions go through one batched forward pass and each opponent policy's through
another, and all games step at once.

Data path: ``worker.py`` processes (no torch) write each game's pending
observation into shared memory; the collector reads a group's rows, runs the
networks on the GPU, and keeps the learner's observations there
(``storage.SampleStore``) for training. Workers are split into groups that take
turns, so one group plays while the collector handles the other.

Each game pairs the learner's seat with an opponent drawn per game:

* ``"self"``  -- the current learner; both seats then produce training samples
* ``"random"`` -- uniform over legal actions
* ``snap_*``  -- frozen copies of the learner, kept on the GPU
* ``"baseline"`` -- optionally, a fixed earlier model that is never evicted, so
  its win rate measures progress across the whole run

Samples are grouped into per-seat trajectories and released, with GAE
advantages, only when the game ends. A trajectory still running when the policy
is updated keeps the log-probs and values it was generated with, so its first
few samples are slightly off-policy; PPO's ratio uses those stored log-probs,
which is the correct behaviour-policy correction.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from multiprocessing.connection import Listener

import numpy as np
import torch

from .encoding import SharedObs, observation_space
from .engine import TRAIN_ROOT, card_table
from .storage import SampleStore

SELF, RANDOM, BASELINE = "self", "random", "baseline"


@dataclass
class Stats:
    decisions: int = 0
    samples: int = 0
    dropped: int = 0  # samples whose observation was overwritten before use
    bot_failures: int = 0  # bot moves replaced by random legal ones
    games: int = 0
    outcomes: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    baseline_by_deck: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    # Bounded: a day-long run finishes millions of games.
    episode_lengths: deque[int] = field(default_factory=lambda: deque(maxlen=20_000))


@dataclass
class Batch:
    """Finished samples, as parallel arrays."""

    ids: np.ndarray
    actions: np.ndarray
    values: np.ndarray
    log_probs: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray

    def __len__(self) -> int:
        return len(self.ids)


def _launch_workers(match_kwargs: dict, n_workers: int, per_worker: int, seed: int, shm: str):
    """Start ``python -m ptcg_rl.worker`` processes and accept their connections.

    Plain subprocesses, not multiprocessing: its children re-import the parent's
    main module, which here is the training script and with it torch (~530 MB
    per worker instead of ~55 MB).
    """
    key = os.urandom(16)
    env = {**os.environ, "PTCG_WORKER_KEY": key.hex(),
           "PYTHONPATH": os.pathsep.join([str(TRAIN_ROOT), os.environ.get("PYTHONPATH", "")])}
    listener = Listener(family="AF_UNIX", authkey=key)
    total = n_workers * per_worker
    procs = [
        subprocess.Popen(
            [sys.executable, "-m", "ptcg_rl.worker", listener.address, str(w), str(per_worker),
             str(seed + w * per_worker), shm, str(w * per_worker), str(total), json.dumps(match_kwargs)],
            env=env,
        )
        for w in range(n_workers)
    ]
    conns = [None] * n_workers
    for _ in range(n_workers):
        conn = listener.accept()
        conns[conn.recv()] = conn
    listener.close()
    return procs, conns


def frozen_copy(policy, device: torch.device):
    """An inference-only copy of an SB3 policy.

    Rebuilt and loaded rather than deep-copied: a trained SB3 policy holds its
    optimizer, whose state deepcopy refuses.
    """
    frozen = type(policy)(**policy._get_constructor_parameters()).to(device)
    frozen.load_state_dict(policy.state_dict())
    frozen.set_training_mode(False)
    for p in frozen.parameters():
        p.requires_grad_(False)
    return frozen


class SelfPlayCollector:
    def __init__(
        self,
        match_kwargs: dict,
        policy,
        n_workers: int,
        matches_per_worker: int,
        device: torch.device,
        gamma: float = 0.995,
        gae_lambda: float = 0.95,
        self_play_frac: float = 0.5,
        max_pool: int = 10,
        active_snapshots: int = 2,
        store_capacity: int = 0,
        deterministic: bool = False,
        pool: dict[str, torch.nn.Module | None] | None = None,
        n_groups: int = 2,
        seed: int = 0,
    ):
        """
        Args:
            policy: the learner's policy (``model.policy``), used live.
            self_play_frac: share of new games played against the current
                learner; the rest draw uniformly from random + the active
                snapshots.
            active_snapshots: snapshots new games can draw at any one time
                (the newest plus a random pick of older ones, re-drawn by
                ``rotate_opponents``). Each one in play costs a forward pass
                per tick, so this caps inference cost as the pool grows.
            store_capacity: samples kept on the GPU; 0 records nothing
                (evaluation). Must exceed the samples recorded while one game is
                in progress plus one rollout, or old trajectories get dropped.
            pool: initial opponents by name (default: just the random player).
            n_groups: worker groups that alternate, overlapping game play with
                inference.
        """
        self.policy, self.device = policy, device
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.self_play_frac, self.max_pool = self_play_frac, max_pool
        self.active_snapshots = active_snapshots
        self.record = store_capacity > 0
        self.deterministic = deterministic
        self.k = matches_per_worker
        self.n = n_workers * matches_per_worker
        self._rng = np.random.default_rng(seed)

        space = observation_space(len(card_table()) - 1)
        self.store = SampleStore(space, store_capacity, device) if self.record else None
        self.shared = SharedObs(space, self.n)
        self._procs, self._conns = _launch_workers(match_kwargs, n_workers, self.k, seed, self.shared.name)
        n_groups = max(1, min(n_groups, n_workers))
        self._groups = [list(range(g, n_workers, n_groups)) for g in range(n_groups)]
        self._pending = [False] * n_groups
        self.pool: dict[str, torch.nn.Module | None] = dict(pool or {RANDOM: None})
        self._active = list(self.pool)

        self._learner_turn = np.zeros(self.n, bool)
        self._opponent = np.full(self.n, SELF, dtype=object)
        self._length = np.zeros(self.n, int)
        # Open trajectories by (game, is the match's learner seat):
        # rows of [store id, action, value, log_prob, reward].
        self._traj: dict[tuple[int, bool], list[list]] = defaultdict(list)
        self._ready: list[Batch] = []
        self._n_ready = 0
        self.stats = Stats()

        for conn in self._conns:
            conn.send(("reset", None))
        for w, conn in enumerate(self._conns):
            for j, learner in enumerate(conn.recv()):
                self._new_game(w * self.k + j, learner)

    # -- opponent pool ----------------------------------------------------------

    def add_opponent(self, name: str, policy: torch.nn.Module) -> None:
        """Freeze a copy of ``policy``; keeps random + the newest ``max_pool``."""
        self.pool[name] = frozen_copy(policy, self.device)
        snapshots = [k for k in self.pool if k not in (RANDOM, BASELINE)]
        if len(snapshots) > self.max_pool:
            del self.pool[snapshots[0]]
        self.rotate_opponents()

    def rotate_opponents(self) -> None:
        """Re-draw which snapshots new games may face: the newest plus random older ones."""
        snapshots = [k for k in self.pool if k not in (RANDOM, BASELINE)]
        active = snapshots[-1:]
        older = snapshots[:-1]
        n_more = min(len(older), max(self.active_snapshots - 1, 0))
        active += [older[i] for i in self._rng.choice(len(older), n_more, replace=False)] if n_more else []
        self._active = [k for k in (RANDOM, BASELINE) if k in self.pool] + active

    def _new_game(self, i: int, learner: bool) -> None:
        if self.record and self._rng.random() < self.self_play_frac:
            self._opponent[i] = SELF
        else:
            self._opponent[i] = self._active[self._rng.integers(len(self._active))]
        self._length[i] = 0
        self._learner_turn[i] = learner

    # -- collection -------------------------------------------------------------

    def collect(self, n_samples: int) -> Batch:
        """Play until ``n_samples`` finished samples are ready, and return them."""
        self.policy.set_training_mode(False)
        self._run(lambda: self._n_ready >= n_samples)
        batch = Batch(*(np.concatenate(parts) for parts in zip(*(
            (b.ids, b.actions, b.values, b.log_probs, b.advantages, b.returns) for b in self._ready
        ))))
        fresh = self.store.count - batch.ids <= self.store.capacity  # not overwritten while waiting
        self.stats.dropped += int((~fresh).sum())
        batch = Batch(*(a[fresh] for a in vars(batch).values()))
        if len(batch) < n_samples:
            self._ready, self._n_ready = [batch], len(batch)
            return self.collect(n_samples)
        head = Batch(*(a[:n_samples] for a in vars(batch).values()))
        tail = Batch(*(a[n_samples:] for a in vars(batch).values()))
        self._ready = [tail] if len(tail) else []
        self._n_ready = len(tail)
        return head

    def play(self, n_games: int) -> None:
        """Play until ``n_games`` more games have finished (evaluation)."""
        self.policy.set_training_mode(False)
        target = self.stats.games + n_games
        self._run(lambda: self.stats.games >= target)

    def _run(self, done) -> None:
        """Alternate worker groups: while one plays, receive and decide for the next."""
        while True:
            for g in range(len(self._groups)):
                if self._pending[g]:
                    self._receive(g)
                if done():
                    for h in range(len(self._groups)):
                        if self._pending[h]:
                            self._receive(h)
                    return
                self._decide_and_send(g)

    @torch.no_grad()
    def _decide_and_send(self, g: int) -> None:
        rows = np.concatenate([np.arange(w * self.k, (w + 1) * self.k) for w in self._groups[g]])
        actions = np.zeros(self.n, np.int64)
        movers = np.where(self._learner_turn[rows], SELF, self._opponent[rows])

        arrays = self.shared.arrays
        for name in np.unique(movers):
            idx = rows[movers == name]
            masks = arrays["mask"][idx]
            if name == RANDOM:
                actions[idx] = [self._rng.choice(np.flatnonzero(m)) for m in masks]
                continue
            obs = {k: torch.as_tensor(arrays[k][idx]).to(self.device, non_blocking=True)
                   for k in self.shared.arrays if k != "mask"}
            policy = self.pool.get(name) if name != SELF else self.policy
            if policy is None:  # snapshot evicted mid-game: the learner stands in
                policy = self.policy
            stored = obs
            # Each policy gets only the keys it was built with, so older
            # architectures (fewer keys) can play against newer ones.
            obs = {k: obs[k] for k in policy.observation_space.spaces}
            if policy is not self.policy:
                dist = policy.get_distribution(obs, action_masks=masks)
                actions[idx] = dist.get_actions(deterministic=self.deterministic).cpu().numpy()
                continue
            act, values, log_probs = policy(obs, deterministic=self.deterministic, action_masks=masks)
            act = act.cpu().numpy()
            actions[idx] = act
            if not self.record:
                continue
            seats = self._learner_turn[idx]
            learning = seats | (self._opponent[idx] == SELF)
            if learning.any():
                sel = torch.as_tensor(np.flatnonzero(learning), device=self.device)
                ids = self.store.add({k: t[sel] for k, t in stored.items()},
                                     torch.as_tensor(masks, device=self.device)[sel])
                values = values.flatten().cpu().numpy()[learning]
                log_probs = log_probs.cpu().numpy()[learning]
                for sid, i, seat, a, v, lp in zip(ids, idx[learning], seats[learning], act[learning],
                                                  values, log_probs):
                    self._traj[(int(i), bool(seat))].append([int(sid), int(a), float(v), float(lp), 0.0])

        for w in self._groups[g]:
            self._conns[w].send(("step", actions[w * self.k:(w + 1) * self.k].tolist()))
        self._pending[g] = True
        self.stats.decisions += len(rows)

    def _receive(self, g: int) -> None:
        for w in self._groups[g]:
            for j, (reward, done, learner, info) in enumerate(self._conns[w].recv()):
                self._apply(w * self.k + j, reward, done, learner, info)
        self._pending[g] = False

    def _apply(self, i: int, reward: float, done: bool, learner: bool, info: dict) -> None:
        # Rewards are from the learner seat's view; the game is zero-sum, so the
        # other seat's is the negation. Each lands on that seat's latest action.
        if self.record:
            for seat, sign in ((True, 1.0), (False, -1.0)):
                traj = self._traj.get((i, seat))
                if traj:
                    traj[-1][4] += sign * reward
        self._length[i] += 1
        if not done:
            self._learner_turn[i] = learner
            return
        if self.record:
            for seat in (True, False):
                self._finish(self._traj.pop((i, seat), []))
        if "outcome" in info:
            # Bot games are decided inside the worker; the collector's draw for them is unused.
            opponent = f"bot:{info['bot']}" if info.get("bot") else self._opponent[i]
            self.stats.outcomes[opponent].append(info["outcome"])
            self.stats.bot_failures += info.get("bot_failures", 0)
            if self._opponent[i] == BASELINE:
                self.stats.baseline_by_deck[info["opp_deck"]].append(info["outcome"])
        self.stats.episode_lengths.append(int(self._length[i]))
        self.stats.games += 1
        self._new_game(i, learner)

    def _finish(self, traj: list[list]) -> None:
        """GAE over a finished trajectory (truncated games are treated as ended)."""
        if not traj:
            return
        if not self.store.valid(traj[0][0]):
            self.stats.dropped += len(traj)
            return
        ids, actions, values, log_probs, rewards = (np.array(c) for c in zip(*traj))
        values = values.astype(np.float32)
        advantages = np.zeros(len(traj), np.float32)
        running, next_value = 0.0, 0.0
        for t in range(len(traj) - 1, -1, -1):
            delta = rewards[t] + self.gamma * next_value - values[t]
            running = delta + self.gamma * self.gae_lambda * running
            advantages[t] = running
            next_value = values[t]
        self._ready.append(Batch(ids, actions, values, log_probs.astype(np.float32),
                                 advantages, advantages + values))
        self._n_ready += len(traj)
        self.stats.samples += len(traj)

    def close(self) -> None:
        for conn in self._conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.shared.close()
