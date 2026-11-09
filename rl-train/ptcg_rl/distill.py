"""Search distillation: ISMCTS improves on the policy, and the policy learns the improvement.

AlphaZero's loop, kept beside PPO (``train.py --distill-workers N``):

* **Searchers** (``python -m ptcg_rl.distill``, started by train.py, CPU only) play
  self-play games with the newest weights: our deck against a training-pool deck,
  both seats this network, picks sampled from the policy as in PPO's games. At a
  fraction (``--distill-prob``) of our deck's real choices they run the submission's
  ISMCTS (``submission/ptcg_policy/ismcts.py``: PUCT with the policy as prior, the
  value head at the end of the turn) for a fixed number of iterations, play its
  most visited move, and keep the position with the search's root visit counts.
  The search sees only what the player sees (determinized hidden cards), exactly
  as on Kaggle, and adds no reward of its own: its values come from the value
  head and finished games.
* **The update** (``PTCGPPO.train``) adds ``distill_coef`` x the cross-entropy
  between those visit distributions and the policy, on a minibatch of recent
  searched positions next to every PPO minibatch. Search finds the multi-pick
  lines one pick at a time misses (Boss's Orders, then the attack that knocks the
  target out), and the policy learns to play them without searching.

The trainer publishes its weights to ``<run>/distill/net`` after every update (a
complete ISMCTS bundle, written once, then only ``policy.npz``); searchers reload
them between games and write samples to ``<run>/distill/samples/*.npz``, which
``DistillBuffer`` reads and deletes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

NET_DIR, SAMPLE_DIR = "net", "samples"


def distill_dir(run: Path) -> Path:
    return run / "distill"


# -- trainer side ---------------------------------------------------------------------------------


def publish(model, deck: list[int], run: Path, search: dict, first: bool = False, gpu: bool = False) -> None:
    """Put the current weights where the searchers load them: the full bundle when ``first``.

    With ``gpu`` the inference server's weights (``policy.pt``, a torch state dict; ``model.zip``
    once, for its architecture) are written instead of the numpy ``policy.npz``.
    """
    from .export import _ismcts_files, _write_bundle, policy_arrays  # export imports train

    net = distill_dir(run) / NET_DIR
    if first or not (net / "arch.json").exists():
        tmp = distill_dir(run) / "net.tmp"
        _write_bundle(model, deck, tmp, value=True, rules=True)
        _ismcts_files(tmp, search)
        if gpu:
            model.save(tmp / "model.zip")
            torch.save(model.policy.state_dict(), tmp / "policy.pt")
        if net.exists():
            shutil.rmtree(net)
        tmp.rename(net)
        return
    if gpu:
        torch.save(model.policy.state_dict(), net / "policy.tmp.pt")
        os.replace(net / "policy.tmp.pt", net / "policy.pt")
        return
    np.savez(net / "policy.tmp.npz", **policy_arrays(model.policy, value=True))
    os.replace(net / "policy.tmp.npz", net / "policy.npz")


def server_address() -> str:
    """The inference server's socket name, after the trainer (in Linux's abstract namespace, see ``_socket``)."""
    return f"ptcg-distill-{os.getpid()}"


def _socket(name: str) -> str:
    return "\0" + name  # abstract: no file, no path-length limit; a command line cannot carry the NUL


def start_server(run: Path, device: torch.device) -> subprocess.Popen:
    """The GPU inference server the searchers send their network evaluations to (``--distill-gpu``)."""
    log = open(distill_dir(run) / "server.log", "a")
    return subprocess.Popen([sys.executable, "-m", "ptcg_rl.distill", str(run), "--serve", server_address(),
                             "--device", str(device)], stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ, "OMP_NUM_THREADS": "2"})


def start_searchers(run: Path, n: int, args: argparse.Namespace, first_worker: int = 0) -> list[subprocess.Popen]:
    """``n`` searcher processes, one CPU core each, in this process group (Ctrl-C reaches them)."""
    (distill_dir(run) / SAMPLE_DIR).mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "CUDA_VISIBLE_DEVICES": ""}
    common = [sys.executable, "-m", "ptcg_rl.distill", str(run), "--prob", str(args.distill_prob),
              "--iterations", str(args.distill_iterations), "--skip-confident", str(args.distill_skip_confident),
              "--uncertain", str(args.distill_uncertain), "--opponent-decks", *args.opponent_decks]
    if args.opponent_weights:
        common += ["--opponent-weights", str(args.opponent_weights)]
    if args.distill_gpu:
        common += ["--server", server_address(), "--threads", str(args.distill_threads)]
    log = open(distill_dir(run) / "searchers.log", "a")
    return [subprocess.Popen(common + ["--worker", str(i)], env=env, stdout=log, stderr=subprocess.STDOUT)
            for i in range(first_worker, first_worker + n)]


class DistillBuffer:
    """The latest searched positions (a ring of ``capacity``), for the update's distillation loss."""

    def __init__(self, folder: Path, space, n_actions: int, capacity: int, device: torch.device, seed: int = 0):
        self.folder, self.capacity, self.device = folder, capacity, device
        self.obs = {k: np.zeros((capacity, *s.shape), dtype=s.dtype) for k, s in space.spaces.items()}
        self.masks = np.zeros((capacity, n_actions), dtype=bool)
        self.targets = np.zeros((capacity, n_actions), dtype=np.float32)
        self.size = self.next = 0
        self.total = self.agree = 0
        self.skipped_files = 0
        self.rng = np.random.default_rng(seed)
        self.recent: deque[tuple[float, int]] = deque(maxlen=20)  # (time, total) at each poll, for the rate

    def __len__(self) -> int:
        return self.size

    def poll(self) -> int:
        """Take in every finished sample file; returns how many samples were added."""
        added = 0
        for path in sorted(self.folder.glob("[!.]*.npz")):
            try:
                with np.load(path) as f:
                    data = {k: f[k] for k in f.files}
            except Exception:  # noqa: BLE001 - a half-written file from a killed searcher
                data = None
            path.unlink(missing_ok=True)
            if data is None or any(f"obs/{k}" not in data or data[f"obs/{k}"].shape[1:] != v.shape[1:]
                                   for k, v in self.obs.items()) or data["mask"].shape[1] != self.masks.shape[1]:
                self.skipped_files += 1  # written for another encoding (e.g. before an upgrade)
                continue
            n = len(data["mask"])
            idx = (self.next + np.arange(n)) % self.capacity
            for k, v in self.obs.items():
                v[idx] = data[f"obs/{k}"]
            self.masks[idx], self.targets[idx] = data["mask"], data["target"]
            self.next = int((self.next + n) % self.capacity)
            self.size = min(self.size + n, self.capacity)
            self.total += n
            self.agree += int(data["agree"].sum())
            added += n
        self.recent.append((time.time(), self.total))
        return added

    def rate(self) -> float:
        """Samples per second over the last polls."""
        if len(self.recent) < 2:
            return 0.0
        (t0, n0), (t1, n1) = self.recent[0], self.recent[-1]
        return (n1 - n0) / (t1 - t0) if t1 > t0 else 0.0

    def sample(self, n: int) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        idx = self.rng.integers(0, self.size, size=n)
        obs = {k: torch.as_tensor(v[idx], dtype=torch.float32, device=self.device) for k, v in self.obs.items()}
        return (obs, torch.as_tensor(self.masks[idx], device=self.device),
                torch.as_tensor(self.targets[idx], device=self.device))


# -- searcher process -----------------------------------------------------------------------------


class Seat:
    """One seat of a searcher's game, played as ``submission/main.py`` plays it, with sampled picks."""

    def __init__(self, deck, net, attacks, meta, rules: dict, enc, search=None, uncertain: float = 0.0):
        self.deck, self.net, self.attacks, self.meta, self.enc, self.search = deck, net, attacks, meta, enc, search
        # > 0: a choice is searched with probability prob x min(1, (1 - top move's probability) / uncertain),
        # so the search budget goes to the choices the policy is least sure of (0: prob for every choice).
        self.uncertain = uncertain
        self.safe_attacks = frozenset(rules["safe_attacks"])
        self.planner = enc.CounterPlanner(meta) if rules.get("counter_planner") else None
        self.opp_seen: set[int] = set()
        self.opp_recent: deque[int] = deque(maxlen=enc.RECENT_SLOTS)
        self.prizes = enc.PrizeMemory(deck)
        self.opp_hand = enc.HandMemory()

    def act(self, obs: dict, rng: random.Random, prob: float, out: list) -> list[int]:
        enc, state = self.enc, obs["current"]
        self.opp_recent.extend(enc.recent_opponent_cards(obs, state["yourIndex"]))
        self.prizes.observe(obs)
        self.opp_hand.observe(obs)
        planned = self.planner.pick(obs) if self.planner is not None else None
        if planned is not None:
            return [planned]
        decision = enc.Decision(obs, self.safe_attacks)
        while not decision.complete:
            mask = decision.mask()
            if mask.sum() == 1:
                if decision.apply(int(mask.argmax())):
                    break
                continue
            self.opp_seen.update(enc.visible_cards(state, 1 - state["yourIndex"]))
            features = enc.encode(decision, self.attacks, my_deck=self.deck, opp_seen=self.opp_seen,
                                  opp_recent=self.opp_recent, card_meta=self.meta, prized=self.prizes.prized,
                                  opp_hand=self.opp_hand.cards())
            logits = self.net({k: v[None].astype(np.float32) for k, v in features.items()})[0]
            logits = np.where(mask, logits, -np.inf)
            p = np.exp(logits - logits.max())
            p /= p.sum()
            action = int(rng.choices(range(len(p)), weights=p)[0])
            share = min(1.0, (1.0 - float(p.max())) / self.uncertain) if self.uncertain > 0 else 1.0
            if self.search is not None and rng.random() < prob * share:
                searched = self.search.choose(obs, decision, self.opp_seen, self.opp_recent, self.prizes.prized,
                                              self.opp_hand.cards())
                visits = self.search.last_visits
                if searched is not None and visits:
                    target = np.zeros(len(mask), dtype=np.float32)
                    for a, v in visits.items():
                        target[a] = v
                    target /= target.sum()
                    out.append((features, mask.copy(), target, int(np.argmax(p)) == searched))
                    action = searched
            if decision.apply(action):
                break
        return decision.picked


def _write(folder: Path, worker: int, k: int, samples: list) -> None:
    keys = samples[0][0].keys()
    arrays = {f"obs/{key}": np.stack([s[0][key] for s in samples]) for key in keys}
    arrays["mask"] = np.stack([s[1] for s in samples])
    arrays["target"] = np.stack([s[2] for s in samples])
    arrays["agree"] = np.array([s[3] for s in samples])
    tmp = folder / f".w{worker}_{k}.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, folder / f"w{worker}_{k}_{int(time.time())}.npz")


# -- GPU inference (--distill-gpu) ----------------------------------------------------------------

AUTHKEY = b"ptcg-distill"
GROUPS = 2  # thread groups per searcher, each with its own server connection (see Batcher)


def _forward(policy, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits over every action (unmasked) and the value, in one pass, as ``net.Net.evaluate`` gives them."""
    feats = policy.extract_features(obs)
    if isinstance(feats, tuple):  # separate actor and critic extractors
        pi, vf = policy.mlp_extractor.forward_actor(feats[0]), policy.mlp_extractor.forward_critic(feats[1])
    else:
        pi, vf = policy.mlp_extractor(feats)
    return policy.action_net(pi), policy.value_net(vf).squeeze(-1)


def serve(run: Path, address: str, device: str) -> None:
    """Inference server: answers every searcher's batch of observations with logits and values.

    It takes whatever requests are waiting, runs them as one batch, and reloads ``policy.pt``
    when the trainer publishes new weights. It stops when the trainer does.
    """
    import threading
    from multiprocessing.connection import Listener, wait

    from .train import load_model

    net = distill_dir(run) / NET_DIR
    parent = os.getppid()
    while not (net / "policy.pt").exists():
        time.sleep(1)
    dev = torch.device(device)
    policy = load_model(net / "model.zip", dev).policy
    policy.set_training_mode(False)
    keys = list(policy.observation_space.spaces)
    stamp = None
    conns, lock = [], threading.Lock()
    listener = Listener(_socket(address), authkey=AUTHKEY)

    def accept() -> None:
        while True:
            conn = listener.accept()
            with lock:
                conns.append(conn)

    threading.Thread(target=accept, daemon=True).start()
    print(f"inference server on {device}", flush=True)
    next_check = 0.0
    while os.getppid() == parent:
        if time.time() >= next_check:
            next_check = time.time() + 1.0
            mtime = (net / "policy.pt").stat().st_mtime
            if mtime != stamp:
                try:
                    policy.load_state_dict(torch.load(net / "policy.pt", map_location=dev))
                    stamp = mtime
                except Exception as e:  # noqa: BLE001 - replaced while reading: next second
                    print(f"weights not loaded yet: {e!r}", flush=True)
        with lock:
            live = list(conns)
        if not live:
            time.sleep(0.05)
            continue
        requests = []
        for conn in wait(live, timeout=0.5):
            try:
                requests.append((conn, conn.recv()))
            except (EOFError, OSError):
                with lock:
                    conns.remove(conn)
        if not requests:
            continue
        sizes = [len(r["global"]) for _, r in requests]
        obs = {k: torch.as_tensor(np.concatenate([r[k] for _, r in requests]), dtype=torch.float32, device=dev)
               for k in keys}
        with torch.no_grad():
            logits, values = _forward(policy, obs)
        logits, values = logits.float().cpu().numpy(), values.float().cpu().numpy()
        at = 0
        for (conn, _), n in zip(requests, sizes):
            try:
                conn.send((logits[at:at + n], values[at:at + n]))
            except (EOFError, OSError):
                pass
            at += n


class Batcher:
    """A searcher's threads take turns (one runs Python at a time, holding ``cond``); a thread
    needing the network waits, and when every thread of its group is waiting, the group's
    requests go to the inference server as one batch. Groups have their own connections and
    the lock is let go while a batch is on the GPU, so one group computes while another waits.
    """

    def __init__(self, conns: list, threads: int):
        import threading

        self.cond = threading.Condition()
        self.conns = conns
        self.groups = [{"active": 0, "waiting": []} for _ in conns]
        for i in range(threads):
            self.groups[i % len(conns)]["active"] += 1

    def run(self, group: int, features: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """Logits and values for ``features``; call while holding ``cond``."""
        g = self.groups[group]
        request = {"features": features, "result": None}
        g["waiting"].append(request)
        if len(g["waiting"]) >= g["active"]:
            self._flush(group)
        else:
            self.cond.wait_for(lambda: request["result"] is not None)
        return request["result"]

    def leave(self, group: int) -> None:
        """A thread is done for good: its group no longer waits for it."""
        g = self.groups[group]
        g["active"] -= 1
        if g["waiting"] and len(g["waiting"]) >= g["active"]:
            self._flush(group)

    def _flush(self, group: int) -> None:
        g, conn = self.groups[group], self.conns[group]
        requests, g["waiting"] = g["waiting"], []
        flat = [f for r in requests for f in r["features"]]
        try:
            conn.send({k: np.stack([f[k] for f in flat]) for k in flat[0]})
            self.cond.release()  # the other groups run while this batch is on the GPU
            try:
                logits, values = conn.recv()
            finally:
                self.cond.acquire()
        except (EOFError, OSError) as e:  # the server is gone: exit, the trainer starts a new searcher
            print(f"inference server lost ({e!r}); exiting", flush=True)
            os._exit(3)
        at = 0
        for r in requests:
            n = len(r["features"])
            r["result"] = (logits[at:at + n], values[at:at + n])
            at += n
        self.cond.notify_all()


class RemoteNet:
    """``net.Net``'s interface (``evaluate``, batched call) answered by the inference server, for one thread."""

    has_value = True

    def __init__(self, batcher: Batcher, group: int):
        self.batcher, self.group = batcher, group

    def evaluate(self, features: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        logits, values = self.batcher.run(self.group, [features])
        return logits[0], float(values[0])

    def __call__(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        n = len(obs["global"])
        return self.batcher.run(self.group, [{k: v[i] for k, v in obs.items()} for i in range(n)])[0]


# -- searcher process -------------------------------------------------------------------------------


def searcher(run: Path, worker: int, prob: float, iterations: int, skip_confident: float,
             opponent_decks: list[str], opponent_weights: Path | None, flush_every: int = 64,
             server: str | None = None, threads: int = 1, uncertain: float = 0.0) -> None:
    """Play searched self-play games forever (until the trainer exits), writing samples as they come.

    With ``server`` the network runs on the GPU inference server and ``threads`` games run at
    once, so their evaluations share batches; otherwise the bundle's numpy network runs here.
    """
    import threading

    from .engine import Game, load_deck

    bundle = distill_dir(run) / NET_DIR
    folder = distill_dir(run) / SAMPLE_DIR
    parent = os.getppid()
    while not (bundle / "policy.npz").exists():
        time.sleep(2)
    sys.path.insert(0, str(bundle))
    import cg.sim  # noqa: E402  the bundle's engine, for the search API (as on Kaggle)
    from ptcg_policy import encoding as enc  # noqa: E402
    from ptcg_policy.ismcts import ISMCTS, EngineSearch  # noqa: E402
    from ptcg_policy.net import load  # noqa: E402

    deck = [int(x) for x in (bundle / "deck.csv").read_text().split()[:60]]
    rules = json.loads((bundle / "rules.json").read_text())
    opp_lists = json.loads((bundle / "opp_decks.json").read_text())
    config = {**json.loads((bundle / "ismcts.json").read_text()), "max_seconds": 60.0,
              "max_iterations": iterations, "skip_confident": skip_confident, "reuse_tree": False}
    weights = json.loads(opponent_weights.read_text()) if opponent_weights else {}
    pool = [(load_deck(p), weights.get(Path(p).stem, 1.0)) for p in opponent_decks]
    pool = [(d, w) for d, w in pool if w > 0]
    seed = worker * 7919 + int(time.time())
    out = {"samples": [], "k": 0}

    def play(rng: random.Random, net, attacks, meta, engine) -> None:
        """One game; samples are flushed to disk every ``flush_every``."""
        search = ISMCTS(net, attacks, meta, deck, opp_lists, engine, config)
        opp_deck = rng.choices([d for d, _ in pool], weights=[w for _, w in pool])[0]
        seat = rng.randrange(2)
        decks = [deck, opp_deck] if seat == 0 else [opp_deck, deck]
        seats = [Seat(decks[s], net, attacks, meta, rules, enc, search if s == seat else None, uncertain)
                 for s in (0, 1)]
        game = Game(*decks, search_input=True)
        try:
            for _ in range(3000):
                if game.done:
                    break
                game.select(seats[game.to_move].act(game.obs, rng, prob, out["samples"]))
                if len(out["samples"]) >= flush_every:
                    _write(folder, worker, out["k"], out["samples"])
                    out["samples"], out["k"] = [], out["k"] + 1
        finally:
            game.close()

    if server is None:
        rng, engine = random.Random(seed), EngineSearch(cg.sim.lib)
        stamp, net = None, None
        while os.getppid() == parent:  # the trainer is gone: stop
            mtime = (bundle / "policy.npz").stat().st_mtime
            if mtime != stamp:
                try:
                    net, attacks, meta = load(bundle)
                    stamp = mtime
                except Exception:  # noqa: BLE001 - replaced while reading; the next game retries
                    if net is None:
                        time.sleep(1)
                        continue
            play(rng, net, attacks, meta, engine)
        return

    from multiprocessing.connection import Client

    _, attacks, meta = load(bundle)  # only the tables: the server holds the weights
    config["max_seconds"] = 600.0  # a search waits for the other threads' evaluations too
    conns = []
    for _ in range(120):  # the server may still be loading its model
        try:
            conns = [Client(_socket(server), authkey=AUTHKEY) for _ in range(GROUPS)]
            break
        except OSError:
            time.sleep(1)
    else:
        raise SystemExit(f"no inference server at {server!r}")
    batcher = Batcher(conns, threads)

    def thread_main(i: int) -> None:
        rng, engine, group = random.Random(seed * 101 + i), EngineSearch(cg.sim.lib), i % GROUPS
        net = RemoteNet(batcher, group)
        with batcher.cond:
            try:
                while os.getppid() == parent:
                    try:
                        play(rng, net, attacks, meta, engine)
                    except Exception as e:  # noqa: BLE001 - one bad game must not stop the searcher
                        print(f"searcher {worker}.{i}: game failed: {e!r}", flush=True)
            finally:
                batcher.leave(group)

    pool_threads = [threading.Thread(target=thread_main, args=(i,), daemon=True) for i in range(threads)]
    for t in pool_threads:
        t.start()
    while os.getppid() == parent and any(t.is_alive() for t in pool_threads):
        time.sleep(5)
    os._exit(0)  # threads still waiting on the server are dropped with the process


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path)
    p.add_argument("--serve", metavar="ADDRESS", help="run the GPU inference server on this socket")
    p.add_argument("--device", default="cuda:0", help="with --serve")
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--prob", type=float, default=0.25)
    p.add_argument("--iterations", type=int, default=64)
    p.add_argument("--skip-confident", type=float, default=0.97)
    p.add_argument("--uncertain", type=float, default=0.0)
    p.add_argument("--opponent-decks", nargs="*", default=[])
    p.add_argument("--opponent-weights", type=Path)
    p.add_argument("--server", metavar="ADDRESS", help="send network evaluations to this inference server")
    p.add_argument("--threads", type=int, default=1, help="with --server: games played at once")
    a = p.parse_args()
    if a.serve:
        serve(a.run, a.serve, a.device)
        return
    searcher(a.run, a.worker, a.prob, a.iterations, a.skip_confident, a.opponent_decks, a.opponent_weights,
             server=a.server, threads=a.threads, uncertain=a.uncertain)


if __name__ == "__main__":
    main()
