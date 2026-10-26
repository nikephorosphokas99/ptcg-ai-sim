"""Worker process hosting several matches.

Deliberately imports nothing beyond the engine bindings and numpy: a worker that
pulls in torch costs ~530 MB instead of ~55 MB, and twenty of those were enough
to push WSL into swap. That is also why ``SelfPlayCollector`` launches this as
``python -m ptcg_rl.worker`` rather than through ``multiprocessing``, whose
children re-import the parent's main module (the training script, and torch).

Observations and masks go into the collector's shared block (``SharedObs``),
one row per game; the pipe carries only rewards and flags.
"""

from __future__ import annotations

import json
import os
import sys
from multiprocessing.connection import Client, Connection

from .encoding import SharedObs, observation_space
from .engine import card_table
from .env import Match, Turn


def run(conn: Connection, match_kwargs: dict, n_matches: int, seed: int,
        shm_name: str, first_row: int, total_rows: int) -> None:
    """Serve commands from ``SelfPlayCollector`` until told to close.

    ``("reset", None)`` -> ``[learner_to_move, ...]``, one per match.
    ``("step", [action, ...])`` (one per match) ->
    ``[(reward, done, learner_to_move, info), ...]``; after a finished game the
    row already holds the next game's first decision.
    ``("weights", {deck: weight})`` -> ``None``, after ``Match.set_opponent_weights``.
    """
    shared = SharedObs(observation_space(len(card_table()) - 1), total_rows, shm_name)
    matches = [Match(**match_kwargs, seed=seed + i) for i in range(n_matches)]

    def publish(j: int, turn: Turn) -> bool:
        shared.put(first_row + j, turn.obs, turn.mask)
        return turn.learner

    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                conn.send([publish(j, m.reset()) for j, m in enumerate(matches)])
            elif cmd == "step":
                out = []
                for j, (m, action) in enumerate(zip(matches, data)):
                    result = m.step(action)
                    turn = m.reset() if result.done else result.turn
                    out.append((result.reward, result.done, publish(j, turn), result.info))
                conn.send(out)
            elif cmd == "weights":
                for m in matches:
                    m.set_opponent_weights(data)
                conn.send(None)
            elif cmd == "close":
                break
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        for m in matches:
            m.close()
        shared.close()
        conn.close()


if __name__ == "__main__":
    # argv: listener address, worker id, matches, seed, shm name, first row, total rows, match kwargs (JSON)
    address, worker_id, n_matches, seed, shm_name, first_row, total_rows, kwargs = sys.argv[1:9]
    conn = Client(address, authkey=bytes.fromhex(os.environ["PTCG_WORKER_KEY"]))
    conn.send(int(worker_id))
    run(conn, json.loads(kwargs), int(n_matches), int(seed), shm_name, int(first_row), int(total_rows))
