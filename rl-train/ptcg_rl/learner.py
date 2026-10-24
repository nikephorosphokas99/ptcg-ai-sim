"""The PPO update in its own process, on its own GPU (``train.py --pipeline``).

The collector's process is busy with Python the whole time it collects (worker
replies, trajectories, batching) and so is the update (kernel launches,
minibatches); in one process they take turns on the GIL, so two threads on two
GPUs ran one after the other. Here the update runs in a separate process with
its own interpreter: while it trains on rollout k, the collector plays rollout
k+1, and both GPUs work at once.

Per update the collector sends the rollout's observations and masks as CUDA
tensors (shared through CUDA IPC, no trip through RAM) plus the PPO targets; the
learner copies them to its GPU, trains, and sends the new weights back (~4 MB).
The learner holds the model that is saved, optimizer state included, so it
writes every checkpoint.
"""

from __future__ import annotations

import signal
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

from .storage import Rollout


def _serve(conn, model_path: str, device: str, settings: dict) -> None:
    """Learner process: load the model, then train / save on request until told to stop."""
    # Ctrl-C / a notebook interrupt goes to the whole process group; the collector's
    # process handles it and then asks this one to save final.zip.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        from stable_baselines3.common.logger import Logger

        from .train import load_model

        if device.startswith("cuda"):
            torch.cuda.set_device(torch.device(device))
        model = load_model(Path(model_path), device)
        model.learning_rate = settings["learning_rate"]
        model._setup_lr_schedule()
        model.target_kl = settings["target_kl"]
        model.amp = settings["amp"]
        model.set_logger(Logger(folder=None, output_formats=[]))
        keys = list(model.observation_space.spaces)
        conn.send(("ready", None))
        while True:
            msg, payload = conn.recv()
            if msg == "train":
                start = time.time()
                obs, masks, arrays, progress, num_timesteps = payload
                rollout = Rollout.from_tensors({k: obs[k] for k in keys}, masks, model.device)
                del obs, masks  # drop the collector's CUDA memory before training
                conn.send(("received", None))
                model.rollout_buffer.load(rollout, np.arange(len(arrays["actions"])), arrays["actions"],
                                          arrays["values"], arrays["log_probs"], arrays["advantages"],
                                          arrays["returns"])
                model.num_timesteps = num_timesteps
                model._current_progress_remaining = progress
                model.train()
                logs = dict(model.logger.name_to_value)
                model.logger.name_to_value.clear()
                weights = {k: v.detach().cpu() for k, v in model.policy.state_dict().items()}
                conn.send(("trained", (weights, logs, time.time() - start)))
            elif msg == "save":
                path, num_timesteps = payload
                model.num_timesteps = num_timesteps
                model.save(path)
                conn.send(("saved", None))
            elif msg == "stop":
                conn.send(("stopped", None))
                return
    except Exception:
        conn.send(("error", traceback.format_exc()))


class RemoteLearner:
    """Handle on the learner process. ``submit`` returns at once; ``result`` waits for the update."""

    def __init__(self, model_path: Path, device: torch.device, settings: dict):
        ctx = mp.get_context("spawn")  # CUDA cannot be used in a forked child
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_serve, args=(child, str(model_path), str(device), settings), daemon=True)
        self._proc.start()
        child.close()
        self._held = None  # the rollout's CUDA tensors stay alive here until the learner has copied them
        self._expect("ready")

    def _expect(self, kind: str):
        try:
            msg, payload = self._conn.recv()
        except EOFError:
            raise RuntimeError(f"learner process exited (code {self._proc.exitcode}) before {kind!r}") from None
        if msg == "error":
            raise RuntimeError(f"learner process failed:\n{payload}")
        if msg != kind:
            raise RuntimeError(f"learner process sent {msg!r}, expected {kind!r}")
        return payload

    def submit(self, rollout: Rollout, arrays: dict[str, np.ndarray], progress: float, num_timesteps: int) -> None:
        self._held = rollout
        self._conn.send(("train", (rollout.obs, rollout.masks, arrays, progress, num_timesteps)))
        self._expect("received")
        self._held = None

    def result(self) -> tuple[dict[str, torch.Tensor], dict, float]:
        """(policy weights on the CPU, the update's train/* log values, seconds it took)."""
        return self._expect("trained")

    def save(self, path: Path, num_timesteps: int) -> None:
        self._conn.send(("save", (str(path), num_timesteps)))
        self._expect("saved")

    def close(self) -> None:
        if self._proc.is_alive():
            try:
                self._conn.send(("stop", None))
                self._expect("stopped")
            except (EOFError, BrokenPipeError, RuntimeError):
                pass
        self._proc.join(timeout=30)
        if self._proc.is_alive():
            self._proc.kill()
