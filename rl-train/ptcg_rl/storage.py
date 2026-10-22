"""Training samples kept on the GPU.

Observations reach the GPU once, for the forward pass that picks the learner's
action, and stay there: ``SampleStore`` keeps them in a ring (ids as int16,
features as float16, ~9 KB a sample), and ``GPURolloutBuffer`` hands SB3's
``MaskablePPO.train()`` minibatches gathered from that ring. With SB3's own
buffer, a 65k-sample update meant ~5 GB of minibatch copies from RAM to VRAM.
"""

from __future__ import annotations

from collections.abc import Generator

import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer, MaskableDictRolloutBufferSamples

from .encoding import N_ACTIONS


class SampleStore:
    """Ring of observations and masks; a sample is addressed by a global id.

    A slot is reused ``capacity`` samples later, so an id is only valid while
    ``count - id <= capacity``; trajectories older than that are dropped.
    """

    def __init__(self, space: spaces.Dict, capacity: int, device: torch.device):
        self.capacity, self.device, self.count = capacity, device, 0
        self.obs = {
            k: torch.zeros((capacity, *s.shape), device=device,
                           dtype=torch.int16 if np.issubdtype(s.dtype, np.integer) else torch.float16)
            for k, s in space.spaces.items()
        }
        self.masks = torch.zeros((capacity, N_ACTIONS), dtype=torch.bool, device=device)

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in [*self.obs.values(), self.masks])

    def add(self, obs: dict[str, torch.Tensor], masks: torch.Tensor) -> np.ndarray:
        n = masks.shape[0]
        ids = np.arange(self.count, self.count + n)
        slots = torch.as_tensor(ids % self.capacity, device=self.device)
        for k, t in obs.items():
            self.obs[k][slots] = t.to(self.obs[k].dtype)
        self.masks[slots] = masks
        self.count += n
        return ids

    def valid(self, oldest_id: int) -> bool:
        return self.count - oldest_id <= self.capacity

    def gather(self, ids: torch.Tensor, keys=None) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        slots = ids % self.capacity
        return {k: self.obs[k][slots] for k in (keys or self.obs)}, self.masks[slots]


class GPURolloutBuffer(MaskableDictRolloutBuffer):
    """Serves ``MaskablePPO.train()`` from a ``SampleStore`` instead of RAM."""

    def reset(self) -> None:
        # Skip the parent's RAM allocation of buffer_size full observations.
        self.pos, self.full, self.generator_ready = 0, False, False
        self.values = self.returns = np.zeros(0, np.float32)

    def load(self, store: SampleStore, ids: np.ndarray, actions: np.ndarray, values: np.ndarray,
             log_probs: np.ndarray, advantages: np.ndarray, returns: np.ndarray) -> None:
        dev = store.device
        self.store = store
        self.keys = list(self.observation_space.spaces)  # the model's own keys; the store may hold more
        self.ids = torch.as_tensor(ids, device=dev)
        self.t_actions = torch.as_tensor(actions, device=dev).float().unsqueeze(1)
        self.t_values = torch.as_tensor(values, device=dev)
        self.t_log_probs = torch.as_tensor(log_probs, device=dev)
        self.t_advantages = torch.as_tensor(advantages, device=dev)
        self.t_returns = torch.as_tensor(returns, device=dev)
        self.values, self.returns = values, returns  # read by train() for explained variance
        self.pos, self.full = len(ids), True

    def get(self, batch_size: int | None = None) -> Generator[MaskableDictRolloutBufferSamples, None, None]:
        n = len(self.ids)
        batch_size = batch_size or n
        perm = torch.randperm(n, device=self.store.device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            obs, masks = self.store.gather(self.ids[idx], self.keys)
            yield MaskableDictRolloutBufferSamples(
                observations=obs,
                actions=self.t_actions[idx],
                old_values=self.t_values[idx],
                old_log_prob=self.t_log_probs[idx],
                advantages=self.t_advantages[idx],
                returns=self.t_returns[idx],
                action_masks=masks,
            )
