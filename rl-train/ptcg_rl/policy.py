"""Features extractor for SB3's ``MultiInputPolicy``.

Each card id becomes a learned embedding concatenated with its static rules
features (``engine.card_table``). Board slots and options keep their position
(the action head scores option ``i`` from slot ``i``); hand, discard piles and
the cards being looked at are order-free, so they are sum-pooled.
"""

from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from .encoding import GLOBAL_FEATURES, MAX_OPTIONS, OPTION_FEATURES, SLOT_FEATURES, SLOTS
from .engine import card_table

_BAGS = ("hand_ids", "my_discard_ids", "opp_discard_ids", "view_ids")


class PTCGExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, card_dim: int = 64, option_dim: int = 32):
        table = torch.as_tensor(card_table())
        n_cards = table.shape[0]
        features_dim = (
            SLOTS * card_dim + len(_BAGS) * card_dim + 3 * card_dim + MAX_OPTIONS * option_dim + 128
        )
        super().__init__(observation_space, features_dim)

        self.register_buffer("card_static", table, persistent=False)
        self.embedding = nn.Embedding(n_cards, 32, padding_idx=0)
        self.card = nn.Sequential(nn.Linear(32 + table.shape[1], card_dim), nn.ReLU())
        self.slot = nn.Sequential(nn.Linear(card_dim + SLOT_FEATURES, card_dim), nn.ReLU())
        self.option = nn.Sequential(
            nn.Linear(2 * card_dim + OPTION_FEATURES, 64), nn.ReLU(), nn.Linear(64, option_dim), nn.ReLU()
        )
        self.globals = nn.Sequential(nn.Linear(GLOBAL_FEATURES, 128), nn.ReLU())

    def _cards(self, ids: torch.Tensor) -> torch.Tensor:
        ids = ids.long().clamp_(0, self.card_static.shape[0] - 1)
        out = self.card(torch.cat([self.embedding(ids), self.card_static[ids]], dim=-1))
        return out * (ids > 0).unsqueeze(-1)  # padding / face-down contributes nothing

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        b = obs["global"].shape[0]
        slots = self.slot(torch.cat([self._cards(obs["slot_ids"]), obs["slot_feats"]], dim=-1))
        bags = [self._cards(obs[k]).sum(dim=1) / 10.0 for k in _BAGS]
        misc = self._cards(obs["misc_ids"])
        options = self.option(torch.cat(
            [self._cards(obs["opt_ids"]), self._cards(obs["opt_target_ids"]), obs["opt_feats"]], dim=-1
        ))
        return torch.cat(
            [slots.reshape(b, -1), *bags, misc.reshape(b, -1), options.reshape(b, -1),
             self.globals(obs["global"])],
            dim=-1,
        )
