"""The trained policy as plain torch, for the submission bundle.

Mirrors ``ptcg_rl.policy.PTCGExtractor`` and the actor half of SB3's
``MaskableActorCriticPolicy`` under the same parameter names, so the training
checkpoint's state dict loads as is; the value head is left out. Kaggle's agent
runtime has torch but not stable-baselines3, which is why this exists.
``ptcg_rl.export`` checks it against the SB3 policy before writing a bundle.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .encoding import GLOBAL_FEATURES, N_ACTIONS, OPTION_FEATURES, SLOT_FEATURES

_BAGS = ("hand_ids", "my_discard_ids", "opp_discard_ids", "view_ids")


class Extractor(nn.Module):
    def __init__(self, card_static: torch.Tensor, embed_dim: int, card_dim: int, option_dim: int):
        super().__init__()
        self.register_buffer("card_static", card_static, persistent=False)
        self.embedding = nn.Embedding(card_static.shape[0], embed_dim, padding_idx=0)
        self.card = nn.Sequential(nn.Linear(embed_dim + card_static.shape[1], card_dim), nn.ReLU())
        self.slot = nn.Sequential(nn.Linear(card_dim + SLOT_FEATURES, card_dim), nn.ReLU())
        self.option = nn.Sequential(
            nn.Linear(2 * card_dim + OPTION_FEATURES, 64), nn.ReLU(), nn.Linear(64, option_dim), nn.ReLU()
        )
        self.globals = nn.Sequential(nn.Linear(GLOBAL_FEATURES, 128), nn.ReLU())

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        b = obs["global"].shape[0]
        table = self.card(torch.cat([self.embedding.weight, self.card_static], dim=-1))
        keep = torch.ones_like(table[:, :1])
        keep[0] = 0.0
        table = table * keep

        def cards(key: str) -> torch.Tensor:
            ids = obs[key].long().clamp(0, table.shape[0] - 1)
            return F.embedding(ids, table, padding_idx=0)

        slots = self.slot(torch.cat([cards("slot_ids"), obs["slot_feats"]], dim=-1))
        bags = [cards(k).sum(dim=1) / 10.0 for k in _BAGS]
        misc = cards("misc_ids")
        options = self.option(torch.cat([cards("opt_ids"), cards("opt_target_ids"), obs["opt_feats"]], dim=-1))
        return torch.cat(
            [slots.reshape(b, -1), *bags, misc.reshape(b, -1), options.reshape(b, -1),
             self.globals(obs["global"])],
            dim=-1,
        )


class PolicyNet(nn.Module):
    """Observation -> logits over the ``N_ACTIONS`` actions (mask them before use)."""

    def __init__(self, card_static: torch.Tensor, arch: dict):
        super().__init__()
        self.features_extractor = Extractor(card_static, arch["embed_dim"], arch["card_dim"], arch["option_dim"])
        layers, width = [], arch["features_dim"]
        for size in arch["pi"]:
            layers += [nn.Linear(width, size), nn.Tanh()]
            width = size
        self.mlp_extractor = nn.Module()
        self.mlp_extractor.policy_net = nn.Sequential(*layers)
        self.action_net = nn.Linear(width, N_ACTIONS)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.action_net(self.mlp_extractor.policy_net(self.features_extractor(obs)))
