"""Policies for MaskablePPO.

Three generations: ``PTCGExtractor`` (with SB3's ``MultiInputPolicy``), one
flat vector into SB3's linear head; ``PointerPolicy``, which scores each option
from its own content and reads the deck memory keys; ``EntityPolicy`` (v3),
attention over board, hand and option tokens with options wired to the slots
they target, attack tactics and card-text features.

Each card id becomes a learned embedding concatenated with its static rules
features (``engine.card_table``). Board slots and options keep their position
(the action head scores option ``i`` from slot ``i``); hand, discard piles and
the cards being looked at are order-free, so they are sum-pooled.
"""

from __future__ import annotations

import torch
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from torch.nn import functional as F

from .encoding import (LEDGER_FEATURES, MAX_OPTIONS, OPT_OPPONENT, OPTION_FEATURES, SLOTS,
                       TACTICAL_FEATURES)
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
        self.slot = nn.Sequential(nn.Linear(card_dim + observation_space["slot_feats"].shape[-1], card_dim), nn.ReLU())
        self.option = nn.Sequential(
            nn.Linear(2 * card_dim + OPTION_FEATURES, 64), nn.ReLU(), nn.Linear(64, option_dim), nn.ReLU()
        )
        self.globals = nn.Sequential(nn.Linear(observation_space["global"].shape[-1], 128), nn.ReLU())

    def card_vectors(self) -> torch.Tensor:
        """Every card's representation, row 0 (padding / face-down) zeroed.

        The card encoder depends on the id alone, so it runs once over the
        ~1,400 known cards per forward pass instead of on each of the ~370 card
        slots of every observation in the batch.
        """
        table = self.card(torch.cat([self.embedding.weight, self.card_static], dim=-1))
        keep = torch.ones_like(table[:, :1])
        keep[0] = 0.0
        return table * keep

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        b = obs["global"].shape[0]
        table = self.card_vectors()

        def cards(key: str) -> torch.Tensor:
            # F.embedding rather than table[ids]: most slots are padding (id 0)
            # or the same few cards, and plain indexing's backward serialises on
            # those rows (~2x slower than before the table, not ~2x faster).
            ids = obs[key].long().clamp_(0, table.shape[0] - 1)
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


# -- pointer policy ---------------------------------------------------------------
#
# The flat policy above hands SB3 one long vector and SB3's action head learns a
# fixed weight per option *slot*: "slot 3" is an attack in one decision and Boss's
# Orders in the next, so the head can only guess what a slot means from its
# position. Here every option gets its own vector (its card, its target, its
# kind), scored against a summary of the board, and STOP has a head of its own.

_POINTER_BAGS = _BAGS + ("my_unseen_ids", "opp_seen_ids")


class PointerExtractor(BaseFeaturesExtractor):
    """Board state (``state_dim``) followed by one vector per option slot (``option_dim``)."""

    def __init__(self, observation_space: spaces.Dict, card_dim: int = 64, state_dim: int = 256,
                 option_dim: int = 128):
        table = torch.as_tensor(card_table())
        super().__init__(observation_space, state_dim + MAX_OPTIONS * option_dim)
        self.state_dim, self.option_dim = state_dim, option_dim
        self.register_buffer("card_static", table, persistent=False)
        self.embedding = nn.Embedding(table.shape[0], 32, padding_idx=0)
        self.card = nn.Sequential(nn.Linear(32 + table.shape[1], card_dim), nn.ReLU())
        self.slot = nn.Sequential(nn.Linear(card_dim + observation_space["slot_feats"].shape[-1], card_dim), nn.ReLU())
        self.globals = nn.Sequential(nn.Linear(observation_space["global"].shape[-1], 128), nn.ReLU())
        board = SLOTS * card_dim + len(_POINTER_BAGS) * card_dim + 3 * card_dim + 128
        self.board = nn.Sequential(nn.Linear(board, 512), nn.ReLU(), nn.Linear(512, state_dim), nn.ReLU())
        self.option = nn.Sequential(
            nn.Linear(2 * card_dim + OPTION_FEATURES, option_dim), nn.ReLU(),
            nn.Linear(option_dim, option_dim), nn.ReLU(),
        )
        # The state also sees what is on offer, pooled over the real options.
        self.fuse = nn.Sequential(nn.Linear(state_dim + option_dim, state_dim), nn.ReLU())

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        b = obs["global"].shape[0]
        table = self.card(torch.cat([self.embedding.weight, self.card_static], dim=-1))
        keep = torch.ones_like(table[:, :1])
        keep[0] = 0.0
        table = table * keep

        def cards(key: str) -> torch.Tensor:
            ids = obs[key].long().clamp_(0, table.shape[0] - 1)
            return F.embedding(ids, table, padding_idx=0)

        slots = self.slot(torch.cat([cards("slot_ids"), obs["slot_feats"]], dim=-1))
        bags = [cards(k).sum(dim=1) / 10.0 for k in _POINTER_BAGS]
        board = torch.cat([slots.reshape(b, -1), *bags, cards("misc_ids").reshape(b, -1),
                           self.globals(obs["global"])], dim=-1)
        state = self.board(board)

        options = self.option(torch.cat([cards("opt_ids"), cards("opt_target_ids"), obs["opt_feats"]], dim=-1))
        real = (obs["opt_feats"].abs().sum(dim=-1, keepdim=True) > 0).float()  # every real option has a type
        pooled = (options * real).sum(dim=1) / real.sum(dim=1).clamp(min=1.0)
        state = self.fuse(torch.cat([state, pooled], dim=-1))
        return torch.cat([state, options.reshape(b, -1)], dim=-1)


class PointerHead(nn.Module):
    """SB3 ``mlp_extractor``: the actor latent *is* the 65 logits; the critic reads the state."""

    def __init__(self, state_dim: int, option_dim: int, hidden: int = 128, value_dim: int = 256):
        super().__init__()
        self.state_dim, self.option_dim = state_dim, option_dim
        self.latent_dim_pi, self.latent_dim_vf = MAX_OPTIONS + 1, value_dim
        self.query = nn.Linear(state_dim, hidden)
        self.score = nn.Sequential(nn.Linear(option_dim + hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.stop = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.value = nn.Sequential(nn.Linear(state_dim, value_dim), nn.Tanh(), nn.Linear(value_dim, value_dim), nn.Tanh())

    def _split(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = features[:, : self.state_dim]
        return state, features[:, self.state_dim:].reshape(-1, MAX_OPTIONS, self.option_dim)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        state, options = self._split(features)
        query = self.query(state).unsqueeze(1).expand(-1, MAX_OPTIONS, -1)
        scores = self.score(torch.cat([options, query], dim=-1)).squeeze(-1)
        return torch.cat([scores, self.stop(state)], dim=-1)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value(features[:, : self.state_dim])

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)


class PointerPolicy(MaskableMultiInputActorCriticPolicy):
    """Masked actor-critic whose logits come straight from ``PointerHead``."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("features_extractor_class", PointerExtractor)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        ext = self.features_extractor
        self.mlp_extractor = PointerHead(ext.state_dim, ext.option_dim)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        self.action_net = nn.Identity()  # the head already outputs one logit per action
        # Near-uniform first policy, as SB3 does for its own action head.
        for layer in (self.mlp_extractor.score[-1], self.mlp_extractor.stop[-1]):
            nn.init.orthogonal_(layer.weight, gain=0.01)
            nn.init.zeros_(layer.bias)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


# -- entity policy (v3) --------------------------------------------------------------
#
# The pointer policy above still reads the board as one flattened vector, and an
# option knows its target only by card id: "Boss's Orders on bench slot 2" did
# not see that slot's HP, energy or tools. Here the board is a set of tokens (one
# per Pokémon with its attachments, per hand card, per pile, plus the stadium,
# the effect being resolved and the global state) mixed by self-attention; each
# option is wired to the encoded tokens of the slots it acts from and targets,
# carries attack tactics (damage after Weakness/Resistance, knock-out, Prizes),
# and options attend to each other and to the board before being scored. Cards
# are described by their printed numbers *and* their rules text
# (``cardtext.card_text_table``).

_ENTITY_PILES = ("my_discard_ids", "opp_discard_ids", "view_ids", "my_unseen_ids", "opp_seen_ids", "opp_recent_ids")


class EntityExtractor(BaseFeaturesExtractor):
    """Global state (``dim``) followed by one token per option slot (``dim`` each), then STOP's (``dim``)
    when ``stop_token``."""

    def __init__(self, observation_space: spaces.Dict, dim: int = 128, heads: int = 4,
                 board_layers: int = 2, option_layers: int = 2, ff: int = 256, stop_token: bool = False):
        from .cardtext import card_text_table

        static = torch.as_tensor(card_table())
        text = torch.as_tensor(card_text_table())
        super().__init__(observation_space, dim + MAX_OPTIONS * dim + (dim if stop_token else 0))
        self.dim = dim
        # STOP as one more option token: a learned vector that goes through the option decoder with the
        # real options (seeing them, what is already picked, and the board) and is scored by the same head.
        # Without it STOP is scored from the board summary alone (models before run D).
        self.stop_token = stop_token
        if stop_token:
            self.stop_embed = nn.Parameter(torch.randn(dim) * 0.02)
        self.register_buffer("card_static", static, persistent=False)
        self.register_buffer("card_text", text, persistent=False)
        self.embedding = nn.Embedding(static.shape[0], 32, padding_idx=0)
        self.card = nn.Sequential(nn.Linear(32 + static.shape[1] + text.shape[1], dim), nn.ReLU(),
                                  nn.Linear(dim, dim))
        # card, attachments, features (as many as the model was built with), side
        self.slot = nn.Linear(2 * dim + observation_space["slot_feats"].shape[-1] + 2, dim)
        self.pile = nn.Linear(dim, dim)
        self.globals = nn.Linear(observation_space["global"].shape[-1], dim)
        # learned type markers: hand, each pile, stadium, context card, effect card
        self.kind = nn.Embedding(1 + len(_ENTITY_PILES) + 3, dim)
        self.board = nn.ModuleList(
            nn.TransformerEncoderLayer(dim, heads, ff, dropout=0.0, batch_first=True, norm_first=True)
            for _ in range(board_layers)
        )
        self.option = nn.Linear(2 * dim + OPTION_FEATURES + observation_space["opt_tactical"].shape[-1], dim)
        self.option_src = nn.Linear(dim, dim, bias=False)
        self.option_tgt = nn.Linear(dim, dim, bias=False)
        self.options = nn.ModuleList(
            nn.TransformerDecoderLayer(dim, heads, ff, dropout=0.0, batch_first=True, norm_first=True)
            for _ in range(option_layers)
        )
        self.state = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())
        self.out_norm = nn.LayerNorm(dim)
        # Deck ledger (v4, when the space has it): one vector per distinct card of our list,
        # added to the global token and to our hand, board and option tokens of that card.
        # The projections start at zero, so a v3 model upgraded to v4 plays as before.
        self.has_ledger = "ledger_ids" in observation_space.spaces
        if self.has_ledger:
            self.ledger = nn.Sequential(nn.Linear(dim + LEDGER_FEATURES, dim), nn.ReLU(), nn.Linear(dim, dim))
            self.ledger_global = nn.Linear(dim, dim, bias=False)
            self.ledger_slot = nn.Linear(dim, dim, bias=False)
            self.ledger_hand = nn.Linear(dim, dim, bias=False)
            self.ledger_option = nn.Linear(dim, dim, bias=False)
            self.zero_ledger()
        # Opponent's known hand (v6, HandMemory): the sum of its cards' vectors, projected into
        # the global token. Starts at zero, so an upgraded v5 model plays exactly as before.
        self.has_opp_hand = "opp_hand_ids" in observation_space.spaces
        if self.has_opp_hand:
            self.opp_hand = nn.Linear(dim, dim, bias=False)
            nn.init.zeros_(self.opp_hand.weight)

    def zero_ledger(self) -> None:
        """Start the ledger's projections at zero (again after SB3's orthogonal init)."""
        for layer in (self.ledger_global, self.ledger_slot, self.ledger_hand, self.ledger_option):
            nn.init.zeros_(layer.weight)

    def card_vectors(self) -> torch.Tensor:
        table = self.card(torch.cat([self.embedding.weight, self.card_static, self.card_text], dim=-1))
        keep = torch.ones_like(table[:, :1])
        keep[0] = 0.0
        return table * keep

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        b, d = obs["global"].shape[0], self.dim
        table = self.card_vectors()

        def cards(x: torch.Tensor) -> torch.Tensor:
            return F.embedding(x.long().clamp(0, table.shape[0] - 1), table, padding_idx=0)

        # Board tokens and their padding mask (True = absent).
        side = torch.zeros(SLOTS, 2, device=table.device)
        side[: SLOTS // 2, 0] = 1.0
        side[SLOTS // 2:, 1] = 1.0
        attach = cards(obs["slot_attach_ids"]).sum(dim=2)
        slots = self.slot(torch.cat([cards(obs["slot_ids"]), attach, obs["slot_feats"],
                                     side.expand(b, -1, -1)], dim=-1))
        slot_absent = obs["slot_feats"][..., 0] <= 0
        # Hand and options are packed from index 0, so a batch only needs as many
        # tokens as its fullest row; masked padding would cost full attention.
        hand_ids = obs["hand_ids"][:, : max(1, int((obs["hand_ids"] > 0).sum(dim=1).max()))]
        hand = cards(hand_ids) + self.kind.weight[0]
        hand_absent = hand_ids <= 0
        glob = self.globals(obs["global"]).unsqueeze(1)
        if self.has_ledger:
            ledger_ids = obs["ledger_ids"].long()
            ledger_present = ledger_ids > 0
            ledger = self.ledger(torch.cat([cards(ledger_ids), obs["ledger_feats"]], dim=-1))
            ledger = ledger * ledger_present.unsqueeze(-1)

            def ledger_of(ids: torch.Tensor) -> torch.Tensor:
                """Each id's ledger vector, zero for cards not in our list."""
                match = (ids.long().unsqueeze(-1) == ledger_ids.unsqueeze(1)) & ledger_present.unsqueeze(1)
                return match.to(ledger.dtype) @ ledger

            mine = torch.zeros(SLOTS, 1, device=table.device)
            mine[: SLOTS // 2] = 1.0
            slots = slots + self.ledger_slot(ledger_of(obs["slot_ids"]) * mine)
            hand = hand + self.ledger_hand(ledger_of(hand_ids))
            count = ledger_present.sum(dim=1, keepdim=True).clamp(min=1)
            glob = glob + self.ledger_global(ledger.sum(dim=1) / count).unsqueeze(1)
        if self.has_opp_hand:
            glob = glob + self.opp_hand(cards(obs["opp_hand_ids"]).sum(dim=1) / 10.0).unsqueeze(1)
        # A pile is the sum of its cards' vectors, i.e. per-card counts times the table:
        # far cheaper than looking up and summing up to 60 vectors per pile.
        width = max(obs[k].shape[1] for k in _ENTITY_PILES)
        ids = torch.stack([F.pad(obs[k], (0, width - obs[k].shape[1])) for k in _ENTITY_PILES], dim=1)
        ids = ids.long().clamp(0, table.shape[0] - 1)  # padding is id 0, whose table row is zero
        counts = torch.zeros(b, len(_ENTITY_PILES), table.shape[0], device=table.device)
        counts.scatter_add_(2, ids, torch.ones_like(ids, dtype=counts.dtype))
        piles = self.pile(counts @ table / 10.0) + self.kind.weight[1:1 + len(_ENTITY_PILES)]
        misc = cards(obs["misc_ids"]) + self.kind.weight[1 + len(_ENTITY_PILES):]
        misc_absent = obs["misc_ids"] <= 0
        tokens = torch.cat([glob, slots, hand, piles, misc], dim=1)
        absent = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=table.device), slot_absent, hand_absent,
                            torch.zeros(b, piles.shape[1], dtype=torch.bool, device=table.device), misc_absent], dim=1)
        for layer in self.board:
            tokens = layer(tokens, src_key_padding_mask=absent)
        state = self.state(tokens[:, 0])
        board_slots = tokens[:, 1:1 + SLOTS]

        # Option tokens, wired to the slots they act from and target.
        def gather(index: torch.Tensor) -> torch.Tensor:
            idx = index.long()
            picked = torch.gather(board_slots, 1, idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, d))
            return picked * (idx >= 0).unsqueeze(-1)

        opt_absent = obs["opt_feats"].abs().sum(dim=-1) <= 0
        n = max(1, int((~opt_absent).sum(dim=1).max()))
        o = {k: obs[k][:, :n] for k in ("opt_ids", "opt_target_ids", "opt_feats", "opt_tactical", "opt_src_slot", "opt_tgt_slot")}
        opt_absent = opt_absent[:, :n].clone()
        if not self.stop_token:  # the STOP token is never masked, so attention always has a key
            opt_absent[:, 0] &= ~opt_absent.all(dim=1)  # never mask every option (attention over nothing is NaN)
        options = self.option(torch.cat([cards(o["opt_ids"]), cards(o["opt_target_ids"]),
                                         o["opt_feats"], o["opt_tactical"]], dim=-1))
        options = options + self.option_src(gather(o["opt_src_slot"])) + self.option_tgt(gather(o["opt_tgt_slot"]))
        if self.has_ledger:
            ours = (o["opt_feats"][..., OPT_OPPONENT] <= 0).unsqueeze(-1)
            options = options + self.ledger_option(ledger_of(o["opt_ids"]) * ours)
        if self.stop_token:
            options = torch.cat([options, self.stop_embed.to(options.dtype).expand(b, 1, d)], dim=1)
            opt_absent = torch.cat([opt_absent, torch.zeros_like(opt_absent[:, :1])], dim=1)
        for layer in self.options:
            options = layer(options, tokens, tgt_key_padding_mask=opt_absent, memory_key_padding_mask=absent)
        options = self.out_norm(options)
        stop = [options[:, n]] if self.stop_token else []
        options = F.pad(options[:, :n], (0, 0, 0, MAX_OPTIONS - n))  # padded slots are masked out anyway
        return torch.cat([state, options.reshape(b, -1), *stop], dim=-1)


class EntityHead(nn.Module):
    """SB3 ``mlp_extractor``: actor latent is the 65 logits (one per option, then STOP).

    With ``stop_token`` STOP's logit comes from its decoder token through the options' ``score`` head;
    otherwise from the state through a head of its own (``stop``)."""

    def __init__(self, dim: int, hidden: int = 128, value_dim: int = 256, stop_token: bool = False):
        super().__init__()
        self.dim, self.stop_token = dim, stop_token
        self.latent_dim_pi, self.latent_dim_vf = MAX_OPTIONS + 1, value_dim
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        if not stop_token:
            self.stop = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.value = nn.Sequential(nn.Linear(dim, value_dim), nn.Tanh(), nn.Linear(value_dim, value_dim), nn.Tanh())

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        state = features[:, : self.dim]
        end = self.dim + MAX_OPTIONS * self.dim
        options = features[:, self.dim:end].reshape(-1, MAX_OPTIONS, self.dim)
        stop = self.score(features[:, end:]) if self.stop_token else self.stop(state)
        return torch.cat([self.score(options).squeeze(-1), stop], dim=-1)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value(features[:, : self.dim])

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)


class EntityPolicy(MaskableMultiInputActorCriticPolicy):
    """Masked actor-critic over ``EntityExtractor`` tokens."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("features_extractor_class", EntityExtractor)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = EntityHead(self.features_extractor.dim, stop_token=self.features_extractor.stop_token)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        self.action_net = nn.Identity()
        heads = [self.mlp_extractor.score] + ([] if self.mlp_extractor.stop_token else [self.mlp_extractor.stop])
        for layer in (h[-1] for h in heads):
            nn.init.orthogonal_(layer.weight, gain=0.01)
            nn.init.zeros_(layer.bias)
        if self.features_extractor.has_ledger:
            self.features_extractor.zero_ledger()
        if getattr(self.features_extractor, "has_opp_hand", False):
            nn.init.zeros_(self.features_extractor.opp_hand.weight)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
