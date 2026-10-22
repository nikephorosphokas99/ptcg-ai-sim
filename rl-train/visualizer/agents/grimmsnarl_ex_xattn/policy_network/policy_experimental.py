from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import PolicyFeatureDataset, transform
from vocab import (
    ABILITY_ID_VOCAB_SIZE,
    AREA_VOCAB_SIZE,
    ATTACK_EFFECT_FLAG_COUNT,
    ATTACK_ID_VOCAB_SIZE,
    CARD_ENERGY_TYPE_VOCAB_SIZE,
    CARD_ID_VOCAB_SIZE,
    CARD_RESISTANCE_VOCAB_SIZE,
    CARD_STAGE_VOCAB_SIZE,
    CARD_TYPE_VOCAB_SIZE,
    CARD_WEAKNESS_VOCAB_SIZE,
    OPTION_TYPE_VOCAB_SIZE,
    SELECT_CONTEXT_VOCAB_SIZE,
    SELECT_TYPE_VOCAB_SIZE,
    SKILL_FLAG_COUNT,
    SPECIAL_CONDITION_VOCAB_SIZE,
    TARGETS_OPPONENT_VOCAB_SIZE,
    AreaType,
    EnergyType,
    embedding_card_id,
)

D = 64

DEFAULT_DROPOUT = 0.1


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float().unsqueeze(-1)
    denom = mask.sum(dim=-2).clamp(min=1.0)
    return (x * mask).sum(dim=-2) / denom


_SUM_SCALE = 10.0


def _masked_mean_sum(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float().unsqueeze(-1)
    summed = (x * mask_f).sum(dim=-2)
    denom = mask_f.sum(dim=-2).clamp(min=1.0)
    return torch.cat([summed / denom, summed / _SUM_SCALE], dim=-1)


MAX_SEQ_LEN = 64

SIDE_STRIDE = 9
NO_SLOT = 2 * SIDE_STRIDE
BOARD_SLOT_VOCAB = NO_SLOT + 1


class ReversePositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = MAX_SEQ_LEN):
        super().__init__()
        self.embed = nn.Embedding(max_len, dim)
        self.max_len = max_len

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=-1, keepdim=True)
        slots = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        positions = (lengths - 1 - slots).clamp(0, self.max_len - 1)
        return x + self.embed(positions)


def _safe_key_padding_mask(valid_mask: torch.Tensor) -> torch.Tensor:
    key_padding_mask = ~valid_mask
    fully_padded = ~valid_mask.any(dim=-1)
    if fully_padded.any():
        key_padding_mask = key_padding_mask.clone()
        key_padding_mask[fully_padded, 0] = False
    return key_padding_mask


def _causal_mask(length: int, device) -> torch.Tensor:
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)


def _last_valid(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    lengths = mask.sum(dim=-1)
    index = (lengths - 1).clamp(min=0)
    last = x.gather(1, index.view(-1, 1, 1).expand(-1, 1, x.shape[-1])).squeeze(1)
    return last * mask.any(dim=-1, keepdim=True)


class CardEmbed(nn.Module):
    def __init__(self, dim=D, dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.id = nn.Embedding(CARD_ID_VOCAB_SIZE, dim)
        self.stage = nn.Embedding(CARD_STAGE_VOCAB_SIZE, dim // 4)
        self.type_embed = nn.Embedding(CARD_TYPE_VOCAB_SIZE, dim // 4)
        self.energy_type = nn.Embedding(CARD_ENERGY_TYPE_VOCAB_SIZE, dim // 4)
        self.weakness = nn.Embedding(CARD_WEAKNESS_VOCAB_SIZE, dim // 4)
        self.resistance = nn.Embedding(CARD_RESISTANCE_VOCAB_SIZE, dim // 4)
        self.ability = nn.Embedding(ABILITY_ID_VOCAB_SIZE, dim // 4)
        self.proj = nn.Linear(dim + 6 * (dim // 4) + 8 + SKILL_FLAG_COUNT, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, fields: dict) -> torch.Tensor:
        card_id = fields["id"]
        zeros = torch.zeros_like(card_id)
        float_zeros = torch.zeros_like(card_id, dtype=torch.float)
        stage = fields.get("stage", zeros)
        stage_norm = fields.get("stage_norm", float_zeros)
        ctype = fields.get("type", zeros)
        energy_type = fields.get("energy_type", zeros)
        weakness = fields.get("weakness", zeros)
        resistance = fields.get("resistance", zeros)
        ability = fields.get("ability_id", zeros)
        flags = torch.stack(
            [
                fields.get("ex", zeros).float(),
                fields.get("mega_ex", zeros).float(),
                fields.get("tera", zeros).float(),
                fields.get("ace_spec", zeros).float(),
                stage_norm,
                fields.get("hp_norm", float_zeros),
                fields.get("retreat_cost_norm", float_zeros),
                fields.get("skill_count_norm", float_zeros),
            ],
            dim=-1,
        )
        skill_flags = fields.get(
            "skill_flags",
            card_id.new_zeros((*card_id.shape, SKILL_FLAG_COUNT), dtype=torch.float),
        )
        out = torch.cat(
            [
                self.id(embedding_card_id(card_id)), self.stage(stage), self.type_embed(ctype),
                self.energy_type(energy_type), self.weakness(weakness),
                self.resistance(resistance), self.ability(ability),
                flags, skill_flags,
            ],
            dim=-1,
        )
        return self.drop(F.relu(self.proj(out)))


def _card_fields(fields: dict, prefix: str) -> dict:
    return {k[len(prefix) + 1:]: v for k, v in fields.items() if k.startswith(prefix + "_")}


class OptionEncoder(nn.Module):
    def __init__(self, dim=D, dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.card = CardEmbed(dim, dropout)
        self.type_embed = nn.Embedding(OPTION_TYPE_VOCAB_SIZE, dim // 4)
        self.area = nn.Embedding(AREA_VOCAB_SIZE, dim // 4)
        self.targets_opponent = nn.Embedding(TARGETS_OPPONENT_VOCAB_SIZE, dim // 4)
        self.attack = nn.Embedding(ATTACK_ID_VOCAB_SIZE, dim // 4)
        self.special_condition = nn.Embedding(SPECIAL_CONDITION_VOCAB_SIZE, dim // 4)
        self.mlp = nn.Sequential(
            nn.Linear(
                dim + 5 * (dim // 4) + 9 + len(EnergyType) + ATTACK_EFFECT_FLAG_COUNT, dim
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, options: dict) -> torch.Tensor:
        card = self.card(_card_fields(options, "card"))
        scalars = torch.stack(
            [
                options["number"], options["count"],
                options["index"].float() / 10.0, options["energy_index"].float() / 10.0,
                options["in_play_index"].float() / 10.0,
                options["serial"].float() / 10.0,
                options["tool_index"].float() / 10.0,
                options["attack_damage_norm"], options["attack_energy_cost_norm"],
            ],
            dim=-1,
        )
        out = torch.cat(
            [
                card,
                self.type_embed(options["type"]),
                self.area(options["area"]),
                self.targets_opponent(options["targets_opponent"]),
                self.attack(options["attack_id_safe"]),
                self.special_condition(options["special_condition_type"]),
                scalars,
                options["attack_energy_type_counts"],
                options["attack_effect_flags"],
            ],
            dim=-1,
        )
        return self.mlp(out)


class SelectionEncoder(nn.Module):
    def __init__(self, dim=D, dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.type_embed = nn.Embedding(SELECT_TYPE_VOCAB_SIZE, dim // 4)
        self.context = nn.Embedding(SELECT_CONTEXT_VOCAB_SIZE, dim // 4)
        self.context_card = CardEmbed(dim, dropout)
        self.effect_card = CardEmbed(dim, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim + 2 * (dim // 4) + 5, dim), nn.ReLU(), nn.Dropout(dropout)
        )

    def forward(self, selection: dict) -> torch.Tensor:
        scalars = torch.stack(
            [
                selection["min_count"], selection["max_count"], selection["remain_damage_counter"],
                selection["remain_energy_cost"], selection["deck_size"],
            ],
            dim=-1,
        )
        out = torch.cat(
            [
                self.type_embed(selection["type"]),
                self.context(selection["context"]),
                self.context_card(_card_fields(selection, "context_card")),
                self.effect_card(_card_fields(selection, "effect_card")),
                scalars,
            ],
            dim=-1,
        )
        return self.mlp(out)


class DecisionChainEncoder(nn.Module):
    def __init__(self, dim=D, nhead=2, layers=8, dropout=DEFAULT_DROPOUT, ff_mult=2):
        super().__init__()
        self.dim = dim
        self.option = OptionEncoder(dim, dropout)
        self.selection = SelectionEncoder(dim, dropout)
        self.in_proj = nn.Linear(3 * dim + 2, dim)
        self.position = ReversePositionalEmbedding(dim)
        encoder_layer = nn.TransformerEncoderLayer(
            dim, nhead, dim_feedforward=ff_mult * dim, batch_first=True, norm_first=True,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, layers)

    def forward(self, decision_chain: dict):
        chain_mask = decision_chain["chain_mask"]
        if chain_mask.shape[1] == 0:
            empty = torch.zeros(chain_mask.shape[0], self.dim, device=chain_mask.device)
            return empty, empty.unsqueeze(1)[:, :0], chain_mask
        option_vecs = self.option(decision_chain["options"])
        option_summary = _masked_mean(option_vecs, decision_chain["options"]["options_mask"])
        selection_summary = self.selection(decision_chain["selection"])
        chosen_summary = self._chosen(option_vecs, decision_chain)
        step = torch.cat(
            [option_summary, chosen_summary, selection_summary,
             decision_chain["turn"].unsqueeze(-1),
             decision_chain["turn_action_count"].unsqueeze(-1)],
            dim=-1,
        )
        step = F.relu(self.in_proj(step))
        step = self.position(step, chain_mask)
        encoded = self.transformer(
            step,
            mask=_causal_mask(step.shape[1], step.device),
            src_key_padding_mask=_safe_key_padding_mask(chain_mask),
        )
        return _last_valid(encoded, chain_mask), encoded, chain_mask

    @staticmethod
    def _chosen(option_vecs: torch.Tensor, decision_chain: dict) -> torch.Tensor:
        targets = decision_chain["target_action"]
        target_mask = decision_chain["target_action_mask"]
        num_options = option_vecs.shape[-2]
        if targets.shape[-1] == 0 or num_options == 0:
            return option_vecs.new_zeros((*option_vecs.shape[:2], option_vecs.shape[-1]))
        index = targets.clamp(0, num_options - 1)
        index = index.unsqueeze(-1).expand(*index.shape, option_vecs.shape[-1])
        chosen = option_vecs.gather(dim=2, index=index)
        return _masked_mean(chosen, target_mask)


class DecisionContextEncoder(nn.Module):
    def __init__(self, dim=D, nhead=2, layers=2, dropout=DEFAULT_DROPOUT, ff_mult=2):
        super().__init__()
        self.dim = dim
        self.option = OptionEncoder(dim, dropout)
        self.selection = SelectionEncoder(dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            dim, nhead, dim_feedforward=ff_mult * dim, batch_first=True, norm_first=True,
            dropout=dropout,
        )
        self.option_attention = nn.TransformerEncoder(encoder_layer, layers)

    def forward(self, decision_context: dict):
        option_vecs = self.option(decision_context["options"]).squeeze(1)
        options_mask = decision_context["options"]["options_mask"].squeeze(1)
        option_vecs = self.option_attention(
            option_vecs, src_key_padding_mask=_safe_key_padding_mask(options_mask)
        )
        option_summary = _masked_mean(option_vecs, options_mask)
        selection_summary = self.selection(decision_context["selection"]).squeeze(1)
        pooled = torch.cat([option_summary, selection_summary], dim=-1)
        return pooled, option_vecs, options_mask


class OptionCrossAttention(nn.Module):
    def __init__(self, dim=D, nhead=2, layers=2, dropout=DEFAULT_DROPOUT, ff_mult=2):
        super().__init__()
        self.dim = dim
        decoder_layer = nn.TransformerDecoderLayer(
            dim, nhead, dim_feedforward=ff_mult * dim, batch_first=True, norm_first=True,
            dropout=dropout,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, layers)
        self.state_marker = nn.Parameter(torch.zeros(dim))

    def zero_residual_branches(self) -> None:
        for layer in self.decoder.layers:
            for projection in (layer.self_attn.out_proj, layer.multihead_attn.out_proj,
                               layer.linear2):
                nn.init.zeros_(projection.weight)
                if projection.bias is not None:
                    nn.init.zeros_(projection.bias)

    def forward(self, option_vecs, options_mask, state, chain_steps, chain_mask,
                memory_extra=None, memory_extra_mask=None):
        memory = torch.cat([(state + self.state_marker).unsqueeze(1), chain_steps], dim=1)
        state_is_real = torch.zeros(
            chain_mask.shape[0], 1, dtype=torch.bool, device=chain_mask.device
        )
        memory_padding = torch.cat([state_is_real, ~chain_mask], dim=1)
        if memory_extra is not None:
            memory = torch.cat([memory, memory_extra], dim=1)
            memory_padding = torch.cat([memory_padding, ~memory_extra_mask], dim=1)
        return self.decoder(
            option_vecs,
            memory,
            tgt_key_padding_mask=_safe_key_padding_mask(options_mask),
            memory_key_padding_mask=memory_padding,
        )


class GlobalStateEncoder(nn.Module):
    def __init__(self, dim=D, dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.first_player = nn.Embedding(3, dim // 4)
        self.result = nn.Embedding(4, dim // 4)
        self.stadium_card = CardEmbed(dim, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim + 2 * (dim // 4) + 6, dim), nn.ReLU(), nn.Dropout(dropout)
        )

    def forward(self, global_state: dict) -> torch.Tensor:
        looking_fields = _card_fields(global_state, "looking_card")
        looking_fields["id"] = looking_fields.pop("ids")
        looking_vecs = self.stadium_card(looking_fields)
        looking_card = _masked_mean(looking_vecs, global_state["looking_card_mask"])
        bits = torch.stack(
            [
                global_state["turn"], global_state["turn_action_count"],
                global_state["stadium_played"].float(), global_state["supporter_played"].float(),
                global_state["energy_attached"].float(), global_state["retreated"].float(),
            ],
            dim=-1,
        )
        out = torch.cat(
            [
                self.stadium_card(_card_fields(global_state, "stadium_card")),
                looking_card,
                self.first_player(global_state["first_player"]),
                self.result(global_state["result"]),
                bits,
            ],
            dim=-1,
        )
        return self.mlp(out)


class PokemonEncoder(nn.Module):
    def __init__(self, dim=D, dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.card = CardEmbed(dim, dropout)
        self.energy = nn.Embedding(len(EnergyType), dim // 4)
        self.energy_card = CardEmbed(dim, dropout)
        self.tool_card = CardEmbed(dim, dropout)
        self.pre_evolution_card = CardEmbed(dim, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dim + 2 * (dim // 4) + 6 * dim + 2, dim), nn.ReLU(),
            nn.Dropout(dropout),
        )

    def _pool_list(self, module, pokemon: dict, prefix: str) -> torch.Tensor:
        vecs = module(_card_fields(pokemon, prefix))
        return _masked_mean_sum(vecs, pokemon[f"{prefix}_mask"])

    def forward(self, pokemon: dict) -> torch.Tensor:
        energy_vecs = self.energy(pokemon["energies"])
        energy_vec = _masked_mean_sum(energy_vecs, pokemon["energies_mask"])
        out = torch.cat(
            [
                self.card(_card_fields(pokemon, "card")),
                energy_vec,
                self._pool_list(self.energy_card, pokemon, "energy_card"),
                self._pool_list(self.tool_card, pokemon, "tool_card"),
                self._pool_list(self.pre_evolution_card, pokemon, "pre_evolution_card"),
                pokemon["hp"].unsqueeze(-1), pokemon["max_hp"].unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.mlp(out)


class PlayerStateEncoder(nn.Module):
    def __init__(self, dim=D, dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.pokemon = PokemonEncoder(dim, dropout)
        self.hand_card = CardEmbed(dim, dropout)
        self.discard_card = CardEmbed(dim, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(7 * dim + 9, dim), nn.ReLU(), nn.Dropout(dropout)
        )

    def _pool_cards(self, module, fields: dict, prefix: str) -> torch.Tensor:
        vecs = module(_card_fields(fields, prefix))
        return _masked_mean_sum(vecs, fields[f"{prefix}_mask"])

    def pokemon_tokens(self, player_state: dict):
        active = (
            self.pokemon(player_state["active_pokemon"])
            * player_state["active_pokemon"]["present"].unsqueeze(-1)
        ).unsqueeze(1)
        bench = self.pokemon(player_state["bench_pokemon"])
        active_mask = player_state["active_pokemon"]["present"].bool().unsqueeze(1)
        bench_mask = player_state["bench_pokemon"]["bench_pokemon_mask"].bool()
        return torch.cat([active, bench], dim=1), torch.cat([active_mask, bench_mask], dim=1)

    def forward(self, player_state: dict) -> torch.Tensor:
        active = (
            self.pokemon(player_state["active_pokemon"])
            * player_state["active_pokemon"]["present"].unsqueeze(-1)
        )
        bench_vecs = self.pokemon(player_state["bench_pokemon"])
        bench_vec = _masked_mean_sum(
            bench_vecs, player_state["bench_pokemon"]["bench_pokemon_mask"]
        )
        hand_vec = self._pool_cards(self.hand_card, player_state, "hand_card")
        discard_vec = self._pool_cards(self.discard_card, player_state, "discard_card")
        status = torch.stack(
            [
                player_state["bench_max"], player_state["deck_count"], player_state["hand_count"],
                player_state["prize_count"], player_state["poisoned"].float(),
                player_state["burned"].float(), player_state["asleep"].float(),
                player_state["paralyzed"].float(), player_state["confused"].float(),
            ],
            dim=-1,
        )
        out = torch.cat([active, bench_vec, hand_vec, discard_vec, status], dim=-1)
        return self.mlp(out)


class OpponentHistoryEncoder(nn.Module):
    def __init__(self, dim=D, nhead=2, layers=8, dropout=DEFAULT_DROPOUT, ff_mult=2):
        super().__init__()
        self.dim = dim
        self.discarded_card = CardEmbed(dim, dropout)
        self.new_pokemon_card = CardEmbed(dim, dropout)
        self.removed_pokemon_card = CardEmbed(dim, dropout)
        self.energy_attached_card = CardEmbed(dim, dropout)
        self.in_proj = nn.Linear(8 * dim + 5 + 5, dim)
        self.position = ReversePositionalEmbedding(dim)
        encoder_layer = nn.TransformerEncoderLayer(
            dim, nhead, dim_feedforward=ff_mult * dim, batch_first=True, norm_first=True,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, layers)

    def forward(self, history: dict) -> torch.Tensor:
        history_mask = history["history_mask"]
        if history_mask.shape[1] == 0:
            return torch.zeros(history_mask.shape[0], self.dim, device=history_mask.device)
        discarded = _masked_mean_sum(
            self.discarded_card(_card_fields(history, "discarded_card")), history["discarded_mask"]
        )
        new_pokemon = _masked_mean_sum(
            self.new_pokemon_card(_card_fields(history, "new_pokemon_card")),
            history["new_pokemon_mask"],
        )
        removed_pokemon = _masked_mean_sum(
            self.removed_pokemon_card(_card_fields(history, "removed_pokemon_card")),
            history["removed_pokemon_mask"],
        )
        energy_attached = _masked_mean_sum(
            self.energy_attached_card(_card_fields(history, "energy_attached_card")),
            history["energy_attached_mask"],
        )
        step = torch.cat(
            [
                discarded, new_pokemon, removed_pokemon, energy_attached,
                history["status_applied"].float(),
                torch.stack(
                    [history["turn"], history["hand_count"], history["deck_count"],
                     history["prize_count"], history["hp_lost"]],
                    dim=-1,
                ),
            ],
            dim=-1,
        )
        step = F.relu(self.in_proj(step))
        step = self.position(step, history_mask)
        encoded = self.transformer(step, src_key_padding_mask=_safe_key_padding_mask(history_mask))
        return _masked_mean(encoded, history_mask)


_MASK_LOGIT = -1e9


_EQUIVALENCE_FIELDS = (
    "type", "area", "targets_opponent", "card_id",
    "number", "count", "attack_id_safe", "in_play_index",
    "special_condition_type",
)


def equivalence_mask(
    options: dict, targets: torch.Tensor, options_mask: torch.Tensor
) -> torch.Tensor:
    feats = torch.stack(
        [options[field].squeeze(1).float() for field in _EQUIVALENCE_FIELDS], dim=-1
    )
    target_index = targets.argmax(dim=-1)
    chosen = feats.gather(
        1, target_index.view(-1, 1, 1).expand(-1, 1, feats.shape[-1])
    )
    return (feats == chosen).all(dim=-1) & options_mask


def masked_selection_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    equivalent: torch.Tensor | None = None,
) -> torch.Tensor:
    safe = logits.masked_fill(~mask, _MASK_LOGIT)
    num_positive = targets.sum(dim=-1)
    single = num_positive == 1

    per_decision = logits.new_zeros(logits.shape[0])
    if single.any():
        if equivalent is None:
            per_decision[single] = F.cross_entropy(
                safe[single], targets[single].argmax(dim=-1), reduction="none"
            )
        else:
            log_probs = F.log_softmax(safe[single], dim=-1)
            group = equivalent[single]
            per_decision[single] = -torch.logsumexp(
                log_probs.masked_fill(~group, _MASK_LOGIT), dim=-1
            )
    multi = ~single
    if multi.any():
        per_option = F.binary_cross_entropy_with_logits(
            logits[multi].masked_fill(~mask[multi], 0.0), targets[multi], reduction="none"
        )
        per_decision[multi] = (per_option * mask[multi]).sum(dim=-1)
    return per_decision.mean()


def masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    safe_logits = logits.masked_fill(~mask, 0.0)
    per_option = F.binary_cross_entropy_with_logits(safe_logits, targets, reduction="none")
    return (per_option * mask).sum() / mask.sum().clamp(min=1)


DEFAULT_THRESHOLD = 0.15


def load_policy(checkpoint, map_location="cpu") -> "PolicyNetwork":
    network = PolicyNetwork()
    network.load_state_dict(torch.load(checkpoint, map_location=map_location))
    network.eval()
    return network


def decode_action(
    logits: torch.Tensor,
    options_mask: torch.Tensor,
    min_count: torch.Tensor,
    max_count: torch.Tensor,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[list[int]]:
    probs = torch.sigmoid(logits).masked_fill(~options_mask, -1.0)
    num_valid = options_mask.sum(-1)

    count = (probs > threshold).sum(-1)
    count = torch.maximum(count, min_count.clamp(min=1))
    count = torch.minimum(count, max_count)
    count = torch.minimum(count, num_valid).clamp(min=0)

    order = probs.argsort(dim=-1, descending=True)
    return [order[i, : int(count[i])].tolist() for i in range(order.shape[0])]


def selection_counts(features: dict) -> tuple[torch.Tensor, torch.Tensor]:
    selection = features["decision_context"]["selection"]
    min_count = (selection["min_count"] * 60.0).round().long().reshape(-1)
    max_count = (selection["max_count"] * 60.0).round().long().reshape(-1)
    return min_count, max_count


class PolicyNetwork(nn.Module):
    def __init__(self, dim=D, dropout=DEFAULT_DROPOUT, nhead=2, chain_layers=8,
                 hist_layers=8, ctx_layers=2, cross_layers=2, ff_mult=2,
                 board_tokens=False, type_conditioned=False):
        super().__init__()
        self.dim = dim
        self.arch = {"dim": dim, "nhead": nhead, "chain_layers": chain_layers,
                     "hist_layers": hist_layers, "ctx_layers": ctx_layers,
                     "cross_layers": cross_layers, "ff_mult": ff_mult,
                     "dropout": dropout, "board_tokens": board_tokens,
                     "type_conditioned": type_conditioned}
        if dim % nhead:
            raise ValueError(f"dim {dim} must be divisible by nhead {nhead}")
        self.decision_chain = DecisionChainEncoder(
            dim, nhead=nhead, layers=chain_layers, dropout=dropout, ff_mult=ff_mult)
        self.decision_context = DecisionContextEncoder(
            dim, nhead=nhead, layers=ctx_layers, dropout=dropout, ff_mult=ff_mult)
        self.global_state = GlobalStateEncoder(dim, dropout)
        self.opponent_history = OpponentHistoryEncoder(
            dim, nhead=nhead, layers=hist_layers, dropout=dropout, ff_mult=ff_mult)
        self.player_state = PlayerStateEncoder(dim, dropout)
        self.fuse = nn.Sequential(nn.Linear(7 * dim, dim), nn.ReLU(), nn.Dropout(dropout))
        self.option_cross = OptionCrossAttention(
            dim, nhead=nhead, layers=cross_layers, dropout=dropout, ff_mult=ff_mult)
        self.score = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, 1))
        self.type_conditioned = type_conditioned
        if type_conditioned:
            self.type_bias = nn.Embedding(OPTION_TYPE_VOCAB_SIZE, 1)
            nn.init.zeros_(self.type_bias.weight)
            self.select_type_embed = nn.Embedding(SELECT_TYPE_VOCAB_SIZE, dim // 4)
            self.select_context_embed = nn.Embedding(SELECT_CONTEXT_VOCAB_SIZE, dim // 4)
            film = nn.Linear(2 * (dim // 4), 2 * dim)
            nn.init.zeros_(film.weight)
            nn.init.zeros_(film.bias)
            self.select_film = film
        self.board_tokens = board_tokens
        if board_tokens:
            self.slot_embed = nn.Embedding(BOARD_SLOT_VOCAB, dim)

    @staticmethod
    def _named_slot(options: dict) -> torch.Tensor:
        def board_pointer(area: torch.Tensor, index: torch.Tensor) -> tuple:
            on_board = (area == AreaType.ACTIVE) | (area == AreaType.BENCH)
            valid = on_board & ((area != AreaType.BENCH) | (index >= 0))
            within = torch.where(area == AreaType.BENCH, index.clamp(min=0) + 1,
                                 torch.zeros_like(index))
            return valid, within

        in_play_valid, in_play_within = board_pointer(
            options["in_play_area"].squeeze(1), options["in_play_index"].squeeze(1)
        )
        area_valid, area_within = board_pointer(
            options["area"].squeeze(1), options["index"].squeeze(1)
        )
        opponent = options["targets_opponent"].squeeze(1)

        side = (opponent == 1).long() * SIDE_STRIDE
        no_slot = torch.full_like(area_within, NO_SLOT)
        slot = torch.where(area_valid, (area_within + side).clamp(max=NO_SLOT - 1), no_slot)
        slot = torch.where(in_play_valid, in_play_within.clamp(max=SIDE_STRIDE - 1), slot)
        return slot

    def _address_board(self, features: dict, option_vecs: torch.Tensor):
        options = features["decision_context"]["options"]
        option_vecs = option_vecs + self.slot_embed(self._named_slot(options))

        own, own_mask = self.player_state.pokemon_tokens(features["state"])
        opp, opp_mask = self.player_state.pokemon_tokens(features["opponent_state"])

        def addressed(tokens: torch.Tensor, offset: int) -> torch.Tensor:
            slots = torch.arange(tokens.shape[1], device=tokens.device)
            slots = slots.clamp(max=SIDE_STRIDE - 1) + offset
            return tokens + self.slot_embed(slots).unsqueeze(0)

        own, opp = addressed(own, 0), addressed(opp, SIDE_STRIDE)
        return option_vecs, torch.cat([own, opp], dim=1), torch.cat([own_mask, opp_mask], dim=1)

    def forward(self, features: dict):
        return self.logits_and_state(features)[0]

    def logits_and_state(self, features: dict):
        state, option_vecs, options_mask = self.encode(features)
        logits = self.score_options(option_vecs, state, features)
        return logits.masked_fill(~options_mask, float("-inf")), state, options_mask

    def encode(self, features: dict):
        ctx_pooled, option_vecs, options_mask = self.decision_context(features["decision_context"])
        chain_state, chain_steps, chain_mask = self.decision_chain(features["decision_chain"])
        state = self.fuse(
            torch.cat(
                [
                    chain_state,
                    ctx_pooled,
                    self.global_state(features["global_state"]),
                    self.opponent_history(features["opponent_history"]),
                    self.player_state(features["state"]),
                    self.player_state(features["opponent_state"]),
                ],
                dim=-1,
            )
        )
        if self.board_tokens:
            option_vecs, memory_extra, memory_extra_mask = self._address_board(
                features, option_vecs
            )
        else:
            memory_extra = memory_extra_mask = None
        option_vecs = self.option_cross(
            option_vecs, options_mask, state, chain_steps, chain_mask,
            memory_extra, memory_extra_mask,
        )
        return state, option_vecs, options_mask

    def score_options(self, option_vecs, state, features: dict):
        query = state.unsqueeze(1).expand(-1, option_vecs.size(1), -1)
        if self.type_conditioned:
            selection = features["decision_context"]["selection"]
            kind = torch.cat(
                [
                    self.select_type_embed(selection["type"].squeeze(1)),
                    self.select_context_embed(selection["context"].squeeze(1)),
                ],
                dim=-1,
            )
            gamma, beta = self.select_film(kind).chunk(2, dim=-1)
            query = query * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        logits = self.score(torch.cat([option_vecs, query], dim=-1)).squeeze(-1)
        if self.type_conditioned:
            logits = logits + self.type_bias(
                features["decision_context"]["options"]["type"].squeeze(1)
            ).squeeze(-1)
        return logits


if __name__ == "__main__":
    from collate import collate_features, pad_stack

    dataset = PolicyFeatureDataset(
        "data/policy_decisions_lucario.parquet", player_name="Majkel1337", transform=transform)
    policy = PolicyNetwork()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    samples = [dataset[i] for i in (77, 78, 79)]
    features = collate_features([obs["features"] for obs, _ in samples])
    num_options = features["decision_context"]["options"]["options_mask"].shape[-1]
    targets = pad_stack(
        [torch.zeros(num_options).index_fill_(0, action, 1.0) for _, action in samples], 0.0
    )
    print("targets:", targets)

    options_mask = features["decision_context"]["options"]["options_mask"].squeeze(1)
    for step in range(100):
        logits = policy(features)
        loss = masked_selection_loss(logits, targets, options_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 10 == 0 or step == 99:
            print(f"step {step}: loss={loss.item():.4f}")
