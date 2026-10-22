from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
from bisect import bisect_right

import torch
from torch.utils.data import Dataset

from observation import DEFAULT_SPEC, ObservationSpec, build_observation
from vocab import (
    HP_CAP,
    EnergyType,
    OptionsVocab,
    SelectionVocab,
    _card_id,
    attack_flags,
    card_type_flags,
    pad_options,
    stack_selections,
)


class PolicyFeatureDataset(Dataset):
    def __init__(
        self,
        parquet_path: str | Path,
        transform: Callable[[dict[str, Any]], Any] | None = None,
        opponent_history_size: int | None = None,
        decision_chain_size: int | None = None,
        player_name: str | Iterable[str] | None = None,
        cached_row_groups: int = 4,
        spec: ObservationSpec = DEFAULT_SPEC,
    ) -> None:
        try:
            import pyarrow.parquet as pq
        except (
            ImportError
        ) as exc:
            raise ImportError(
                "PolicyFeatureDataset requires pyarrow. Install it with "
                "`python -m pip install pyarrow`."
            ) from exc

        parquet_path = Path(parquet_path)
        if not parquet_path.is_file():
            raise FileNotFoundError(f"Parquet dataset not found: {parquet_path}")

        self._parquet = pq.ParquetFile(parquet_path)
        self._row_group_offsets = [0]
        for row_group in range(self._parquet.num_row_groups):
            self._row_group_offsets.append(
                self._row_group_offsets[-1]
                + self._parquet.metadata.row_group(row_group).num_rows
            )
        self._table_cache: OrderedDict[int, Any] = OrderedDict()
        self._table_cache_size = max(1, cached_row_groups)
        self._row_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._row_cache_size = 4096
        self.transform = transform
        overrides = {}
        if opponent_history_size is not None:
            overrides["opponent_history_size"] = opponent_history_size
        if decision_chain_size is not None:
            overrides["decision_chain_size"] = decision_chain_size
        self.spec = spec.variant(**overrides) if overrides else spec
        self.player_name = player_name

        self._row_indexes: list[int] | None = None
        if player_name is not None:
            wanted = (
                [player_name] if isinstance(player_name, str) else list(player_name)
            )
            names = self._parquet.read(columns=["player_name"]).to_pandas()[
                "player_name"
            ]
            self._row_indexes = names.index[names.isin(wanted)].tolist()

    def __len__(self) -> int:
        if self._row_indexes is None:
            return self._row_group_offsets[-1]
        return len(self._row_indexes)

    def raw_index(self, idx: int) -> int:
        return idx if self._row_indexes is None else self._row_indexes[idx]

    def episode_ids(self) -> list[Any]:
        column = self._parquet.read(columns=["episode_id"]).to_pandas()["episode_id"]
        if self._row_indexes is None:
            return column.tolist()
        return column.iloc[self._row_indexes].tolist()

    @staticmethod
    def _touch(cache: OrderedDict, key: int, value: Any, limit: int) -> Any:
        cache[key] = value
        if len(cache) > limit:
            cache.popitem(last=False)
        return value

    def _read_row(self, idx: int) -> dict[str, Any]:
        row = self._row_cache.get(idx)
        if row is not None:
            self._row_cache.move_to_end(idx)
            return row

        row_group = bisect_right(self._row_group_offsets, idx) - 1
        table = self._table_cache.get(row_group)
        if table is None:
            table = self._touch(
                self._table_cache,
                row_group,
                self._parquet.read_row_group(row_group),
                self._table_cache_size,
            )
        else:
            self._table_cache.move_to_end(row_group)

        offset = idx - self._row_group_offsets[row_group]
        row = table.slice(offset, 1).to_pylist()[0]
        return self._touch(self._row_cache, idx, row, self._row_cache_size)

    def __getitem__(self, sample_idx: int) -> tuple[Any, torch.Tensor]:
        if not 0 <= sample_idx < len(self):
            raise IndexError(f"Dataset index out of range: {sample_idx}")

        idx = self.raw_index(sample_idx)
        row = self._read_row(idx)
        observation = build_observation(self._read_row, idx, row, self.spec)
        if self.transform is not None:
            observation = {
                "features": self.transform(observation),
                "meta": observation["meta"],
            }
        return observation, torch.tensor(row["target_action"], dtype=torch.long)


def decision_collate(
    batch: list[tuple[Any, torch.Tensor]],
) -> tuple[list[Any], list[torch.Tensor]]:
    observations, actions = zip(*batch)
    return list(observations), list(actions)


def pad_target_actions(decision_chain):
    per_decision = [
        torch.tensor(decision["target_action"], dtype=torch.long) for decision in decision_chain
    ]
    chain_len = len(per_decision)
    max_targets = max((t.numel() for t in per_decision), default=0)
    target_action = torch.full((chain_len, max_targets), -1, dtype=torch.long)
    mask = torch.zeros((chain_len, max_targets), dtype=torch.bool)
    for i, t in enumerate(per_decision):
        target_action[i, : t.numel()] = t
        mask[i, : t.numel()] = True
    return target_action, mask


def _with_card_flags(fields: dict, id_key: str, prefix: str, card_flag_fields=None) -> dict:
    fields.update(
        {
            f"{prefix}_{flag}": value
            for flag, value in card_type_flags(fields[id_key], card_flag_fields).items()
        }
    )
    return fields


def _card_id_list_with_flags(cards, prefix, card_flag_fields=None):
    ids = torch.tensor(
        [_card_id(card["id"] if card is not None else None) for card in cards],
        dtype=torch.long,
    )
    return _with_card_flags(
        {f"{prefix}_card_id": ids}, f"{prefix}_card_id", f"{prefix}_card", card_flag_fields
    )


def transform_pokemon(pokemon):
    fields = {
        "card_id": torch.tensor(_card_id(pokemon["id"]), dtype=torch.long),
        "serial": torch.tensor(pokemon["serial"], dtype=torch.long),
        "hp": _normalize(torch.tensor(pokemon["hp"], dtype=torch.long), _NORMALIZATION_CAPS["hp"]),
        "max_hp": _normalize(
            torch.tensor(pokemon["maxHp"], dtype=torch.long), _NORMALIZATION_CAPS["max_hp"]
        ),
        "appear_this_turn": torch.tensor(int(pokemon["appearThisTurn"]), dtype=torch.long),
        "energies": torch.tensor(pokemon["energies"], dtype=torch.long),
    }
    _with_card_flags(fields, "card_id", "card")
    fields.update(
        _card_id_list_with_flags(
            pokemon["energyCards"],
            "energy",
            card_flag_fields=(
                "type", "energy_type", "ace_spec",
                "skill_count", "skill_count_norm", "ability_id", "skill_flags",
            ),
        )
    )
    fields.update(_card_id_list_with_flags(pokemon["tools"], "tool"))
    fields.update(_card_id_list_with_flags(pokemon["preEvolution"], "pre_evolution"))
    return fields


_EMPTY_POKEMON = {
    "id": None, "serial": 0, "hp": 0, "maxHp": 0, "appearThisTurn": False,
    "energies": [], "energyCards": [], "tools": [], "preEvolution": [],
}


def transform_active_pokemon(active):
    present = bool(active) and active[0] is not None
    fields = transform_pokemon(active[0] if present else _EMPTY_POKEMON)
    fields["present"] = torch.tensor(int(present), dtype=torch.long)
    return fields


def transform_player_state(player_state):
    active_pokemon = transform_active_pokemon(player_state["active"])
    bench_pokemon = [transform_pokemon(p) for p in player_state["bench"]]

    fields = {
        "active_pokemon": active_pokemon,
        "bench_pokemon": bench_pokemon,
        "bench_max": _normalize(
            torch.tensor(player_state["benchMax"], dtype=torch.long), _NORMALIZATION_CAPS["bench_max"]
        ),
        "deck_count": _normalize(
            torch.tensor(player_state["deckCount"], dtype=torch.long), _NORMALIZATION_CAPS["deck_count"]
        ),
        "hand_count": _normalize(
            torch.tensor(player_state["handCount"], dtype=torch.long), _NORMALIZATION_CAPS["hand_count"]
        ),
        "prize_count": _normalize(
            torch.tensor(len(player_state["prize"]), dtype=torch.long), _NORMALIZATION_CAPS["prize_count"]
        ),
        "poisoned": torch.tensor(int(player_state["poisoned"]), dtype=torch.long),
        "burned": torch.tensor(int(player_state["burned"]), dtype=torch.long),
        "asleep": torch.tensor(int(player_state["asleep"]), dtype=torch.long),
        "paralyzed": torch.tensor(int(player_state["paralyzed"]), dtype=torch.long),
        "confused": torch.tensor(int(player_state["confused"]), dtype=torch.long),
    }
    fields.update(_card_id_list_with_flags(player_state["discard"], "discard"))
    fields.update(_card_id_list_with_flags(player_state["hand"] or [], "hand"))
    return fields


_STATUS_CONDITIONS = ("poisoned", "burned", "asleep", "paralyzed", "confused")


def _pad_card_list_chain(card_lists, prefix, card_flag_fields=None):
    chain_len = len(card_lists)
    max_width = max((len(cards) for cards in card_lists), default=0)
    ids = torch.zeros((chain_len, max_width), dtype=torch.long)
    mask = torch.zeros((chain_len, max_width), dtype=torch.bool)
    for i, cards in enumerate(card_lists):
        for j, card in enumerate(cards):
            ids[i, j] = _card_id(card["id"])
            mask[i, j] = True
    return _with_card_flags(
        {f"{prefix}_card_id": ids, f"{prefix}_mask": mask}, f"{prefix}_card_id", f"{prefix}_card",
        card_flag_fields,
    )


def _pad_pokemon_list_chain(pokemon_lists, prefix):
    chain_len = len(pokemon_lists)
    max_width = max((len(pokemon) for pokemon in pokemon_lists), default=0)
    card_ids = torch.zeros((chain_len, max_width), dtype=torch.long)
    serials = torch.zeros((chain_len, max_width), dtype=torch.long)
    hps = torch.zeros((chain_len, max_width), dtype=torch.long)
    max_hps = torch.zeros((chain_len, max_width), dtype=torch.long)
    mask = torch.zeros((chain_len, max_width), dtype=torch.bool)
    for i, pokemon_list in enumerate(pokemon_lists):
        for j, pokemon in enumerate(pokemon_list):
            card_ids[i, j] = _card_id(pokemon["id"])
            serials[i, j] = pokemon["serial"]
            hps[i, j] = pokemon["hp"]
            max_hps[i, j] = pokemon["maxHp"]
            mask[i, j] = True
    fields = {
        f"{prefix}_card_id": card_ids,
        f"{prefix}_serial": serials,
        f"{prefix}_hp": _normalize(hps, _NORMALIZATION_CAPS["hp"]),
        f"{prefix}_max_hp": _normalize(max_hps, _NORMALIZATION_CAPS["max_hp"]),
        f"{prefix}_mask": mask,
    }
    return _with_card_flags(fields, f"{prefix}_card_id", f"{prefix}_card")


def _pad_energy_attached_chain(energy_attached_lists):
    chain_len = len(energy_attached_lists)
    max_width = max((len(events) for events in energy_attached_lists), default=0)
    card_ids = torch.zeros((chain_len, max_width), dtype=torch.long)
    serials = torch.zeros((chain_len, max_width), dtype=torch.long)
    mask = torch.zeros((chain_len, max_width), dtype=torch.bool)
    new_energy_type_counts = torch.zeros((chain_len, max_width, len(EnergyType)), dtype=torch.long)
    for i, events in enumerate(energy_attached_lists):
        for j, event in enumerate(events):
            card_ids[i, j] = _card_id(event["id"])
            serials[i, j] = event["serial"]
            mask[i, j] = True
            for energy_type in event["new_energy_types"]:
                new_energy_type_counts[i, j, energy_type] += 1
    fields = {
        "energy_attached_card_id": card_ids,
        "energy_attached_serial": serials,
        "energy_attached_mask": mask,
        "energy_attached_new_energy_type_counts": _normalize(
            new_energy_type_counts, _NORMALIZATION_CAPS["energy_attach_count"]
        ),
    }
    return _with_card_flags(fields, "energy_attached_card_id", "energy_attached_card")


def _pad_status_applied_chain(status_lists):
    chain_len = len(status_lists)
    out = torch.zeros((chain_len, len(_STATUS_CONDITIONS)), dtype=torch.bool)
    for i, statuses in enumerate(status_lists):
        for status in statuses:
            out[i, _STATUS_CONDITIONS.index(status)] = True
    return out


def transform_opponent_history(history):
    turns = torch.tensor([entry["turn"] for entry in history], dtype=torch.long)
    hand_counts = torch.tensor([entry["hand_count"] for entry in history], dtype=torch.long)
    deck_counts = torch.tensor([entry["deck_count"] for entry in history], dtype=torch.long)
    prize_counts = torch.tensor([entry["prize_count"] for entry in history], dtype=torch.long)
    hp_lost = torch.tensor([entry["hp_lost"] for entry in history], dtype=torch.long)

    fields = {
        "turn": _normalize(turns, _NORMALIZATION_CAPS["turn"]),
        "hand_count": _normalize(hand_counts, _NORMALIZATION_CAPS["hand_count"]),
        "deck_count": _normalize(deck_counts, _NORMALIZATION_CAPS["deck_count"]),
        "prize_count": _normalize(prize_counts, _NORMALIZATION_CAPS["prize_count"]),
        "hp_lost": _normalize(hp_lost, _NORMALIZATION_CAPS["hp_lost"]),
        "status_applied": _pad_status_applied_chain([entry["status_applied"] for entry in history]),
    }
    fields.update(_pad_card_list_chain([entry["discarded_cards"] for entry in history], "discarded"))
    fields.update(_pad_pokemon_list_chain([entry["new_pokemon"] for entry in history], "new_pokemon"))
    fields.update(
        _pad_pokemon_list_chain([entry["removed_pokemon"] for entry in history], "removed_pokemon")
    )
    fields.update(_pad_energy_attached_chain([entry["energy_attached"] for entry in history]))
    return fields


def transform_global_state(global_state):
    stadium = global_state["stadium"]
    stadium_card_id = _card_id(stadium[0]["id"]) if stadium else 0
    looking = global_state["looking"] or []
    looking_card_ids = torch.tensor(
        [_card_id(card["id"] if card is not None else None) for card in looking],
        dtype=torch.long,
    )
    turn = torch.tensor(global_state["turn"], dtype=torch.long)
    turn_action_count = torch.tensor(global_state["turnActionCount"], dtype=torch.long)
    fields = {
        "turn": _normalize(turn, _NORMALIZATION_CAPS["turn"]),
        "turn_action_count": _normalize(turn_action_count, _NORMALIZATION_CAPS["turn_action_count"]),
        "first_player": torch.tensor(global_state["firstPlayer"] + 1, dtype=torch.long),
        "stadium_card_id": torch.tensor(stadium_card_id, dtype=torch.long),
        "stadium_played": torch.tensor(int(global_state["stadiumPlayed"]), dtype=torch.long),
        "supporter_played": torch.tensor(int(global_state["supporterPlayed"]), dtype=torch.long),
        "energy_attached": torch.tensor(int(global_state["energyAttached"]), dtype=torch.long),
        "retreated": torch.tensor(int(global_state["retreated"]), dtype=torch.long),
        "looking_card_ids": looking_card_ids,
        "result": torch.tensor(global_state["result"] + 1, dtype=torch.long),
    }
    _with_card_flags(fields, "stadium_card_id", "stadium_card")
    _with_card_flags(fields, "looking_card_ids", "looking_card")
    return fields


_NORMALIZATION_CAPS = {
    "number": 10.0,
    "count": 3.0,
    "min_count": 60.0,
    "max_count": 60.0,
    "remain_damage_counter": 10.0,
    "remain_energy_cost": 6.0,
    "deck_size": 60.0,
    "turn": 120.0,
    "turn_action_count": 60.0,
    "bench_max": 10.0,
    "deck_count": 60.0,
    "hand_count": 60.0,
    "prize_count": 6.0,
    "hp": HP_CAP,
    "max_hp": HP_CAP,
    "hp_lost": 1000.0,
    "energy_attach_count": 8.0,
}


def _normalize(value: torch.Tensor, cap: float) -> torch.Tensor:
    return (value.float() / cap).clamp(0.0, 1.0)


def _with_normalized_options(fields: dict) -> dict:
    fields["number"] = _normalize(fields["number"], _NORMALIZATION_CAPS["number"])
    fields["count"] = _normalize(fields["count"], _NORMALIZATION_CAPS["count"])
    return fields


def _with_normalized_selection(fields: dict) -> dict:
    for key in ("min_count", "max_count", "remain_damage_counter", "remain_energy_cost", "deck_size"):
        fields[key] = _normalize(fields[key], _NORMALIZATION_CAPS[key])
    return fields


def _with_attack_flags(fields: dict) -> dict:
    fields.update(attack_flags(fields["attack_id"]))
    return fields


def _options_with_card_flags(options):
    return _with_attack_flags(
        _with_normalized_options(_with_card_flags(pad_options(options), "card_id", "card"))
    )


def _selections_with_card_flags(selections):
    fields = stack_selections(selections)
    _with_card_flags(fields, "context_card_id", "context_card")
    _with_card_flags(fields, "effect_card_id", "effect_card")
    return _with_normalized_selection(fields)


def transform_decision_context(decision_context):
    options = OptionsVocab.from_options(decision_context["options"])
    selection = SelectionVocab.from_selection(decision_context["selection"])
    return {
        "options": _options_with_card_flags([options]),
        "selection": _selections_with_card_flags([selection]),
    }


def transform_decision_chain(decision_chain):
    turns = torch.tensor([decision["turn"] for decision in decision_chain], dtype=torch.long)
    turn_action_counts = torch.tensor(
        [decision["turn_action_count"] for decision in decision_chain], dtype=torch.long
    )
    target_action, target_action_mask = pad_target_actions(decision_chain)
    options = [OptionsVocab.from_options(decision["options"]) for decision in decision_chain]
    selections = [SelectionVocab.from_selection(decision["selection"]) for decision in decision_chain]
    return {
        "turn": _normalize(turns, _NORMALIZATION_CAPS["turn"]),
        "turn_action_count": _normalize(turn_action_counts, _NORMALIZATION_CAPS["turn_action_count"]),
        "target_action": target_action,
        "target_action_mask": target_action_mask,
        "options": _options_with_card_flags(options),
        "selection": _selections_with_card_flags(selections),
    }

def transform(row):
    return {
        "decision_chain": transform_decision_chain(row["features"]["decision_chain"]),
        "decision_context": transform_decision_context(row["features"]["decision_context"]),
        "global_state": transform_global_state(row["features"]["global_state"]),
        "opponent_history": transform_opponent_history(row["features"]["opponent_history"]),
        "opponent_state": transform_player_state(row["features"]["opponent_state"]),
        "state": transform_player_state(row["features"]["state"]),
    }