import torch

from dataset import _EMPTY_POKEMON, transform_pokemon

_UNMASKED_CARD_GROUPS = (
    "hand_card", "discard_card", "tool_card", "energy_card",
    "pre_evolution_card", "looking_card",
)


def pad_stack(tensors: list[torch.Tensor], pad_value) -> torch.Tensor:
    ndim = tensors[0].dim()
    max_shape = [max(t.shape[d] for t in tensors) for d in range(ndim)]
    out = tensors[0].new_full((len(tensors), *max_shape), pad_value)
    for i, t in enumerate(tensors):
        index = (i,) + tuple(slice(0, s) for s in t.shape)
        out[index] = t
    return out


def lengths_mask(lengths: list[int], max_len: int) -> torch.Tensor:
    mask = torch.zeros(len(lengths), max_len, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    return mask


def _pad_value_for(tensor: torch.Tensor):
    return False if tensor.dtype == torch.bool else 0


def collate_flat_dict(dicts: list[dict]) -> dict:
    out = {}
    for key in dicts[0].keys():
        tensors = [d[key] for d in dicts]
        out[key] = pad_stack(tensors, _pad_value_for(tensors[0]))
    for prefix in _UNMASKED_CARD_GROUPS:
        id_key = f"{prefix}_id" if f"{prefix}_id" in out else f"{prefix}_ids"
        if id_key in out:
            lengths = [d[id_key].shape[-1] for d in dicts]
            out[f"{prefix}_mask"] = lengths_mask(lengths, out[id_key].shape[-1])
    if "energies" in out:
        lengths = [d["energies"].shape[-1] for d in dicts]
        out["energies_mask"] = lengths_mask(lengths, out["energies"].shape[-1])
    return out


def _collate_with_nested(dicts: list[dict], nested: dict) -> dict:
    flat = [{k: v for k, v in d.items() if k not in nested} for d in dicts]
    out = collate_flat_dict(flat)
    for key, fn in nested.items():
        out[key] = fn([d[key] for d in dicts])
    return out


def collate_options(dicts: list[dict]) -> dict:
    return collate_flat_dict(dicts)


def collate_selection(dicts: list[dict]) -> dict:
    return collate_flat_dict(dicts)


def collate_pokemon(dicts: list[dict]) -> dict:
    return collate_flat_dict(dicts)


def collate_bench_pokemon(bench_lists: list[list[dict]]) -> dict:
    lengths = [len(bench) for bench in bench_lists]
    max_bench = max(lengths, default=0)
    empty = transform_pokemon(_EMPTY_POKEMON)

    if max_bench == 0:
        batched = collate_pokemon([empty for _ in bench_lists])
        out = {k: v.unsqueeze(1)[:, :0] for k, v in batched.items()}
        out["bench_pokemon_mask"] = lengths_mask(lengths, 0)
        return out

    flat = []
    for bench in bench_lists:
        flat.extend(bench + [empty] * (max_bench - len(bench)))

    batched_flat = collate_pokemon(flat)
    batch_size = len(bench_lists)
    out = {k: v.view(batch_size, max_bench, *v.shape[1:]) for k, v in batched_flat.items()}
    out["bench_pokemon_mask"] = lengths_mask(lengths, max_bench)
    return out


def collate_active_pokemon(dicts: list[dict]) -> dict:
    return collate_pokemon(dicts)


def collate_player_state(dicts: list[dict]) -> dict:
    return _collate_with_nested(
        dicts, {"active_pokemon": collate_active_pokemon, "bench_pokemon": collate_bench_pokemon}
    )


def collate_decision_chain(dicts: list[dict]) -> dict:
    lengths = [d["turn"].shape[0] for d in dicts]
    out = _collate_with_nested(dicts, {"options": collate_options, "selection": collate_selection})
    out["chain_mask"] = lengths_mask(lengths, out["turn"].shape[1])
    return out


def collate_decision_context(dicts: list[dict]) -> dict:
    return {
        "options": collate_options([d["options"] for d in dicts]),
        "selection": collate_selection([d["selection"] for d in dicts]),
    }


def collate_global_state(dicts: list[dict]) -> dict:
    return collate_flat_dict(dicts)


def collate_opponent_history(dicts: list[dict]) -> dict:
    lengths = [d["turn"].shape[0] for d in dicts]
    out = collate_flat_dict(dicts)
    out["history_mask"] = lengths_mask(lengths, out["turn"].shape[1])
    return out


def collate_features(samples: list[dict]) -> dict:
    return {
        "decision_chain": collate_decision_chain([s["decision_chain"] for s in samples]),
        "decision_context": collate_decision_context([s["decision_context"] for s in samples]),
        "global_state": collate_global_state([s["global_state"] for s in samples]),
        "opponent_history": collate_opponent_history([s["opponent_history"] for s in samples]),
        "opponent_state": collate_player_state([s["opponent_state"] for s in samples]),
        "state": collate_player_state([s["state"] for s in samples]),
    }
