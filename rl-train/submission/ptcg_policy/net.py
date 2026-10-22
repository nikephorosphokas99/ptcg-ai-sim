"""The trained policy's forward pass in numpy, for the submission bundle.

Kaggle's agent runtime has numpy (a dependency of kaggle-environments) but not
torch: a bundle that imported torch failed on every decision there. So the
actor is re-implemented here with numpy only, reading the training model's own
parameter names from ``policy.npz`` and its shape from ``arch.json``.
``ptcg_rl.export`` checks it against the SB3 policy before writing a bundle.

Generations: ``kind == "entity"`` mirrors ``ptcg_rl.policy.EntityPolicy`` (v3,
attention over board / hand / option tokens); ``"pointer"`` mirrors
``PointerPolicy``; ``"flat"`` the first ``PTCGExtractor`` + SB3 MLP policy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_BAGS = ("hand_ids", "my_discard_ids", "opp_discard_ids", "view_ids")
_POINTER_BAGS = _BAGS + ("my_unseen_ids", "opp_seen_ids")
_ENTITY_PILES = ("my_discard_ids", "opp_discard_ids", "view_ids", "my_unseen_ids", "opp_seen_ids", "opp_recent_ids")
_SLOTS = 18


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class Net:
    """Observation (batched dict of arrays) -> logits over ``N_ACTIONS``; mask before use."""

    def __init__(self, params: dict[str, np.ndarray], arch: dict):
        self.p = {k: v.astype(np.float32) for k, v in params.items()}
        self.arch = arch

    def _lin(self, name: str, x: np.ndarray) -> np.ndarray:
        return x @ self.p[f"{name}.weight"].T + self.p[f"{name}.bias"]

    def _card_table(self) -> np.ndarray:
        """Every card's vector; row 0 (padding / face-down) is zero."""
        p = self.p
        rows = np.concatenate([p["features_extractor.embedding.weight"], p["card_static"]], axis=-1)
        table = _relu(self._lin("features_extractor.card.0", rows))
        table[0] = 0.0
        return table

    def __call__(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        kind = self.arch.get("kind")
        if kind == "entity":
            return np.stack([self._entity({k: v[i] for k, v in obs.items()}) for i in range(obs["global"].shape[0])])
        return self._pointer(obs) if kind == "pointer" else self._flat(obs)

    # -- entity (v3): one sample at a time, padding tokens left out altogether ---------

    def _attention(self, name: str, q_in: np.ndarray, kv_in: np.ndarray) -> np.ndarray:
        w, b = self.p[f"{name}.in_proj_weight"], self.p[f"{name}.in_proj_bias"]
        d, heads = q_in.shape[-1], self.arch["heads"]
        q = q_in @ w[:d].T + b[:d]
        k = kv_in @ w[d:2 * d].T + b[d:2 * d]
        v = kv_in @ w[2 * d:].T + b[2 * d:]
        hd = d // heads
        split = lambda x: x.reshape(x.shape[0], heads, hd).transpose(1, 0, 2)  # noqa: E731
        att = _softmax(split(q) @ split(k).transpose(0, 2, 1) / np.sqrt(hd))
        out = (att @ split(v)).transpose(1, 0, 2).reshape(q_in.shape[0], d)
        return self._lin(f"{name}.out_proj", out)

    def _ln(self, name: str, x: np.ndarray) -> np.ndarray:
        return _layer_norm(x, self.p[f"{name}.weight"], self.p[f"{name}.bias"])

    def _ff(self, name: str, x: np.ndarray) -> np.ndarray:
        return self._lin(f"{name}.linear2", _relu(self._lin(f"{name}.linear1", x)))

    def _entity(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        p, fe = self.p, "features_extractor"
        rows = np.concatenate([p[f"{fe}.embedding.weight"], p["card_static"], p["card_text"]], axis=-1)
        table = self._lin(f"{fe}.card.2", _relu(self._lin(f"{fe}.card.0", rows)))
        table[0] = 0.0
        cards = lambda ids: table[np.clip(np.asarray(ids).astype(np.int64), 0, len(table) - 1)]  # noqa: E731
        kind = p[f"{fe}.kind.weight"]

        side = np.zeros((_SLOTS, 2), dtype=np.float32)
        side[: _SLOTS // 2, 0] = 1.0
        side[_SLOTS // 2:, 1] = 1.0
        slots = self._lin(f"{fe}.slot", np.concatenate(
            [cards(obs["slot_ids"]), cards(obs["slot_attach_ids"]).sum(axis=1), obs["slot_feats"], side], axis=-1))
        present = np.flatnonzero(obs["slot_feats"][:, 0] > 0)
        hand = cards(obs["hand_ids"][obs["hand_ids"] > 0]) + kind[0]
        piles = np.stack([table[np.clip(obs[k].astype(np.int64), 0, len(table) - 1)].sum(axis=0) / 10.0
                          for k in _ENTITY_PILES])
        piles = self._lin(f"{fe}.pile", piles) + kind[1:1 + len(_ENTITY_PILES)]
        misc_at = np.flatnonzero(obs["misc_ids"] > 0)
        misc = cards(obs["misc_ids"][misc_at]) + kind[1 + len(_ENTITY_PILES) + misc_at]
        glob = self._lin(f"{fe}.globals", obs["global"][None])
        tokens = np.concatenate([glob, slots[present], hand, piles, misc], axis=0)
        for i in range(self.arch["board_layers"]):
            name = f"{fe}.board.{i}"
            tokens = tokens + self._attention(f"{name}.self_attn", self._ln(f"{name}.norm1", tokens), self._ln(f"{name}.norm1", tokens))
            tokens = tokens + self._ff(name, self._ln(f"{name}.norm2", tokens))
        state = _relu(self._lin(f"{fe}.state.1", self._ln(f"{fe}.state.0", tokens[0])))
        slot_out = np.zeros((_SLOTS, tokens.shape[1]), dtype=np.float32)
        slot_out[present] = tokens[1:1 + len(present)]

        real = np.abs(obs["opt_feats"]).sum(axis=-1) > 0
        n = max(1, int(real.sum()))  # real options are packed first
        gather = lambda idx: np.where((idx >= 0)[:, None], slot_out[np.clip(idx, 0, _SLOTS - 1)], 0.0)  # noqa: E731
        src, tgt = obs["opt_src_slot"][:n].astype(np.int64), obs["opt_tgt_slot"][:n].astype(np.int64)
        options = self._lin(f"{fe}.option", np.concatenate(
            [cards(obs["opt_ids"][:n]), cards(obs["opt_target_ids"][:n]), obs["opt_feats"][:n], obs["opt_tactical"][:n]], axis=-1))
        options = options + gather(src) @ p[f"{fe}.option_src.weight"].T + gather(tgt) @ p[f"{fe}.option_tgt.weight"].T
        keep = np.flatnonzero(real[:n]) if real[:n].any() else np.array([0])
        for i in range(self.arch["option_layers"]):
            name = f"{fe}.options.{i}"
            x = self._ln(f"{name}.norm1", options)
            options = options + self._attention(f"{name}.self_attn", x, x[keep])
            options = options + self._attention(f"{name}.multihead_attn", self._ln(f"{name}.norm2", options), tokens)
            options = options + self._ff(name, self._ln(f"{name}.norm3", options))
        options = self._ln(f"{fe}.out_norm", options)

        head = "mlp_extractor"
        # Padded option slots are always masked in play; score them as the torch model does
        # (its scorer sees zero vectors there) so the full distributions match exactly.
        padded = self._lin(f"{head}.score.2", _relu(self._lin(f"{head}.score.0", np.zeros((1, options.shape[1]), np.float32))))[0, 0]
        logits = np.full(65, padded, dtype=np.float32)
        logits[:n] = self._lin(f"{head}.score.2", _relu(self._lin(f"{head}.score.0", options)))[:, 0]
        logits[-1] = self._lin(f"{head}.stop.2", _relu(self._lin(f"{head}.stop.0", state)))[0]
        return logits

    def _common(self, obs: dict[str, np.ndarray], bags: tuple[str, ...]):
        table = self._card_table()
        b = obs["global"].shape[0]
        cards = lambda key: table[np.clip(obs[key].astype(np.int64), 0, len(table) - 1)]  # noqa: E731
        slots = _relu(self._lin("features_extractor.slot.0",
                                np.concatenate([cards("slot_ids"), obs["slot_feats"]], axis=-1)))
        bag_vecs = [cards(k).sum(axis=1) / 10.0 for k in bags]
        options = np.concatenate([cards("opt_ids"), cards("opt_target_ids"), obs["opt_feats"]], axis=-1)
        globals_ = _relu(self._lin("features_extractor.globals.0", obs["global"]))
        return b, cards, slots, bag_vecs, options, globals_

    def _pointer(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        b, cards, slots, bags, options, globals_ = self._common(obs, _POINTER_BAGS)
        board = np.concatenate([slots.reshape(b, -1), *bags, cards("misc_ids").reshape(b, -1), globals_], axis=-1)
        state = _relu(self._lin("features_extractor.board.2", _relu(self._lin("features_extractor.board.0", board))))
        options = _relu(self._lin("features_extractor.option.2", _relu(self._lin("features_extractor.option.0", options))))
        real = (np.abs(obs["opt_feats"]).sum(axis=-1, keepdims=True) > 0).astype(np.float32)
        pooled = (options * real).sum(axis=1) / np.maximum(real.sum(axis=1), 1.0)
        state = _relu(self._lin("features_extractor.fuse.0", np.concatenate([state, pooled], axis=-1)))

        query = self._lin("mlp_extractor.query", state)
        query = np.broadcast_to(query[:, None, :], (b, options.shape[1], query.shape[-1]))
        hidden = _relu(self._lin("mlp_extractor.score.0", np.concatenate([options, query], axis=-1)))
        scores = self._lin("mlp_extractor.score.2", hidden)[..., 0]
        stop = self._lin("mlp_extractor.stop.2", _relu(self._lin("mlp_extractor.stop.0", state)))
        return np.concatenate([scores, stop], axis=-1)

    def _flat(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        b, cards, slots, bags, options, globals_ = self._common(obs, _BAGS)
        options = _relu(self._lin("features_extractor.option.2", _relu(self._lin("features_extractor.option.0", options))))
        x = np.concatenate([slots.reshape(b, -1), *bags, cards("misc_ids").reshape(b, -1),
                            options.reshape(b, -1), globals_], axis=-1)
        for i in range(len(self.arch["pi"])):
            x = np.tanh(self._lin(f"mlp_extractor.policy_net.{2 * i}", x))
        return self._lin("action_net", x)


def load(bundle: Path) -> tuple[Net, np.ndarray, np.ndarray | None]:
    """The bundle's network, attack table and card meta table (for ``encoding.encode``)."""
    with np.load(bundle / "policy.npz", allow_pickle=False) as data:
        params = {k: data[k] for k in data.files}
    arch = json.loads((bundle / "arch.json").read_text())
    meta = params.get("card_meta")
    return Net(params, arch), params["attack_table"], None if meta is None else meta.astype(np.int64)
