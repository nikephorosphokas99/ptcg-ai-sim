import os
import random
import sys
from pathlib import Path

try:
    _ROOT = Path(__file__).resolve().parent
except NameError:
    _ROOT = Path("/kaggle_simulations/agent")
    if not _ROOT.is_dir():
        _ROOT = Path.cwd()

for _candidate in (_ROOT, Path("/kaggle_simulations/agent")):
    if not _candidate.is_dir():
        continue
    for _entry in (_candidate, _candidate / "policy_network"):
        if _entry.is_dir() and str(_entry) not in sys.path:
            sys.path.insert(0, str(_entry))

from cg.api import Observation, to_observation_class

_NEG_INF = float("-inf")


def _resolve(filename: str) -> str:
    if os.path.exists(filename):
        return filename
    for base in (_ROOT, Path("/kaggle_simulations/agent")):
        candidate = base / filename
        if candidate.exists():
            return str(candidate)
    return filename


def read_deck_csv() -> list[int]:
    with open(_resolve("deck.csv"), "r") as file:
        lines = file.read().split("\n")
    return [int(lines[i]) for i in range(60)]


def _load_policy_module():
    import importlib.util
    import sys as _sys

    if "_bundle_policy" in _sys.modules:
        return _sys.modules["_bundle_policy"]
    source = Path(_resolve("policy_network/policy_experimental.py"))
    spec = importlib.util.spec_from_file_location("_bundle_policy", source)
    module = importlib.util.module_from_spec(spec)
    _sys.modules["_bundle_policy"] = module
    spec.loader.exec_module(module)
    return module


def _build_actor_critic():
    import json
    import torch.nn as nn

    _policy_module = _load_policy_module()
    PolicyNetwork = _policy_module.PolicyNetwork

    ARCH = {}
    meta = Path(_resolve("actor_critic.pt") + ".meta.json")
    if meta.exists():
        ARCH = json.loads(meta.read_text()).get("arch") or {}
    D = ARCH.get("dim", _policy_module.D)

    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.policy = PolicyNetwork(**ARCH)
            self.value = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, 1))
            self.stop = nn.Linear(D, 1)

        def forward(self, features):
            policy = self.policy
            ctx_pooled, option_vecs, options_mask = policy.decision_context(
                features["decision_context"]
            )
            import torch

            if hasattr(policy, "logits_and_state"):
                logits, state, options_mask = policy.logits_and_state(features)
                return logits, self.stop(state).squeeze(-1), options_mask

            chain = policy.decision_chain(features["decision_chain"])
            chain_state, chain_steps, chain_mask = (
                chain if isinstance(chain, tuple) else (chain, None, None)
            )
            state = policy.fuse(
                torch.cat(
                    [
                        chain_state,
                        ctx_pooled,
                        policy.global_state(features["global_state"]),
                        policy.opponent_history(features["opponent_history"]),
                        policy.player_state(features["state"]),
                        policy.player_state(features["opponent_state"]),
                    ],
                    dim=-1,
                )
            )
            if chain_steps is not None:
                option_vecs = policy.option_cross(
                    option_vecs, options_mask, state, chain_steps, chain_mask
                )
            query = state.unsqueeze(1).expand(-1, option_vecs.size(1), -1)
            logits = policy.score(torch.cat([option_vecs, query], dim=-1)).squeeze(-1)
            return logits, self.stop(state).squeeze(-1), options_mask

    return ActorCritic()


def _greedy_action(logits, stop_logit, options_mask, min_count, max_count,
                   floor=True):
    import torch

    available = options_mask[0].clone()
    min_c, max_c = int(min_count[0]), int(max_count[0])
    stop_column = options_mask.shape[1]
    action: list[int] = []
    while len(action) < max_c:
        masked = logits.masked_fill(~available.unsqueeze(0), _NEG_INF)
        stop_open = len(action) >= min_c and not (floor and not action)
        stop = stop_logit if stop_open else torch.full_like(stop_logit, _NEG_INF)
        step = torch.cat([masked, stop.unsqueeze(-1)], dim=-1)[0]
        if not torch.isfinite(step).any():
            break
        choice = int(step.argmax())
        if choice == stop_column:
            break
        action.append(choice)
        available[choice] = False
    return action


class _Policy:
    def __init__(self) -> None:
        self.network = None
        self.extractor = None
        self.episode = 0
        self.failed = False

    def reset(self) -> None:
        self.episode += 1
        if self.extractor is not None:
            self.extractor.reset(episode_id=self.episode)

    def _load(self) -> None:
        import torch

        from live import LiveFeatureExtractor

        self.network = _build_actor_critic()
        state = torch.load(_resolve("actor_critic.pt"), map_location="cpu",
                           weights_only=True)
        self.network.load_state_dict(state)
        self.network.eval()
        for parameter in self.network.parameters():
            parameter.requires_grad_(False)
        if self.extractor is None:
            from observation import ObservationSpec

            self.extractor = LiveFeatureExtractor(
                spec=ObservationSpec(opponent_history_size=30,
                                     decision_chain_size=30)
            )
            self.extractor.reset(episode_id=self.episode)

    def act(self, obs_dict: dict) -> list[int]:
        import torch

        from collate import collate_features
        from dataset import transform

        selection_counts = _load_policy_module().selection_counts

        if self.network is None:
            self._load()

        observation = self.extractor(obs_dict)
        features = collate_features([transform(observation)])
        with torch.no_grad():
            logits, stop_logit, options_mask = self.network(features)
        min_count, max_count = selection_counts(features)
        action = _greedy_action(logits, stop_logit, options_mask,
                                min_count, max_count, floor=_FLOOR)
        self.extractor.record_action(action)
        return action


_FLOOR = os.environ.get("SUBMISSION_DECODE_FLOOR", "1") != "0"

_policy = _Policy()


def _fallback(obs: "Observation") -> list[int]:
    count = random.randint(obs.select.minCount, obs.select.maxCount)
    count = min(count, len(obs.select.option))
    return random.sample(range(len(obs.select.option)), count)


def agent(obs_dict: dict) -> list[int]:
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        _policy.reset()
        return read_deck_csv()

    if _policy.failed:
        return _fallback(obs)

    try:
        action = _policy.act(obs_dict)
    except Exception as exc:  # noqa: BLE001 — a live match must not crash
        print(f"[agent] policy failed ({exc!r}) — falling back", file=sys.stderr)
        _policy.failed = True
        return _fallback(obs)

    if (not (obs.select.minCount <= len(action) <= obs.select.maxCount)
            or len(set(action)) != len(action)):
        print(
            f"[agent] decoded {len(action)} options outside "
            f"[{obs.select.minCount}, {obs.select.maxCount}] — falling back",
            file=sys.stderr,
        )
        return _fallback(obs)
    return action
