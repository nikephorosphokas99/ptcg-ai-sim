from typing import Any

from features import split_observation
from observation import DEFAULT_SPEC, ObservationSpec, build_observation


class LiveFeatureExtractor:
    def __init__(
        self,
        opponent_history_size: int | None = None,
        decision_chain_size: int | None = None,
        player_names: tuple[str | None, str | None] = (None, None),
        spec: ObservationSpec = DEFAULT_SPEC,
    ) -> None:
        overrides = {}
        if opponent_history_size is not None:
            overrides["opponent_history_size"] = opponent_history_size
        if decision_chain_size is not None:
            overrides["decision_chain_size"] = decision_chain_size
        self.spec = spec.variant(**overrides) if overrides else spec
        self.player_names = player_names
        self._episode_id = -1
        self._rows: list[dict[str, Any]] = []

    def reset(self, episode_id: int | None = None) -> None:
        self._episode_id = self._episode_id + 1 if episode_id is None else episode_id
        self._rows = []

    @property
    def rows(self) -> list[dict[str, Any]]:
        return self._rows

    def _read_row(self, index: int) -> dict[str, Any]:
        return self._rows[index]

    def __call__(self, obs: dict[str, Any]) -> dict[str, Any]:
        if obs.get("select") is None:
            raise ValueError(
                "obs has no 'select' — there is no decision to featurise "
                "(the episode is over or this observation is inactive)."
            )
        state, selection, options = split_observation(obs)
        player_index = state["yourIndex"]
        row = {
            "episode_id": self._episode_id,
            "frame_index": len(self._rows),
            "player_index": player_index,
            "player_name": self.player_names[player_index],
            "state": state,
            "selection": selection,
            "options": options,
            "target_action": None,
        }
        self._rows.append(row)
        return build_observation(self._read_row, len(self._rows) - 1, row, self.spec)

    def record_action(self, action: list[int]) -> None:
        if not self._rows:
            raise RuntimeError("record_action called before any observation was extracted.")
        self._rows[-1]["target_action"] = list(action)
