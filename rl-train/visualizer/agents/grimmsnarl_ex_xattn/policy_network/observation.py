from dataclasses import dataclass, replace
from typing import Any, Callable

from features import decision_chain, extract_features, opponent_history

OPPONENT_FRAMES = "opponent_frames"
OWN_FRAMES = "own_frames"


@dataclass(frozen=True)
class ObservationSpec:
    opponent_history_size: int = 60
    decision_chain_size: int = 60
    opponent_history_source: str = OWN_FRAMES

    def __post_init__(self) -> None:
        if self.opponent_history_source not in (OPPONENT_FRAMES, OWN_FRAMES):
            raise ValueError(
                f"opponent_history_source must be {OPPONENT_FRAMES!r} or "
                f"{OWN_FRAMES!r}, got {self.opponent_history_source!r}"
            )

    def variant(self, **changes) -> "ObservationSpec":
        return replace(self, **changes)


DEFAULT_SPEC = ObservationSpec()

LEARNER_SPEC = ObservationSpec(opponent_history_size=30, decision_chain_size=30)


def build_observation(
    read_row: Callable[[int], dict[str, Any]],
    idx: int,
    row: dict[str, Any],
    spec: ObservationSpec = DEFAULT_SPEC,
) -> dict[str, Any]:
    player_index = row["player_index"]
    features = extract_features(
        row["state"], row["selection"], row["options"], player_index
    )
    if spec.opponent_history_size > 0:
        features["opponent_history"] = opponent_history(
            read_row, idx, row, spec.opponent_history_size,
            source=spec.opponent_history_source,
        )
    if spec.decision_chain_size > 0:
        features["decision_chain"] = decision_chain(
            read_row, idx, row, spec.decision_chain_size
        )
    meta = {
        "episode_id": row["episode_id"],
        "frame_index": row["frame_index"],
        "player_index": player_index,
        "player_name": row.get("player_name"),
    }
    return {"features": features, "meta": meta}
