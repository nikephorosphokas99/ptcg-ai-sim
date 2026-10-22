from typing import Any, Callable


OPTION_FIELDS = (
    "type", "number", "area", "index", "playerIndex", "toolIndex",
    "energyIndex", "count", "inPlayArea", "inPlayIndex", "attackId",
    "cardId", "serial", "specialConditionType",
)

SELECTION_FIELDS = (
    "type", "context", "minCount", "maxCount", "remainDamageCounter",
    "remainEnergyCost", "deck", "contextCard", "effect",
)

GLOBAL_STATE_FIELDS = (
    "turn", "turnActionCount", "firstPlayer", "stadium", "stadiumPlayed",
    "supporterPlayed", "energyAttached", "retreated", "looking", "result",
)


def _strip_player_index(node: Any) -> Any:
    if isinstance(node, dict):
        if "id" in node and "serial" in node:
            node = {key: value for key, value in node.items() if key != "playerIndex"}
        return {key: _strip_player_index(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_strip_player_index(item) for item in node]
    return node


_AREA_DECK, _AREA_HAND, _AREA_DISCARD = 1, 2, 3
_AREA_ACTIVE, _AREA_BENCH = 4, 5
_AREA_STADIUM, _AREA_LOOKING = 7, 12


def resolve_option_card(
    state: dict[str, Any], selection: dict[str, Any], option: dict[str, Any],
    player_index: int,
) -> int | None:
    index = option.get("index")
    if index is None:
        return None
    index = int(index)
    if index < 0:
        return None
    area = option.get("area")
    area = _AREA_HAND if area is None else int(area)
    player = state["players"][player_index]
    try:
        if area == _AREA_HAND:
            return int(player["hand"][index]["id"])
        if area == _AREA_DISCARD:
            return int(player["discard"][index]["id"])
        if area == _AREA_ACTIVE:
            return int(player["active"][index]["id"])
        if area == _AREA_BENCH:
            return int(player["bench"][index]["id"])
        if area == _AREA_DECK:
            return int(selection["deck"][index]["id"])
        if area == _AREA_STADIUM:
            return int(state["stadium"][index]["id"])
        if area == _AREA_LOOKING:
            return int(state["looking"][index]["id"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return None


def _remap_options(
    options: list[dict[str, Any]], player_index: int,
    state: dict[str, Any] | None = None, selection: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    remapped = []
    for option in options:
        resolved = (
            resolve_option_card(state, selection or {}, option, player_index)
            if state is not None and option.get("cardId") is None
            else None
        )
        option = {field: option.get(field) for field in OPTION_FIELDS}
        if option.get("cardId") is None and resolved is not None:
            option["cardId"] = resolved
        raw = option.pop("playerIndex", None)
        option["targets_opponent"] = None if raw is None else raw != player_index
        remapped.append(option)
    return remapped


def board_state(state: dict[str, Any], player_index: int) -> dict[str, Any]:
    return _strip_player_index(state["players"][player_index])


def _global_state(state: dict[str, Any]) -> dict[str, Any]:
    return _strip_player_index({field: state[field] for field in GLOBAL_STATE_FIELDS})


STATUS_CONDITIONS = ("poisoned", "burned", "asleep", "paralyzed", "confused")

EMPTY_BOARD_STATE: dict[str, Any] = {
    "active": [], "bench": [], "benchMax": 0, "deckCount": 0,
    "discard": [], "prize": [], "handCount": 0, "hand": None,
    "poisoned": False, "burned": False, "asleep": False,
    "paralyzed": False, "confused": False,
}


def _pokemon_by_serial(board: dict[str, Any]) -> dict[int, dict[str, Any]]:
    pokemon = [p for p in board["active"] if p is not None] + board["bench"]
    return {p["serial"]: p for p in pokemon}


def diff_board_state(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    previous_pokemon = _pokemon_by_serial(previous)
    current_pokemon = _pokemon_by_serial(current)
    previous_serials, current_serials = previous_pokemon.keys(), current_pokemon.keys()
    kept_serials = previous_serials & current_serials

    previous_discard_serials = {card["serial"] for card in previous["discard"]}
    energy_attached = [
        {
            "serial": serial,
            "id": current_pokemon[serial]["id"],
            "new_energy_types": current_pokemon[serial]["energies"][len(previous_pokemon[serial]["energies"]):],
        }
        for serial in kept_serials
        if len(current_pokemon[serial]["energies"]) > len(previous_pokemon[serial]["energies"])
    ]

    return {
        "hand_count": current["handCount"],
        "deck_count": current["deckCount"],
        "prize_count": len(current["prize"]),
        "discarded_cards": [
            card for card in current["discard"] if card["serial"] not in previous_discard_serials
        ],
        "new_pokemon": [current_pokemon[serial] for serial in current_serials - previous_serials],
        "removed_pokemon": [previous_pokemon[serial] for serial in previous_serials - current_serials],
        "energy_attached": energy_attached,
        "hp_lost": sum(
            max(0, previous_pokemon[s]["hp"] - current_pokemon[s]["hp"]) for s in kept_serials
        ),
        "status_applied": [
            condition for condition in STATUS_CONDITIONS
            if current[condition] and not previous[condition]
        ],
    }


def decision_context(
    selection: dict[str, Any], options: list[dict[str, Any]], player_index: int,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "selection": _strip_player_index(selection),
        "options": _remap_options(options, player_index, state, selection),
    }


def extract_features(
    state: dict[str, Any],
    selection: dict[str, Any],
    options: list[dict[str, Any]],
    player_index: int,
) -> dict[str, Any]:
    return {
        "state": board_state(state, player_index),
        "opponent_state": board_state(state, 1 - player_index),
        "global_state": _global_state(state),
        "decision_context": decision_context(selection, options, player_index, state),
    }


def extract_features_from_observation(observation: dict[str, Any]) -> dict[str, Any]:
    state, selection, options = split_observation(observation)
    return extract_features(state, selection, options, state["yourIndex"])


def split_observation(
    observation: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    state = observation["current"]
    select = observation.get("select") or {}
    selection = {field: select.get(field) for field in SELECTION_FIELDS}
    return state, selection, list(select.get("option") or [])


def opponent_history(
    read_row: Callable[[int], dict[str, Any]],
    idx: int,
    row: dict[str, Any],
    size: int,
    source: str = "own_frames",
) -> list[dict[str, Any]]:
    if size <= 0:
        return []
    episode_id, player_index = row["episode_id"], row["player_index"]
    opponent_index = 1 - player_index
    frame_owner = opponent_index if source == "opponent_frames" else player_index
    snapshots: list[tuple[int, dict[str, Any]]] = []
    seen_turns: set[int] = set()
    cursor = idx - 1
    while cursor >= 0 and len(snapshots) < size:
        prior = read_row(cursor)
        if prior["episode_id"] != episode_id:
            break
        turn = prior["state"]["turn"]
        if prior["player_index"] == frame_owner and turn not in seen_turns:
            seen_turns.add(turn)
            snapshots.append((turn, board_state(prior["state"], opponent_index)))
        cursor -= 1
    snapshots.reverse()

    history = []
    previous_board = EMPTY_BOARD_STATE
    for turn, board in snapshots:
        history.append({"turn": turn, **diff_board_state(previous_board, board)})
        previous_board = board
    return history


def decision_chain(
    read_row: Callable[[int], dict[str, Any]],
    idx: int,
    row: dict[str, Any],
    size: int,
) -> list[dict[str, Any]]:
    if size <= 0:
        return []
    episode_id, player_index = row["episode_id"], row["player_index"]
    chain = []
    cursor = idx - 1
    while cursor >= 0 and len(chain) < size:
        prior = read_row(cursor)
        if prior["episode_id"] != episode_id:
            break
        if prior["player_index"] == player_index:
            chain.append({
                "turn": prior["state"]["turn"],
                "turn_action_count": prior["state"]["turnActionCount"],
                **decision_context(
                    prior["selection"], prior["options"], player_index, prior["state"]
                ),
                "target_action": prior["target_action"],
            })
        cursor -= 1
    chain.reverse()
    return chain
