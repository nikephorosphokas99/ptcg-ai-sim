class ISMCTSNode:

    def __init__(self) -> None:
        pass




_POLICY = _Policy()


def agent(obs_dict: dict) -> list[int]:
    """Return the deck on the first call, then option indices for each decision.

    Each returned list holds distinct indices, ``0 <= i < len(options)``, and
    its length is within ``[minCount, maxCount]``.
    """
    if obs_dict.get("select") is None:
        _POLICY.new_game()
        return read_deck_csv()
    return _POLICY.act(obs_dict)

