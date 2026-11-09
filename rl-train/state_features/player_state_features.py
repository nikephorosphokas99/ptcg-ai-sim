# Game has 2 state. Opponent & Player.
# Player state are listed as:
# - benched_pokemons: [ids, ids .. N] max: 8
# - active_pokemon: id
# - hands: [ids, ids, ids ... N]
# NOTE: prize & deck are guessable via trainer/pokemon abilities. For example:
#       - tatsugiri's ability: look up 6 cards at top deck & take 1 supporter available
#       - Team rocket's petrel: look up all trainer card in the deck and take 1 trainer
#       - spikemuth's gym: look all marnie's pokemon in the deck and take 1
#         etc.
# - prize: [None, ids, ids] (None means unguessed, ids = guessed prize)
# - deck_list: [None, ids, ids.. (60 - total_prize_cont - hand_count - discard_count)] (None means unguessed, ids = guessed deck card)


from dataclasses import dataclass


@dataclass
class PlayerState:
    active_pokemon: int
    benched_pokemons: list[int]  # max: 8
    # for card & special tokens reference, look at @card_tokenizer.py
    # hand_cards: list[int]
    # prize_cards: list[int]
    # dec_cards: list[int]
    # hand cards
    total_hand_count: int #numerical
    pokemon_hand_cards: list[int]
    supporter_hand_cards: list[int]
    stadium_hand_cards: list[int]
    tool_hand_cards: list[int]
    item_hand_cards: list[int]

    # guess deck
    total_deck_count: int
    guessed_deck_count: int #numerical
    unguessed_deck_count: int
    guessed_deck_pokemon_hand_cards: list[int]
    guessed_deck_supporter_hand_cards: list[int]
    guessed_deck_stadium_hand_cards: list[int]
    guessed_deck_tool_hand_cards: list[int]
    guessed_deck_item_hand_cards: list[int]

    # guess prize
    total_prize_count: int
    guessed_prize_count: int #numerical
    unguessed_prize_count: int
    guessed_prize_pokemon_hand_cards: list[int]
    guessed_prize_supporter_hand_cards: list[int]
    guessed_prize_stadium_hand_cards: list[int]
    guessed_prize_tool_hand_cards: list[int]
    guessed_prize_item_hand_cards: list[int]



