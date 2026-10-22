import os
import sys
from collections import defaultdict

from cg.api import (
    AreaType,
    CardType,
    EnergyType,
    Observation,
    SelectContext,
    OptionType,
    Card,
    Pokemon,
    all_card_data,
    to_observation_class,
)

"""
Mega Venusaur ex / Grass Deck
V1 rule-based agent for the Mega Venusaur ex / Grass Deck.

Design goals:
- Use the V1 Venusaur deck list and its dedicated card constants.
- Keep the submission format and CABT API usage compatible with the competition environment.
- Prioritize Bulbasaur -> Ivysaur -> Mega Venusaur ex setup.
- Use Teal Mask Ogerpon ex as an early attacker / energy-draw support.
- Use search and draw trainers aggressively in the early game.
- Avoid ending the turn while useful actions are available.
"""

# ---------------------------------------------------------------------
# 1. Deck loading
# ---------------------------------------------------------------------
# This fallback deck is the same as deck.csv. It is useful if the file is
# missing during local tests, while Kaggle submission normally reads deck.csv.
EMBEDDED_DECK = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    650, 650, 650, 650,
    651, 651, 651, 651,
    652, 652, 652,
    1071,
    96, 96, 96, 96,
    1126,
    1122, 1122, 1122, 1122,
    1094, 1094, 1094, 1094,
    1121, 1121, 1121, 1121,
    1123, 1123, 1123, 1123,
    1229, 1229, 1229, 1229,
    1227, 1227, 1227, 1227,
    1182, 1182,
    1261, 1261, 1261, 1261,
]

file_path = "deck.csv"
if not os.path.exists(file_path):
    file_path = "/kaggle_simulations/agent/" + file_path

my_deck = []
try:
    with open(file_path, "r") as file:
        csv = [line.strip() for line in file.read().splitlines() if line.strip()]
    for i in range(60):
        my_deck.append(int(csv[i]))
except Exception:
    # Fallback for local debugging. In official submission, deck.csv should exist.
    my_deck = EMBEDDED_DECK[:]

# Fetch card metadata database and create an ID-to-Card lookup table.
all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}

# ---------------------------------------------------------------------
# 2. Deck constants
# ---------------------------------------------------------------------
Basic_Grass_Energy = 1  # ×13

Bulbasaur = 650  # フシギダネ ×4
Ivysaur = 651  # フシギソウ ×4
Mega_Venusaur_ex = 652  # メガフシギバナex ×3

Meowth_ex = 1071  # ニャースex ×1
Teal_Mask_Ogerpon_ex = 96  # オーガポン みどりのめんex ×4

Precious_Carry = 1126  # プレシャスキャリ ×1
Pokegear_3_0 = 1122  # ポケギア3.0 ×4
Bug_Catching_Set = 1094  # むしとりセット ×4
Ultra_Ball = 1121  # ハイパーボール ×4
Switch = 1123  # ポケモンいれかえ ×4

Mitsuru_Consideration = 1229  # ミツルの思いやり ×4
Lillie_Determination = 1227  # リーリエの決心 ×4
Boss_Orders = 1182  # ボスの指令 ×2
Vitality_Forest = 1261  # 活力の森 ×4

MAIN_LINE = {Bulbasaur, Ivysaur, Mega_Venusaur_ex}
BASIC_POKEMON = {Bulbasaur, Teal_Mask_Ogerpon_ex, Meowth_ex}
SEARCH_CARDS = {Precious_Carry, Pokegear_3_0, Bug_Catching_Set, Ultra_Ball}
DRAW_SUPPORTERS = {Mitsuru_Consideration, Lillie_Determination}
STADIUMS = {Vitality_Forest}


class AttackPlan:
    attacker = -1      # 0 = active, 1.. = bench index + 1
    target = -1        # 0 = active, 1.. = bench index + 1
    attack_index = -1
    remain_hp = -1
    energy = False


plan = AttackPlan()
pre_turn = 0
ability_used = False


# ---------------------------------------------------------------------
# 3. Helper functions
# ---------------------------------------------------------------------
def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    """Safely extract a Card or Pokemon object from a specific zone."""
    ps = obs.current.players[player_index]
    try:
        match area:
            case AreaType.DECK:
                return obs.select.deck[index]
            case AreaType.HAND:
                return ps.hand[index]
            case AreaType.DISCARD:
                return ps.discard[index]
            case AreaType.ACTIVE:
                return ps.active[index]
            case AreaType.BENCH:
                return ps.bench[index]
            case AreaType.PRIZE:
                return ps.prize[index]
            case AreaType.STADIUM:
                return obs.current.stadium[index]
            case AreaType.LOOKING:
                return obs.current.looking[index]
            case _:
                return None
    except Exception:
        return None


def card_name(card_id: int) -> str:
    try:
        return card_table[card_id].name
    except Exception:
        return ""


def prize_count(pokemon: Pokemon) -> int:
    """How many Prize cards the opponent takes if this Pokemon is Knocked Out."""
    data = card_table[pokemon.id]
    count = 3 if data.megaEx else 2 if data.ex else 1
    for card in pokemon.energyCards:
        if card.id == 12:  # Legacy Energy
            count -= 1
    for card in pokemon.tools:
        if card.id == 1172 and "Lillie" in data.name:  # Lillie’s Pearl
            count -= 1
    return max(0, count)


def pokemon_score(pokemon: Pokemon) -> int:
    """Heuristically evaluates the value of targeting an opponent's Pokemon."""
    data = card_table[pokemon.id]

    score = prize_count(pokemon) * 1000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100

    if data.stage2:
        score += 250
    elif data.stage1:
        score += 130

    # Low-priority support Pokemon examples from the sample agent.
    if pokemon.id in (144, 322, 323, 337):  # Squawkabilly ex, Noctowl, Fan Rotom, Archaludon ex
        score -= 200
    if pokemon.id == 112 and len(pokemon.energies) >= 1:  # Munkidori
        score += 300

    score += pokemon.hp
    return score


def field_attacker_score(pokemon: Pokemon | None, active: bool = False) -> int:
    """Scores our own Pokemon as a potential attacker / active candidate."""
    if pokemon is None:
        return -10_000

    energy_count = len(pokemon.energies)
    score = 0

    if pokemon.id == Mega_Venusaur_ex:
        score = 2500 + energy_count * 350
    elif pokemon.id == Ivysaur:
        score = 1300 + energy_count * 180
    elif pokemon.id == Bulbasaur:
        score = 650 + energy_count * 120
    elif pokemon.id == Teal_Mask_Ogerpon_ex:
        # Strong early-game support attacker. Keep useful even before Venusaur is ready.
        score = 1800 + energy_count * 250
    elif pokemon.id == Meowth_ex:
        score = 700 + energy_count * 120
    else:
        score = 100 + energy_count * 100

    if active:
        score += 120

    # Avoid overexposing important ex/Mega ex when we are close to losing,
    # but still value them if they are the only good attacker.
    try:
        if card_table[pokemon.id].megaEx:
            score -= 60
        elif card_table[pokemon.id].ex:
            score -= 25
    except Exception:
        pass

    return score


def estimated_energy_goal(pokemon: Pokemon | None) -> int:
    """Very rough energy target for our deck."""
    if pokemon is None:
        return 0
    if pokemon.id == Mega_Venusaur_ex:
        return 3
    if pokemon.id == Ivysaur:
        return 3
    if pokemon.id == Bulbasaur:
        return 2
    if pokemon.id == Teal_Mask_Ogerpon_ex:
        return 2
    if pokemon.id == Meowth_ex:
        return 1
    return 1


def energy_score(pokemon: Pokemon | None, active: bool) -> int:
    """Score energy attachment targets."""
    if pokemon is None:
        return -1

    energy_count = len(pokemon.energies)
    goal = estimated_energy_goal(pokemon)

    score = 8000
    if active:
        score += 25

    if pokemon.id == Mega_Venusaur_ex:
        score += 500
    elif pokemon.id == Ivysaur:
        score += 360
    elif pokemon.id == Bulbasaur:
        score += 280
    elif pokemon.id == Teal_Mask_Ogerpon_ex:
        # Ogerpon converts Grass Energy into tempo/draw in many lines.
        score += 430
    elif pokemon.id == Meowth_ex:
        score += 120
    else:
        score -= 100

    if energy_count < goal:
        score += (goal - energy_count) * 120
    else:
        score -= 200 + energy_count * 20

    # If the precomputed plan says this Pokemon needs one more energy, prioritize it.
    return score


def setup_active_score(card: Card | Pokemon, first_player: bool) -> int:
    """Initial Active selection."""
    if card is None:
        return 0

    if card.id == Teal_Mask_Ogerpon_ex:
        # Usually the safest strong opener.
        return 50
    if card.id == Meowth_ex:
        return 35
    if card.id == Bulbasaur:
        return 30
    return 5


def bench_play_score(card_id: int, field_counts: dict[int, int]) -> int:
    """Score Basic Pokemon bench deployment."""
    if card_id == Bulbasaur:
        if field_counts[Bulbasaur] + field_counts[Ivysaur] + field_counts[Mega_Venusaur_ex] >= 3:
            return -1
        return 20000
    if card_id == Teal_Mask_Ogerpon_ex:
        if field_counts[Teal_Mask_Ogerpon_ex] >= 2:
            return -1
        return 19800
    if card_id == Meowth_ex:
        if field_counts[Meowth_ex] >= 1:
            return -1
        return 12000
    return 10000


def to_hand_score(
    card: Card | Pokemon | None,
    field_counts: dict[int, int],
    hand_counts: dict[int, int],
    discard_counts: dict[int, int],
    state,
) -> int:
    """Score choosing a card from search effects."""
    if card is None:
        return 0

    score = 200 - hand_counts[card.id] * 40

    if card.id == Bulbasaur:
        if field_counts[Bulbasaur] + field_counts[Ivysaur] + field_counts[Mega_Venusaur_ex] == 0:
            score += 170
        elif field_counts[Bulbasaur] + field_counts[Ivysaur] + field_counts[Mega_Venusaur_ex] < 2:
            score += 80
        else:
            score -= 120

    elif card.id == Ivysaur:
        if field_counts[Bulbasaur] >= 1:
            score += 170
        else:
            score -= 80

    elif card.id == Mega_Venusaur_ex:
        if field_counts[Ivysaur] >= 1:
            score += 220
        elif field_counts[Bulbasaur] >= 1:
            score += 80
        else:
            score -= 100

    elif card.id == Teal_Mask_Ogerpon_ex:
        if field_counts[Teal_Mask_Ogerpon_ex] == 0:
            score += 140
        elif field_counts[Teal_Mask_Ogerpon_ex] == 1:
            score += 50
        else:
            score -= 120

    elif card.id == Meowth_ex:
        if field_counts[Meowth_ex] == 0:
            score += 40
        else:
            score -= 120

    elif card.id == Basic_Grass_Energy:
        if not state.energyAttached:
            score += 100
        else:
            score += 20

    elif card.id in SEARCH_CARDS:
        score += 60

    elif card.id in DRAW_SUPPORTERS:
        if not state.supporterPlayed:
            score += 80
        else:
            score -= 80

    elif card.id == Boss_Orders:
        if not state.supporterPlayed:
            score += 60
        else:
            score -= 80

    elif card.id == Switch:
        score += 25

    elif card.id == Vitality_Forest:
        score += 50

    return score


def discard_score(card: Card | Pokemon | None, field_counts: dict[int, int], hand_counts: dict[int, int]) -> int:
    """Score cards to discard if the current context asks us to discard cards."""
    if card is None:
        return 0

    # Higher score = more disposable.
    if card.id == Basic_Grass_Energy:
        return 180 if hand_counts[Basic_Grass_Energy] >= 2 else 30
    if card.id == Vitality_Forest:
        return 150 if hand_counts[Vitality_Forest] >= 2 else 40
    if card.id == Switch:
        return 130 if hand_counts[Switch] >= 2 else 40
    if card.id == Pokegear_3_0:
        return 120
    if card.id == Bug_Catching_Set:
        return 100
    if card.id == Precious_Carry:
        return 60
    if card.id == Ultra_Ball:
        return 70

    # Do not casually discard the evolution line.
    if card.id == Mega_Venusaur_ex:
        return -220
    if card.id == Ivysaur:
        return -180
    if card.id == Bulbasaur:
        return -120

    if card.id == Teal_Mask_Ogerpon_ex:
        return -80 if field_counts[Teal_Mask_Ogerpon_ex] == 0 else 40
    if card.id == Meowth_ex:
        return 30

    if card.id in DRAW_SUPPORTERS:
        return 80
    if card.id == Boss_Orders:
        return 20

    return 0


def context_name(context) -> str:
    try:
        return context.name
    except Exception:
        return str(context)


def best_bench_attacker_index(my_state, active_score: int) -> int:
    """Return 1+bench_index if a bench Pokemon is better than active; otherwise -1."""
    best_index = -1
    best_score = active_score
    for idx, pokemon in enumerate(my_state.bench):
        if pokemon is None:
            continue
        score = field_attacker_score(pokemon, active=False)
        if score > best_score + 220:
            best_score = score
            best_index = idx + 1
    return best_index


# ---------------------------------------------------------------------
# 4. Main agent
# ---------------------------------------------------------------------
def agent(obs_dict: dict) -> list[int]:
    """Main Agent Function.

    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount
    inclusive, with no duplicate elements.
    """
    obs = to_observation_class(obs_dict)

    if obs.select is None:
        # Initial selection: return a 60-card deck list.
        return my_deck

    state = obs.current
    select = obs.select
    context = select.context
    ctx_name = context_name(context)

    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]
    my_prize = len(my_state.prize)

    global plan
    global pre_turn
    global ability_used

    if pre_turn != state.turn:
        pre_turn = state.turn
        plan = AttackPlan()
        ability_used = False

    field_counts = defaultdict(int)
    hand_counts = defaultdict(int)
    discard_counts = defaultdict(int)

    for card in my_state.active + my_state.bench:
        if card is not None:
            field_counts[card.id] += 1

    for card in my_state.hand:
        if card is not None:
            hand_counts[card.id] += 1

    for card in my_state.discard:
        if card is not None:
            discard_counts[card.id] += 1

    stadium_id = 0
    for card in state.stadium:
        if card is not None:
            stadium_id = card.id

    # Precompute a light attack/switch plan.
    can_attack = False
    can_switch = False
    can_op_switch = False

    if context == SelectContext.MAIN:
        for o in select.option:
            if o.type == OptionType.PLAY:
                card = get_card(obs, AreaType.HAND, o.index, my_index)
                if card is None:
                    continue
                if card.id == Switch:
                    can_switch = True
                elif card.id == Boss_Orders:
                    can_op_switch = True
            elif o.type == OptionType.RETREAT:
                can_switch = True
            elif o.type == OptionType.ATTACK:
                can_attack = True

        active = my_state.active[0] if len(my_state.active) > 0 else None
        active_score = field_attacker_score(active, active=True)
        bench_attacker = best_bench_attacker_index(my_state, active_score)

        if can_switch and bench_attacker >= 1:
            plan.attacker = bench_attacker
        else:
            plan.attacker = 0

        # Choose target. If Boss is playable, we may target bench; otherwise active.
        op_cards = [op_state.active[0]]
        for pokemon in op_state.bench:
            op_cards.append(pokemon)

        best_target_score = -1
        best_target_index = 0
        for j, op_pokemon in enumerate(op_cards):
            if op_pokemon is None:
                continue
            if j != 0 and not can_op_switch:
                continue
            score = pokemon_score(op_pokemon)
            if j == 0:
                score += 180
            if best_target_score < score:
                best_target_score = score
                best_target_index = j
        plan.target = best_target_index

        # Energy plan: if the selected attacker is short of its rough target, prioritize energy.
        planned_pokemon = active
        if plan.attacker >= 1 and plan.attacker - 1 < len(my_state.bench):
            planned_pokemon = my_state.bench[plan.attacker - 1]
        if planned_pokemon is not None:
            plan.energy = len(planned_pokemon.energies) < estimated_energy_goal(planned_pokemon)

    # -----------------------------------------------------------------
    # Score all legal options.
    # -----------------------------------------------------------------
    scores = []

    for option_index, o in enumerate(select.option):
        score = 0

        if o.type == OptionType.NUMBER:
            score = o.number

        elif o.type == OptionType.YES:
            # Generally prefer optional beneficial effects.
            score = 1

        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)

            if card is not None:
                energy_count = len(card.energies) if isinstance(card, Pokemon) else 0

                if context == SelectContext.SWITCH or context == SelectContext.TO_ACTIVE:
                    # Choose own active Pokemon or opponent target.
                    if o.playerIndex == my_index:
                        score += field_attacker_score(card, active=True)
                        score += energy_count * 2
                        if o.index == plan.attacker - 1:
                            score += 5000
                    else:
                        score += pokemon_score(card)
                        if o.index == plan.target - 1:
                            score += 5000

                elif context == SelectContext.SETUP_ACTIVE_POKEMON:
                    score = setup_active_score(card, state.firstPlayer == my_index)

                elif context == SelectContext.TO_HAND:
                    score = to_hand_score(card, field_counts, hand_counts, discard_counts, state)

                elif context == SelectContext.ATTACH_FROM:
                    score = energy_score(card, o.area == AreaType.ACTIVE)

                elif "DISCARD" in ctx_name or "TRASH" in ctx_name:
                    score = discard_score(card, field_counts, hand_counts)

                else:
                    # Generic card selection fallback.
                    # If selecting from opponent field, prefer valuable targets.
                    if isinstance(card, Pokemon) and o.playerIndex != my_index:
                        score = pokemon_score(card)
                    else:
                        score = to_hand_score(card, field_counts, hand_counts, discard_counts, state)

        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            if card is None:
                score = -1
            else:
                data = card_table[card.id]

                if data.cardType == CardType.POKEMON:
                    score = bench_play_score(card.id, field_counts)

                else:
                    score = 10000

                    if card.id == Switch:
                        if plan.attacker >= 1:
                            score = 6200
                        else:
                            score = -1

                    elif card.id == Boss_Orders:
                        if plan.target >= 1:
                            score = 4300
                        else:
                            score = -1

                    elif card.id == Vitality_Forest:
                        if stadium_id == Vitality_Forest:
                            score = -1
                        else:
                            score = 5200

                    elif card.id == Precious_Carry:
                        # Very strong early-game setup card.
                        if state.turn <= 2:
                            score = 8200
                        else:
                            score = 4800

                    elif card.id == Ultra_Ball:
                        score = 7000
                        # If no basic/evolution target is needed, reduce overuse.
                        if (
                            field_counts[Bulbasaur] >= 1
                            and field_counts[Ivysaur] >= 1
                            and field_counts[Mega_Venusaur_ex] >= 1
                        ):
                            score -= 1200

                    elif card.id == Bug_Catching_Set:
                        score = 6800

                    elif card.id == Pokegear_3_0:
                        score = 5900 if not state.supporterPlayed else 2600

                    elif card.id == Mitsuru_Consideration:
                        score = 3400 if not state.supporterPlayed else -1

                    elif card.id == Lillie_Determination:
                        score = 3500 if not state.supporterPlayed else -1

        elif o.type == OptionType.ATTACH:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)

            if card is not None and pokemon is not None:
                score = energy_score(pokemon, o.inPlayArea == AreaType.ACTIVE)

                # If this attachment completes our planned attacker, boost.
                target_index = 0 if o.inPlayArea == AreaType.ACTIVE else 1 + o.inPlayIndex
                if target_index == plan.attacker and plan.energy:
                    score += 500

        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            evolve_card = get_card(obs, AreaType.HAND, o.index, my_index)

            score = 9000
            if pokemon is not None:
                score += len(pokemon.energies) * 50

            if evolve_card is not None:
                if evolve_card.id == Mega_Venusaur_ex:
                    score += 1400
                elif evolve_card.id == Ivysaur:
                    score += 900

        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)

            if card is not None and card.id == Teal_Mask_Ogerpon_ex:
                # Use Ogerpon's tempo/draw ability as much as possible when legal.
                score = 32000
            elif card is not None and card.id == Vitality_Forest:
                score = 500
            else:
                score = 30000

        elif o.type == OptionType.RETREAT:
            if plan.attacker >= 1:
                score = 4100
            else:
                score = -1

        elif o.type == OptionType.ATTACK:
            # When an attack is legal, it should usually be used before ending the turn.
            # Later attack options are often stronger, so add a tiny option-index preference.
            score = 20000 + option_index

        else:
            score = 0

        scores.append(score)

    # Select options in descending order of score.
    desc_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]

    # Track Ogerpon ability usage for card-selection heuristics.
    if context == SelectContext.MAIN and len(desc_indices) > 0:
        best_option = select.option[desc_indices[0]]
        if best_option.type == OptionType.ABILITY:
            card = get_card(obs, best_option.area, best_option.index, my_index)
            if card is not None and card.id == Teal_Mask_Ogerpon_ex:
                ability_used = True

    return desc_indices[:select.maxCount]
