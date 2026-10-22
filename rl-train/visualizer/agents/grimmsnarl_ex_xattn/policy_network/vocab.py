import re
import sys
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from cg.api import all_attack, all_card_data  # noqa: E402


class EnergyType(IntEnum):
    COLORLESS = 0
    GRASS = 1
    FIRE = 2
    WATER = 3
    LIGHTNING = 4
    PSYCHIC = 5
    FIGHTING = 6
    DARKNESS = 7
    METAL = 8
    DRAGON = 9
    RAINBOW = 10
    TEAM_ROCKET = 11


class CardType(IntEnum):
    POKEMON = 0
    ITEM = 1
    TOOL = 2
    SUPPORTER = 3
    STADIUM = 4
    BASIC_ENERGY = 5
    SPECIAL_ENERGY = 6


class SpecialConditionType(IntEnum):
    POISON = 0
    BURN = 1
    SLEEP = 2
    PARALYZE = 3
    CONFUSE = 4


class AreaType(IntEnum):
    DECK = 1
    HAND = 2
    DISCARD = 3
    ACTIVE = 4
    BENCH = 5
    PRIZE = 6
    STADIUM = 7
    ENERGY = 8
    TOOL = 9
    PRE_EVOLUTION = 10
    PLAYER = 11
    LOOKING = 12


class OptionType(IntEnum):
    NUMBER = 0
    YES = 1
    NO = 2
    CARD = 3
    TOOL_CARD = 4
    ENERGY_CARD = 5
    ENERGY = 6
    PLAY = 7
    ATTACH = 8
    EVOLVE = 9
    ABILITY = 10
    DISCARD = 11
    RETREAT = 12
    ATTACK = 13
    END = 14
    SKILL = 15
    SPECIAL_CONDITION = 16


class SelectType(IntEnum):
    MAIN = 0
    CARD = 1
    ATTACHED_CARD = 2
    CARD_OR_ATTACHED_CARD = 3
    ENERGY = 4
    SKILL = 5
    ATTACK = 6
    EVOLVE = 7
    COUNT = 8
    YES_NO = 9
    SPECIAL_CONDITION = 10


class SelectContext(IntEnum):
    MAIN = 0
    SETUP_ACTIVE_POKEMON = 1
    SETUP_BENCH_POKEMON = 2
    SWITCH = 3
    TO_ACTIVE = 4
    TO_BENCH = 5
    TO_FIELD = 6
    TO_HAND = 7
    DISCARD = 8
    TO_DECK = 9
    TO_DECK_BOTTOM = 10
    TO_PRIZE = 11
    NOT_MOVE = 12
    DAMAGE_COUNTER = 13
    DAMAGE_COUNTER_ANY = 14
    DAMAGE = 15
    REMOVE_DAMAGE_COUNTER = 16
    HEAL = 17
    EVOLVES_FROM = 18
    EVOLVES_TO = 19
    DEVOLVE = 20
    ATTACH_FROM = 21
    ATTACH_TO = 22
    DETACH_FROM = 23
    LOOK = 24
    EFFECT_TARGET = 25
    DISCARD_ENERGY_CARD = 26
    DISCARD_TOOL_CARD = 27
    SWITCH_ENERGY_CARD = 28
    DISCARD_CARD_OR_ATTACHED_CARD = 29
    DISCARD_ENERGY = 30
    TO_HAND_ENERGY = 31
    TO_DECK_ENERGY = 32
    SWITCH_ENERGY = 33
    SKILL_ORDER = 34
    ATTACK = 35
    DISABLE_ATTACK = 36
    EVOLVE = 37
    DRAW_COUNT = 38
    DAMAGE_COUNTER_COUNT = 39
    REMOVE_DAMAGE_COUNTER_COUNT = 40
    IS_FIRST = 41
    MULLIGAN = 42
    ACTIVATE = 43
    FIRST_EFFECT = 44
    MORE_DEVOLVE = 45
    COIN_HEAD = 46
    AFFECT_SPECIAL_CONDITION = 47
    RECOVER_SPECIAL_CONDITION = 48


TRAINED_CARD_POOL = 1267
TRAINED_ATTACK_POOL = 1556

_ALL_CARDS = all_card_data()
_ALL_ATTACKS = all_attack()
MAX_CARD_ID = min(TRAINED_CARD_POOL, max(card.cardId for card in _ALL_CARDS))
MAX_ATTACK_ID = min(TRAINED_ATTACK_POOL, max(attack.attackId for attack in _ALL_ATTACKS))
CARD_ID_VOCAB_SIZE = MAX_CARD_ID + 1
ATTACK_ID_VOCAB_SIZE = MAX_ATTACK_ID + 1
CARD_TABLE_SIZE = max(card.cardId for card in _ALL_CARDS) + 1
ATTACK_TABLE_SIZE = max(attack.attackId for attack in _ALL_ATTACKS) + 1


def embedding_card_id(card_id: torch.Tensor) -> torch.Tensor:
    return torch.where(card_id < CARD_ID_VOCAB_SIZE, card_id, torch.zeros_like(card_id))


class CardStage(IntEnum):
    NOT_APPLICABLE = 0
    BASIC = 1
    STAGE1 = 2
    STAGE2 = 3


CARD_STAGE_VOCAB_SIZE = len(CardStage)

CARD_TYPE_VOCAB_SIZE = len(CardType) + 1
CARD_ENERGY_TYPE_VOCAB_SIZE = len(EnergyType) + 1

CARD_WEAKNESS_VOCAB_SIZE = len(EnergyType) + 1
CARD_RESISTANCE_VOCAB_SIZE = len(EnergyType) + 1


def _normalize_rules_text(text: str) -> str:
    return " ".join(text.lower().replace("’", "'").replace("‘", "'").split())


_SKILL_PATTERNS: tuple[tuple[str, str], ...] = (
    ("draw", r"draw \d|draw a card|draw card"),
    ("search_deck", r"search your deck"),
    ("shuffle", r"shuffle"),
    ("to_hand", r"into your hand|to your hand"),
    ("discard", r"discard"),
    ("heal", r"heal"),
    ("damage_counter", r"damage counter"),
    ("switch", r"switch"),
    ("attach_energy", r"attach .{0,40}energy|energy .{0,20}attach"),
    ("provides_energy", r"provides"),
    ("prevent", r"prevent|no damage|isn't affected|is not affected"),
    ("coin_flip", r"flip a coin"),
    ("once_per_turn", r"once during your turn"),
    ("knocked_out", r"knocked out"),
    ("evolve", r"evolve|evolution"),
    ("retreat", r"retreat"),
    ("special_condition", r"poison|burn|asleep|paralyz|confus"),
    ("reveal", r"reveal"),
    ("bench", r"bench"),
    ("active_spot", r"active spot"),
    ("prize", r"prize"),
    ("targets_opponent", r"opponent"),
    ("extra_hp", r"gets \+\d+ hp"),
    ("cost_hand", r"discard a card from your hand|discard \d+ cards? from your hand"),
    ("restriction", r"can't attack|can't retreat|can't use"),
)

SKILL_FLAG_NAMES = tuple(name for name, _ in _SKILL_PATTERNS)
SKILL_FLAG_COUNT = len(SKILL_FLAG_NAMES)

SKILL_COUNT_CAP = 4
SKILL_COUNT_VOCAB_SIZE = SKILL_COUNT_CAP + 1


def _build_skill_tables():
    names: dict[str, int] = {}
    count = torch.zeros(CARD_TABLE_SIZE, dtype=torch.long)
    ability = torch.zeros(CARD_TABLE_SIZE, dtype=torch.long)
    flags = torch.zeros((CARD_TABLE_SIZE, SKILL_FLAG_COUNT), dtype=torch.long)
    compiled = [(index, re.compile(pattern)) for index, (_, pattern) in enumerate(_SKILL_PATTERNS)]

    in_pool = sorted(_ALL_CARDS, key=lambda card: card.cardId > MAX_CARD_ID)
    for card in in_pool:
        if not card.skills:
            continue
        count[card.cardId] = min(len(card.skills), SKILL_COUNT_CAP)
        skill_name = card.skills[0].name.strip()
        if card.cardId <= MAX_CARD_ID:
            ability[card.cardId] = names.setdefault(skill_name, len(names) + 1)
        else:
            ability[card.cardId] = names.get(skill_name, 0)
        for skill in card.skills:
            text = _normalize_rules_text(skill.text)
            for index, pattern in compiled:
                if pattern.search(text):
                    flags[card.cardId, index] = 1
    return count, ability, flags, len(names) + 1


(
    _CARD_SKILL_COUNT,
    _CARD_ABILITY_ID,
    _CARD_SKILL_FLAGS,
    ABILITY_ID_VOCAB_SIZE,
) = _build_skill_tables()

_CARD_SKILL_COUNT_NORM = _CARD_SKILL_COUNT.float() / SKILL_COUNT_CAP
_CARD_SKILL_FLAGS_FLOAT = _CARD_SKILL_FLAGS.float()

HP_CAP = 400.0
RETREAT_COST_CAP = 5.0
ENERGY_COST_CAP = 6.0

_CARD_STAGE = torch.zeros(CARD_TABLE_SIZE, dtype=torch.long)
_CARD_TYPE = torch.zeros(CARD_TABLE_SIZE, dtype=torch.long)
_CARD_ENERGY_TYPE = torch.zeros(CARD_TABLE_SIZE, dtype=torch.long)
_CARD_EX = torch.zeros(CARD_TABLE_SIZE, dtype=torch.long)
_CARD_MEGA_EX = torch.zeros(CARD_TABLE_SIZE, dtype=torch.long)
_CARD_TERA = torch.zeros(CARD_TABLE_SIZE, dtype=torch.long)
_CARD_ACE_SPEC = torch.zeros(CARD_TABLE_SIZE, dtype=torch.long)
_CARD_HP = torch.zeros(CARD_TABLE_SIZE, dtype=torch.long)
_CARD_RETREAT_COST = torch.zeros(CARD_TABLE_SIZE, dtype=torch.long)
_CARD_WEAKNESS = torch.zeros(CARD_TABLE_SIZE, dtype=torch.long)
_CARD_RESISTANCE = torch.zeros(CARD_TABLE_SIZE, dtype=torch.long)
for _card in _ALL_CARDS:
    if _card.basic:
        _stage = CardStage.BASIC
    elif _card.stage1:
        _stage = CardStage.STAGE1
    elif _card.stage2:
        _stage = CardStage.STAGE2
    else:
        _stage = CardStage.NOT_APPLICABLE
    _CARD_STAGE[_card.cardId] = _stage
    _CARD_TYPE[_card.cardId] = _card.cardType + 1
    _CARD_ENERGY_TYPE[_card.cardId] = _card.energyType + 1
    _CARD_EX[_card.cardId] = int(_card.ex)
    _CARD_MEGA_EX[_card.cardId] = int(_card.megaEx)
    _CARD_TERA[_card.cardId] = int(_card.tera)
    _CARD_ACE_SPEC[_card.cardId] = int(_card.aceSpec)
    _CARD_HP[_card.cardId] = _card.hp
    _CARD_RETREAT_COST[_card.cardId] = _card.retreatCost
    _CARD_WEAKNESS[_card.cardId] = 0 if _card.weakness is None else _card.weakness + 1
    _CARD_RESISTANCE[_card.cardId] = 0 if _card.resistance is None else _card.resistance + 1
del _card, _stage

_CARD_STAGE_NORM = _CARD_STAGE.float() / (len(CardStage) - 1)

_CARD_HP_NORM = (_CARD_HP.float() / HP_CAP).clamp(0.0, 1.0)
_CARD_RETREAT_COST_NORM = (_CARD_RETREAT_COST.float() / RETREAT_COST_CAP).clamp(0.0, 1.0)

_ATTACK_EFFECT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("damage_scaling", r"for each"),
    ("more_damage", r"more damage"),
    ("discard", r"discard"),
    ("next_turn", r"next turn"),
    ("coin_flip", r"flip a coin|flip \d+ coins"),
    ("shuffle", r"shuffle"),
    ("special_condition", r"poison|burn|asleep|paralyz|confus"),
    ("ignore_weak_res", r"weakness and resistance"),
    ("search_deck", r"search your deck"),
    ("damage_counter", r"damage counter"),
    ("discard_own_energy", r"discard all energy from this|discard .{0,30}energy from this pok"),
    ("bench_damage", r"damage to each|damage to \d+ of your opponent|to each of your opponent"),
    ("attack_lock", r"can't use|can't attack|cannot attack"),
    ("self_damage", r"damage to itself"),
    ("to_hand", r"into your hand|to your hand"),
    ("reveal", r"reveal"),
    ("heal", r"heal"),
    ("draw", r"draw"),
    ("retreat", r"retreat"),
    ("from_discard_pile", r"discard pile"),
    ("switch", r"switch"),
    ("prevent", r"prevent|no damage|isn't affected"),
    ("basic_pokemon", r"basic pok"),
    ("knocked_out", r"knocked out"),
    ("prize", r"prize"),
    ("attach_energy", r"attach .{0,40}energy"),
    ("move_energy", r"move .{0,30}energy"),
    ("targets_opponent", r"opponent"),
)

ATTACK_EFFECT_FLAG_NAMES = tuple(name for name, _ in _ATTACK_EFFECT_PATTERNS)
ATTACK_EFFECT_FLAG_COUNT = len(ATTACK_EFFECT_FLAG_NAMES)


def _build_attack_tables() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    damage = torch.zeros(ATTACK_TABLE_SIZE, dtype=torch.long)
    energy_cost = torch.zeros(ATTACK_TABLE_SIZE, dtype=torch.long)
    energy_type_counts = torch.zeros((ATTACK_TABLE_SIZE, len(EnergyType)), dtype=torch.long)
    effect_flags = torch.zeros((ATTACK_TABLE_SIZE, ATTACK_EFFECT_FLAG_COUNT), dtype=torch.long)
    compiled = [
        (index, re.compile(pattern))
        for index, (_, pattern) in enumerate(_ATTACK_EFFECT_PATTERNS)
    ]
    for attack in _ALL_ATTACKS:
        damage[attack.attackId] = attack.damage
        energy_cost[attack.attackId] = len(attack.energies)
        for energy_type in attack.energies:
            energy_type_counts[attack.attackId, energy_type] += 1
        if attack.text and attack.text.strip():
            text = _normalize_rules_text(attack.text)
            for index, pattern in compiled:
                if pattern.search(text):
                    effect_flags[attack.attackId, index] = 1
    return damage, energy_cost, energy_type_counts, effect_flags


(
    _ATTACK_DAMAGE,
    _ATTACK_ENERGY_COST,
    _ATTACK_ENERGY_TYPE_COUNTS,
    _ATTACK_EFFECT_FLAGS,
) = _build_attack_tables()

_ATTACK_EFFECT_FLAGS_FLOAT = _ATTACK_EFFECT_FLAGS.float()

_ATTACK_DAMAGE_NORM = (_ATTACK_DAMAGE.float() / HP_CAP).clamp(0.0, 1.0)
_ATTACK_ENERGY_COST_NORM = (_ATTACK_ENERGY_COST.float() / ENERGY_COST_CAP).clamp(0.0, 1.0)
_ATTACK_ENERGY_TYPE_COUNTS_NORM = (
    _ATTACK_ENERGY_TYPE_COUNTS.float() / ENERGY_COST_CAP
).clamp(0.0, 1.0)


def attack_flags(attack_id: torch.Tensor) -> dict[str, torch.Tensor]:
    safe = attack_id.clamp(min=0)
    return {
        "attack_id_safe": torch.where(safe < ATTACK_ID_VOCAB_SIZE, safe, torch.zeros_like(safe)),
        "attack_damage_norm": _ATTACK_DAMAGE_NORM[safe],
        "attack_energy_cost_norm": _ATTACK_ENERGY_COST_NORM[safe],
        "attack_energy_type_counts": _ATTACK_ENERGY_TYPE_COUNTS_NORM[safe],
        "attack_effect_flags": _ATTACK_EFFECT_FLAGS_FLOAT[safe],
    }


_CARD_FLAG_TABLES: dict[str, torch.Tensor] = {
    "stage": _CARD_STAGE,
    "stage_norm": _CARD_STAGE_NORM,
    "type": _CARD_TYPE,
    "energy_type": _CARD_ENERGY_TYPE,
    "ex": _CARD_EX,
    "mega_ex": _CARD_MEGA_EX,
    "tera": _CARD_TERA,
    "ace_spec": _CARD_ACE_SPEC,
    "hp_norm": _CARD_HP_NORM,
    "retreat_cost_norm": _CARD_RETREAT_COST_NORM,
    "weakness": _CARD_WEAKNESS,
    "resistance": _CARD_RESISTANCE,
    "skill_count": _CARD_SKILL_COUNT,
    "skill_count_norm": _CARD_SKILL_COUNT_NORM,
    "ability_id": _CARD_ABILITY_ID,
    "skill_flags": _CARD_SKILL_FLAGS_FLOAT,
}


def card_type_flags(
    card_id: torch.Tensor, fields: tuple[str, ...] | None = None
) -> dict[str, torch.Tensor]:
    tables = _CARD_FLAG_TABLES
    return {flag: tables[flag][card_id] for flag in (fields or tables)}

OPTION_TYPE_VOCAB_SIZE = len(OptionType)
AREA_VOCAB_SIZE = len(AreaType) + 1
SELECT_TYPE_VOCAB_SIZE = len(SelectType)
SELECT_CONTEXT_VOCAB_SIZE = len(SelectContext) + 1

TARGETS_OPPONENT_VOCAB_SIZE = 3

SPECIAL_CONDITION_VOCAB_SIZE = len(SpecialConditionType) + 1

NO_VALUE = -1


def _area(value: int | None) -> int:
    return 0 if value is None else int(value)


def _card_id(value: int | None) -> int:
    return 0 if value is None else int(value)


def _magnitude(value: int | None) -> int:
    return NO_VALUE if value is None else int(value)


def _targets_opponent(value: bool | None) -> int:
    return 2 if value is None else int(value)


def _special_condition(value: int | None) -> int:
    return 0 if value is None else int(value) + 1


@dataclass
class OptionsVocab:
    type: torch.Tensor
    number: torch.Tensor
    area: torch.Tensor
    index: torch.Tensor
    targets_opponent: torch.Tensor
    tool_index: torch.Tensor
    energy_index: torch.Tensor
    count: torch.Tensor
    in_play_area: torch.Tensor
    in_play_index: torch.Tensor
    attack_id: torch.Tensor
    card_id: torch.Tensor
    serial: torch.Tensor
    special_condition_type: torch.Tensor

    @classmethod
    def from_options(cls, options: list[dict[str, Any]]) -> "OptionsVocab":
        return cls(
            type=torch.tensor([int(o["type"]) for o in options], dtype=torch.long),
            number=torch.tensor([_magnitude(o["number"]) for o in options], dtype=torch.long),
            area=torch.tensor([_area(o["area"]) for o in options], dtype=torch.long),
            index=torch.tensor([_magnitude(o["index"]) for o in options], dtype=torch.long),
            targets_opponent=torch.tensor(
                [_targets_opponent(o["targets_opponent"]) for o in options], dtype=torch.long
            ),
            tool_index=torch.tensor([_magnitude(o["toolIndex"]) for o in options], dtype=torch.long),
            energy_index=torch.tensor([_magnitude(o["energyIndex"]) for o in options], dtype=torch.long),
            count=torch.tensor([_magnitude(o["count"]) for o in options], dtype=torch.long),
            in_play_area=torch.tensor([_area(o["inPlayArea"]) for o in options], dtype=torch.long),
            in_play_index=torch.tensor([_magnitude(o["inPlayIndex"]) for o in options], dtype=torch.long),
            attack_id=torch.tensor([_magnitude(o["attackId"]) for o in options], dtype=torch.long),
            card_id=torch.tensor([_card_id(o["cardId"]) for o in options], dtype=torch.long),
            serial=torch.tensor([_magnitude(o["serial"]) for o in options], dtype=torch.long),
            special_condition_type=torch.tensor(
                [_special_condition(o["specialConditionType"]) for o in options], dtype=torch.long
            ),
        )


@dataclass
class SelectionVocab:
    type: torch.Tensor
    context: torch.Tensor
    min_count: torch.Tensor
    max_count: torch.Tensor
    remain_damage_counter: torch.Tensor
    remain_energy_cost: torch.Tensor
    deck_size: torch.Tensor
    context_card_id: torch.Tensor
    effect_card_id: torch.Tensor

    @classmethod
    def from_selection(cls, selection: dict[str, Any]) -> "SelectionVocab":
        deck = selection["deck"]
        context_card = selection["contextCard"]
        effect = selection["effect"]
        return cls(
            type=torch.tensor(int(selection["type"]), dtype=torch.long),
            context=torch.tensor(int(selection["context"]), dtype=torch.long),
            min_count=torch.tensor(int(selection["minCount"]), dtype=torch.long),
            max_count=torch.tensor(int(selection["maxCount"]), dtype=torch.long),
            remain_damage_counter=torch.tensor(int(selection["remainDamageCounter"]), dtype=torch.long),
            remain_energy_cost=torch.tensor(int(selection["remainEnergyCost"]), dtype=torch.long),
            deck_size=torch.tensor(0 if deck is None else len(deck), dtype=torch.long),
            context_card_id=torch.tensor(_card_id(context_card and context_card["id"]), dtype=torch.long),
            effect_card_id=torch.tensor(_card_id(effect and effect["id"]), dtype=torch.long),
        )


def pad_options(per_decision_options: list[OptionsVocab]) -> dict[str, torch.Tensor]:
    chain_len = len(per_decision_options)
    max_options = max((o.type.numel() for o in per_decision_options), default=0)

    def stack(field: str, fill: int) -> torch.Tensor:
        out = torch.full((chain_len, max_options), fill, dtype=torch.long)
        for i, options in enumerate(per_decision_options):
            values = getattr(options, field)
            out[i, : values.numel()] = values
        return out

    mask = torch.zeros((chain_len, max_options), dtype=torch.bool)
    for i, options in enumerate(per_decision_options):
        mask[i, : options.type.numel()] = True

    magnitude_fields = (
        "number", "index", "energy_index", "count",
        "in_play_index", "attack_id", "serial", "tool_index",
    )
    zero_filled_fields = (
        "area", "in_play_area", "targets_opponent", "card_id", "special_condition_type",
    )

    result = {"type": stack("type", 0), "options_mask": mask}
    for field in magnitude_fields:
        result[field] = stack(field, NO_VALUE)
    for field in zero_filled_fields:
        result[field] = stack(field, 0)
    return result


def stack_selections(selections: list[SelectionVocab]) -> dict[str, torch.Tensor]:
    fields = (
        "type", "context", "min_count", "max_count", "remain_damage_counter",
        "remain_energy_cost", "deck_size", "context_card_id", "effect_card_id",
    )
    if not selections:
        return {field: torch.empty(0, dtype=torch.long) for field in fields}
    return {field: torch.stack([getattr(s, field) for s in selections]) for field in fields}
