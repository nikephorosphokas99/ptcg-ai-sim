"""Static features for Items, Pokemon Tools, Stadiums and Energy cards.

Usage (from the folder that contains the `cg` package):
    python ptcg_engine_playground/features/trainer_energy_features.py "<path/to/CardImpl.h>" <out_dir>
Writes item_features.csv, tool_features.csv, stadium_features.csv, energy_features.csv.

Tags are rule-based on the engine's effect DSL (CardImpl.h); review edge cases by hand.
Like Supporters, the card-ID embedding is meant to learn whatever these tags miss.
"""
import csv, os, re, sys
sys.path.insert(0, os.getcwd())
from cg import api

SRC, OUT = sys.argv[1], sys.argv[2]
src = open(SRC, encoding="utf-8").read()
names = {c.cardId: c for c in api.all_card_data()}

ENERGY = ["Colorless", "Grass", "Fire", "Water", "Lightning", "Psychic",
          "Fighting", "Darkness", "Metal", "Dragon"]          # engine EnergyType 0..9

# ---- shared "what does it do" tags (same vocabulary as supporter_features.py) ----
ACTION_TAGS = {
    "draw":             r"effectDraw|\bDraw\b|DrawUntil|DrawTargetCount",
    "search_deck":      r"DeckToHand|DeckToBench|LookAndTo|LookDeck|DeckSelect|DeckAttach",
    "recover_discard":  r"TrashToHand|TrashToDeck|targetTrash|AttachBasicEnergyTrash",
    "bench_from_deck":  r"DeckToBench|ToBench",
    "gust":             r"SwitchEnemyBench",
    "switch_self":      r"effectSwitch\(Me\)",
    "energy_accel":     r"SelectAttach|AttachTo|AttachFrom|AttachSelectedCard|DeckAttach",
    "energy_move":      r"EnergySwitch|SelectSwitchEnergy",
    "heal":             r"Heal|RemoveDamageCounter",
    "damage_buff":      r"DamageChange|PlayerDamageChange",
    "damage_counters":  r"DamageCounter\)?\.eVal|\(DamageCounter,",
    "prize_mod":        r"TakePrizeCount|PrizeToHand|KoPrize|ToPrize",
    "opp_hand_disrupt": r"(ToTrash|ToDeck\w*|ToLooking), Enemy\)\.targetHand|Draw, Enemy",
    "opp_board_disrupt":r"(ToTrash|ToHand), (Both|Enemy)\)\.targetAttached|TrashEnergy\(\d+, Enemy",
    "protection":       r"NoDamage|NoEffectEnemy",
    "special_cond":     r"Confuse|Poison|Burn|Sleep|Paralyze",
    "evolve":           r"Evolve",
    "deck_order":       r"LookDeck|DeckBottom|ToDeckReverse, Me\)\.targetLooking",
    "self_mill":        r"DeckToTrash",
}
COMMON_FLAGS = {
    "has_hand_cost":    r"costHandTrash",
    "has_play_cond":    r"\.condition|\.exist",
    "coin_flip":        r"Coin",
    "archetype_locked": r"TeamRocket|Lillie|Ethan|Hop|Cynthia|Marnie|Steven|TargetType::N\b|targetName|NameContains|MegaEx|Terastal|targetEnergyType|Ancient|Future",
}

def blocks(kind):
    starts = list(re.finditer(r'CreateCard\((\d+), u8"[^"]*", (\w+)', src))
    for i, m in enumerate(starts):
        if m.group(2) != kind:
            continue
        end = starts[i + 1].start() if i + 1 < len(starts) else len(src)
        raw = src[m.start():end]
        dsl = "\n".join(l for l in raw.splitlines()[1:]
                        if not re.search(r"textEn|nameEn|playSkill|delaySkill", l))
        en = re.search(r'textEn\(u8"[^"]*", u8"((?:[^"\\]|\\.)*)"', raw)
        yield int(m.group(1)), dsl, (en.group(1).replace("\\n", " ") if en else "")

def flag(pat, dsl):
    return int(bool(re.search(pat, dsl)))

def val(pat, dsl, agg=sum):
    """Aggregate of eVal(...) that follows each match of `pat` on the same DSL call chain."""
    vals = [int(m.group(m.lastindex))
            for m in re.finditer(pat + r"[^.\n]*\)?(?:\.\w+\([^)]*\))*?\.eVal\((-?\d+)\)", dsl)]
    return agg(vals) if vals else 0

def base(cid, dsl, text):
    c = names.get(cid)
    return {"card_id": cid, "name": c.name if c else "?", "ace_spec": int(c.aceSpec) if c else 0}

def modifiers(dsl):
    """Continuous stat changes shared by Tools, Stadiums and Special Energy."""
    return {
        "hp_delta":          val(r"MaxHpChange", dsl),
        "retreat_delta":     val(r"RetreatCostChange", dsl),
        "no_retreat":        flag(r"NoRetreatCost", dsl),
        "dmg_dealt_delta":   val(r"(?<![A-Za-z])((?:Player)?DamageChange\w*)", dsl),
        "dmg_taken_delta":   val(r"Take\w*DamageChange", dsl),
        "attack_cost_delta": val(r"AttackCostChange\w*", dsl) - val(r"AttackCostDown", dsl),
        "prize_delta":       val(r"KoPrize\w*", dsl),
        "no_ability":        flag(r"NoAbility", dsl),
        "no_tool_effect":    flag(r"NoToolEffect", dsl),
        "no_special_cond":   flag(r"NoSpecialCondition", dsl),
    }

def finish(row, dsl, text):
    row["effect_ops"] = " ".join(re.findall(r"\.(effect\w*|cost\w*|condition\w*|trigger\w*)", dsl))
    row["text"] = text
    return row

out = {}

# ---- Items ----
rows = []
for cid, dsl, text in blocks("Item"):
    r = base(cid, dsl, text)
    r.update({k: flag(p, dsl) for k, p in ACTION_TAGS.items()})
    r.update({k: flag(p, dsl) for k, p in COMMON_FLAGS.items()})
    r.update({
        "fossil":            flag(r"\.fossil\(\)", dsl),   # can be played as a Basic Pokemon
        "not_first_turn":    flag(r"NotFirstTurn", dsl),
        "first_turn_2nd_player_only": flag(r"FirstTurnSecondPlayer", dsl),
        "ends_turn":         flag(r"TurnEnd\(\)", dsl),
        "affects_opponent":  flag(r"\((?:EffectType::)?\w+, (?:TargetPlayer::)?(?:Enemy|Both)\)|SwitchEnemyBench|, Enemy\)", dsl),
    })
    nums = [int(x) for x in re.findall(r"(?:eVal|Draw|ToHand\w*|ToBench\w*|TrashToHand|Heal\w*|maxSelect|AndShuffle)\((\d+)", dsl)]
    r["magnitude"] = max([n for n in nums if n < 99], default=0)
    rows.append(finish(r, dsl, text))
out["item_features.csv"] = rows

# ---- Pokemon Tools ----
rows = []
for cid, dsl, text in blocks("Tool"):
    r = base(cid, dsl, text)
    r.update({
        "mode_continuous": flag(r"effectToolContinual|\.effect\(\w+, (Both|Me|Enemy)\)\.eVal", dsl),
        "mode_trigger":    flag(r"triggerAttachedCard|\.trigger\(", dsl),
        "mode_activated":  flag(r"toolSkillOnceTurn", dsl),
        "trig_on_damaged": flag(r"TriggerType::Damaged", dsl),
        "trig_on_ko":      flag(r"TriggerType::(Pre)?Ko", dsl),
        "trig_on_turn_end":flag(r"TriggerType::TurnEnd", dsl),
        "active_only":     flag(r"Active", dsl),
        "self_discard":    flag(r"effectMe\(ToTrash\)", dsl),
        "retaliate_counters": val(r"effectTriggerObject\(DamageCounter\)", dsl),
        "grants_attack":   flag(r"TechnicalMachine|\.attack\(", dsl),
        "restricted_holder": flag(COMMON_FLAGS["archetype_locked"] + r"|targetNotRule|RetreatCost, 4|Poison", dsl),
        "has_play_cond":   flag(r"\.condition", dsl),
    })
    r.update(modifiers(dsl))
    r.update({k: flag(ACTION_TAGS[k], dsl) for k in ("draw", "search_deck", "energy_accel", "special_cond", "energy_move")})
    rows.append(finish(r, dsl, text))
out["tool_features.csv"] = rows

# ---- Stadiums ----
rows = []
for cid, dsl, text in blocks("Stadium"):
    r = base(cid, dsl, text)
    r.update({
        "symmetric":          int(flag(r"Both|stadiumActivateSkill", dsl) or "each player" in text.lower()
                                  or "both yours and your opponent" in text.lower()),
        "activated_once_per_turn": flag(r"stadiumActivateSkillOnceTurn", dsl),
        "mode_continuous":    1 - flag(r"stadiumActivateSkillOnceTurn|\.trigger\(", dsl),
        "mode_trigger":       flag(r"\.trigger\(", dsl),
        "bench_capacity":     val(r"BenchCapacity", dsl, agg=max),
        "ends_turn":          flag(r"effectTurnEnd", dsl),
    })
    r.update(modifiers(dsl))
    r.update({k: flag(p, dsl) for k, p in ACTION_TAGS.items()
              if k in ("draw", "search_deck", "recover_discard", "bench_from_deck", "heal",
                       "switch_self", "special_cond", "evolve", "deck_order", "damage_counters", "protection")})
    r.update({k: flag(p, dsl) for k, p in COMMON_FLAGS.items()})
    rows.append(finish(r, dsl, text))
out["stadium_features.csv"] = rows

# ---- Energy (basic + special) ----
rows = []
for kind in ("BasicEnergy", "SpecialEnergy"):
    for cid, dsl, text in blocks(kind):
        r = base(cid, dsl, text)
        r["basic"] = int(kind == "BasicEnergy")
        m = re.search(r"(?:basicEnergy|specialEnergy)\(([^,)]+)(?:,\s*(\d+))?", dsl)
        spec, count = (m.group(1), int(m.group(2) or 1)) if m else ("", 0)
        provides = [t for t in ENERGY if re.search(rf"\b{t}\b", spec)]
        r["provides_any_type"] = int("All" in spec)
        for t in ENERGY:
            r[f"provides_{t.lower()}"] = int(t in provides or "All" in spec)
        r["provides_count"] = count
        r.update({
            "mode_continuous":  flag(r"effectEnergyContinual", dsl),
            "trig_on_attach":   flag(r"TriggerType::Attach", dsl),
            "trig_on_damaged":  flag(r"TriggerType::Damaged", dsl),
            "trig_on_ko":       flag(r"TriggerType::Ko", dsl),
            "on_play_effect":   flag(r"^\s*\.effect(Draw|DeckTo)", dsl),
            "discard_end_turn": flag(r"trashMyTurnEnd", dsl),
            "recur_from_trash": flag(r"canActivateTrash", dsl),
            "retaliate_counters": val(r"effectTriggerObject\(DamageCounter\)", dsl),
            "restricted_holder": flag(r"targetEnergyType|onlyTeamRocket|TargetType::Area", dsl),
            "protection":       flag(ACTION_TAGS["protection"], dsl),
        })
        r.update(modifiers(dsl))
        rows.append(finish(r, dsl, text))
out["energy_features.csv"] = rows

for fn, data in out.items():
    with open(os.path.join(OUT, fn), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    print(f"{fn}: {len(data)} rows")
