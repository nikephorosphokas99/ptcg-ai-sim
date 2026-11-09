"""Extract supporter feature vectors from the engine's CardImpl.h DSL.

Usage: python supporter_features.py <path/to/CardImpl.h> <out.csv>
Tags are rule-based on the effect DSL; review the CSV and hand-fix edge cases.
"""
import csv, re, sys

SRC, OUT = sys.argv[1], sys.argv[2]
text = open(SRC, encoding="utf-8").read()
starts = [m for m in re.finditer(r'CreateCard\((\d+), u8"[^"]*", (\w+)', text)]

# tag -> regex over the card's DSL block
TAGS = {
    "draw":            r"effectDraw|\bDraw\b|DrawUntil",
    "hand_reset":      r"(ToDeckReverseAndShuffle|ToDeckReverse|ToTrash), Me\)\.targetHand\(\)\s*\n(?!.*maxSelect)|canHandToDeckAndShuffle",
    "search_deck":     r"DeckToHand|DeckToBench|targetDeck\(\)|LookDeck|LookAndToHand",
    "recover_discard": r"TrashToHand|targetTrash|AttachBasicEnergyTrash",
    "gust":            r"SwitchEnemyBench",
    "switch_self":     r"effectSwitch\(Me\)",
    "energy_accel":    r"SelectAttach|AttachTo|AttachFrom|AttachSelectedCard",
    "energy_move":     r"EnergySwitch|SelectSwitchEnergy",
    "heal":            r"\bHeal|RemoveDamageCounter",
    "damage_buff":     r"DamageChange|PlayerDamageChange",
    "prize_mod":       r"TakePrizeCount|PrizeToHand",
    "opp_hand_disrupt":r"(ToTrash|ToDeck\w*|ToLooking), Enemy\)\.targetHand|Draw, Enemy",
    "opp_board_disrupt":r"Enemy\)\.targetAttached|Enemy\)\.targetPokemon",
    "protection":      r"NoDamage|NoEffect\w*Next",
    "special_cond":    r"Confuse|Poison|Burn|Sleep|Paralyze",
    "evolve":          r"Evolves",
    "deck_order":      r"ToDeckReverse, Me\)\.targetLooking|DeckBottom",
}
FLAGS = {
    "has_hand_cost":     r"costHandTrash",
    "has_play_cond":     r"\.condition\(|conditionPrize|conditionNotFirstTurn|condition(Less|Greater)",
    "prize_dependent":   r"targetPrize\(\)|conditionPrizeCount",
    "ko_dependent":      r"KoPreEnemyTurn",
    "can_play_turn1":    r"canPlayFirstTurn",
    "affects_opponent":  r"\.effect\w*\([A-Za-z:]+, (TargetPlayer::)?Enemy\)|SwitchEnemyBench",
    "coin_flip":         r"Coin",
    "archetype_locked":  r"TeamRocket|Lillie|Ethan|targetName|MegaEx|Terastal|targetEnergyType|EthanPokemon",
    "choose_one":        r"secondEffectStartIndex",
}

rows = []
for i, m in enumerate(starts):
    if m.group(2) != "Supporter":
        continue
    end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
    block = text[m.start():end]
    dsl = "\n".join(l for l in block.splitlines() if not re.search(r"textEn|nameEn|playSkill|delaySkill", l))
    en = re.search(r'textEn\(u8"([^"]*)", u8"((?:[^"\\]|\\.)*)"', block)
    name, eng_text = (en.group(1), en.group(2).replace("\\n", " ")) if en else ("?", "")
    nums = [int(x) for x in re.findall(r"(?:effectDraw|eVal|DrawUntil|ToHand\w*|TrashToHand|maxSelect|multiSelect|Heal\w*)\((\d+)\)", dsl)]
    row = {"card_id": int(m.group(1)), "name": name}
    for k, pat in {**TAGS, **FLAGS}.items():
        row[k] = int(bool(re.search(pat, dsl)))
    row["magnitude"] = max(nums) if nums else 0   # largest numeric param (draw/search/heal amount)
    row["effect_ops"] = " ".join(re.findall(r"\.(effect\w*|cost\w*|condition\w*)", dsl))
    row["text"] = eng_text
    rows.append(row)

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(len(rows), "supporters ->", OUT)
