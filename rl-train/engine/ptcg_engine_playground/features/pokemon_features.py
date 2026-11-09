"""Static Pokemon feature table built from the engine's Python API (cg.api).

Usage (from the folder that contains the `cg` package):
    python pokemon_features.py <out_dir>
Writes pokemon_features.csv (one row per Pokemon) and pokemon_attacks.csv (one row per attack).
Effects of attacks/abilities are left as IDs: the agent learns them via embeddings in self-play.
"""
import csv, json, os, sys
from collections import defaultdict
sys.path.insert(0, os.getcwd())
from cg import api

N_TYPES = 12  # EnergyType 0..11 (COLORLESS..TEAM_ROCKET)
OUT = sys.argv[1] if len(sys.argv) > 1 else "."

cards = api.all_card_data()
attacks = {a.attackId: a for a in api.all_attack()}
pokemon = [c for c in cards if c.cardType == api.CardType.POKEMON]

# name -> ids (any card type, so Fossil items resolve too)
ids_by_name = defaultdict(list)
for c in cards:
    ids_by_name[c.name].append(c.cardId)
post = defaultdict(list)
for c in pokemon:
    if c.evolvesFrom:
        for pid in ids_by_name.get(c.evolvesFrom, []):
            post[pid].append(c.cardId)

def cost_vec(energies):
    v = [0] * N_TYPES
    for e in energies:
        v[int(e)] += 1
    return v

rows, atk_rows = [], []
for c in pokemon:
    stage = 2 if c.stage2 else 1 if c.stage1 else 0
    prize = 3 if c.megaEx else 2 if c.ex else 1
    atks = [attacks[a] for a in c.attacks]
    rows.append({
        "card_id": c.cardId, "name": c.name,
        # categorical
        "type": int(c.energyType), "stage": stage,
        "weakness": -1 if c.weakness is None else int(c.weakness),
        "resistance": -1 if c.resistance is None else int(c.resistance),
        "ex": int(c.ex), "mega_ex": int(c.megaEx), "tera": int(c.tera),
        "has_ability": int(len(c.skills) > 0),
        "ability_ids": json.dumps([f"{c.cardId}:{i}" for i in range(len(c.skills))]),
        "attack_ids": json.dumps(c.attacks),
        "pre_evolution_ids": json.dumps(ids_by_name.get(c.evolvesFrom, []) if c.evolvesFrom else []),
        "post_evolution_ids": json.dumps(sorted(post.get(c.cardId, []))),
        # numerical
        "hp": c.hp, "prize_value": prize, "retreat_cost": c.retreatCost,
        "n_attacks": len(atks),
        "max_damage": max([a.damage for a in atks], default=0),
        "min_attack_cost": min([len(a.energies) for a in atks], default=0),
    })
    for a in atks:
        atk_rows.append({
            "card_id": c.cardId, "attack_id": a.attackId, "name": a.name,
            "damage": a.damage, "cost_total": len(a.energies),
            **{f"cost_{t.name.lower()}": n for t, n in zip(api.EnergyType, cost_vec(a.energies))},
            "has_effect": int(bool(a.text)), "text": a.text,
        })

for fn, data in [("pokemon_features.csv", rows), ("pokemon_attacks.csv", atk_rows)]:
    with open(os.path.join(OUT, fn), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
print(len(rows), "pokemon,", len(atk_rows), "attacks ->", OUT)
