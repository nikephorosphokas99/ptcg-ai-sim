import pandas as pd, json, re, sys
EN = sys.argv[1]; F = sys.argv[2]
df = pd.read_csv(EN)
TYPE_COL = 'Stage (Pokémon)/Type (Energy and Trainer)'
SYM = {'C':0,'G':1,'R':2,'W':3,'L':4,'P':5,'F':6,'D':7,'M':8,'N':9}
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[’‘`]", "'", str(s))).strip()
name_norm = lambda s: re.sub(r"EX$", " ex", norm(s)).replace("  ", " ")

files = {'Pokémon':'pokemon_features.csv','Item':'item_features.csv','Supporter':'supporter_features.csv',
         'Pokémon Tool':'tool_features.csv','Stadium':'stadium_features.csv','Energy':'energy_features.csv'}
feat = {}
for t, f in files.items():
    d = pd.read_csv(f"{F}/{f}")
    for _, r in d.iterrows(): feat[int(r.card_id)] = (t, r)
atk = pd.read_csv(f"{F}/pokemon_attacks.csv")
name_by_id = {i: norm(r['name']) for i, (t, r) in feat.items()}

errs = []
def err(cid, what, en, ours): errs.append((cid, name_by_id.get(cid, '?'), what, en, ours))

for cid, g in df.groupby('Card ID'):
    row = g.iloc[0]
    en_type = row[TYPE_COL]
    en_group = 'Pokémon' if 'Pokémon' in en_type and en_type != 'Pokémon Tool' else ('Energy' if 'Energy' in en_type else en_type)
    if cid not in feat: err(cid, 'missing', en_group, None); continue
    t, r = feat[cid]
    if t != en_group: err(cid, 'card_type', en_group, t)
    if name_norm(row['Card Name']) != name_norm(r['name']): err(cid, 'name', row['Card Name'], r['name'])
    if t == 'Energy':
        if int(r.basic) != int(en_type == 'Basic Energy'): err(cid, 'basic', en_type, r.basic)
    if t != 'Pokémon': continue
    # ---- Pokémon fields ----
    stage = {'Basic Pokémon':0,'Stage 1 Pokémon':1,'Stage 2 Pokémon':2}[en_type]
    if stage != r.stage: err(cid, 'stage', stage, r.stage)
    if int(row.HP) != r.hp: err(cid, 'hp', row.HP, r.hp)
    rt = 0 if pd.isna(row.Retreat) else int(row.Retreat)
    if rt != r.retreat_cost: err(cid, 'retreat', rt, r.retreat_cost)
    for col, ours in [('Weakness', r.weakness), ('Resistance (Type)', r.resistance)]:
        en = -1 if pd.isna(row[col]) else SYM[row[col].strip('{}')]
        if en != ours: err(cid, col, row[col], ours)
    if pd.notna(row.Type) and row.Type.strip('{}') in SYM and SYM[row.Type.strip('{}')] != r.type:
        err(cid, 'type', row.Type, r.type)
    rule = row.Rule if pd.notna(row.Rule) else ''
    if int('ex' in rule) != r.ex: err(cid, 'ex', rule, r.ex)
    if int(rule == 'Mega Pokémon ex') != r.mega_ex: err(cid, 'mega_ex', rule, r.mega_ex)
    cat = row.Category if pd.notna(row.Category) else ''
    if int(cat.startswith('Tera')) != r.tera: err(cid, 'tera', cat, r.tera)
    pre = row['Previous stage']
    pre_names = sorted({name_by_id.get(i, '?') for i in json.loads(r.pre_evolution_ids)})
    if (pd.notna(pre) and pre_names != [norm(pre)]) or (pd.isna(pre) and pre_names):
        err(cid, 'pre_evolution', pre, pre_names)
    # ---- attacks: EN rows that have a Cost are attacks; others are abilities/rules ----
    en_atk = g[g.Cost.notna()]
    ours = atk[atk.card_id == cid]
    if sorted(map(norm, en_atk['Move Name'])) != sorted(map(norm, ours['name'])):
        err(cid, 'attack_names', list(en_atk['Move Name']), list(ours['name'])); continue
    for _, a in en_atk.iterrows():
        o = ours[ours['name'].map(norm) == norm(a['Move Name'])].iloc[0]
        en_dmg = 0 if pd.isna(a.Damage) else int(re.sub(r'\D', '', str(a.Damage)) or 0)
        if en_dmg != o.damage: err(cid, f"damage:{a['Move Name']}", a.Damage, o.damage)
        en_mod = {'×':'x','+':'+','-':'-'}.get(str(a.Damage)[-1:], '') if pd.notna(a.Damage) else ''
        our_mod = '' if pd.isna(o.damage_mod) else o.damage_mod
        if en_mod != our_mod: err(cid, f"damage_mod:{a['Move Name']}", a.Damage, our_mod)
        syms = re.findall(r'\{(\w)\}', a.Cost) + ['C'] * a.Cost.count('●')
        en_cost = [syms.count(k) for k in 'CGRWLPFDMN']
        our_cost = [o[f'cost_{n}'] for n in ['colorless','grass','fire','water','lightning','psychic','fighting','darkness','metal','dragon']]
        if en_cost != our_cost: err(cid, f"cost:{a['Move Name']}", a.Cost, our_cost)
    # abilities: EN non-attack rows excluding rule-only rows like [Tera]
    en_ab = g[g.Cost.isna() & g['Move Name'].notna() & g['Move Name'].astype(str).str.startswith('[Ability]')]
    if len(en_ab) != r.has_ability * len(json.loads(r.ability_ids)) and not (len(en_ab) == 0 and r.has_ability == 0):
        err(cid, 'ability_count', len(en_ab), len(json.loads(r.ability_ids)))

extra = set(feat) - set(df['Card ID'])
print(f"EN cards: {df['Card ID'].nunique()}  feature rows: {len(feat)}  extra in features: {sorted(extra)}")
print(f"attack rows: EN {df.Cost.notna().sum()} vs ours {len(atk)}")
print(f"mismatches: {len(errs)}")
e = pd.DataFrame(errs, columns=['card_id','name','field','EN_csv','features'])
print(e.field.str.split(':').str[0].value_counts().to_string() if len(e) else '')
print(e.head(40).to_string())
e.to_csv('mismatches.csv', index=False)
