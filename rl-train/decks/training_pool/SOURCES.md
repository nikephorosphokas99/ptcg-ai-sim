# Training deck pool

Opponent decks drawn for training games (`--opponent-decks decks/training_pool/*.csv`).

* The 13 visualizer decks (`visualizer/decks/`), translated to official card ids.
* `alakazam_5th`, `crustle_bot`, `archaludon` — the lists the rule-based bots in `benchmarks/` play.
* `festival_lead`, `hydrapple` — recovered from the repo's git history (commit 1b69148).
* `teal_mask_ogerpon` — bai fang's Teal Mask Ogerpon ex / Hydrapple ex / Meganium list, read from
  Kaggle episode 125180090 (their first action in the replay).
* `team_rocket` — a Team Rocket's Mewtwo ex / Spidops list. Two cards are not in this engine:
  Earthen Vessel x3 became Energy Search x3, Bravery Charm became a 2nd Rescue Board.
* `dragapult_blaziken` — Dragapult ex / Blaziken ex (with Munkidori, Shaymin).
* `dragapult_dusknoir` — Dragapult ex / Dusknoir. Both converted with `tools/decklist_to_csv.py`;
  `dragapult` itself is the Crushing Hammer build the Dragapult bot plays.
* `shardul_gharat_grimmsnarl` — Shardul Gharat's Grimmsnarl ex (Unfair Stamp) list from Kaggle episode
  125180985; the same 60 cards as `grimmsnarl_ex`, added again to double how often it is drawn.
* `ogerpon_leaderboard` — the Teal Mask Ogerpon list bai fang and EkremKutukculer played (episodes
  125180090, 125181129); the same cards as `teal_mask_ogerpon`, likewise doubled.
* `lucario_gravity_mountain` — a Mega Lucario ex build with Gravity Mountain, Air Balloon and
  Maximum Belt (52/60 cards shared with `mega_lucario_ex`, which stays as it is).
* `dragapult_raihan` — Raihan Ramadistra's Dragapult ex / Munkidori (Crushing Hammer, Dunsparce, Risky Ruins)
  list from Kaggle episode 125184564; the agent's own deck (`decks/main_agent.csv`) from 2026-10-22 until
  2026-10-24, when it was replaced by a Dragapult ex / Dusknoir list (Crushing Hammer, Jamming Tower, Patrat).
* `marnie_grimmsnarl_ex` — the agent's previous deck (Marnie's Grimmsnarl ex), kept as an opponent.
* `dhelmise_banette` — the meta Dhelmise / Banette / Sinistcha list, from per-deck average card counts
  (2026-10-23) rounded to 60 (Night Stretcher and Explorer's Guidance rounded up to fill it; cards
  averaging under 0.5 copies left out). Telepathic Psychic Energy is "Telepath Psychic Energy" (POR 87)
  in this engine. `dhelmise` (4 Sinistcha, 4 Explorer's Guidance, Spiritomb) stays as it is.

Added 2026-10-24 from the 160 Kaggle episodes of submission `dragapult_final_216M`, for the
archetypes that were 19% of its games but had no list here. Each is the opponent's first action
(their deck) in the episodes named, unchanged:

* `festival_thwackey` — Grookey / Thwackey / Dipplin / Seaking with Festival Grounds; Liu Zeyu1212
  and four other players (episodes 125241769, 125243583, ...). `festival_lead` is a different build.
* `lopunny_froslass_dudunsparce` — Mega Lopunny ex / Mega Froslass ex / Dudunsparce with Battle Cage
  and Mist Energy; ShimonHori, YG JH, clora (125245549, 125248506, ...).
* `lopunny_kangaskhan_froslass` — Mega Lopunny ex / Mega Kangaskhan ex / Mega Froslass ex on Water;
  Tys TANA (125249038).
* `starmie_cinderace` — Mega Starmie ex / Cinderace with Crushing Hammer; yayaswya (125247982, 125285922).
* `froslass_starmie` — Mega Froslass ex / Mega Starmie ex with Surfing Beach; Mozune4002 (125259181, 125260043).
* `kangaskhan_slowking` — Mega Kangaskhan ex / Slowking / Latias ex; teppei (125248596).
* `teal_mask_ogerpon_mono` — four Teal Mask Ogerpon ex, 17 Grass Energy; vrmichalski (125251035).
* `crustle_cook` — Crustle with Cook, Waitress and Jumbo Ice Cream; Jules Evers (125248746). Crustle
  walls Dragapult ex, and the agent lost 20 of its 25 games against it by decking out.
* `mega_abomasnow` — Mega Abomasnow ex / Kyogre; Sliver-Ze (125285200).
* `ionos_bellibolt` — Iono's Bellibolt ex / Kilowattrel; tai (125257227).

## Weights

`weights.json` (`--opponent-weights`) draws decks by how often their archetype came up in those
391 games and how badly the agent did against it: share x 2 x (1 - win rate), clipped to
[0.4, 1.6], split between the pool decks of the archetype, with 1% for archetypes not met on
Kaggle. Mega Lucario ex (16% of games, 29% won), Ogerpon / Hydrapple (9%, 31%) and Crustle
(10%, 35%) get the most; Alakazam (19%, 69%) and Archaludon (5%, 81%) less than their share.
Decks missing from the file are drawn with weight 1.0.
