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
