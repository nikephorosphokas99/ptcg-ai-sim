# Benchmark opponents

Rule-based agents published as Kaggle notebooks in the PTCG AI Battle Challenge,
used here only to measure our agents (`python -m ptcg_rl.benchmark`). Each folder
holds the notebook's `main.py` unchanged and the deck it plays; `cg` is not
copied in: the harness puts the official bindings
(`rl-train/engine/sample_submission/cg`) on the agent's import path.

| Folder | Notebook | Deck |
| --- | --- | --- |
| `archaludon_starmie` | a-sample-archaludon-75-wr-vs-my-1300-starmie | the notebook's `deck.csv` |
| `alakazam_5th` | rule-based-not-psychic-alakazam-best-5th | the notebook's `deck.csv` |
| `crustle_bot` | beating-the-day-1-1-crustle-bot | the notebook's `deck.csv` |
| `venusaur_v1` | mega-venusaur-ex-deck-v1-rule-based-agent | the notebook's hard-coded list |
| `dragapult_sample` | a-sample-rule-based-agent-dragapult-ex-deck | not in the notebook (a Kaggle dataset): `visualizer/decks/dragapult.csv`, the official sample list its code is written for |
| `lucario_sample` | a-sample-rule-based-agent-mega-lucario-ex-deck | likewise: `visualizer/decks/mega_lucario_ex.csv` |
| `iono_sample` | a-sample-rule-based-agent-iono-s-deck | **missing** — add `deck.csv` to enable |
| `abomasnow_sample` | a-sample-rule-based-agent-mega-abomasnow-ex-deck | **missing** — add `deck.csv` to enable |

A folder without `deck.csv` is skipped.
