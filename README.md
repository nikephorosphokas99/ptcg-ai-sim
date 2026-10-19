# Pokémon TCG — self-play RL agent

Reinforcement learning for the Pokémon Trading Card Game on the official engine
from the [Pokémon TCG AI Battle Challenge](https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground).
The agent is trained with MaskablePPO (Stable-Baselines3 + sb3-contrib) by
self-play against a pool of its own earlier snapshots and a random player.

    cd rl-train
    pip install -r requirements.txt
    python -m ptcg_rl.train --run runs/first --n-envs 16 --timesteps 20_000_000
    tensorboard --logdir runs/first/tb
    python -m ptcg_rl.evaluate runs/first/final.zip --games 50

## Layout

| | |
| --- | --- |
| `rl-train/engine/` | the official competition package, unchanged: C++ source, prebuilt `libcg` binaries, card data |
| `rl-train/decks/main_agent.csv` | the deck the agent learns to pilot (60 card ids) |
| `rl-train/decks/opponents/` | opponent decks drawn from each episode; empty files are skipped |
| `rl-train/ptcg_rl/engine.py` | ctypes bindings with one engine handle per game |
| `rl-train/ptcg_rl/env.py` | single-agent Gymnasium env; the opponent plays inside `step` |
| `rl-train/ptcg_rl/encoding.py` | observation encoding, action mask, multi-select handling |
| `rl-train/ptcg_rl/policy.py` | features extractor: card embeddings + static card features |
| `rl-train/ptcg_rl/train.py` | MaskablePPO over `SubprocVecEnv`, self-play snapshot pool |
| `rl-train/ptcg_rl/evaluate.py` | win rate per opponent deck |

## Notes

**Parallel games.** The engine keeps each battle in its own heap object, so many
games can run in one process. The competition's `cg/game.py` wrapper stores the
battle pointer on a class attribute, which limits it to one game per process;
`ptcg_rl/engine.py` keeps a pointer per game instead. Training runs one env per
worker process.

**Card ids** are the official engine's, as listed in
`rl-train/engine/EN_Card_Data_R2_full.csv`. `load_deck` rejects a deck containing
an id the engine does not implement.

**Not reproducible from a seed.** The engine seeds its RNG from
`std::random_device`, so `reset(seed=...)` fixes only the env's own choices
(seat, opponent deck, opponent policy).

## Licensing

`rl-train/engine/` is the competition's engine package, provided for competition
use only under the licence in its `LICENSES/` directory. Pokémon, card names and
card text are trademarks and copyright of Pokémon/Nintendo/Creatures/GAME FREAK.
