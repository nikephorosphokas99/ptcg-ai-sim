# Pokémon TCG — self-play RL agent

Reinforcement learning for the Pokémon Trading Card Game on the official engine
from the [Pokémon TCG AI Battle Challenge](https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground).
The agent is trained with MaskablePPO (Stable-Baselines3 + sb3-contrib) by
self-play against a pool of its own earlier snapshots and a random player,
by default in mirror matches with the agent's own deck.

    cd rl-train
    pip install -r requirements.txt
    python -m ptcg_rl.train --run runs/first --timesteps 50_000_000 \
        --opponent-decks decks/training_pool/*.csv --opponent-weights decks/training_pool/weights.json \
        --wandb-project ptcg-ai-sim          # optional; needs `wandb login` or WANDB_API_KEY
    tensorboard --logdir runs/first/tb
    python -m ptcg_rl.evaluate runs/first/final.zip --games 500
    python -m ptcg_rl.export runs/first/final.zip      # -> submissions/first_final.zip
    python -m ptcg_rl.benchmark submissions/first_final.zip   # vs the competition's rule-based agents

## Layout

| | |
| --- | --- |
| `rl-train/engine/` | the official competition package, unchanged: C++ source, prebuilt `libcg` binaries, card data |
| `rl-train/decks/main_agent.csv` | the deck the agent learns to pilot (60 card ids) |
| `rl-train/decks/opponents/` | optional opponent decks; without any, every game is a mirror match |
| `rl-train/ptcg_rl/engine.py` | ctypes bindings with one engine handle per game |
| `rl-train/ptcg_rl/encoding.py` | observation encoding, action mask, multi-select handling |
| `rl-train/ptcg_rl/env.py` | `Match`: one game, stepped for whichever seat is to move |
| `rl-train/ptcg_rl/worker.py` | game process hosting many matches; never imports torch |
| `rl-train/ptcg_rl/collector.py` | `SelfPlayCollector`: advances every game one decision per tick, opponent pool |
| `rl-train/ptcg_rl/storage.py` | GPU sample store and the rollout buffer that feeds `MaskablePPO.train()` from it |
| `rl-train/ptcg_rl/policy.py` | policies: `EntityPolicy` (v3, default), `PointerPolicy`, and the first-generation flat `PTCGExtractor` |
| `rl-train/ptcg_rl/cardtext.py` | card rules text as fixed features (TF-IDF + SVD, plus mechanic flags) |
| `rl-train/ptcg_rl/bots.py` | the rule-based bots as in-process training opponents |
| `rl-train/decks/training_pool/` | the opponent decks used in training, and `weights.json` for how often each is drawn |
| `rl-train/ptcg_rl/train.py` | MaskablePPO training loop, snapshot pool, checkpoints |
| `rl-train/ptcg_rl/evaluate.py` | win rate against the random player or a checkpoint |
| `rl-train/ptcg_rl/export.py` | builds and checks a submission zip from a trained model |
| `rl-train/ptcg_rl/benchmark.py` | plays a submission against `benchmarks/` (rule-based agents from Kaggle notebooks) |
| `rl-train/benchmarks/` | those opponents, one folder each; see `benchmarks/SOURCES.md` |
| `rl-train/submission/` | the submission's `main.py` and torch-only network, copied into each zip |

## Notes

**Where things run.** Games run on CPU in `--workers` processes (CPUs - 2 by
default), `--matches-per-worker` games each (64, so 1,152 games on 20 CPUs).
Workers write each game's pending observation into shared memory. The main
process runs the learner, every frozen opponent and the PPO update on the GPU,
batched across all games. Workers are plain `python -m ptcg_rl.worker`
subprocesses: `multiprocessing` children re-import the training script, and with
it torch, which costs ~530 MB per worker instead of ~55 MB.

**Why not an SB3 `VecEnv`.** A `VecEnv` steps only when every game is waiting on
the learner, so one game in the middle of a long opponent turn holds up the
batch: ~20 synchronous rounds per learner step, mostly idle. The collector
instead advances every game one decision per tick, whoever is to move, and keeps
each seat's trajectory until the game ends. SB3 still does the PPO update
(`MaskablePPO.train()`), fed by `storage.GPURolloutBuffer`.

**Policies.** `--arch entity` (v3, the default) reads the board as tokens: one
per Pokémon with its attached tools and energy, per hand card, per pile (both
discards, cards being viewed, own unseen cards, opponent cards seen, recent
opponent cards from the log), plus the stadium, the effect being resolved and the
global state, mixed by self-attention. Every option is wired to the encoded
tokens of the board slots it acts from and targets, carries attack tactics
(damage after Weakness and Resistance against the opponent's Active, knock-out,
Prize cards), and options attend to each other and to the board before being
scored. Cards are described by learned embeddings, printed numbers and their
rules text (`cardtext.py`). `--arch pointer` scores options from their own
content against a flattened board; `--arch flat` is the first generation.
Observation keys only ever get added, and each policy reads only the keys it was
built with, so all generations play each other in training (`--baseline`) and in
`evaluate.py`.

**Bots in training.** `--bot-frac` sends that share of games against the
rule-based agents in `benchmarks/`, each with its own deck, played inside the
game workers (`bots.py`). Their `cg` gets a stand-in `cg.sim` that reuses the
worker's engine: a second `GameInitialize()` in a process that runs battles
crashes the engine (`buffer full. capacity:7`). Each game loads a fresh copy of
the bot (they keep per-game state in module globals); a bot that errors or loops
in its own turn is replaced by a random legal move or has the game ended.
Benchmark numbers against bots used in training are optimistic.

**Opponents.** Opponent decks come from `--opponent-decks` (none means mirror matches),
drawn in proportion to `--opponent-weights` (`decks/training_pool/weights.json`: each archetype's
share of Kaggle games, weighted towards the matchups the last submission lost; see
`decks/training_pool/SOURCES.md`). Half the games (`--self-play-frac`) are against the current
learner, and both seats then produce training samples. The rest are against the
random player or a frozen snapshot: one is taken every `--snapshot-every` steps,
the newest `--max-pool` are kept on the GPU, and new games draw from
`--active-snapshots` of them at a time (each one in play costs a forward pass
per tick). `--baseline model.zip` keeps one fixed earlier model in the pool for
the whole run, logged as `win_rate/vs_baseline`.

**Rules.** Two kinds of decision are played by rule, not by the policy, for both
seats in training (`Match`, `rules=True`) and in the submission (`rules.json` in the
bundle). Damage counters placed "in any way you like" (Phantom Dive) go where they
take the most Prize cards (`encoding.CounterPlanner`): the engine asks for them one
counter at a time and the policy cannot see where the earlier ones went, so the 216M
policy put all six on one Pokémon in 93% of its Kaggle Phantom Dives; on those same
positions the planner takes 715 Prize cards to its 509. And a turn that offers an attack
without a drawback (`engine.safe_attack_ids`) cannot be ended without attacking.
`--no-rules` (train) and `--no-rules` (export) turn both off.

**Logging.** TensorBoard under `<run>/tb`; with `--wandb-project` the same values go
to Weights & Biases, and `--wandb-log-model` uploads each checkpoint and `final.zip`
as model artifacts. `vs_deck/<deck>` is the learner's win rate against each opponent
deck (games against itself and its snapshots).

**Two GPUs (Kaggle T4 x2).** `--device cuda:0 --collect-device cuda:1 --pipeline --amp`:
the PPO update runs in fp16 (`ppo.PTCGPPO`) on GPU 0 while GPU 1 plays the next
rollout's games with a copy of the weights, refreshed after each update, so an
iteration takes max(collect, train) rather than their sum. Splitting the update
itself across both GPUs (DDP) would only shorten the update half and leave
collection waiting. If fp16 keeps overflowing, the update falls back to fp32 on its own.

**Memory and disk.** Training samples are stored on the GPU as int16/float16,
~9 KB each (~11 KB for v3); `--store` (default 6 x `--rollout`) is ~3.4 GB of VRAM with the
defaults. On an 8 GB GPU, v3 needs `--store 196608 --batch-size 1024`: past the
VRAM, WSL spills into shared memory and an update slows ~6x. It has to cover the samples recorded while one game is in progress
plus one rollout; `rollout/dropped_samples` in TensorBoard counts any that are
overwritten too early. System RAM stays around 5-6 GB. A run writes
`latest.zip` at each snapshot, a checkpoint every `--save-every` steps and
`final.zip` (~50 MB each), plus TensorBoard logs.

**Throughput.** On a 20-CPU WSL machine with an RTX 3070: ~80-100 games/s
(~15k training samples/s), including the PPO update. The CPU-only ceiling,
with random play and no network at all, is ~450 games/s; what remains is
mostly Python overhead around GPU calls in the main process.

**Parallel games.** The engine keeps each battle in its own heap object, so many
games can run in one process. The competition's `cg/game.py` wrapper stores the
battle pointer on a class attribute, which limits it to one game per process;
`ptcg_rl/engine.py` keeps a pointer per game instead.

**Submissions.** `export.py` writes a zip with `main.py` at its root, the deck,
`policy.pt` (actor weights only), a torch-only copy of the network and the
training-time `encoding.py`, plus the official `cg/` bindings. It needs only
torch and numpy at run time. Before zipping it checks that the bundle's network
matches the SB3 policy on real positions, and plays complete games with the
bundle loaded the way the harness loads it (no `__file__`, no SB3 or
gymnasium). Any error inside `agent()` falls back to a random legal move rather
than forfeiting the match.

**Benchmark.** Self-play win rates only say how an agent does against itself.
`benchmark.py` runs a submission zip, exactly as submitted, against rule-based
agents other competitors published, each with its own deck, and reports win
rate per opponent. Each agent runs in its own process (`agent_host.py`) behind
the official `cg` bindings with `State.round`/`State.draw` declared: the engine
emits both, the shipped `api.py` does not, and agents written for the
competition's later `api.py` read `obs.current.round` (best-of-three rounds).

**Card ids** are the official engine's, as listed in
`rl-train/engine/EN_Card_Data_R2_full.csv`. `load_deck` rejects a deck containing
an id the engine does not implement.

**Not reproducible from a seed.** The engine seeds its RNG from
`std::random_device`; seeds fix only each match's own choices (seat, deck).

## Licensing

`rl-train/engine/` is the competition's engine package, provided for competition
use only under the licence in its `LICENSES/` directory. Pokémon, card names and
card text are trademarks and copyright of Pokémon/Nintendo/Creatures/GAME FREAK.
