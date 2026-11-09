"""Self-play training: our own collector feeding MaskablePPO's update.

    cd rl-train
    python -m ptcg_rl.train --run runs/first --timesteps 50_000_000

Games run in ``--workers`` CPU processes, ``--matches-per-worker`` each, and
advance one decision per tick (``collector.py``). The learner and every
opponent run on ``--device`` (the GPU by default). Each update uses
``--rollout`` samples from finished games and SB3's ``MaskablePPO.train()``.

Opponents: ``--self-play-frac`` of games are played against the current learner
(both seats then train it); the rest against the random player or a frozen
snapshot, one taken every ``--snapshot-every`` steps and kept on the GPU.

Collection and the update run in turn by default. With ``--pipeline`` the update
runs in a separate process (``learner.py``) on ``--device`` while this process
keeps collecting on ``--collect-device``: with two GPUs (Kaggle T4 x2: ``--device
cuda:0 --collect-device cuda:1``) both work at once and an iteration takes
max(collect, train) instead of their sum. (Two threads in one process do not do
this: both halves are Python-bound and take turns on the GIL.) The rollout being
collected uses weights one update older than the learner's; PPO's ratio uses the
stored log-probs, so that is corrected for like any off-policy sample. ``--amp``
runs the update in fp16 (``ppo.PTCGPPO``), for the T4's tensor cores.

On disk a run keeps ``latest.zip`` (rewritten at each snapshot, for
``--resume``), a checkpoint every ``--save-every`` steps, ``final.zip`` and
TensorBoard logs. With ``--wandb-project`` every logged value also goes to
Weights & Biases (``WANDB_API_KEY`` or ``wandb login`` for the credentials), and
with ``--wandb-log-model`` the checkpoints are uploaded as model artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict, deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.utils import LinearSchedule, get_device
from stable_baselines3.common.vec_env import DummyVecEnv

from .collector import BASELINE, FIXED, RANDOM, SELF, SelfPlayCollector, frozen_copy
from .distill import DistillBuffer, distill_dir, publish, start_searchers, start_server
from .encoding import N_ACTIONS, observation_space
from .plots import eval_chart, wilson
from .engine import AGENT_DECK, card_table, default_opponent_decks, load_deck
from .env import Match
from .learner import RemoteLearner
from .policy import EntityPolicy, PointerPolicy, PTCGExtractor
from .ppo import PTCGPPO
from .storage import GPURolloutBuffer, Rollout


ITERATION_CAP = 300.0  # seconds; an update taking longer than this was a pause (see main)


class SpacesOnly(gym.Env):
    """Carries the spaces MaskablePPO is built from; the collector does the playing."""

    def __init__(self, space=None):
        self.observation_space = space or observation_space(len(card_table()) - 1)
        self.action_space = gym.spaces.Discrete(N_ACTIONS)

    def reset(self, *, seed=None, options=None):
        return {k: np.zeros(s.shape, s.dtype) for k, s in self.observation_space.spaces.items()}, {}

    def step(self, action):
        raise RuntimeError("SpacesOnly is not playable; use SelfPlayCollector")

    def action_masks(self) -> np.ndarray:
        return np.ones(N_ACTIONS, bool)


def spaces_env(space=None) -> DummyVecEnv:
    """A stand-in env; ``space`` defaults to the current encoding's full space."""
    return DummyVecEnv([lambda: SpacesOnly(space)])


def load_model(path: Path, device, **kwargs) -> PTCGPPO:
    """Load a model of any generation: each keeps the observation keys it was built with."""
    model = PTCGPPO.load(path, device=device,
                             custom_objects={"rollout_buffer_class": GPURolloutBuffer}, **kwargs)
    model.set_env(spaces_env(model.observation_space))
    return model


def upgrade_encoding(model: PTCGPPO) -> list[str]:
    """Give an entity model the observation keys and features the encoding has gained since it was saved.

    The policy is rebuilt for the full space and keeps every weight it had; the new
    inputs' modules start at zero (``EntityExtractor``), and a feature appended to an
    existing key (e.g. the Stadium's owner in ``global``) gets zero columns in the layer
    reading it, so it plays exactly as before until training finds a use for them.
    Adam's moments carry over by parameter name. Returns the keys added or widened.
    """
    space = observation_space(len(card_table()) - 1)
    old_spaces = model.observation_space.spaces
    added = [k for k in space.spaces if k not in old_spaces]
    widened = [k for k in space.spaces if k in old_spaces and space[k].shape[-1] > old_spaces[k].shape[-1]]
    if not added and not widened:
        return []
    old = model.policy
    new = old.__class__(space, model.action_space, model.lr_schedule, **model.policy_kwargs).to(model.device)
    target = new.state_dict()

    def fit(name: str, t: torch.Tensor) -> torch.Tensor:
        """An input layer that reads a widened feature: the old weights, zero for the new columns."""
        want = target[name].shape
        if t.shape == want:
            return t
        if t.dim() != 2 or t.shape[0] != want[0] or t.shape[1] > want[1]:
            raise SystemExit(f"cannot upgrade this model: {name} is {tuple(t.shape)}, now {tuple(want)}")
        return torch.cat([t, t.new_zeros(want[0], want[1] - t.shape[1])], dim=1)

    state = {k: fit(k, v) if k in target else v for k, v in old.state_dict().items()}
    missing, unexpected = new.load_state_dict(state, strict=False)
    if unexpected or not all(k.split(".", 1)[1].startswith(("ledger", "opp_hand")) for k in missing):
        raise SystemExit(f"cannot upgrade this model: missing {missing}, unexpected {unexpected}")
    names = {id(t): n for n, t in old.named_parameters()}
    moments = {names[id(t)]: old.optimizer.state[t] for t in old.parameters() if t in old.optimizer.state}
    for n, t in new.named_parameters():
        if n in moments:
            new.optimizer.state[t] = {k: fit(n, v) if v.dim() == 2 else v.clone() for k, v in moments[n].items()}
    model.policy = new
    model.observation_space = model.rollout_buffer.observation_space = space
    model.set_env(spaces_env(space))
    return added + widened


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, default=Path("runs/ppo"))
    p.add_argument("--agent-deck", default=str(AGENT_DECK))
    p.add_argument("--opponent-decks", nargs="*", default=default_opponent_decks(),
                   help="opponent deck CSVs; none means a mirror match")
    p.add_argument("--opponent-weights", type=Path,
                   help="JSON {deck file stem: relative weight} for drawing opponent decks "
                        "(e.g. decks/training_pool/weights.json); default: every deck equally")
    p.add_argument("--adaptive-opponents", action="store_true",
                   help="after each update, scale every deck's --opponent-weights by max(1 - win rate, "
                        "--adaptive-floor) over its last --adaptive-window games, so lost matchups are "
                        "played more (logged as opp_weight/<deck>)")
    p.add_argument("--adaptive-window", type=int, default=300, help="recent games per deck for its win rate")
    p.add_argument("--adaptive-floor", type=float, default=0.2,
                   help="smallest factor a deck's weight is scaled by, however often it is won")
    p.add_argument("--opp-min-share", type=float, default=0.0,
                   help="with --adaptive-opponents, every deck is drawn for at least this percent of games, "
                        "so a matchup rare in --opponent-weights still gets practice when it is lost")
    p.add_argument("--sacrifice-refund", action="store_true",
                   help="don't count the Prize card given up for the learner's own self-Knock-Out Abilities "
                        "(Cursed Blast) against it in --prize-reward (see env.Match)")
    p.add_argument("--no-rules", action="store_true",
                   help="let the policy place 'any way you like' damage counters and end turns with an "
                        "attack available, instead of the rules env.Match applies by default")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                   help="game processes (default: CPUs - 2)")
    p.add_argument("--matches-per-worker", type=int, default=64,
                   help="games per process; more games make bigger GPU batches")
    p.add_argument("--timesteps", type=int, default=50_000_000,
                   help="steps to train (added to the model's count when resuming)")
    p.add_argument("--max-hours", type=float,
                   help="stop after this much training time (pauses such as sleep do not count); "
                        "schedules then follow time, not steps")
    p.add_argument("--rollout", type=int, default=65_536, help="samples per PPO update")
    p.add_argument("--batch-size", type=int, default=0, help="default: 2048 for entity, 4096 otherwise")
    p.add_argument("--epochs", type=int, default=0, help="default: 3 for entity, 4 otherwise")
    p.add_argument("--store", type=int, default=0,
                   help="samples kept on the GPU, ~9 KB each (default: 6 x rollout); must cover "
                        "the samples recorded while one game is played, plus one rollout")
    p.add_argument("--self-play-frac", type=float, default=0.5,
                   help="share of games against the current learner")
    p.add_argument("--no-random-opponent", action="store_true",
                   help="leave the random player out of the opponent pool: with no --baseline, every game is "
                        "self-play, against the current learner or one of its snapshots")
    p.add_argument("--snapshot-every", type=int, default=1_000_000)
    p.add_argument("--save-every", type=int, default=5_000_000)
    p.add_argument("--max-pool", type=int, default=10, help="snapshots kept as opponents")
    p.add_argument("--active-snapshots", type=int, default=2,
                   help="snapshots new games draw from at once, re-drawn every update")
    p.add_argument("--prize-reward", type=float, default=0.05)
    p.add_argument("--energy-penalty", type=float, default=0.0,
                   help="reward subtracted for a basic Energy attached where it helps no attack (see Match)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-end", type=float, help="decay the learning rate linearly to this by the end")
    p.add_argument("--target-kl", type=float,
                   help="stop an update's epochs early once its KL passes 1.5x this")
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.995,
                   help="discount per decision (every pick is a step) for a new model; a resumed one keeps its own")
    p.add_argument("--device", default="auto", help="where the PPO update runs")
    p.add_argument("--collect-device",
                   help="where game inference, the sample store and snapshots live (default: --device); "
                        "with two GPUs, e.g. --device cuda:0 --collect-device cuda:1 --pipeline")
    p.add_argument("--pipeline", action="store_true",
                   help="run the update in its own process on --device while the next rollout is collected "
                        "on --collect-device (both GPUs busy at once)")
    p.add_argument("--amp", action="store_true", help="run the PPO update in fp16 autocast (CUDA only)")
    p.add_argument("--resume", type=Path, help="a saved model .zip to continue from")
    p.add_argument("--upgrade-encoding", action="store_true",
                   help="with --resume: add the observation keys the encoding gained since the model was saved "
                        "(e.g. the v4 deck ledger), starting their weights at zero (see upgrade_encoding)")
    p.add_argument("--arch", choices=("entity", "pointer", "flat"), default="entity",
                   help="policy for a new model (ignored with --resume): entity = v3 attention over "
                        "board/hand/option tokens; pointer = per-option head with deck memory; flat = first generation")
    # Entity network size (new models only; a resumed model keeps the size it was saved with).
    p.add_argument("--dim", type=int, default=128, help="entity: token width")
    p.add_argument("--heads", type=int, default=4, help="entity: attention heads (must divide --dim)")
    p.add_argument("--board-layers", type=int, default=2, help="entity: self-attention layers over the board")
    p.add_argument("--option-layers", type=int, default=2, help="entity: decoder layers over the options")
    p.add_argument("--ff", type=int, default=256, help="entity: feed-forward width inside each layer")
    p.add_argument("--bots", nargs="*", default=None,
                   help="rule-based agent bundles to train against (default: every benchmarks/* with a deck)")
    p.add_argument("--bot-frac", type=float, default=0.0,
                   help="share of games against a rule-based bot, playing its own deck")
    p.add_argument("--baseline", type=Path,
                   help="a fixed model kept in the opponent pool for the whole run (win_rate/vs_baseline); "
                        "evaluation only with --eval-frac")
    p.add_argument("--fixed-opponents", nargs="*", type=Path, default=[],
                   help="more fixed models kept in the opponent pool for the whole run, e.g. earlier agents "
                        "of other archetypes (win_rate/vs_fixed, vs_fixed_by_name/<file stem>); evaluation "
                        "only with --eval-frac")
    p.add_argument("--distill-workers", type=int, default=0,
                   help="search distillation (ptcg_rl.distill): this many CPU processes play self-play games, "
                        "run ISMCTS at some of our choices, and the update learns the search's visit counts")
    p.add_argument("--distill-prob", type=float, default=0.25, help="share of our real choices a searcher searches")
    p.add_argument("--distill-iterations", type=int, default=48, help="ISMCTS iterations per searched choice")
    p.add_argument("--distill-skip-confident", type=float, default=0.97,
                   help="choices whose top move the policy gives at least this probability are not searched")
    p.add_argument("--distill-uncertain", type=float, default=0.0,
                   help="> 0: search a choice with probability --distill-prob x min(1, (1 - top move's probability) "
                        "/ this), so the search goes to the choices the policy is least sure of")
    p.add_argument("--distill-horizon", default="turn_end", choices=["turn_end", "next_turn", "decision"],
                   help="where a searcher's simulation stops for the value head: the end of the current turn, or "
                        "next_turn = after the opponent's reply, at our next turn (sees their attack)")
    p.add_argument("--distill-coef", type=float, default=1.0, help="weight of the distillation cross-entropy")
    p.add_argument("--distill-batch", type=int, default=256, help="searched positions per distillation minibatch")
    p.add_argument("--distill-reuse", type=float, default=8.0,
                   help="times each searched position is trained on, on average (sets minibatches per update)")
    p.add_argument("--distill-buffer", type=int, default=20_000, help="latest searched positions kept")
    p.add_argument("--distill-gpu", action="store_true",
                   help="searchers send their network evaluations to one GPU inference server (on --device), "
                        "--distill-threads games per searcher at once sharing batches, instead of numpy on the CPU")
    p.add_argument("--distill-threads", type=int, default=32, help="with --distill-gpu: games per searcher at once")
    p.add_argument("--eval-frac", type=float, default=0.0,
                   help="share of games played for evaluation only against the random player, --baseline and "
                        "--fixed-opponents: never trained on, and new training games then face only the "
                        "learner and its snapshots (eval/win_rate_*, eval/reward_* over --eval-window games)")
    p.add_argument("--eval-window", type=int, default=500, help="recent evaluation games per opponent in eval/*")
    p.add_argument("--chart-every-min", type=float, default=15,
                   help="with --eval-frac and W&B: log the evaluation chart (win rates with a 95%% band, reward "
                        "per training game; ptcg_rl.plots) as charts/eval this often; 0 for never")
    p.add_argument("--wandb-project", help="log to this Weights & Biases project (off when not given)")
    p.add_argument("--wandb-entity", help="W&B team or user (default: the API key's)")
    p.add_argument("--wandb-name", help="W&B run name (default: the --run folder name)")
    p.add_argument("--wandb-id", help="W&B run id to continue, e.g. when resuming a run")
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    p.add_argument("--wandb-log-model", action="store_true",
                   help="upload checkpoints to W&B as versions of the model artifact --wandb-artifact: "
                        "latest.zip every --backup-every-min minutes, each --save-every checkpoint and final.zip")
    p.add_argument("--wandb-artifact", default="ptcg-checkpoint",
                   help="W&B artifact the checkpoints are uploaded to; the same name across runs, so "
                        "<artifact>:latest is always the newest model to resume from")
    p.add_argument("--backup-every-min", type=float, default=30.0,
                   help="with --wandb-log-model, also save and upload latest.zip this often (0: off), "
                        "so a crashed or killed session loses at most this much training")
    args = p.parse_args()

    (args.run / "checkpoints").mkdir(parents=True, exist_ok=True)
    args.batch_size = args.batch_size or (2048 if args.arch == "entity" else 4096)
    args.epochs = args.epochs or (3 if args.arch == "entity" else 4)
    opponent_weights = json.loads(args.opponent_weights.read_text()) if args.opponent_weights else None
    if opponent_weights:
        opponent_weights = {k: float(v) for k, v in opponent_weights.items() if not k.startswith("_")}
        decks = {Path(d).stem for d in args.opponent_decks}
        unknown = sorted(set(opponent_weights) - decks)
        if unknown:
            print(f"--opponent-weights names decks not in --opponent-decks (ignored): {unknown}")
    match_kwargs = dict(
        agent_deck=args.agent_deck,
        opponent_decks=args.opponent_decks,
        opponent_weights=opponent_weights,
        prize_reward=args.prize_reward,
        energy_penalty=args.energy_penalty,
        rules=not args.no_rules,
        sacrifice_refund=args.sacrifice_refund,
    )
    if args.bot_frac > 0:
        from .benchmark import BENCHMARKS, compat_cg

        bots = args.bots if args.bots is not None else [
            str(d) for d in sorted(BENCHMARKS.iterdir()) if (d / "main.py").is_file() and (d / "deck.csv").is_file()]
        match_kwargs.update(bots=[str(Path(b).resolve()) for b in bots], bot_frac=args.bot_frac,
                            bot_cg=str(compat_cg()))
        print(f"{args.bot_frac:.0%} of games against bots: {[Path(b).name for b in bots]}")
    Match(**match_kwargs).close()  # fail on a bad deck here, not in N workers

    try:
        import tensorboard  # noqa: F401
        tb_log = str(args.run / "tb")
    except ImportError:
        tb_log = None

    wandb_run = None
    if args.wandb_project and args.wandb_mode != "disabled":
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name or args.run.name,
            id=args.wandb_id, resume="allow" if args.wandb_id else None, mode=args.wandb_mode,
            config={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            dir=str(args.run),
        )
        wandb_run.define_metric("time/total_timesteps")
        wandb_run.define_metric("*", step_metric="time/total_timesteps")

    learner_device = get_device(args.device)
    collect_device = get_device(args.collect_device) if args.collect_device else learner_device
    # With --pipeline this process only collects: its model is the actor, on the collecting
    # GPU, and the learner process trains its own copy on --device.
    device = collect_device if args.pipeline else learner_device
    if args.resume:
        model = load_model(args.resume, device, tensorboard_log=tb_log)
        if args.upgrade_encoding:
            print(f"encoding upgraded, new inputs: {upgrade_encoding(model) or 'none'}", flush=True)
    else:
        if args.arch == "entity":
            if args.dim % args.heads:
                raise SystemExit(f"--heads {args.heads} must divide --dim {args.dim}")
            policy, policy_kwargs = EntityPolicy, dict(features_extractor_kwargs=dict(
                dim=args.dim, heads=args.heads, board_layers=args.board_layers,
                option_layers=args.option_layers, ff=args.ff, stop_token=True))
        elif args.arch == "pointer":
            policy, policy_kwargs = PointerPolicy, {}
        else:
            policy = "MultiInputPolicy"
            policy_kwargs = dict(features_extractor_class=PTCGExtractor, net_arch=dict(pi=[512, 256], vf=[512, 256]))
        model = PTCGPPO(
            policy,
            spaces_env(),
            learning_rate=args.lr,
            n_steps=args.rollout,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            gamma=args.gamma,
            gae_lambda=0.95,
            ent_coef=args.ent_coef,
            rollout_buffer_class=GPURolloutBuffer,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tb_log,
            device=device,
            verbose=0,
        )
    # Applied on resume too, so a continued run can change its schedule.
    model.learning_rate = LinearSchedule(args.lr, args.lr_end, 1.0) if args.lr_end is not None else args.lr
    model._setup_lr_schedule()
    model.target_kl = args.target_kl
    model.amp = args.amp
    total, _ = model._setup_learn(args.timesteps, reset_num_timesteps=not args.resume, tb_log_name="PPO")
    # Training time, not wall time: when the machine sleeps (Windows suspends WSL)
    # the clock keeps running but nothing trains. One update takes seconds, so an
    # update that appears to take longer than ITERATION_CAP was a pause and only
    # counts up to the cap. --max-hours and the learning-rate decay follow this.
    trained = 0.0
    budget = args.max_hours * 3600 if args.max_hours else None

    # Without --pipeline but with the update on another GPU, the collector plays with its
    # own copy of the weights, refreshed after every update.
    separate = collect_device != device
    actor = frozen_copy(model.policy, collect_device) if separate else model.policy
    collector = SelfPlayCollector(
        match_kwargs, actor, args.workers, args.matches_per_worker, collect_device,
        gamma=model.gamma, gae_lambda=model.gae_lambda,
        self_play_frac=args.self_play_frac, max_pool=args.max_pool,
        active_snapshots=args.active_snapshots,
        store_capacity=args.store or 6 * model.n_steps,
        pool={RANDOM: None} if args.eval_frac > 0 else {} if args.no_random_opponent else None,
        eval_frac=args.eval_frac,
    )
    if args.baseline:
        collector.add_opponent(BASELINE, load_model(args.baseline, collect_device).policy)
    for path in args.fixed_opponents:  # a run's final.zip is named after its folder
        name = path.parent.name if path.stem in ("final", "latest", "model") else path.stem
        collector.add_opponent(FIXED + name, load_model(path, collect_device).policy)
    # --adaptive-opponents: the latest outcomes per deck, which scale its --opponent-weights.
    prior_weights = {Path(d).stem: (opponent_weights or {}).get(Path(d).stem, 1.0) for d in args.opponent_decks}
    recent_by_deck = {deck: deque(maxlen=args.adaptive_window) for deck in prior_weights}
    # --eval-frac: the latest evaluation outcomes per opponent.
    recent_eval: dict[str, deque] = defaultdict(lambda: deque(maxlen=args.eval_window))
    next_chart = time.monotonic() + 60 * args.chart_every_min

    def kind_of(name: str) -> str:
        """The win_rate/<kind> an opponent counts towards."""
        if name.startswith("bot:"):
            return "vs_bot"
        if name.startswith(FIXED):
            return "vs_fixed"
        return {RANDOM: "vs_random", SELF: "vs_self", BASELINE: "vs_baseline"}.get(name, "vs_snapshot")

    def label_of(name: str) -> str:
        """An evaluation opponent's name in eval/*."""
        return f"vs_{name[len(FIXED):]}" if name.startswith(FIXED) else kind_of(name)
    print(f"policy {type(model.policy).__name__} ({sum(p.numel() for p in model.policy.parameters()):,} parameters)")
    print(f"{collector.n} games across {args.workers} workers; games on {collect_device}, update on {learner_device}"
          f"{' (fp16)' if args.amp else ''}{' in its own process, in parallel' if args.pipeline else ''}; "
          f"{model.n_steps} samples per update; sample store {collector.store.nbytes / 2**30:.1f} GB")

    learner = None
    if args.pipeline:
        init = args.run / "learner_init.zip"
        model.save(init)
        learner = RemoteLearner(init, learner_device, dict(
            learning_rate=model.learning_rate, target_kl=args.target_kl, amp=args.amp))
        init.unlink()

    def save(path: Path) -> None:
        # The learner's model is the trained one, optimizer state included.
        if learner is not None:
            learner.save(path, model.num_timesteps)
        else:
            model.save(path)

    def timed_train() -> float:
        start = time.time()
        # Collection and the update share the GPU here: hand the cached blocks of one back
        # before the other, so an 8 GB card does not spill into (WSL's much slower) shared memory.
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        if device.type == "cuda":
            model.logger.record("time/gpu_peak_gb", torch.cuda.max_memory_allocated(device) / 2**30)
            torch.cuda.empty_cache()
        return time.time() - start

    # --distill-workers: searcher processes and the buffer of their searched positions.
    searchers, server, distill, distill_credit, agree_seen = [], None, None, 0.0, (0, 0)
    search_config = {"horizon": args.distill_horizon}
    if args.distill_workers:
        if args.pipeline:
            raise SystemExit("--distill-workers does not support --pipeline yet")
        samples_dir = distill_dir(args.run) / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        for stale in samples_dir.glob("*.npz"):  # from an earlier segment, maybe another encoding
            stale.unlink()
        publish(model, load_deck(args.agent_deck), args.run, search_config, first=True, gpu=args.distill_gpu)
        if args.distill_gpu:
            server = start_server(args.run, device)
        distill = DistillBuffer(samples_dir, model.observation_space, model.action_space.n, args.distill_buffer,
                                device)
        model.distill, model.distill_coef, model.distill_batch = distill, args.distill_coef, args.distill_batch
        searchers = start_searchers(args.run, args.distill_workers, args)
        print(f"search distillation: {args.distill_workers} searchers, {args.distill_iterations} ISMCTS iterations "
              f"at {args.distill_prob:.0%} of our choices"
              f"{f' (x uncertainty / {args.distill_uncertain})' if args.distill_uncertain else ''}, horizon {args.distill_horizon}"
              f", coef {args.distill_coef}, reuse {args.distill_reuse}"
              f"{f', network on {device} ({args.distill_threads} games per searcher)' if args.distill_gpu else ''}",
              flush=True)

    samples = None  # with --pipeline, the rollout collected during the previous update

    backup_every = args.backup_every_min * 60 if wandb_run is not None and args.wandb_log_model else 0
    last_backup = time.time()
    next_snapshot = model.num_timesteps + args.snapshot_every
    next_save = model.num_timesteps + args.save_every
    try:
        while model.num_timesteps < total and (budget is None or trained < budget):
            t0 = time.time()
            decisions0, games0 = collector.stats.decisions, collector.stats.games
            collect_s = 0.0
            if samples is None:
                samples = collector.collect(model.n_steps)
                collect_s = time.time() - t0
            keys = list(model.observation_space.spaces)
            model.num_timesteps += len(samples)
            if budget is None:
                model._update_current_progress_remaining(model.num_timesteps, total)
            else:
                model._current_progress_remaining = max(0.0, 1.0 - trained / budget)
            arrays = dict(actions=samples.actions, values=samples.values, log_probs=samples.log_probs,
                          advantages=samples.advantages, returns=samples.returns)

            train_logs = {}
            if learner is not None:
                # The learner copies this rollout to its GPU and trains while we play the next one.
                learner.submit(Rollout(collector.store, samples.ids, collect_device, keys), arrays,
                               model._current_progress_remaining, model.num_timesteps)
                c0 = time.time()
                samples = collector.collect(model.n_steps)
                collect_s = time.time() - c0
                weights, train_logs, train_s = learner.result()
                model.policy.load_state_dict(weights)  # the collector's actor
            else:
                if separate:
                    source, ids = Rollout(collector.store, samples.ids, device, keys), np.arange(len(samples))
                else:
                    source, ids = collector.store, samples.ids
                model.rollout_buffer.load(source, ids, **arrays)
                if distill is not None:
                    # Each new searched position is trained on --distill-reuse times, on average.
                    distill_credit += args.distill_reuse * distill.poll() / args.distill_batch
                    model.distill_steps = min(int(distill_credit), 4 * model.n_steps // model.batch_size)
                    distill_credit -= model.distill_steps
                train_s = timed_train()
                if distill is not None:
                    publish(model, load_deck(args.agent_deck), args.run, search_config, gpu=args.distill_gpu)
                samples = None
                if actor is not model.policy:
                    actor.load_state_dict(model.policy.state_dict())
            t2 = time.time()
            trained += min(t2 - t0, ITERATION_CAP)

            stats = collector.stats
            games = stats.games - games0
            log = model.logger
            for key, value in train_logs.items():
                log.record(key, value)
            log.record("time/collect_s", collect_s)
            log.record("time/train_s", train_s)
            log.record("time/iteration_s", t2 - t0)
            log.record("time/games_per_s", games / (t2 - t0))
            log.record("time/samples_per_s", model.n_steps / (t2 - t0))
            log.record("time/decisions_per_s_collecting", (stats.decisions - decisions0) / max(collect_s, 1e-6))
            log.record("rollout/games", games)
            log.record("rollout/decisions_per_game", np.mean(list(stats.episode_lengths)[-max(games, 1):]))
            log.record("rollout/dropped_samples", stats.dropped)
            log.record("rollout/bot_failures", stats.bot_failures)
            # Evaluation opponents are charted only as eval/* over --eval-window games: a few games
            # per update made win_rate/<kind> jump between 50% and 100% on one loss.
            evaluators = set(collector.evaluators())
            kinds: dict[str, list[float]] = {}
            for name, outcomes in stats.outcomes.items():
                if name in evaluators:
                    continue
                kind = kind_of(name)
                if name.startswith("bot:"):
                    log.record(f"vs_bot_by_name/{name[4:]}", float(np.mean(outcomes)))
                if name.startswith(FIXED):
                    log.record(f"vs_fixed_by_name/{name[len(FIXED):]}", float(np.mean(outcomes)))
                kinds.setdefault(kind, []).extend(outcomes)
            for kind, outcomes in kinds.items():
                log.record(f"win_rate/{kind}", float(np.mean(outcomes)))
            # The learner seat's reward per finished game it trains on (self-play and snapshots);
            # evaluation games are scored by win rate alone.
            trained_returns = [r for name, returns in stats.returns.items() if name not in evaluators for r in returns]
            if trained_returns:
                log.record("reward/training_games", float(np.mean(trained_returns)))
            for name in evaluators:
                recent_eval[name].extend(stats.outcomes.get(name, []))
            for name, recent in recent_eval.items():
                if not recent:
                    continue
                rate = float(np.mean(recent))
                lo, hi = wilson(rate, len(recent))
                log.record(f"eval/win_rate_{label_of(name)}", rate)
                log.record(f"eval/win_rate_low_{label_of(name)}", float(lo))  # Wilson 95% interval
                log.record(f"eval/win_rate_high_{label_of(name)}", float(hi))
                log.record(f"eval/games_{label_of(name)}", len(recent))
            for deck, outcomes in stats.baseline_by_deck.items():
                log.record(f"vs_baseline_by_deck/{deck}", float(np.mean(outcomes)))
            for deck, outcomes in stats.by_deck.items():
                log.record(f"vs_deck/{deck}", float(np.mean(outcomes)))
                if deck in recent_by_deck:
                    recent_by_deck[deck].extend(outcomes)
            if args.adaptive_opponents and prior_weights:
                # Win rate with one win and one loss added, so a deck not met yet counts as 50%.
                weights = {deck: w * max(1.0 - (sum(recent_by_deck[deck]) + 1) / (len(recent_by_deck[deck]) + 2),
                                         args.adaptive_floor)
                           for deck, w in prior_weights.items()}
                if args.opp_min_share > 0:
                    floor = args.opp_min_share / 100 * sum(weights.values())
                    weights = {deck: max(w, floor) for deck, w in weights.items()}
                collector.set_opponent_weights(weights)
                for deck, w in weights.items():
                    log.record(f"opp_weight/{deck}", 100 * w / sum(weights.values()))
            # Per finished game, what the learner seat did: usage/play Dusknoir, usage/ability Dusknoir, ...
            for key, count in stats.usage.items():
                log.record(f"usage/{key}", count / max(stats.usage_games, 1))
            # The plays the v5 features are about, as rates (env.Match._note_tactics).
            u = stats.usage
            ratios = {
                "bench_guard_when_threatened": ("turns Bench threatened, guard in play", "turns Bench threatened"),
                "attacks_blocked_to_0": ("attacks blocked to 0", "attacks"),
                "attacks_wasted": ("attacks blocked to 0, no Bench damage", "attacks"),
                "wall_attacks_that_hit": ("attacks vs a wall that did damage", "attacks vs a wall"),
                "wall_turns_with_answer": ("turns vs a wall, answer in play", "turns vs a wall"),
            }
            for key in [k for k in u if k.startswith("tactics: turns Marnie's ") and k.endswith(" in play")]:
                name = key[len("tactics: turns "):-len(" in play")]
                ratios[f"over_3_energy/{name}"] = (f"turns {name} over 3 Energy", f"turns {name} in play")
            for key, (part, whole) in ratios.items():
                if u.get(f"tactics: {whole}"):
                    log.record(f"tactics/{key}", u.get(f"tactics: {part}", 0) / u[f"tactics: {whole}"])
            if u.get("tactics: attacks vs a wall"):
                for k in [k for k in u if k.startswith("tactics: attacks vs a wall by ")]:
                    log.record(f"tactics/wall_attacker/{k[len('tactics: attacks vs a wall by '):]}",
                               u[k] / u["tactics: attacks vs a wall"])
            stats.outcomes.clear()
            stats.returns.clear()
            stats.baseline_by_deck.clear()
            stats.by_deck.clear()
            stats.usage.clear()
            stats.usage_games = 0
            if distill is not None:
                total_new, agree_new = distill.total - agree_seen[0], distill.agree - agree_seen[1]
                agree_seen = (distill.total, distill.agree)
                log.record("distill/buffer", len(distill))
                log.record("distill/positions_total", distill.total)
                log.record("distill/positions_per_s", distill.rate())
                if total_new:  # how often the policy's top move was the search's choice, before learning it
                    log.record("distill/policy_agrees", agree_new / total_new)
                if server is not None and server.poll() is not None:  # its searchers exit and are restarted
                    print(f"inference server exited (code {server.returncode}); restarting it", flush=True)
                    server = start_server(args.run, device)
                for i, proc in enumerate(searchers):
                    if proc.poll() is not None:  # a crashed searcher: start a new one
                        print(f"searcher {i} exited (code {proc.returncode}); restarting it", flush=True)
                        searchers[i] = start_searchers(args.run, 1, args, first_worker=i)[0]
                log.record("distill/searchers", sum(proc.poll() is None for proc in searchers))
            log.record("time/total_timesteps", model.num_timesteps)
            log.record("time/trained_hours", trained / 3600)
            if wandb_run is not None:
                wandb_run.log({k: float(v) for k, v in log.name_to_value.items()
                               if isinstance(v, (int, float, np.number))})
            log.dump(model.num_timesteps)
            if (wandb_run is not None and tb_log and any(recent_eval.values()) and args.chart_every_min > 0
                    and time.monotonic() >= next_chart):
                next_chart = time.monotonic() + 60 * args.chart_every_min
                try:  # a chart is never worth a crashed run
                    chart = eval_chart(args.run, args.run / "eval.png")
                    if chart is not None:
                        wandb_run.log({"charts/eval": wandb.Image(str(chart)),
                                       "time/total_timesteps": model.num_timesteps})
                except Exception as e:  # noqa: BLE001
                    print(f"evaluation chart failed: {e!r}", flush=True)
            vs_base = f" | vs baseline {np.mean(kinds['vs_baseline']):.2f}" if "vs_baseline" in kinds and not evaluators else ""
            vs_base += f" | vs bots {np.mean(kinds['vs_bot']):.2f}" if "vs_bot" in kinds else ""
            for name, recent in recent_eval.items():  # rolling, over the last --eval-window games
                if recent:
                    vs_base += f" | {label_of(name).replace('_', ' ')} {np.mean(recent):.2f} ({len(recent)})"
            print(f"{model.num_timesteps:>13,} steps | {games / (t2 - t0):6.1f} games/s | "
                  f"collect {collect_s:4.1f}s train {train_s:4.1f}s"
                  f"{f' in parallel: {t2 - t0:4.1f}s' if learner else ''}{vs_base}", flush=True)

            collector.rotate_opponents()
            if model.num_timesteps >= next_snapshot:
                next_snapshot += args.snapshot_every
                collector.add_opponent(f"snap_{model.num_timesteps}", model.policy)
                save(args.run / "latest.zip")
            if model.num_timesteps >= next_save:
                next_save += args.save_every
                checkpoint = args.run / "checkpoints" / f"step_{model.num_timesteps}.zip"
                save(checkpoint)
                _log_model(wandb_run, args, checkpoint, model.num_timesteps)
                last_backup = time.time()
            if backup_every and time.time() - last_backup >= backup_every:
                # Off-Kaggle backup: a killed session keeps everything up to here in W&B.
                save(args.run / "latest.zip")
                _log_model(wandb_run, args, args.run / "latest.zip", model.num_timesteps)
                last_backup = time.time()
    finally:
        try:
            save(args.run / "final.zip")
        except Exception as e:  # the learner died: keep its last weights, without optimizer state
            print(f"learner could not save ({e}); saving the collector's copy of the weights")
            model.save(args.run / "final.zip")
        if learner is not None:
            learner.close()
        for proc in searchers + ([server] if server is not None else []):
            proc.terminate()
        collector.close()
        if wandb_run is not None:
            _log_model(wandb_run, args, args.run / "final.zip", model.num_timesteps, final=True)
            wandb_run.finish()


def _log_model(run, args, path: Path, steps: int, final: bool = False) -> None:
    """Upload a saved model as a new version of the W&B artifact ``--wandb-artifact``.

    Only with ``--wandb-log-model``; never fails the run. The upload happens in the
    background, from a copy W&B takes now, so ``path`` may be overwritten right after.
    """
    if run is None or not args.wandb_log_model or not path.exists():
        return
    try:
        import wandb

        artifact = wandb.Artifact(re.sub(r"[^A-Za-z0-9_.-]", "-", args.wandb_artifact), type="model",
                                  metadata={"timesteps": steps, "file": path.name, "run": run.name,
                                            "run_id": run.id, "final": final})
        artifact.add_file(str(path), name="model.zip")
        run.log_artifact(artifact, aliases=["latest", f"step_{steps}"] + (["final"] if final else []))
        print(f"W&B backup: {path.name} at {steps:,} steps -> {args.wandb_artifact}:latest", flush=True)
    except Exception as e:  # a network hiccup must not cost a training run
        print(f"W&B model upload failed: {e}")


if __name__ == "__main__":
    main()
