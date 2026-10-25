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
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.utils import LinearSchedule, get_device
from stable_baselines3.common.vec_env import DummyVecEnv

from .collector import BASELINE, RANDOM, SELF, SelfPlayCollector, frozen_copy
from .encoding import N_ACTIONS, observation_space
from .engine import AGENT_DECK, card_table, default_opponent_decks
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, default=Path("runs/ppo"))
    p.add_argument("--agent-deck", default=str(AGENT_DECK))
    p.add_argument("--opponent-decks", nargs="*", default=default_opponent_decks(),
                   help="opponent deck CSVs; none means a mirror match")
    p.add_argument("--opponent-weights", type=Path,
                   help="JSON {deck file stem: relative weight} for drawing opponent decks "
                        "(e.g. decks/training_pool/weights.json); default: every deck equally")
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
    p.add_argument("--device", default="auto", help="where the PPO update runs")
    p.add_argument("--collect-device",
                   help="where game inference, the sample store and snapshots live (default: --device); "
                        "with two GPUs, e.g. --device cuda:0 --collect-device cuda:1 --pipeline")
    p.add_argument("--pipeline", action="store_true",
                   help="run the update in its own process on --device while the next rollout is collected "
                        "on --collect-device (both GPUs busy at once)")
    p.add_argument("--amp", action="store_true", help="run the PPO update in fp16 autocast (CUDA only)")
    p.add_argument("--resume", type=Path, help="a saved model .zip to continue from")
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
                   help="a fixed model kept in the opponent pool for the whole run (win_rate/vs_baseline)")
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
    else:
        if args.arch == "entity":
            if args.dim % args.heads:
                raise SystemExit(f"--heads {args.heads} must divide --dim {args.dim}")
            policy, policy_kwargs = EntityPolicy, dict(features_extractor_kwargs=dict(
                dim=args.dim, heads=args.heads, board_layers=args.board_layers,
                option_layers=args.option_layers, ff=args.ff))
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
            gamma=0.995,
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
    )
    if args.baseline:
        collector.add_opponent(BASELINE, load_model(args.baseline, collect_device).policy)
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
        model.train()
        return time.time() - start

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
                train_s = timed_train()
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
            kinds: dict[str, list[float]] = {}
            for name, outcomes in stats.outcomes.items():
                kind = {RANDOM: "vs_random", SELF: "vs_self", BASELINE: "vs_baseline"}.get(name, "vs_snapshot")
                if name.startswith("bot:"):
                    kind = "vs_bot"
                    log.record(f"vs_bot_by_name/{name[4:]}", float(np.mean(outcomes)))
                kinds.setdefault(kind, []).extend(outcomes)
            for kind, outcomes in kinds.items():
                log.record(f"win_rate/{kind}", float(np.mean(outcomes)))
            for deck, outcomes in stats.baseline_by_deck.items():
                log.record(f"vs_baseline_by_deck/{deck}", float(np.mean(outcomes)))
            for deck, outcomes in stats.by_deck.items():
                log.record(f"vs_deck/{deck}", float(np.mean(outcomes)))
            # Per finished game, what the learner seat did: usage/play Dusknoir, usage/ability Dusknoir, ...
            for key, count in stats.usage.items():
                log.record(f"usage/{key}", count / max(stats.usage_games, 1))
            stats.outcomes.clear()
            stats.baseline_by_deck.clear()
            stats.by_deck.clear()
            stats.usage.clear()
            stats.usage_games = 0
            log.record("time/total_timesteps", model.num_timesteps)
            log.record("time/trained_hours", trained / 3600)
            if wandb_run is not None:
                wandb_run.log({k: float(v) for k, v in log.name_to_value.items()
                               if isinstance(v, (int, float, np.number))})
            log.dump(model.num_timesteps)
            vs_base = f" | vs baseline {np.mean(kinds['vs_baseline']):.2f}" if "vs_baseline" in kinds else ""
            vs_base += f" | vs bots {np.mean(kinds['vs_bot']):.2f}" if "vs_bot" in kinds else ""
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
