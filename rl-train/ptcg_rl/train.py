"""Self-play training with MaskablePPO over parallel engine processes.

    cd rl-train
    python -m ptcg_rl.train --run runs/first --n-envs 16 --timesteps 20_000_000

Every ``--snapshot-every`` steps the current policy is saved into the run's
``pool/`` and added to every env's opponent pool, so the learner keeps facing
older copies of itself alongside the random player.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from .engine import AGENT_DECK, default_opponent_decks
from .env import PTCGEnv
from .policy import PTCGExtractor


class SelfPlayCallback(BaseCallback):
    """Snapshot the policy into the opponent pool at a fixed interval."""

    def __init__(self, pool_dir: Path, every: int):
        super().__init__()
        self.pool_dir, self.every = pool_dir, every
        self._next = every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.every
            path = self.pool_dir / f"snap_{self.num_timesteps}.zip"
            self.model.save(path)
            self.training_env.env_method("add_opponent", str(path))
        return True


class OutcomeCallback(BaseCallback):
    """Log win rate per opponent and per opponent deck once per rollout."""

    def __init__(self):
        super().__init__()
        self._outcomes: dict[str, list[float]] = defaultdict(list)

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "outcome" in info:
                opp = "random" if info["opponent"] == "random" else "snapshot"
                self._outcomes[f"win_rate/vs_{opp}"].append(info["outcome"])
                self._outcomes[f"win_rate_deck/{info['opp_deck']}"].append(info["outcome"])
        return True

    def _on_rollout_end(self) -> None:
        for key, values in self._outcomes.items():
            self.logger.record(key, sum(values) / len(values))
        self._outcomes.clear()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, default=Path("runs/ppo"))
    p.add_argument("--agent-deck", default=str(AGENT_DECK))
    p.add_argument("--opponent-decks", nargs="+",
                   default=default_opponent_decks())
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--timesteps", type=int, default=20_000_000)
    p.add_argument("--snapshot-every", type=int, default=500_000)
    p.add_argument("--prize-reward", type=float, default=0.05)
    p.add_argument("--n-steps", type=int, default=512, help="rollout length per env")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", type=Path, help="a saved model .zip to continue from")
    p.add_argument("--dummy", action="store_true", help="run envs in-process (debugging)")
    args = p.parse_args()

    pool_dir = args.run / "pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    env_kwargs = dict(
        agent_deck=args.agent_deck,
        opponent_decks=args.opponent_decks,
        prize_reward=args.prize_reward,
    )
    PTCGEnv(**env_kwargs).close()  # fail on a bad deck here, not in N workers

    make = lambda: PTCGEnv(**env_kwargs)  # noqa: E731
    if args.dummy:
        venv = DummyVecEnv([make] * args.n_envs)
    else:
        venv = SubprocVecEnv([make] * args.n_envs, start_method="forkserver")
    venv = VecMonitor(venv)

    try:
        import tensorboard  # noqa: F401
        tb_log = str(args.run / "tb")
    except ImportError:
        tb_log = None

    if args.resume:
        model = MaskablePPO.load(args.resume, env=venv, device=args.device, tensorboard_log=tb_log)
    else:
        model = MaskablePPO(
            "MultiInputPolicy",
            venv,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=4,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=args.ent_coef,
            policy_kwargs=dict(
                features_extractor_class=PTCGExtractor,
                net_arch=dict(pi=[512, 256], vf=[512, 256]),
            ),
            tensorboard_log=tb_log,
            device=args.device,
            verbose=1,
        )

    callbacks = CallbackList([
        SelfPlayCallback(pool_dir, args.snapshot_every),
        OutcomeCallback(),
        CheckpointCallback(max(args.snapshot_every // args.n_envs, 1), str(args.run / "checkpoints")),
    ])
    try:
        model.learn(args.timesteps, callback=callbacks, reset_num_timesteps=not args.resume)
    finally:
        model.save(args.run / "final")
        venv.close()


if __name__ == "__main__":
    main()
