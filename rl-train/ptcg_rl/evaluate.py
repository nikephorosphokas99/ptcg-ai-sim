"""Win rate of a trained model against the random player or another checkpoint.

    python -m ptcg_rl.evaluate runs/ppo/final.zip --games 500
    python -m ptcg_rl.evaluate runs/ppo/final.zip --opponent runs/ppo/checkpoints/step_5000000.zip
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from stable_baselines3.common.utils import get_device

from .collector import RANDOM, SelfPlayCollector
from .engine import AGENT_DECK, default_opponent_decks
from .train import load_model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    p.add_argument("--opponent", default=RANDOM, help='"random" or a checkpoint .zip')
    p.add_argument("--agent-deck", default=str(AGENT_DECK))
    p.add_argument("--opponent-decks", nargs="*", default=default_opponent_decks(),
                   help="opponent deck CSVs; none means a mirror match")
    p.add_argument("--games", type=int, default=500)
    p.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    p.add_argument("--matches-per-worker", type=int, default=16)
    p.add_argument("--deterministic", action="store_true",
                   help="both sides play their top-ranked move, as a submission does")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = get_device(args.device)
    model = load_model(args.model, device)
    if args.opponent == RANDOM:
        pool = {RANDOM: None}
    else:
        pool = {Path(args.opponent).stem: load_model(args.opponent, device).policy}
    collector = SelfPlayCollector(
        dict(agent_deck=args.agent_deck, opponent_decks=args.opponent_decks, prize_reward=0.0),
        model.policy, args.workers, args.matches_per_worker, device,
        deterministic=args.deterministic, pool=pool,
    )
    try:
        collector.play(args.games)
    finally:
        collector.close()

    outcomes = [o for v in collector.stats.outcomes.values() for o in v]
    print(f"win rate {np.mean(outcomes):.3f} over {len(outcomes)} games vs {args.opponent} "
          f"(draws count 0.5; games still running at the end are not counted)")
    for deck, results in sorted(collector.stats.by_deck.items(), key=lambda kv: np.mean(kv[1])):
        print(f"  vs {deck:32s} {np.mean(results):.3f} over {len(results)} games")


if __name__ == "__main__":
    main()
