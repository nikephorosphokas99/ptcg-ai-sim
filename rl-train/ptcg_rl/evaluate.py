"""Win rate of a trained model against an opponent, per opponent deck.

    python -m ptcg_rl.evaluate runs/ppo/final.zip --games 50
    python -m ptcg_rl.evaluate runs/ppo/final.zip --opponent runs/ppo/pool/snap_500000.zip
"""

from __future__ import annotations

import argparse

import numpy as np
from sb3_contrib import MaskablePPO

from .engine import AGENT_DECK, default_opponent_decks
from .env import PTCGEnv


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    p.add_argument("--opponent", default="random", help='"random" or a checkpoint .zip')
    p.add_argument("--agent-deck", default=str(AGENT_DECK))
    p.add_argument("--opponent-decks", nargs="+",
                   default=default_opponent_decks())
    p.add_argument("--games", type=int, default=20, help="games per opponent deck")
    p.add_argument("--deterministic", action="store_true")
    args = p.parse_args()

    model = MaskablePPO.load(args.model, device="cpu")
    total = []
    for deck in args.opponent_decks:
        env = PTCGEnv(args.agent_deck, [deck], opponent_pool=[args.opponent], prize_reward=0.0)
        outcomes = []
        for _ in range(args.games):
            obs, _ = env.reset()
            while True:
                action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=args.deterministic)
                obs, _, terminated, truncated, info = env.step(int(action))
                if terminated or truncated:
                    outcomes.append(info.get("outcome", 0.5))
                    break
        env.close()
        total += outcomes
        print(f"{env.opp_deck_name:<24} {np.mean(outcomes):.2f}")
    print(f"{'overall':<24} {np.mean(total):.2f}  ({len(total)} games vs {args.opponent})")


if __name__ == "__main__":
    main()
