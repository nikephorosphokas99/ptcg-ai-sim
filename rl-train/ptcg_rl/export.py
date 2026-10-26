"""Package a trained model as a competition submission zip.

    cd rl-train
    python -m ptcg_rl.export runs/first/final.zip
    python -m ptcg_rl.export runs/first/final.zip --visualizer selfplay_ppo_50m   # also add it to the web visualizer

Writes ``submissions/<name>/`` and ``submissions/<name>.zip`` with ``main.py``
at the root, next to ``deck.csv``, ``policy.npz`` + ``arch.json`` (actor weights only, for numpy), the
``ptcg_policy`` package (the numpy ``submission/ptcg_policy/net.py`` plus this repo's
``encoding.py``, so the agent sees exactly the observations it trained on) and
the official ``cg`` bindings, as in the sample submission, and ``rules.json``
(the decisions training plays by rule; see ``env.Match``). With
``--visualizer NAME`` the same checked bundle is also installed as an opponent
in ``visualizer/agents/NAME`` (without ``cg``, which the agent does not use),
with an ``ARCH.json`` for the opponent dialog.

Before zipping it checks that:

1. the numpy network in the bundle gives the SB3 policy's action
   log-probabilities on real game positions, and
2. the bundle, loaded the way kaggle_environments loads it (from another
   working directory, ``main.py`` exec'd with no ``__file__``, its folder
   removed from ``sys.path`` afterwards, in a fresh interpreter without torch,
   stable-baselines3 or gymnasium), plays complete games through the official engine with every
   selection accepted. ``main.py`` has no fallback move, so any error fails
   the check.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import deque
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from torch import nn

from .encoding import RECENT_SLOTS, Decision, encode, recent_opponent_cards, visible_cards
from .engine import AGENT_DECK, TRAIN_ROOT, Game, attack_table, card_meta_table, load_deck, safe_attack_ids
from .policy import EntityPolicy, PointerPolicy
from .train import load_model

TEMPLATE = TRAIN_ROOT / "submission"
VISUALIZER_AGENTS = TRAIN_ROOT / "visualizer" / "agents"
OFFICIAL_CG = TRAIN_ROOT / "engine" / "sample_submission" / "cg"

PLAY_CHECK = r'''
import os, random, sys
for blocked in ("torch", "gymnasium", "stable_baselines3", "sb3_contrib", "ptcg_rl"):
    sys.modules[blocked] = None  # Kaggle's agent runtime has none of these
bundle = sys.argv[2]
# The engine for this check comes from the bundle's cg; the agent must not lean on it being importable.
sys.path.insert(0, bundle)
from cg.game import battle_finish, battle_select, battle_start
sys.path.remove(bundle)

# Load main.py exactly as kaggle_environments.agent.get_last_callable does: from
# another working directory, bundle dir appended to sys.path, exec'd in a bare
# dict (no __file__), the dir popped again, the last callable taken.
path = os.path.join(bundle, "main.py")
sys.path.append(os.path.dirname(path))
env = {}
exec(compile(open(path).read(), path, "exec"), env)
sys.path.pop()
agent = [v for v in env.values() if callable(v)][-1]
assert agent.__name__ == "agent", f"last callable in main.py is {agent.__name__!r}, not agent"

deck = agent({"select": None})
assert deck == [int(x) for x in open(os.path.join(bundle, "deck.csv")).read().split()[:60]], "agent returned a different deck"
games, wins, decisions = int(sys.argv[1]), 0, 0
for g in range(games):
    seat = g % 2
    obs, start = battle_start(deck, deck)
    assert obs is not None, f"engine rejected the deck: {start.errorType}"
    while obs["current"]["result"] < 0:
        select = obs["select"]
        if obs["current"]["yourIndex"] == seat:
            picks = agent(obs)
            decisions += 1
        else:
            picks = random.sample(range(len(select["option"])), select["maxCount"])
        obs = battle_select(picks)  # raises if the selection is illegal
    wins += obs["current"]["result"] == seat
    battle_finish()
print(f"{games} games, {decisions} agent decisions, won {wins}/{games} vs random")
'''


_VALUE_PARAMS = ("mlp_extractor.value_net", "mlp_extractor.value", "value_net")


def _arch(policy) -> dict:
    ext = policy.features_extractor
    if isinstance(policy, EntityPolicy):
        return {
            "kind": "entity",
            "dim": ext.dim,
            "heads": ext.board[0].self_attn.num_heads,
            "board_layers": len(ext.board),
            "option_layers": len(ext.options),
        }
    if isinstance(policy, PointerPolicy):
        return {
            "kind": "pointer",
            "embed_dim": ext.embedding.embedding_dim,
            "card_dim": ext.card[0].out_features,
            "state_dim": ext.state_dim,
            "option_dim": ext.option_dim,
            "hidden": policy.mlp_extractor.query.out_features,
        }
    if policy.activation_fn is not nn.Tanh or not policy.share_features_extractor:
        raise ValueError("submission/ptcg_policy/net.py assumes Tanh layers and a shared extractor")
    return {
        "embed_dim": ext.embedding.embedding_dim,
        "card_dim": ext.card[0].out_features,
        "option_dim": ext.option[2].out_features,
        "features_dim": ext.features_dim,
        "pi": list(policy.net_arch["pi"]),
        "kind": "flat",
    }


def _ismcts_files(out: Path, config: dict) -> None:
    """ismcts.json switches main.py's search on; opp_decks.json holds the deck lists it determinizes from."""
    decks = [load_deck(p) for p in sorted((TRAIN_ROOT / "decks" / "training_pool").glob("*.csv"))]
    decks += [load_deck(p) for p in sorted((TRAIN_ROOT / "benchmarks").glob("*/deck.csv"))]
    (out / "opp_decks.json").write_text(json.dumps(decks))
    (out / "ismcts.json").write_text(json.dumps(config, indent=2) + "\n")


def _check_value(model: MaskablePPO, deck: list[int], bundle: Path, positions: int = 50) -> float:
    """Largest difference between the bundle's value head and SB3's, on real positions."""
    sys.path.insert(0, str(bundle))
    try:
        net_spec = importlib.util.spec_from_file_location("bundle_value_net", bundle / "ptcg_policy" / "net.py")
        net_module = importlib.util.module_from_spec(net_spec)
        net_spec.loader.exec_module(net_module)
    finally:
        sys.path.remove(str(bundle))
    net, _, meta = net_module.load(bundle)
    attacks, rng, worst, seen = attack_table(), np.random.default_rng(1), 0.0, 0
    with torch.no_grad():
        while seen < positions:
            game = Game(deck, deck)
            while not game.done and seen < positions:
                d = Decision(game.obs)
                features = encode(d, attacks, my_deck=deck, opp_seen=set(), opp_recent=(), card_meta=meta)
                _, ours = net.evaluate({k: v.astype(np.float32) for k, v in features.items()})
                sb3_obs = {k: torch.as_tensor(features[k][None].astype(np.float32)) for k in model.observation_space.spaces}
                theirs = float(model.policy.predict_values(sb3_obs)[0, 0])
                worst, seen = max(worst, abs(ours - theirs)), seen + 1
                while not d.complete:
                    if d.apply(int(rng.choice(np.flatnonzero(d.mask())))):
                        break
                game.select(d.picked)
            game.close()
    return worst


def _write_bundle(model: MaskablePPO, deck: list[int], out: Path, value: bool = False, rules: bool = True) -> None:
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(TEMPLATE, out, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(OFFICIAL_CG, out / "cg", ignore=shutil.ignore_patterns("__pycache__"))
    (out / "deck.csv").write_text("\n".join(map(str, deck)) + "\n")

    package = out / "ptcg_policy"
    (package / "__init__.py").write_text("")
    source = (TRAIN_ROOT / "ptcg_rl" / "encoding.py").read_text()
    # The bundle has no engine module: its constants are written in as literals.
    engine_import = re.search(r"^from \.engine import \(([^)]*)\)\n", source, re.MULTILINE)
    if engine_import is None or "from .engine" in source[engine_import.end():]:
        raise ValueError("encoding.py's engine import changed; update export._write_bundle")
    from . import engine
    names = re.findall(r"\w+", engine_import[1])
    constants = "".join(f"{name} = {getattr(engine, name)!r}\n" for name in names)
    (package / "encoding.py").write_text(source.replace(engine_import[0], constants))

    policy = model.policy
    keep = ("features_extractor.", "mlp_extractor.policy_net.", "action_net.",
            "mlp_extractor.query.", "mlp_extractor.score.", "mlp_extractor.stop.")
    if value:  # the lookahead scores positions with the critic
        keep += ("mlp_extractor.value.", "value_net.")
    # numpy, not torch: Kaggle's agent runtime has no torch (see submission/ptcg_policy/net.py).
    arrays = {k: v.detach().cpu().numpy() for k, v in policy.state_dict().items() if k.startswith(keep)}
    arrays["card_static"] = policy.features_extractor.card_static.cpu().numpy()
    if hasattr(policy.features_extractor, "card_text"):
        arrays["card_text"] = policy.features_extractor.card_text.cpu().numpy()
    arrays["attack_table"] = attack_table()
    arrays["card_meta"] = card_meta_table()
    np.savez(out / "policy.npz", **arrays)
    (out / "arch.json").write_text(json.dumps(_arch(policy), indent=2) + "\n")
    if rules:  # the decisions training's Match plays by rule (env.Match, rules=True)
        (out / "rules.json").write_text(json.dumps(
            {"counter_planner": True, "safe_attacks": sorted(safe_attack_ids())}) + "\n")


def _check_parity(model: MaskablePPO, deck: list[int], bundle: Path, positions: int = 300) -> float:
    """Largest log-probability difference between the bundle's network and SB3's, on real positions."""
    spec = importlib.util.spec_from_file_location(
        "bundle_ptcg_policy", bundle / "ptcg_policy" / "__init__.py",
        submodule_search_locations=[str(bundle / "ptcg_policy")])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["bundle_ptcg_policy"] = pkg
    spec.loader.exec_module(pkg)
    net_spec = importlib.util.spec_from_file_location("bundle_ptcg_policy.net", bundle / "ptcg_policy" / "net.py")
    net_module = importlib.util.module_from_spec(net_spec)
    sys.modules["bundle_ptcg_policy.net"] = net_module
    net_spec.loader.exec_module(net_module)

    net, _, meta = net_module.load(bundle)

    attacks, rng, worst, seen = attack_table(), np.random.default_rng(0), 0.0, 0
    with torch.no_grad():
        while seen < positions:
            game = Game(deck, deck)
            opp_seen: list[set[int]] = [set(), set()]
            recent: list[deque] = [deque(maxlen=RECENT_SLOTS), deque(maxlen=RECENT_SLOTS)]
            while not game.done and seen < positions:
                d = Decision(game.obs)
                mover = game.to_move
                opp_seen[mover].update(visible_cards(game.obs["current"], 1 - mover))
                recent[mover].extend(recent_opponent_cards(game.obs, mover))
                while not d.complete:
                    features = encode(d, attacks, my_deck=deck, opp_seen=opp_seen[mover],
                                      opp_recent=recent[mover], card_meta=meta)
                    batch = {k: v[None].astype(np.float32) for k, v in features.items()}
                    # SB3's distribution holds log-probabilities, the bundle raw logits.
                    ours = torch.log_softmax(torch.as_tensor(net(batch)), dim=-1)
                    sb3_obs = {k: torch.as_tensor(batch[k]) for k in model.observation_space.spaces}
                    theirs = model.policy.get_distribution(sb3_obs).distribution.logits
                    worst = max(worst, float((ours - theirs).abs().max()))
                    seen += 1
                    if d.apply(int(rng.choice(np.flatnonzero(d.mask())))):
                        break
                game.select(d.picked)
            game.close()
    return worst


def _install_in_visualizer(bundle: Path, model: MaskablePPO, model_path: Path, name: str) -> Path:
    target = VISUALIZER_AGENTS / name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(bundle, target, ignore=shutil.ignore_patterns("cg", "__pycache__"))
    policy = model.policy
    actor = [m for k, m in policy.named_modules()
             if isinstance(m, nn.Linear) and not k.startswith(_VALUE_PARAMS)]
    arch = _arch(policy)
    family = {"entity": "Entity-attention network (MaskablePPO actor)",
              "pointer": "Pointer network with deck memory (MaskablePPO actor)"}.get(
        arch["kind"], "Card-embedding MLP (MaskablePPO actor)")
    (target / "ARCH.json").write_text(json.dumps({
        "family": family,
        "parameters": sum(p.numel() for k, p in policy.named_parameters() if not k.startswith(_VALUE_PARAMS)),
        "dim": arch.get("dim", arch.get("card_dim")),
        "layers": len(actor),
        "trainedBy": f"self-play PPO, {model.num_timesteps:,} steps",
        "checkpoint": f"{model_path.resolve().parent.name}/{model_path.name} -> policy.npz",
    }, indent=2) + "\n")
    return target


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", type=Path, help="a trained MaskablePPO .zip")
    p.add_argument("--deck", default=str(AGENT_DECK), help="the deck the model was trained to pilot")
    p.add_argument("--name", help="bundle name (default: <run>_<model>)")
    p.add_argument("--out", type=Path, default=TRAIN_ROOT / "submissions")
    p.add_argument("--check-games", type=int, default=20)
    p.add_argument("--visualizer", metavar="NAME", help="also install as a visualizer opponent under this name")
    p.add_argument("--value", action="store_true",
                   help="also ship the critic (value head) for search-based agents; checked against SB3")
    p.add_argument("--ismcts", action="store_true",
                   help="ISMCTS bundle (ptcg_policy/ismcts.py): ships the value head, ismcts.json, opp_decks.json")
    p.add_argument("--ismcts-seconds", type=float, default=1.0, help="most thinking time per searched pick")
    p.add_argument("--no-rules", action="store_true",
                   help="leave out rules.json, for a model trained with --no-rules")
    args = p.parse_args()

    name = args.name or f"{args.model.resolve().parent.name}_{args.model.stem}"
    bundle, archive = args.out / name, args.out / f"{name}.zip"
    deck = load_deck(args.deck)
    model = load_model(args.model, "cpu")

    args.value = args.value or args.ismcts
    _write_bundle(model, deck, bundle, value=args.value, rules=not args.no_rules)
    diff = _check_parity(model, deck, bundle)
    print(f"parity: bundle vs SB3 log-probabilities differ by at most {diff:.2e}")
    if diff > 1e-4:
        raise SystemExit("bundle network does not match the trained policy")
    if args.value:
        vdiff = _check_value(model, deck, bundle)
        print(f"value parity: bundle vs SB3 values differ by at most {vdiff:.2e}")
        if vdiff > 1e-4:
            raise SystemExit("bundle value head does not match the trained critic")
    if args.ismcts:
        _ismcts_files(bundle, {"max_seconds": args.ismcts_seconds})

    check = subprocess.run([sys.executable, "-I", "-c", PLAY_CHECK, str(args.check_games), str(bundle.resolve())],
                           cwd=bundle.parent, capture_output=True, text=True)
    print(check.stdout.strip() or check.stderr.strip()[-2000:])
    if check.returncode != 0:
        raise SystemExit("bundle failed its play check")

    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(bundle.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                zf.write(path, path.relative_to(bundle).as_posix())
    print(f"wrote {archive} ({archive.stat().st_size / 2**20:.1f} MB)")
    if args.visualizer:
        print(f"installed visualizer opponent {_install_in_visualizer(bundle, model, args.model, args.visualizer)}")
    shutil.rmtree(bundle)


if __name__ == "__main__":
    main()
