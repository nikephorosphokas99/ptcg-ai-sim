"""Write ARCH.json for a registered agent: what its policy actually is.

The dialog shows each opponent's architecture, and the honest source for that is
the checkpoint itself — tensor shapes give the width, the layer indices give the
depth, and the total gives the parameter count. Reading it needs torch, which
the server deliberately does not import (only agent workers do), so this runs
once when a bundle is registered and leaves a small JSON beside it.

    python describe_agent.py agents/grimmsnarl_ex_xattn
    python describe_agent.py agents/*            # all of them
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def _module_defaults(bundle: Path) -> dict:
    source = bundle / "policy_network" / "policy_experimental.py"
    if not source.is_file():
        return {}
    tree = ast.parse(source.read_text())
    constants, defaults = {}, {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
        if isinstance(node, ast.ClassDef) and node.name == "PolicyNetwork":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    args = item.args.args[1:]
                    pad = len(args) - len(item.args.defaults)
                    for arg, value in zip(args[pad:], item.args.defaults):
                        try:
                            defaults[arg.arg] = ast.literal_eval(value)
                        except ValueError:
                            name = getattr(value, "id", None)
                            if name in constants:
                                defaults[arg.arg] = constants[name]
    return defaults


def describe(bundle: Path) -> dict:
    import torch

    weights = sorted(bundle.glob("*.pt"))
    if not weights:
        raise SystemExit(f"{bundle}: no checkpoint")
    state = torch.load(weights[0], map_location="cpu", weights_only=True)

    depths: dict[str, int] = defaultdict(int)
    for key in state:
        found = re.match(r"policy\.([a-z_]+)\..*layers\.(\d+)\.", key)
        if found:
            depths[found.group(1)] = max(depths[found.group(1)], int(found.group(2)) + 1)

    defaults = _module_defaults(bundle)
    meta = {}
    submission = bundle / "SUBMISSION.json"
    if submission.is_file():
        meta = json.loads(submission.read_text())

    params = sum(v.numel() for v in state.values())
    width = defaults.get("dim")
    for key, tensor in state.items():
        if key.endswith("card.id.weight"):
            width = tensor.shape[1]
            break

    arch = {
        "family": "Cross-attention pointer transformer",
        "parameters": int(params),
        "dim": int(width) if width else None,
        "heads": defaults.get("nhead"),
        "ffMult": defaults.get("ff_mult"),
        "dropout": defaults.get("dropout"),
        "towers": {name: int(n) for name, n in sorted(depths.items())},
        "layers": int(sum(depths.values())) or None,
        "trainedBy": meta.get("source"),
        "decisionChain": meta.get("decision_chain_size"),
        "opponentHistory": meta.get("opponent_history_size"),
        "checkpoint": weights[0].name,
    }
    return {k: v for k, v in arch.items() if v not in (None, {}, [])}


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for name in sys.argv[1:]:
        bundle = Path(name)
        if not (bundle / "main.py").is_file():
            continue
        arch = describe(bundle)
        (bundle / "ARCH.json").write_text(json.dumps(arch, indent=2) + "\n")
        print(f"{bundle.name}: {arch['parameters']:,} parameters, "
              f"d{arch.get('dim')} · {arch.get('heads')} heads · {arch.get('layers')} layers")


if __name__ == "__main__":
    main()
