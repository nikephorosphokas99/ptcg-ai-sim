"""Plot a run's evaluation over training, with a shaded 95% confidence band (``ptcg_rl.plots``).

    python tools/plot_eval.py runs/local-1m-ledger            # writes runs/local-1m-ledger/eval.png
    python tools/plot_eval.py runs/local-1m-ledger --out eval.png --smooth 5

Reads the run's TensorBoard logs, so it works while training runs; training logs the same
chart to W&B as ``charts/eval``.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ptcg_rl.plots import eval_chart  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path, help="the run folder (holds tb/)")
    p.add_argument("--out", type=Path, help="image to write (default: <run>/eval.png)")
    p.add_argument("--smooth", type=int, default=1, help="moving average over this many updates")
    args = p.parse_args()
    out = eval_chart(args.run, args.out or args.run / "eval.png", args.smooth)
    if out is None:
        raise SystemExit(f"no eval/win_rate_* in {args.run / 'tb'} (train with --eval-frac)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
