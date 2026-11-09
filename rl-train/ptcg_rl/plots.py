"""The evaluation chart: win rate against each evaluation opponent over training, with a
shaded 95% band, and the learner's reward per training game below it.

Drawn from a run's TensorBoard logs (``<run>/tb``, every segment of a resumed run), by
``tools/plot_eval.py`` and by ``train.py`` for W&B (``charts/eval``, every ``--chart-every-min``).
Each ``eval/win_rate_vs_*`` point is the win rate over the opponent's last ``--eval-window``
evaluation games (``eval/games_vs_*`` of them); the band is the Wilson 95% interval for that
many games, i.e. how far the true win rate could be from the measured one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def wilson(p, n, z: float = 1.96):
    """Lower and upper 95% bounds of a win rate ``p`` measured over ``n`` games."""
    p, n = np.asarray(p, dtype=float), np.maximum(np.asarray(n, dtype=float), 1)
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) < k:
        return x
    return np.convolve(np.pad(x, (k - 1, 0), mode="edge"), np.ones(k) / k, mode="valid")


def eval_chart(run: Path, out: Path, smooth: int = 1) -> Path | None:
    """Write the chart for ``run`` to ``out``; None if the run has no evaluation games logged yet."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    logs = []
    for events in sorted((Path(run) / "tb").rglob("events.out.tfevents.*")):
        ea = EventAccumulator(str(events), size_guidance={"scalars": 0})
        ea.Reload()
        logs.append(ea)

    def series(tag: str) -> tuple[np.ndarray, np.ndarray]:
        points = {e.step: e.value for ea in logs if tag in ea.Tags()["scalars"] for e in ea.Scalars(tag)}
        steps = sorted(points)
        return np.array(steps, dtype=float), np.array([points[s] for s in steps], dtype=float)

    tags = sorted({t for ea in logs for t in ea.Tags()["scalars"] if t.startswith("eval/win_rate_vs_")})
    if not tags:
        return None
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    for tag in tags:
        name = tag[len("eval/win_rate_"):]
        steps, rate = series(tag)
        game_steps, games = series(f"eval/games_{name}")
        n = np.interp(steps, game_steps, games) if len(games) else np.full_like(rate, 500)
        lo, hi = wilson(rate, n)
        x = steps / 1e6
        line, = top.plot(x, _smooth(rate, smooth), lw=1.6, label=name.replace("_", " "))
        top.fill_between(x, _smooth(lo, smooth), _smooth(hi, smooth), color=line.get_color(), alpha=0.2)
    top.axhline(0.5, color="grey", lw=0.8, ls="--")
    top.set_ylim(0, 1)
    top.set_ylabel("win rate (last eval games, 95% band)")
    top.set_title(f"{Path(run).name}: evaluation (never trained on)")
    top.legend(loc="lower right")
    top.grid(alpha=0.3)

    steps, reward = series("reward/training_games")
    if len(steps):
        k = max(smooth, 20)
        bottom.plot(steps / 1e6, reward, color="lightgrey", lw=0.8, label="per update")
        bottom.plot(steps / 1e6, _smooth(reward, k), color="black", lw=1.6, label=f"moving average ({k} updates)")
        bottom.legend(loc="lower right")
    bottom.axhline(0.0, color="grey", lw=0.8, ls="--")
    bottom.set_ylabel("reward per game (training games)")
    bottom.set_xlabel("training steps (millions)")
    bottom.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return Path(out)
