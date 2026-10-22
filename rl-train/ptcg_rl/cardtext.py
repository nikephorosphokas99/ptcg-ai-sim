"""What each card's text says, as fixed features (the v3 policy's card input).

``engine.card_table`` gives the network a card's printed numbers; nothing in it
says what an Ability or attack effect *does*, so the network could only learn
that Shaymin protects the Bench by losing to it often enough. Here every card's
rules text (Abilities, attack names and effects, trainer effects) becomes:

* a dense summary: TF-IDF over the card pool's vocabulary, reduced with SVD to
  ``TEXT_DIM`` dimensions (numpy only; deterministic for a given engine build);
* ``KEYWORDS`` flags for mechanics that decide games.

Row 0 (padding / face-down) is zero. The table is static, so the submission
ships it inside ``policy.npz``.
"""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

import numpy as np

from .engine import all_attacks, all_cards

TEXT_DIM = 48
KEYWORDS = (
    ("prevent", r"\bprevent"),
    ("bench", r"\bbench"),
    ("damage counter", r"damage counter"),
    ("switch", r"\bswitch"),
    ("search deck", r"search your deck"),
    ("draw", r"\bdraw"),
    ("discard", r"\bdiscard"),
    ("heal", r"\bheal"),
    ("attach energy", r"attach .{0,40}energy"),
    ("opponent's hand", r"opponent.s hand"),
    ("coin", r"\bcoin"),
    ("special condition", r"asleep|burned|confused|paralyzed|poisoned"),
    ("can't", r"can.t"),
    ("once during your turn", r"once during your turn"),
    ("prize", r"\bprize"),
    ("each", r"\beach\b"),
)
TEXT_FEATURES = TEXT_DIM + len(KEYWORDS)

_TOKEN = re.compile(r"[a-z][a-z']+")


def card_text(card: dict, attacks: dict[int, dict]) -> str:
    parts = [f"{s['name']} {s['text']}" for s in card["skills"]]
    for aid in card["attacks"]:
        a = attacks.get(aid)
        if a:
            parts.append(f"{a['name']} {a['text']}")
    return " ".join(parts).replace("’", "'").lower()


@lru_cache(maxsize=1)
def card_text_table() -> np.ndarray:
    cards, attacks = all_cards(), all_attacks()
    ids = sorted(cards)
    texts = [card_text(cards[i], attacks) for i in ids]
    docs = [_TOKEN.findall(t) for t in texts]

    df = Counter(w for d in docs for w in set(d))
    vocab = sorted(w for w, n in df.items() if n >= 2)
    index = {w: j for j, w in enumerate(vocab)}
    tf = np.zeros((len(ids), len(vocab)), dtype=np.float64)
    for r, d in enumerate(docs):
        for w in d:
            j = index.get(w)
            if j is not None:
                tf[r, j] += 1.0
    idf = np.log((1 + len(ids)) / (1 + np.array([df[w] for w in vocab]))) + 1.0
    x = np.log1p(tf) * idf
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)
    u, s, _ = np.linalg.svd(x - x.mean(axis=0), full_matrices=False)
    dense = u[:, :TEXT_DIM] * s[:TEXT_DIM]
    dense /= np.abs(dense).max(axis=0, keepdims=True) + 1e-9
    # Sign of each SVD component is arbitrary; pin it so the table is reproducible.
    dense *= np.sign(dense[np.abs(dense).argmax(axis=0), np.arange(TEXT_DIM)])

    table = np.zeros((max(cards) + 1, TEXT_FEATURES), dtype=np.float32)
    for r, (cid, t) in enumerate(zip(ids, texts)):
        table[cid, :TEXT_DIM] = dense[r]
        table[cid, TEXT_DIM:] = [float(bool(re.search(p, t))) for _, p in KEYWORDS]
    return table
