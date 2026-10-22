"""Turn a PTCG Live-style decklist into a deck CSV of official engine card ids.

    cd rl-train
    python tools/decklist_to_csv.py list.txt decks/training_pool/my_deck.csv

Lines look like ``4 Dreepy TWM 128``; section headers ("Pokémon: 23", ...) and
blank lines are ignored. Each card is matched by set code and number in
``engine/EN_Card_Data_R2_full.csv``, then by name (newer printings of a card
share its engine id); basic energy maps to ids 1-8. The deck is written only if
all 60 cards resolve and the engine accepts it; otherwise the unresolved lines
are reported so substitutes can be chosen.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ptcg_rl.engine import TRAIN_ROOT, Game, all_cards  # noqa: E402

BASIC_ENERGY = {"grass": 1, "fire": 2, "water": 3, "lightning": 4, "psychic": 5,
                "fighting": 6, "darkness": 7, "metal": 8}
LINE = re.compile(r"^\s*(\d+)\s+(.+?)\s+([A-Z0-9-]+)\s+(\d+)\s*$")


def _norm(s: str) -> str:
    return s.replace("’", "'").replace("é", "e").lower().strip()


def resolve(text: str) -> tuple[list[int], list[str], list[str]]:
    """(card ids, one report line per entry, unresolved entries)."""
    cards = all_cards()
    by_set, by_name = {}, {}
    with open(TRAIN_ROOT / "engine" / "EN_Card_Data_R2_full.csv", encoding="utf-8-sig") as f:
        for r in list(csv.reader(f))[1:]:
            by_set.setdefault((r[2], r[3].lstrip("0")), set()).add(int(r[0]))
            by_name.setdefault(_norm(r[1]), set()).add(int(r[0]))
    for i, c in cards.items():
        by_name.setdefault(_norm(c["name"]), set()).add(i)

    ids, report, missing = [], [], []
    for line in text.splitlines():
        m = LINE.match(line)
        if not m:
            continue
        n, name, set_code, number = int(m[1]), m[2], m[3], m[4].lstrip("0")
        energy = re.match(r"(\w+) Energy$", name)
        if energy and energy[1].lower() in BASIC_ENERGY:
            cid, how = BASIC_ENERGY[energy[1].lower()], "basic energy"
        else:
            hit, how = {i for i in by_set.get((set_code, number), set()) if i in cards}, "set+number"
            if not hit:
                hit, how = {i for i in by_name.get(_norm(name), set()) if i in cards}, "name"
            if not hit:
                missing.append(line.strip())
                report.append(f"{n} {name:<30} -> NOT IN THIS ENGINE")
                continue
            cid = min(hit)
        ids += [cid] * n
        report.append(f"{n} {name:<30} -> {cid:>4} {cards[cid]['name']:<30} ({how})")
    return ids, report, missing


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    ids, report, missing = resolve(src.read_text(encoding="utf-8"))
    print("\n".join(report))
    print(f"{len(ids)} cards resolved; not in this engine: {missing or 'none'}")
    if missing or len(ids) != 60:
        sys.exit("deck not written: choose substitutes for the missing cards (or fix the count)")
    game = Game(ids, ids)  # raises if the engine rejects the deck
    game.close()
    out.write_text("\n".join(map(str, ids)) + "\n")
    print(f"engine accepts it; wrote {out}")


if __name__ == "__main__":
    main()
