---
title: Pokémon TCG — Play vs a Trained Agent
emoji: 🃏
colorFrom: red
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Play the competition Pokémon TCG against a trained agent
---

# Play the competition's Pokémon TCG against a trained agent

You pilot one seat in the browser; a policy trained by behavioural cloning
pilots the other. No game rule is implemented in the web app: every legal move,
every coin flip and every card effect comes from the competition's own C++
engine, which this image compiles from source at build time. The UI only ever
offers the exact option list the engine put in front of your seat.

**New game** → pick a deck (five prebuilt lists, or build your own over the
whole 1267-card pool), pick an opponent, pick a seat.

Cards are the real scans, and the opponent's turn is *replayed* one action at a
time — each card it plays is held up with its art — because a board that changes
all at once tells you nothing about what it decided.

### Notes on running here

* The native engine keeps one battle per process, so this Space runs **one
  engine process per player**: your game is yours, and someone else opening the
  page does not touch it. A handful of games run at once (`PTCG_MAX_SESSIONS`);
  beyond that the page says the server is full rather than resetting a stranger's
  match. A game left untouched for 15 minutes is reclaimed.
* Your session lives in `localStorage`, so a reload returns you to your game.
* The first move of a game takes a few seconds: the opponent bundle is loading
  torch and its checkpoint in its own process.

### Attribution

The engine in `ptcg_engine/` is the competition's own package, included under the
licence in its `src/LICENSES/` directory so that the simulator runs; it is not
open-source. Pokémon, card names, card text and card artwork are trademarks and
copyright of Pokémon / Nintendo / Creatures / GAME FREAK. This is a
non-commercial research demo. Everything else — the server, the front end — is my
own work.
