# `deploy/` — the simulator as a Hugging Face Space

    hf auth login                                   # once
    ./deploy/push_to_space.sh <user>/<space> [--private]

That is the whole deployment. The script stages `simulator/` at the Space repo's
root, drops the `Dockerfile` and the Space card on top, and uploads; the Hub then
builds the image and runs it. First upload moves the 683 MB of card art and takes
a while, later ones are seconds because the Hub skips files it already has by
hash.

| file | what it is |
| --- | --- |
| `Dockerfile` | two stages: compile `libcg.so` from `ptcg_engine/src`, then a runtime image with no compiler in it |
| `requirements.txt` | the server and the bundle's torch, CPU wheels only |
| `SPACE_README.md` | becomes the Space's `README.md` — its YAML header *is* the Space's configuration (`sdk: docker`, `app_port: 7860`) |
| `.gitattributes` | LFS rules for the scans (`*.webp`) and the checkpoint (`*.pt`) |

`.dockerignore` sits at `simulator/` instead, because that is the build context
root and the only place Docker reads it from.
| `push_to_space.sh` | stage → create repo if needed → upload |

## The two things that make this non-trivial

**The engine is C++ and no `.so` is in git.** So the image builds it, exactly as a
local checkout does (`build_engine.sh`), in a first stage whose compiler is then
thrown away. The binary is copied into `ptcg_engine/` and into every agent
bundle's `cg/`, because a bundle runs in its own interpreter and loads its own.

**A game is a process, not a request handler.** `cg.sim.Battle` keeps the native
battle pointer on a class attribute, so one process plays one game — which is
fine at a keyboard and wrong on a URL two people can open. The server therefore
holds no battle: it routes each request to the caller's own `game_worker.py`
process (`sessions.py`), and that worker spawns the agent bundle in a third. Two
visitors play two games because they are in different processes.

The cost is memory: roughly 1 GB per game once torch is loaded, against 16 GB on
a free CPU Space. Hence the two knobs, set in the `Dockerfile` and overridable in
the Space's variables:

| variable | default here | what it does |
| --- | --- | --- |
| `PTCG_MAX_SESSIONS` | 6 | games at once; beyond it `/api/new` answers 503 with "the server is full" rather than resetting somebody's match |
| `PTCG_IDLE_TIMEOUT` | 900 | seconds a game may sit untouched before its processes are reclaimed |

A session is identified by an id the page generates and sends as `X-Session`,
kept in `localStorage`. Not a cookie, deliberately: a Space is normally viewed in
an iframe on `huggingface.co`, where the Space's own cookies are third-party and
dropped by default in current browsers, and the sessions would silently collapse
into one. A header behaves the same in the iframe as on the direct
`*.hf.space` URL. Closing the tab posts `/api/leave`, so a slot is not held for
15 minutes by someone who has gone.

## Before making it public

Two things travel with this deployment that do not travel with the repository,
and both are worth a deliberate decision rather than a default:

* **The card scans.** This repo does not distribute them (`.gitignore`, and the
  note in the root README) — the artwork is Pokémon/Nintendo/Creatures/GAME
  FREAK's. Uploading `card_images_en/` to a *public* Space does publish it. A
  **private** Space (`--private`) keeps it to you and still plays from anywhere,
  which is the option to take unless you have a reason not to. For a public demo,
  the honest configuration is without the art: the board falls back to drawn text
  cards and everything else works.

      ./deploy/push_to_space.sh <user>/<space> --private          # art included
      # public: stage without it, then upload
      hf upload <user>/<space> . --repo-type space --exclude 'card_images_en/*'

* **The engine.** `ptcg_engine/` is the competition's package under its own
  licence (`src/LICENSES/`), included so the thing runs. It is not open-source,
  and a public Space is a public copy of it.

The Space card in `SPACE_README.md` carries the attribution either way.

## Hardware

The free **CPU basic** tier is enough: the engine is C++, and a checkpoint
forward on 2.7 M parameters is milliseconds on a CPU. There is nothing to gain
from a GPU here. What the free tier does do is **sleep** after inactivity — the
first visit then waits for a cold start (the image is cached, so this is startup,
not a rebuild).

## Checking a deployment

    curl -s https://<user>-<space>.hf.space/api/sessions
    # {"live":0,"max":6,"idleTimeout":900.0,"playing":0}

`live` is how many games are running. If the page loads but a game will not
start, that endpoint and the Space's **Logs** tab are where the reason is: a
worker that could not import the engine (the build failed) or a bundle that could
not load its checkpoint say so there.

## Running the same image locally

    docker build -t ptcg-sim -f deploy/Dockerfile .    # from simulator/
    docker run --rm -p 7860:7860 ptcg-sim
    # then open http://127.0.0.1:7860

Worth doing before a push: it is the same build the Hub runs, and a compile error
surfaces in a minute instead of after a 683 MB upload.
