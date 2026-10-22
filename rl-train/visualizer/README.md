# `rl-train/visualizer/` — play the game yourself, against a trained agent

An interactive web front end on the same engine the agents train on: the
official competition binaries in `rl-train/engine/sample_submission/cg`, so card
ids, rules and decks match `ptcg_rl` exactly. Nothing is built here.

You pilot one seat in a browser; a registered agent from `agents/` pilots the
other. No game rule is implemented here: every legal move, every coin flip and
every card effect comes from the engine, and the UI only ever offers the exact
option list the engine put in front of the seat you are sitting in.

## Run it

    cd rl-train/visualizer
    pip install fastapi "uvicorn[standard]" pyarrow pillow torch   # once
    python fetch_card_images.py      # once: card art, needs git-lfs (conda install -c conda-forge git-lfs)
    python server.py --port 8000
    # then open http://127.0.0.1:8000

Card art goes in `card_images_en/` (~800 MB, git-ignored), named by official
card id. `fetch_card_images.py` pulls the scans from Git LFS at the commit that
last carried them and renames them through `card_image_map.json`: the art was
named by an older custom engine whose ids differ for the newest sets (Special
Red Card is 1413 officially, 1513 in the art), so the map matches cards by name,
type and HP. Three cards have no art (Transformation Tome, Tremendous Bomb,
Backtrack Badge) and draw as text cards. `PTCG_CARD_IMAGES=/path` overrides the
folder; without any art the board draws text cards and everything else works.

## Opponents

* `selfplay_ppo_50m` — the self-play MaskablePPO agent from `rl-train`
  (`runs/first/final.zip`, 50M steps), piloting Marnie's Grimmsnarl ex
  (`decks/main_agent.csv`). It is exactly the submission bundle: install another
  model the same way with
  `python -m ptcg_rl.export <model.zip> --visualizer <name>` from `rl-train/`.
* `grimmsnarl_ex_xattn` — the earlier behavioural-cloning transformer.

Decks in `decks/` all use official ids; the seven lists written for the older
engine were translated card by card (by name, type and HP, and by attack text
where several printings share a name).

`deploy/` (Dockerfile, Space scripts) still describes the older layout that
built a custom engine from source, and has not been updated for this one.

**New game** opens a dialog rather than a row of dropdowns, because a decklist
is 60 cards and the only useful way to show one is to show the cards. The five
prebuilt lists — Crustle, Grimmsnarl ex, Alakazam, Mega Lucario ex, Dragapult —
each render as one tile per distinct card with its count in the corner
(`[image] x4`), so a list reads at a glance. Pick a deck, an opponent and a seat, then **Start game**.

### Choosing an opponent

The opponent is not a dropdown entry but a card of its own: the archetype's
headline Pokémon (picked from the agent's decklist — the most numerous Pokémon
ex in it), the architecture that pilots it, and the 60 cards it plays.

The architecture is read from the checkpoint rather than described by hand.
`describe_agent.py` walks the state dict — tensor shapes give the width, layer
indices give the depth of each tower, the total gives the parameter count — and
takes the head count and feed-forward multiplier from the constructor defaults
in the bundle's own policy module, parsed with `ast` so that describing one
bundle cannot import a module name a second bundle also uses. The result lands
in `ARCH.json` beside the weights:

    python describe_agent.py agents/grimmsnarl_ex_xattn

    Cross-attention pointer transformer
    Parameters        2,685,443       Decision chain     30 frames
    Embedding dim     64              Opponent history   30 frames
    Attention heads   2               Trained by         behavioural cloning
    Layers            20 across 4 towers (chain 8, context 2, history 8, cross 2)

Run it once when registering a bundle; the server reads the JSON, so it never
imports torch itself.

Every line of the sheet carries a **(?)** that explains the term on hover — what
an embedding dimension is, what each tower reads, what behavioural cloning means
against PPO. The numbers say nothing to a reader who does not already have the
vocabulary, and the sheet exists so they do not need it.

### Create your own…

The last entry swaps the preview for a builder over the whole 1267-card pool:

- **Search** by name, ability text or attack text, with a card-type filter.
- **Add to Deck** / **Add x4** for the selected card; double-click a result to
  add one; hover +/− on a deck tile to adjust a count; **Clear** to start over.
- **Start from a prebuilt deck…** copies a legal list in as a base — the fastest
  way to build is to edit something that already works.
- **Export CSV** copies the deck as one card id per line (the same `deck.csv`
  format the bundles ship); **Import CSV** pastes one back.

The legality line updates on every change and states what is still wrong in the
engine's own terms — `54/60`, `5 copies of Munkidori (max 4)`, `2 ACE SPEC cards
(max 1)`, `no Basic Pokémon`. Those are the rules `ApiBattleStart` enforces
(`engine.deck_check` mirrors them), so a deck the dialog calls legal is one the
engine will accept; `/api/new` re-checks server-side and refuses anything else.
Deep link `#build` opens straight into the builder.

## The opponent is a policy *and* its deck

There is no separate "agent deck" choice. A checkpoint was trained piloting the
60 cards packaged next to it, and handing it a different list measures neither
the policy nor the deck — so one dropdown entry names both.

Opponents are **registered**: one directory each under `simulator/agents/`,
holding the full inference code the competition would run.

    simulator/agents/grimmsnarl_ex_xattn/
        main.py          def agent(obs_dict) -> list[int]   <- all the simulator calls
        deck.csv         the 60 card ids it was trained on
        actor_critic.pt  the weights
        policy_network/  the generation of the policy modules those weights fit
        cg/              the bundle's own engine bindings

To add an opponent, drop another bundle directory in; the dropdown is built from
whatever is there (`opponents.discover_bundles`). Nothing else is scanned. The
simulator knows only `main.py` and its `agent()` function, so any agent that
answers a competition observation can be registered without the simulator
learning anything about it.

Bundles elsewhere in the repo are deliberately not discovered any more. Several
of them fail to load or never answer, and one of those was what left **New game**
stuck on "starting…" — the server was blocked forever on a worker that would
never reply. Both waits are now bounded (`START_TIMEOUT`, `MOVE_TIMEOUT`), the
worker sends a `ready` handshake once its policy is loaded, and a bundle that
cannot load fails the *start* with its own error message instead of hanging.

Each bundle still runs in its own process (`agent_worker.py`, one JSON
observation per line in, one selection per line out): bundles import their policy
modules by bare name and ship their own `cg`/`libcg.so`, so whichever loaded
first in a shared interpreter would capture those names and run the next
bundle's weights through the wrong architecture. If a bundle raises mid-game the
worker answers with a legal random move — a weak move beats ending your game —
and the lapse reaches the browser console as `agentError`.

## The board

The page is laid out as the physical playmat, not as two lists of cards: the
two Active Spots face each other across the centre line, each player's five
Bench slots sit behind their Active, Prizes stack on the outer left and the deck
and discard pile on the outer right, with your hand along the bottom and the
Stadium on the centre line. The opponent's half is the same grid mirrored.
Empty slots are drawn rather than omitted, so a bench of two reads as two of
five, and a taken prize leaves a gap in the 2x3 block.

Cards are the real scans from `card_images_en/`, one file per card id, served
by the app at `/cards/`. The picture is the card; overlaid on it is only what
the print cannot know — current HP and the damage taken, attached Energy as
pips, tools, Special Conditions, and a "new" marker on a Pokémon played this
turn. Hovering any face-up card enlarges the art next to its full rules text,
attacks and costs. A card with no file on disk falls back to the drawn text
face, which is also what face-down backs use, so the board still works if the
directory is absent.

The opponent's hand is a row of face-down backs, one per card, the way it looks
across a table — counting them is the same information the number was, in the
form the game gives it.

What the mat alone cannot show sits in the nameplates: the four once-per-turn
rights (energy attach, supporter, stadium, retreat) are struck through as this
turn spends them, next to hand and deck counts and any Special Conditions.

Two prompts in a row can look identical — Hilda searches the deck twice, once
for an Evolution Pokémon and once for an Energy card, under the same title —
and both allow taking nothing. A second click meant for the first prompt used to
land on the second and silently answer it "nothing", so Hilda fetched only half
of what it should. Clicks are now ignored for 400 ms after a selection is sent,
and the button reads **Take nothing** rather than an enabled `Confirm (0/1)`, so
skipping a search is something you choose rather than something you fall into.

**Confirm** is wired in `renderChoice`, next to the state it sends, rather than
once at startup — a startup wiring line was lost in an edit and left every
multi-select confirm (Ultra Ball's discard, a bench search) enabled but dead,
which no amount of checking *whether the button was reachable* could catch. The
regression check is now `full_game.py`-style: play a whole game through the UI
and assert it reaches a result.

The choice panel and the log split the column on fixed terms: the log takes a
constant slice of the viewport (`clamp(150px, 28vh, 320px)`) and scrolls inside
it, and the choice panel keeps the rest with a 220px floor. Letting the log size
itself meant it grew all game and squeezed the option list — down to 11 pixels by
turn four — which is what made **End turn** look missing when there was nothing
left to do. (It is always offered: 103 of 103 Main decisions across two games
carried an End-turn option.) With 400 log lines, twice what the server ever
sends, the option box measures the same as it does on turn one.

The panel is also pinned to the viewport and never taller than it — the option
list and the log scroll inside their own boxes — so **Confirm** is always on
screen. It used to sit below the fold on a short window, which reads as a button
that cannot be clicked. Clicking a board slot that several options share now
takes the next one not yet picked (and gives one back when they are all taken),
so a slot click always moves a multi-select towards the count Confirm needs. And
when the engine refuses a selection that passed the count check,
`engine.raw_select` keeps its error code (`cg.game.battle_select` collapses them
all into an empty `IndexError`) and the reason appears next to the prompt with
the picks intact, instead of as an empty message in the corner.

The option list is grouped under headings — Pokémon, Evolve, Energy, Pokémon
Tool, Trainer, Ability, Attack, Misc — in the order a turn takes rather than the
order the rules happened to walk. The grouping is by what the *card* is, not by
the engine's option type: playing a Basic and playing a Supporter are both
`PLAY`, and a player reading the list thinks of them as different moves. A
prompt whose options all fall in one group keeps its flat list, since a lone
heading says nothing.

Every legal choice is both a row in the side panel and a highlight on the board:
the slots an option touches are outlined, hovering one highlights the matching
row, and clicking the slot takes the option when only one lives there. Options
that name two places at once -- attaching or evolving, where a card leaves your
hand and lands on a Pokémon -- light up both. Option text is resolved the same
way, so a bench target reads "Attach Basic {D} Energy to Munkidori (bench)"
rather than "a card". Discard piles are clickable, and hovering any face-up card
shows its full rules text, attacks and costs.

## Watching what happened

A turn the opponent took is otherwise invisible: by the time the board comes
back, the Poké Pad has been played, the attack has resolved, and all you have is
a changed number. So each view carries the events behind its log lines
(`render.log_events`) and the board acts them out in order:

- a card played is held up in a banner with its art — the agent's whole turn
  replays as a sequence of these, which is what makes its decisions legible;
- an attack announces itself (name and damage) while the attacking Pokémon
  lunges;
- HP changes float off the Pokémon they happened to — red `-120`, green `+80` —
  and the card flashes as it takes them;
- evolutions pop, attachments flash, coin flips and Special Conditions get a
  banner.

**Check which build you are running.** The header shows `build <n>` and the
console prints the same on load. If that number does not change after you
restart the server, the page is running from cache and nothing described here is
loaded. The asset URLs are versioned by file mtime (`/static/app.js?v=…`) so
that cannot happen any more: a changed file gets a changed URL, which no cache
can answer. `no-cache` headers alone were not enough — a browser that cached
`app.js` *before* those headers existed kept serving its old copy without
asking, which is how a paced animation could still look instant on your screen
while measuring correctly on mine.

**Your options are locked while the opponent plays.** A turn arrives as a batch
of events, and acting immediately used to cancel the rest of the replay — the
very thing it exists to show, and easy to do by accident. The panel reads
*Opponent is playing…* until the replay ends.

There is deliberately nothing on screen to skip with. A button in the panel and
a click-anywhere-over-the-board shortcut both existed and both were too easy to
hit by accident, and losing the turn you were waiting to watch is the one thing
this must not do. The **S** key still cuts a replay short — a key is not
something you press by mistake while watching.

**A step is one move: the board, what was done to it, the board again.** The
engine hands over a turn as one batch, but it makes it one decision at a time —
so each decision is recorded with the log entries it wrote *and* the board it
left behind, and the browser replays them in that shape:

    board  →  the move, narrated  →  the board it produced  →  hold  →  next

That pairing is the engine's own, which is why it needs no alignment. An earlier
version sent a flat event list and separately-positioned boards and tried to
line the two up by counting log entries; every miscount showed as a board that
lagged the narration and then jumped — the whole turn arriving at once at the
end. Slices come from the human's own batch, so nothing here decides what may be
shown: the engine already redacted it for this seat.

**The board moves with the narration, one action at a time.** This is the part
that took three attempts to get right. The pop-ups were paced from the start,
but the board was drawn from the *end-of-turn* position the moment the response
arrived — so the banner said "played Poké Pad" over cards that had already made
every move of the turn. Pacing an overlay over a board that has finished moving
is not an animation.

`engine.Battle.advance` now takes a board snapshot after **each** of the agent's
decisions (`render.board_snapshot`, hands stripped — the agent's is private and
the human's is not in that observation at all), and each snapshot is tagged with
its position in the log. The browser rewinds to the board as it stood before the
agent moved and walks it forward as the narration plays, so the cards arrive one
by one with the sentence describing them, and the real position is drawn at the
end. Measured from the DOM:

    +31497 ms  SAY    Opponent · Card moved · hand → bench
    +31497 ms  BOARD  opp active=Snorunt bench=[Munkidori] hand=6
    +32602 ms  SAY    Opponent · Draw · drew a card
    +32603 ms  BOARD  opp ... hand=5
    +33604 ms  SAY    Opponent · Spikemuth Gym · played

**The turn that ends the match is still shown.** Normally the human's own next
observation carries the whole of the agent's turn, so nothing has to be kept as
it happens. When the match *ends* on the agent's turn there is no next
observation, and its final one holds only the slice since the agent's previous
decision — which is how a game could end on "You lose" with the attack, the
damage and the Knock Out that caused it never shown at all. The turn is now also
accumulated as it happens, redacted through `public_logs`, and used only when
the human is never asked again; the human's own view supersedes it otherwise, so
a turn is never shown twice.

**A Knock Out resolves in order.** The attack is announced, the damage lands,
the Pokémon goes to the discard pile, the prize is taken — and only then does
the verdict appear. Three separate faults used to collapse that into one jump:

- The batch opened on the board the human's own action had *already* reached, so
  an attack could show its damage before it was animated. `select` now records
  the position from before the choice was applied.
- Every board landed past the last event, so nothing moved until the replay was
  over. The agent's first observation reaches back over everything the browser
  has already been shown since the agent last moved; `take_logs` counts exactly
  that, and `take_steps` subtracts it.
- The whole final turn was *filtered away*: when a match ends on the agent's
  turn its observation is the only record, and `public_logs` dropped every entry
  attributed to that seat. It now redacts instead — a draw and a move into a
  private zone lose their card identity, everything the table can see is kept —
  so the attack that ends the game is still narrated.

The end of the match is itself an event, so the verdict is the closing beat of
the replay rather than an announcement over a turn still being played out.

**Cards travel between zones.** A snapshot swap alone means a drawn card simply
appears in hand and a Knocked Out Pokémon simply stops existing, so each move
also sends a copy of the card flying from where it was to where it is going —
deck to hand on a draw (fanned, up to four, when several are drawn at once),
board to discard on a Knock Out, shrinking and fading on the way. The travelling
card is face-up only when the move is public and the card is known: a card going
into the opponent's hand flies face-down, because it is not ours to show. Only
the engine's own movement entries are animated, never the "played" line as well,
so a card is never sent twice or to the wrong pile.

**Every action gets at least a second.** The opponent's whole turn arrives in one
response, so this pacing is the only thing between you and a board that changes
all at once: `MIN_BEAT` is 1 s at normal speed, and the actions worth dwelling on
get more (an attack 1.8 s, a played card 1.4 s). There is no length-based
compression — a turn with twelve actions in it takes twelve beats to watch,
which is the point. The replay covers the whole turn, not just its highlights:
draws, deck searches, shuffles, switches, mulligans and card movements are shown
alongside plays and attacks, which is what makes a Supporter like Lillie's
Determination or Unfair Stamp *visible* rather than a board that silently
changed. Runs of bookkeeping are merged so this stays watchable — six draws
become one "drew 6 cards" beat, and three 10-damage counters on one Pokémon
become a single -30 — which takes a 37-event turn down to 8 beats. **Replay speed** in the header (slow / normal / fast,
remembered in `localStorage`) is how you shorten it, and the status bar counts
the steps: `replaying the opponent's turn — 3/9`.

Events name their card by `serial`, which is what the rendered board carries, so
an animation anchors to the right slot without any position being sent — and a
card that has already left play simply floats its number from the centre. A
long agent turn plays at double speed, and a newer view cancels whatever is
still queued. All of it is overlay: the board underneath is already the current
position, so an interrupted or dropped animation can never leave it wrong.

One gap is honest rather than papered over: **the engine emits no log entry for
an Ability**, so an opponent's ability use is not in the observation and nothing
is shown for it. Your own is announced from your click, which is the only place
that information exists.

A Pokémon in play can be *face-down*, which the engine reports as a slot that
exists but holds nothing identifiable (`active == [None]`, as against `[]` for a
genuinely empty Active Spot). That is where both players stand through Set Up,
and where the opponent stays until the reveal. The board draws a card back for
it — collapsing the two states made Set Up look like an empty mat and the
Pokémon appear from nowhere on the first turn. Your own Active is face-down to
you as well during Set Up: the engine does not put its identity in your
observation, exactly as the card lies face-down on the table.

## Imperfect information

The engine emits observation JSON for whichever seat owns the current decision,
with the other seat's hand, deck and prize identities erased
(`State::erasePlayerData`, `ToJson.h`). This server keeps **only the human
seat's** observations (`engine.Battle.view_obs`) and throws the agent's away, so
the payload the browser receives has no opponent hand, no deck order and no
prize contents in it — there is nothing hidden client-side that a devtools
window could uncover. Your own prizes are face-down to you too, exactly as in
the physical game. While the agent is choosing, the board you see is the last
position you legitimately observed.

The one place this needs care is a match that *ends* on the agent's turn: that
result lives in the agent's final observation. `engine.public_logs` takes only
the entries attributed to you plus the `RESULT` entry from it, and drops the
rest.

## Files

| file | what it does |
| --- | --- |
| `server.py` | FastAPI app: `/api/config`, `/api/decks`, `/api/cards`, `/api/deck/check`, `/api/new`, `/api/state`, `/api/select`, plus `/cards/` art |
| `engine.py` | one live battle, driven seat by seat; deck loading; log filtering |
| `render.py` | raw observation → the view model the browser draws (card names, option sentences, log lines) |
| `opponents.py` | the registry under `agents/`, and the pipe to a bundle's process |
| `agents/` | one directory per registered opponent (`main.py`, `deck.csv`, weights, `ARCH.json`) |
| `describe_agent.py` | reads a checkpoint and writes the `ARCH.json` the picker shows |
| `agent_worker.py` | one bundle, loaded in its own interpreter, answering selections |
| `static/` | the page (`index.html`, `app.js`, `style.css`), served `no-cache` so an edited asset survives a plain reload |
| `ptcg_engine/` | the whole engine: `src/` (C++) and the ctypes bindings; `build_engine.sh` makes `libcg.so` |
| `decks/` | the prebuilt decklists, one card id per line |
| `sessions.py` | the table of live sessions: one worker process per player, capped and reaped |
| `game_worker.py` | one battle, in its own interpreter, answering the server over a pipe |
| `deploy/` | the Dockerfile, Space card and push script that put this on Hugging Face |
| `card_images_en/` | card scans, kept out of git (683 MB) — see above |

`render.py` began as the draft in `training_PPO_grimsnarl_ex/webplay/`; its log
renderer was rewritten here against the real `LogType` fields (`value`, `head`,
`isRecover`, the `*_REVERSE` variants).

## One game per process — one process per player

`cg.sim.Battle` keeps the native battle pointer on a class attribute, so a
process holds exactly one battle. That is a hard constraint of the engine, not a
choice, and it used to make the server single-game: one module-level battle
behind a lock, where starting a game ended the previous one. Fine at your own
keyboard, wrong the moment the URL is one two people can open — the second
visitor's *New game* would silently reset the first's.

So the server holds no battle at all. It holds a table of `game_worker.py`
processes, one per session, and routes each request to the caller's own
(`sessions.py`). The battle logic did not change: `game_worker.py` is the
`_battle`/`_view` code that used to sit in `server.py`, moved behind a pipe. The
isolation is the process boundary rather than any new locking, which is the only
kind the engine allows.

    browser  --X-Session: abc-->  server.py  --pipe-->  game_worker.py  --pipe-->  agent_worker.py
                                     |                  (one battle)              (torch, one bundle)
                                     +--> another session's worker, another game

Read-only endpoints — the card pool, the decklists, the opponent sheets — are
still answered in the server process, because they only read the engine's static
tables and no battle is involved.

A session is identified by an id the page generates and sends as `X-Session`,
kept in `localStorage` so a reload returns you to your game. Deliberately not a
cookie: deployed as a Space the page runs in an iframe on `huggingface.co`, where
its own cookies are third-party and dropped by default, and every visitor would
collapse into one session. Closing the tab posts `/api/leave` so a slot is freed
at once rather than at the idle timeout.

Two knobs, because a game costs a worker process plus a torch process — roughly
1 GB once loaded — and running out of memory kills every game rather than the one
that walked away:

| variable | default | what it does |
| --- | --- | --- |
| `PTCG_MAX_SESSIONS` | 4 | games at once; beyond it `/api/new` answers 503 saying the server is full, instead of resetting someone's match |
| `PTCG_IDLE_TIMEOUT` | 1200 | seconds a game may sit untouched before its processes are reclaimed |

`GET /api/sessions` reports `{"live": n, "max": n, "playing": n}` — the one call
worth making against a deployment that is misbehaving.

## Playing it from anywhere

For a game or two with a friend, put the local server on the internet through a
Cloudflare quick tunnel — no account, no domain, no card, and no published
bandwidth cap, which matters because a single game pulls a few MB of card scans:

    python server.py            # one terminal
    ./tunnel.sh                 # another; prints an https://….trycloudflare.com

Install `cloudflared` once, anywhere on `PATH`:

    curl -fsSL -o ~/.local/bin/cloudflared \
      https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    chmod +x ~/.local/bin/cloudflared

That address is public and unauthenticated, and it is random and different every
time — that is what a quick tunnel is. `tunnel.sh` runs cloudflared's metrics
server so `/api/public-url` can read the hostname back, which is what makes a
room invitation carry a link to *this* machine rather than to the guest's own.
For a hostname that stays put, put a domain on Cloudflare, make a named tunnel
(`cloudflared tunnel create`) and set `PTCG_PUBLIC_URL`; that wins over both
quick tunnels. If the link is slow, `PTCG_LIGHT_IMAGES=1` re-encodes the scans.

`deploy/` turns this directory into a Hugging Face Space (Docker SDK):

    hf auth login
    ./deploy/push_to_space.sh <user>/<space> [--private]

The image compiles the engine from `ptcg_engine/src` in a build stage — the same
`build_engine.sh` a checkout runs — and carries the card scans, which go up once
via LFS and are skipped by hash on later pushes. `deploy/README.md` has the
details, the environment variables and how to run the same image locally.
