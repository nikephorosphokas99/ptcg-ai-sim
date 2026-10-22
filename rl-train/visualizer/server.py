from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import opponents as agent_lib  # noqa: E402
import engine  # noqa: E402
import render  # noqa: E402
import replay_view  # noqa: E402
import rooms  # noqa: E402
import sessions  # noqa: E402

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    yield
    sessions.registry.close_all()
    rooms.registry.close_all()


app = FastAPI(title="PTCG player-vs-agent simulator", lifespan=_lifespan)

from fastapi.middleware.gzip import GZipMiddleware  # noqa: E402

app.add_middleware(GZipMiddleware, minimum_size=4096)

SESSION_HEADER = "X-Session"


class NewGame(BaseModel):
    agent: str
    humanDeck: str | None = None
    humanCards: list[int] | None = None
    seat: int = -1
    playerName: str | None = None


class Choice(BaseModel):
    options: list[int]


def _session_reply(response: Response, sid: str, reply: dict) -> dict:
    response.headers[SESSION_HEADER] = sid
    if not reply.get("ok"):
        raise HTTPException(reply.get("status", 500), reply.get("detail", "game error"))
    view = reply["view"]
    view["session"] = sid
    return view


@app.get("/api/config")
def config() -> dict:
    return {"decks": engine.deck_names(), "agents": agent_lib.available_agents(),
            "sessions": sessions.registry.stats()}


def _deck_entry(card_id: int, count: int) -> dict:
    return dict(render.card_info(card_id), count=count)


def _deck_listing(cards: list[int]) -> list[dict]:
    counts: dict[int, int] = {}
    for card_id in cards:
        counts[card_id] = counts.get(card_id, 0) + 1
    order = {"pokemon": 0, "item": 1, "tool": 2, "supporter": 3, "stadium": 4, "energy": 5}
    rows = [_deck_entry(card_id, count) for card_id, count in counts.items()]
    rows.sort(key=lambda r: (order.get(r["kind"], 9), r["name"]))
    return rows


def _archetype(cards: list[int]) -> dict:
    counts: dict[int, int] = {}
    for card_id in cards:
        counts[card_id] = counts.get(card_id, 0) + 1
    best = None
    for card_id, count in counts.items():
        card = render.card_info(card_id)
        if card["kind"] != "pokemon":
            continue
        rank = (bool(card["megaEx"]), bool(card["ex"]), count, -card_id)
        if best is None or rank > best[0]:
            best = (rank, card)
    if best is None:
        return {}
    card = best[1]
    return {"name": card["name"], "image": card["image"], "cardId": card["id"]}


@app.get("/api/agents")
def agents() -> dict:
    out = []
    for name, bundle in agent_lib.discover_bundles().items():
        cards = engine.read_deck(bundle / "deck.csv")
        out.append({
            "id": name,
            "archetype": _archetype(cards),
            "arch": agent_lib.agent_arch(bundle),
            "deck": _deck_listing(cards),
            "deckTotal": len(cards),
        })
    return {"agents": out}


@app.get("/api/decks")
def decks() -> dict:
    out = []
    for name in engine.deck_names():
        cards = engine.deck_by_name(name)
        out.append({"name": name, "total": len(cards), "cards": _deck_listing(cards)})
    return {"decks": out}


@app.get("/api/cards")
def cards() -> dict:
    return {"cards": [render.card_info(card_id) for card_id in sorted(render.card_db())]}


@app.post("/api/deck/check")
def deck_check(cards: list[int]) -> dict:
    return {"problems": engine.deck_check(cards, render.card_db()),
            "listing": _deck_listing(cards)}


@app.post("/api/new")
def new_game(request: NewGame, response: Response,
             x_session: str | None = Header(default=None)) -> dict:
    try:
        sid, session = sessions.registry.open(x_session)
    except sessions.SessionBusy as exc:
        raise HTTPException(503, str(exc)) from exc
    except (RuntimeError, OSError) as exc:
        raise HTTPException(500, f"cannot start a game process: {exc}") from exc
    try:
        reply = session.request("new", request.model_dump())
    except sessions.SessionGone as exc:
        sessions.registry.drop(sid)
        raise HTTPException(503, str(exc)) from exc
    return _session_reply(response, sid, reply)


def _forward(op: str, payload: dict, response: Response, sid: str | None) -> dict:
    session = sessions.registry.get(sid)
    if session is None:
        raise HTTPException(409, "no game in progress")
    try:
        reply = session.request(op, payload)
    except sessions.SessionGone as exc:
        sessions.registry.drop(sid)
        raise HTTPException(409, f"the game ended unexpectedly: {exc}") from exc
    return _session_reply(response, sid, reply)


@app.get("/api/state")
def state(response: Response, x_session: str | None = Header(default=None)) -> dict:
    return _forward("state", {}, response, x_session)


@app.post("/api/select")
def select(choice: Choice, response: Response,
           x_session: str | None = Header(default=None)) -> dict:
    return _forward("select", choice.model_dump(), response, x_session)


@app.post("/api/leave")
def leave(x_session: str | None = Header(default=None)) -> dict:
    sessions.registry.drop(x_session)
    return {"ok": True}


PLAYER_HEADER = "X-Player"


class RoomName(BaseModel):
    name: str


class RoomJoin(BaseModel):
    room: str
    name: str


class RoomReady(BaseModel):
    room: str
    ready: bool = True
    deck: str | None = None
    cards: list[int] | None = None


class RoomRef(BaseModel):
    room: str


class RoomChoice(BaseModel):
    room: str
    options: list[int]


def _room_call(fn):
    try:
        return fn()
    except rooms.RoomError as exc:
        raise HTTPException(exc.status, exc.detail) from exc
    except sessions.SessionGone as exc:
        raise HTTPException(409, f"the game ended unexpectedly: {exc}") from exc


def _room_game_raw(room: rooms.Room, token: str | None, op: str) -> dict:
    me = room.player(token)
    if room.worker is None or me.seat is None:
        raise rooms.RoomError(409, "the game has not started")
    reply = room.worker.request(op, {"seat": me.seat})
    if not reply.get("ok"):
        raise rooms.RoomError(reply.get("status", 500), reply.get("detail", "game error"))
    return reply["view"]


def _room_game(room: rooms.Room, token: str | None, op: str, payload: dict) -> dict:
    me = room.player(token)
    if room.worker is None or me.seat is None:
        raise rooms.RoomError(409, "the game has not started")
    reply = room.worker.request(op, dict(payload, seat=me.seat))
    if not reply.get("ok"):
        raise rooms.RoomError(reply.get("status", 500), reply.get("detail", "game error"))
    view = reply["view"]
    view["room"] = room.id
    return view


@app.post("/api/room/create")
def room_create(request: RoomName) -> dict:
    def run():
        room, me = rooms.registry.create(request.name)
        return {"room": room.id, "token": me.token, "lobby": room.lobby(me)}
    return _room_call(run)


@app.post("/api/room/join")
def room_join(request: RoomJoin) -> dict:
    def run():
        room, me = rooms.registry.join(request.room, request.name)
        return {"room": room.id, "token": me.token, "lobby": room.lobby(me)}
    return _room_call(run)


@app.get("/api/room/info")
def room_info(room: str) -> dict:
    def run():
        r = rooms.registry.get(room)
        return {"room": r.id, "host": r.players[0].name, "full": len(r.players) == 2,
                "status": r.status}
    return _room_call(run)


LONG_POLL = float(os.environ.get("PTCG_LONG_POLL", "25"))
POLL_STEP = 0.3


def _lobby_with_tag(r: rooms.Room, me: rooms.Player) -> dict:
    lobby = r.lobby(me)
    lobby["tag"] = hashlib.sha1(json.dumps(lobby, sort_keys=True).encode()).hexdigest()[:12]
    return lobby


@app.get("/api/room/lobby")
def room_lobby(room: str, since: str | None = None,
               x_player: str | None = Header(default=None)) -> dict:
    def run():
        r = rooms.registry.get(room)
        me = r.player(x_player)
        deadline = time.monotonic() + (LONG_POLL if since else 0)
        while True:
            lobby = _lobby_with_tag(r, me)
            if lobby["tag"] != since or time.monotonic() >= deadline:
                return lobby
            time.sleep(POLL_STEP)
            r = rooms.registry.get(room)
    return _room_call(run)


@app.post("/api/room/ready")
def room_ready(request: RoomReady, x_player: str | None = Header(default=None)) -> dict:
    def run():
        r = rooms.registry.get(request.room)
        with r.lock:
            me = r.player(x_player)
            if r.status != "lobby":
                raise rooms.RoomError(409, "the game has already started")
            if request.ready:
                if request.cards is not None:
                    deck = [int(c) for c in request.cards]
                    label = "own deck"
                elif request.deck is not None:
                    try:
                        deck = engine.deck_by_name(request.deck)
                    except KeyError as exc:
                        raise rooms.RoomError(400, f"unknown deck {exc}") from exc
                    label = request.deck
                else:
                    raise rooms.RoomError(400, "choose a deck first")
                problems = engine.deck_check(deck, render.card_db())
                if problems:
                    raise rooms.RoomError(400, "illegal deck: " + "; ".join(problems))
                me.deck, me.deck_label, me.ready = deck, label, True
            else:
                me.ready = False
            r.error = ""
            if len(r.players) == 2 and all(p.ready for p in r.players):
                try:
                    r.start()
                except (rooms.RoomError, RuntimeError, OSError) as exc:
                    for p in r.players:
                        p.ready = False
                    r.error = getattr(exc, "detail", str(exc))
            return _lobby_with_tag(r, me)
    return _room_call(run)


@app.get("/api/room/state")
def room_state(room: str, since: int | None = None,
               x_player: str | None = Header(default=None)) -> dict:
    def run():
        r = rooms.registry.get(room)
        if since is not None:
            deadline = time.monotonic() + LONG_POLL
            while time.monotonic() < deadline:
                probe = _room_game_raw(r, x_player, "pvp_version")
                if probe["version"] != since:
                    break
                time.sleep(POLL_STEP)
                r = rooms.registry.get(room)
        return _room_game(r, x_player, "pvp_state", {})
    return _room_call(run)


@app.post("/api/room/select")
def room_select(choice: RoomChoice, x_player: str | None = Header(default=None)) -> dict:
    return _room_call(lambda: _room_game(rooms.registry.get(choice.room), x_player,
                                         "pvp_select", {"options": choice.options}))


@app.post("/api/room/leave")
def room_leave(request: RoomRef, x_player: str | None = Header(default=None)) -> dict:
    def run():
        r = rooms.registry.get(request.room)
        me = r.player(x_player)
        if r.status == "lobby" and me is not r.players[0]:
            with r.lock:
                r.players.remove(me)
                for p in r.players:
                    p.ready = False
        else:
            rooms.registry.drop(r.id)
        return {"ok": True}
    return _room_call(run)


def _ngrok_url() -> str | None:
    try:
        with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=0.5) as r:
            tunnels = json.loads(r.read()).get("tunnels", [])
    except (OSError, ValueError):
        return None
    for t in tunnels:
        if str(t.get("public_url", "")).startswith("https://"):
            return t["public_url"]
    return None


CLOUDFLARED_METRICS = os.environ.get("PTCG_CLOUDFLARED_METRICS", "127.0.0.1:20241")


def _cloudflared_url() -> str | None:
    try:
        with urllib.request.urlopen(f"http://{CLOUDFLARED_METRICS}/quicktunnel", timeout=0.5) as r:
            host = json.loads(r.read()).get("hostname")
    except (OSError, ValueError):
        return None
    return f"https://{host}" if host else None


@app.get("/api/public-url")
def public_url() -> dict:
    url = (os.environ.get("PTCG_PUBLIC_URL")
           or _ngrok_url()
           or _cloudflared_url())
    return {"url": url.rstrip("/") if url else None}


@app.get("/api/replays")
def replays() -> dict:
    return {"replays": replay_view.listing()}


@app.get("/api/replays/view")
def replay(file: str, seat: int = 0) -> dict:
    if seat not in (0, 1):
        raise HTTPException(400, "seat must be 0 or 1")
    try:
        path = replay_view.resolve(file)
    except FileNotFoundError as exc:
        raise HTTPException(404, "no such replay") from exc
    try:
        return replay_view.frames(path, seat)
    except (OSError, ValueError, KeyError, IndexError) as exc:
        raise HTTPException(422, f"cannot read this replay: {type(exc).__name__}: {exc}") from exc


@app.get("/api/sessions")
def session_stats() -> dict:
    return sessions.registry.stats()


def _asset_stamp() -> str:
    newest = 0.0
    for name in ("app.js", "style.css", "index.html"):
        path = _HERE / "static" / name
        if path.is_file():
            newest = max(newest, path.stat().st_mtime)
    return str(int(newest))


@app.get("/api/version")
def version() -> dict:
    return {"build": _asset_stamp()}


@app.get("/")
def index() -> HTMLResponse:
    html = (_HERE / "static" / "index.html").read_text()
    stamp = _asset_stamp()
    html = html.replace("/static/app.js", f"/static/app.js?v={stamp}")
    html = html.replace("/static/style.css", f"/static/style.css?v={stamp}")
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


class _FreshStatic(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/static", _FreshStatic(directory=_HERE / "static"), name="static")

LIGHT_IMAGES = os.environ.get("PTCG_LIGHT_IMAGES", "") not in ("", "0", "false")
LIGHT_DIR = _HERE / ".image_cache"

if LIGHT_IMAGES and render.IMAGE_DIR.is_dir():
    from fastapi.responses import FileResponse

    @app.get("/cards/{name}")
    def light_card(name: str) -> FileResponse:
        source = render.IMAGE_DIR / Path(name).name
        if not source.is_file():
            raise HTTPException(404, "no such card image")
        target = LIGHT_DIR / (source.stem + ".webp")
        if not target.is_file() or target.stat().st_mtime < source.stat().st_mtime:
            from PIL import Image
            LIGHT_DIR.mkdir(exist_ok=True)
            Image.open(source).convert("RGB").save(target, format="WEBP", quality=80, method=4)
        return FileResponse(target, media_type="image/webp",
                            headers={"Cache-Control": "public, max-age=604800"})
elif render.IMAGE_DIR.is_dir():
    app.mount("/cards", StaticFiles(directory=render.IMAGE_DIR), name="cards")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
