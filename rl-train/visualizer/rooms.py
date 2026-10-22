from __future__ import annotations

import os
import random
import secrets
import threading
import time

import sessions

MAX_ROOMS = int(os.environ.get("PTCG_MAX_ROOMS", "8"))
ROOM_IDLE_TIMEOUT = float(os.environ.get("PTCG_ROOM_IDLE_TIMEOUT", str(30 * 60)))
NAME_MAX = 24


class RoomError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def clean_name(name: str | None) -> str:
    name = " ".join((name or "").split())[:NAME_MAX]
    if not name:
        raise RoomError(400, "please enter a name")
    return name


class Player:
    def __init__(self, name: str) -> None:
        self.name = name
        self.token = secrets.token_urlsafe(16)
        self.deck: list[int] | None = None
        self.deck_label = ""
        self.ready = False
        self.seat: int | None = None


class Room:
    def __init__(self, host_name: str) -> None:
        self.id = secrets.token_urlsafe(6)
        self.players: list[Player] = [Player(host_name)]
        self.worker: sessions.Session | None = None
        self.lock = threading.Lock()
        self.touched = time.monotonic()
        self.status = "lobby"
        self.error = ""

    def player(self, token: str | None) -> Player:
        for p in self.players:
            if token and secrets.compare_digest(p.token, token):
                return p
        raise RoomError(403, "you are not in this room")

    def lobby(self, me: Player) -> dict:
        return {
            "room": self.id,
            "status": self.status,
            "you": self.players.index(me),
            "players": [
                {"name": p.name, "ready": p.ready, "deck": p.deck_label, "host": i == 0}
                for i, p in enumerate(self.players)
            ],
            "full": len(self.players) == 2,
            "error": self.error,
        }

    def start(self) -> None:
        order = self.players[:]
        random.shuffle(order)
        for seat, p in enumerate(order):
            p.seat = seat
        worker = sessions.Session("room-" + self.id)
        reply = worker.request("pvp_new", {
            "decks": [p.deck for p in order],
            "names": [p.name for p in order],
        })
        if not reply.get("ok"):
            worker.close()
            raise RoomError(reply.get("status", 500), reply.get("detail", "could not start"))
        self.worker = worker
        self.status = "playing"

    def close(self) -> None:
        if self.worker is not None:
            self.worker.close()
            self.worker = None


class RoomRegistry:
    def __init__(self) -> None:
        self._rooms: dict[str, Room] = {}
        self._lock = threading.Lock()

    def _reap(self) -> None:
        now = time.monotonic()
        for rid, room in list(self._rooms.items()):
            if room.lock.locked():
                continue
            dead = room.worker is not None and room.worker.process.poll() is not None
            if dead or now - room.touched > ROOM_IDLE_TIMEOUT:
                del self._rooms[rid]
                room.close()

    def create(self, name: str) -> tuple[Room, Player]:
        name = clean_name(name)
        with self._lock:
            self._reap()
            if len(self._rooms) >= MAX_ROOMS:
                raise RoomError(503, f"{MAX_ROOMS} rooms are already open on this server — "
                                     f"try again in a few minutes")
            room = Room(name)
            self._rooms[room.id] = room
            return room, room.players[0]

    def get(self, rid: str | None) -> Room:
        with self._lock:
            self._reap()
            room = self._rooms.get(rid or "")
        if room is None:
            raise RoomError(404, "this room does not exist or has closed")
        room.touched = time.monotonic()
        return room

    def join(self, rid: str, name: str) -> tuple[Room, Player]:
        name = clean_name(name)
        room = self.get(rid)
        with room.lock:
            if len(room.players) >= 2:
                raise RoomError(409, "this room already has two players")
            if name.casefold() == room.players[0].name.casefold():
                name = f"{name} (2)"
            player = Player(name)
            room.players.append(player)
            return room, player

    def drop(self, rid: str) -> None:
        with self._lock:
            room = self._rooms.pop(rid, None)
        if room is not None:
            room.close()

    def count(self) -> int:
        with self._lock:
            self._reap()
            return len(self._rooms)

    def close_all(self) -> None:
        with self._lock:
            rooms = list(self._rooms.values())
            self._rooms.clear()
        for room in rooms:
            room.close()


registry = RoomRegistry()
