from __future__ import annotations

import json
import os
import secrets
import select
import subprocess
import sys
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent

BOOT_TIMEOUT = 60.0
NEW_TIMEOUT = 240.0
REQUEST_TIMEOUT = 90.0

MAX_SESSIONS = int(os.environ.get("PTCG_MAX_SESSIONS", "4"))
IDLE_TIMEOUT = float(os.environ.get("PTCG_IDLE_TIMEOUT", str(20 * 60)))


class SessionBusy(Exception):
    pass


class SessionGone(Exception):
    pass


def new_session_id() -> str:
    return secrets.token_urlsafe(16)


class Session:
    def __init__(self, sid: str) -> None:
        self.sid = sid
        self.lock = threading.Lock()
        self.touched = time.monotonic()
        self.started = time.monotonic()
        self.playing = False
        self.process = subprocess.Popen(
            [sys.executable, str(_HERE / "game_worker.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1, cwd=str(_HERE),
        )
        hello = json.loads(self._readline(BOOT_TIMEOUT, "starting up"))
        if not hello.get("ready"):
            self.close()
            raise RuntimeError(f"game worker failed to start: {hello.get('error')}")

    def _readline(self, timeout: float, what: str) -> str:
        ready, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not ready:
            self.close()
            raise SessionGone(f"the game stopped responding while {what} "
                              f"(no answer in {timeout:.0f}s)")
        line = self.process.stdout.readline()
        if not line:
            code = self.process.poll()
            self.close()
            raise SessionGone(f"the game exited while {what} (code {code})")
        return line

    def request(self, op: str, payload: dict | None = None) -> dict:
        with self.lock:
            self.touched = time.monotonic()
            if self.process.poll() is not None:
                raise SessionGone(f"the game exited (code {self.process.returncode})")
            body = dict(payload or {}, op=op)
            timeout = NEW_TIMEOUT if op == "new" else REQUEST_TIMEOUT
            try:
                self.process.stdin.write(json.dumps(body) + "\n")
                self.process.stdin.flush()
            except (OSError, ValueError) as exc:
                self.close()
                raise SessionGone(f"the game is no longer reachable: {exc}") from exc
            reply = json.loads(self._readline(timeout, f"handling {op}"))
            if op == "new" and reply.get("ok"):
                self.playing = True
            self.touched = time.monotonic()
            return reply

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.process.stdin.close()
            except OSError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


class Registry:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def _reap(self) -> None:
        now = time.monotonic()
        for sid, session in list(self._sessions.items()):
            if session.lock.locked():
                continue
            dead = session.process.poll() is not None
            idle = now - session.touched > IDLE_TIMEOUT
            if dead or idle:
                del self._sessions[sid]
                session.close()

    def get(self, sid: str | None) -> Session | None:
        if not sid:
            return None
        with self._lock:
            self._reap()
            return self._sessions.get(sid)

    def open(self, sid: str | None) -> tuple[str, Session]:
        with self._lock:
            self._reap()
            if sid and sid in self._sessions:
                return sid, self._sessions[sid]
            if len(self._sessions) >= MAX_SESSIONS:
                raise SessionBusy(
                    f"{MAX_SESSIONS} games are already running on this server — "
                    f"try again in a few minutes, or run your own copy"
                )
            sid = sid or new_session_id()
            session = Session(sid)
            self._sessions[sid] = session
            return sid, session

    def drop(self, sid: str | None) -> None:
        if not sid:
            return
        with self._lock:
            session = self._sessions.pop(sid, None)
        if session is not None:
            session.close()

    def stats(self) -> dict:
        with self._lock:
            self._reap()
            return {
                "live": len(self._sessions),
                "max": MAX_SESSIONS,
                "idleTimeout": IDLE_TIMEOUT,
                "playing": sum(1 for s in self._sessions.values() if s.playing),
            }

    def close_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.close()


registry = Registry()
