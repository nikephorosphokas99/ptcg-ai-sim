#!/usr/bin/env bash
# Put the local server on the public internet through a Cloudflare quick tunnel.
#
# A quick tunnel needs no Cloudflare account, no domain and no credit card, and
# has no published bandwidth cap -- which is the reason to prefer it here, since
# a single game pulls a few MB of card scans and ngrok's free plan allows 1 GB a
# month. Set PTCG_LIGHT_IMAGES=1 on the server if the link is still slow.
#
#   ./tunnel.sh           # tunnels http://127.0.0.1:8000
#   ./tunnel.sh 8001      # ...or another port
#
# The address is random and changes every restart: that is what a quick tunnel
# is. For a hostname that stays put you need a domain on Cloudflare and a named
# tunnel (`cloudflared tunnel create`), and then PTCG_PUBLIC_URL on the server.
#
# The metrics server is not optional here. cloudflared prints the address to its
# log and nowhere else, so /api/public-url asks the metrics server for it --
# without one, room invitations would carry a link to the guest's own machine.
set -euo pipefail

port="${1:-8000}"
metrics="${PTCG_CLOUDFLARED_METRICS:-127.0.0.1:20241}"

command -v cloudflared >/dev/null || {
  echo "cloudflared not found. Install it with:" >&2
  echo "  curl -fsSL -o ~/.local/bin/cloudflared \\" >&2
  echo "    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64" >&2
  echo "  chmod +x ~/.local/bin/cloudflared" >&2
  exit 1
}

curl -fsS -o /dev/null --max-time 2 "http://127.0.0.1:${port}/" || {
  echo "nothing is answering on 127.0.0.1:${port} -- start the server first:" >&2
  echo "  python server.py --port ${port}" >&2
  exit 1
}

echo "tunnelling http://127.0.0.1:${port} ..."
cloudflared tunnel --url "http://127.0.0.1:${port}" --metrics "$metrics" &
cloudflared_pid=$!
trap 'kill "$cloudflared_pid" 2>/dev/null || true' EXIT

# Wait for the hostname rather than guessing how long the handshake takes.
for _ in $(seq 1 60); do
  # `|| true`: the metrics server is not up for the first second or two, and
  # under `set -e` a failing assignment would end the script -- killing the
  # tunnel through the trap before it ever came up.
  host="$(curl -fsS --max-time 1 "http://${metrics}/quicktunnel" 2>/dev/null \
          | sed -n 's/.*"hostname":"\([^"]*\)".*/\1/p')" || true
  [ -n "${host:-}" ] && break
  sleep 1
done

if [ -z "${host:-}" ]; then
  echo "cloudflared did not report a hostname; see its output above" >&2
  wait "$cloudflared_pid"
fi

echo
echo "  https://${host}"
echo
echo "That address is public and unauthenticated: anyone who has it can play,"
echo "watch your saved replays and read the decks. Ctrl-C closes it."
wait "$cloudflared_pid"
