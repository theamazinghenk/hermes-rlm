#!/bin/zsh
# Reliable Hermes gateway restart.
#
# Why this exists: `launchctl kickstart -k` starts the replacement while the
# old instance's sockets can still be in TIME_WAIT. The api_server treats a
# bind failure as PERMANENT (retryable=False in gateway/platforms/
# api_server.py), so the gateway then runs for hours WITHOUT its API —
# observed 2026-08-06: 9 hours, health down, nobody noticed because Telegram
# kept working.
#
# So: stop, wait until the port is genuinely free, start, verify health, and
# retry once if the API did not come up.
#
#   hermes-restart.sh [label] [port]
#   hermes-restart.sh --detached [label] [port]   # survives its caller
#
# --detached matters when an agent restarts its own gateway: SIGTERM
# propagates to the process group, so a foreground restart kills the very
# process that issued it.
set -u

DETACHED=0
if [[ "${1:-}" == "--detached" ]]; then DETACHED=1; shift; fi
LABEL="${1:-ai.hermes.gateway}"
PORT="${2:-8642}"
DOMAIN="gui/$(id -u)"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
LOG="$HOME/.hermes/logs/hermes-restart.log"
mkdir -p "$(dirname "$LOG")"

log() { printf '%s %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$*" | tee -a "$LOG"; }

if (( DETACHED )); then
  # setsid detaches from this process group so the caller's own death (or the
  # SIGTERM it is about to receive) cannot take the restart down with it.
  setsid "$0" "$LABEL" "$PORT" >>"$LOG" 2>&1 &
  disown 2>/dev/null
  log "restart detached for $LABEL (follow: tail -f $LOG)"
  exit 0
fi

port_holders() { lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null; }

log "== restart $LABEL (port $PORT)"
launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null
sleep 2

# Wait for the port to clear; kill only leftover gateway processes, never
# anything else that happens to hold it.
for i in {1..30}; do
  holders="$(port_holders)"
  [[ -z "$holders" ]] && break
  for pid in ${=holders}; do
    cmd="$(ps -p "$pid" -o command= 2>/dev/null)"
    case "$cmd" in
      *hermes_cli.main*gateway*)
        log "killing stale gateway listener pid=$pid"
        kill -TERM "$pid" 2>/dev/null
        sleep 2
        kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null
        ;;
      *) log "port held by a non-gateway process pid=$pid — leaving it alone" ;;
    esac
  done
  sleep 1
done

[[ -n "$(port_holders)" ]] && log "warn: port $PORT still held after 30s; starting anyway"

if command -v launchctl >/dev/null 2>&1; then
  launchctl bootstrap "$DOMAIN" "$PLIST" 2>/dev/null || launchctl kickstart "$DOMAIN/$LABEL" 2>/dev/null
elif command -v systemctl >/dev/null 2>&1; then
  systemctl --user restart "$LABEL" 2>/dev/null || systemctl restart "$LABEL" 2>/dev/null
else
  # Container: the gateway is PID 1 and the container supervisor restarts it.
  log "no launchctl/systemctl — signalling PID 1 for supervisor restart"
  kill -TERM 1 2>/dev/null
fi
log "service started, waiting for health"

health_ok() { curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status": "ok"'; }

for attempt in 1 2; do
  for i in {1..60}; do
    health_ok && { log "healthy after ${i}0s (attempt $attempt)"; exit 0; }
    sleep 10
  done
  # The api_server gives up permanently on a bind failure, so a second pass
  # is the only cure: by now the port is certainly free.
  log "warn: no health after 10 min (attempt $attempt) — kickstarting once more"
  launchctl kickstart -k "$DOMAIN/$LABEL" 2>/dev/null
done

log "ERROR: gateway did not become healthy — check ~/.hermes/logs/gateway.error.log"
exit 1
