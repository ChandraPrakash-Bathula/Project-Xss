#!/usr/bin/env bash
# Start, stop and check the three local servers.
#   ./run.sh start [off|observe|block]   default: off
#   ./run.sh stop | status | logs
# Everything binds to 127.0.0.1 only.
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python
MODE="${2:-off}"
mkdir -p logs

APP_PORT="${APP_PORT:-5000}"

is_up() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

# A port being occupied does NOT mean our service is there. An unrelated app once held
# :5000 and the proxy happily forwarded every request to it, returning 404s that looked
# like "nothing executed". Always confirm identity with a known route.
is_ours() {  # port, path
  [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$1$2" 2>/dev/null)" = "200" ]
}

whose_port() {
  local pid; pid=$(lsof -nP -iTCP:"$1" -sTCP:LISTEN 2>/dev/null | tail -1 | awk '{print $2}')
  [ -n "$pid" ] && lsof -p "$pid" 2>/dev/null | awk '/cwd/{print $NF}' | head -1
}

wait_for() {  # port, name, probe-path, max seconds
  for _ in $(seq 1 "${4:-15}"); do
    if is_ours "$1" "$3"; then echo "  ok   $2 on :$1"; return 0; fi
    sleep 1
  done
  if is_up "$1"; then
    echo "  FAIL :$1 is held by something else, not $2"
    echo "       owner cwd: $(whose_port "$1")"
    echo "       free it, or re-run with a different port, e.g. APP_PORT=5010 ./run.sh start"
  else
    echo "  FAIL $2 did not come up on :$1 (see logs/)"
  fi
  return 1
}

start() {
  [ -x "$PY" ] || { echo "no venv found. run: python3 -m venv venv && ./venv/bin/pip install -r requirements.txt"; exit 1; }
  case "$MODE" in off|observe|block) ;; *) echo "mode must be off, observe or block"; exit 1 ;; esac
  if [ "$MODE" != "off" ] && [ ! -f models/baseline.joblib ]; then
    echo "mode '$MODE' needs models/baseline.joblib. run: $PY ml/train.py"; exit 1
  fi

  echo "starting (proxy mode: $MODE, app port: $APP_PORT)"
  is_ours "$APP_PORT" /reflect-text?q=ping || \
    { APP_PORT="$APP_PORT" $PY app/vuln_app.py >logs/app.out 2>&1 & }
  wait_for "$APP_PORT" "vulnerable app" "/reflect-text?q=ping" || exit 1

  if is_ours 8080 /reflect-text?q=ping; then echo "  ..   proxy already on :8080, leaving it alone"
  else
    MODE="$MODE" LOG=logs/proxy.jsonl APP_PORT="$APP_PORT" \
      $PY proxy/proxy.py >logs/proxy.out 2>&1 &
    wait_for 8080 "proxy ($MODE)" "/reflect-text?q=ping" || exit 1
  fi

  is_ours 5001 / || { APP_PORT="$APP_PORT" $PY teach/playground.py >logs/playground.out 2>&1 & }
  wait_for 5001 "playground" "/" || exit 1

  echo
  echo "  open  http://127.0.0.1:5001/          playground"
  echo "  open  http://127.0.0.1:5001/results   results and figure"
  echo "  stop  ./run.sh stop"
}

stop() {
  pkill -f 'teach/playground.py' 2>/dev/null
  pkill -f 'proxy/proxy.py'      2>/dev/null
  pkill -f 'app/vuln_app.py'     2>/dev/null
  sleep 1; echo "stopped"; status
}

status() {
  check() {  # port, path, name
    if is_ours "$1" "$2"; then echo "  up    :$1  $3"
    elif is_up "$1"; then echo "  BUSY  :$1  held by something else ($(whose_port "$1"))"
    else echo "  down  :$1  $3"; fi
  }
  check "$APP_PORT" "/reflect-text?q=ping" "vulnerable app"
  check 8080 "/reflect-text?q=ping" "proxy"
  check 5001 "/" "playground"
}

case "${1:-start}" in
  start)  start ;;
  stop)   stop ;;
  status) status ;;
  logs)   tail -n 20 logs/app.out logs/proxy.out logs/playground.out 2>/dev/null ;;
  *)      echo "usage: ./run.sh [start [off|observe|block] | stop | status | logs]" ;;
esac
