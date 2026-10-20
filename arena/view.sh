#!/bin/bash
# Open the arena Inspect logs in the browser from the laptop. Idempotent: starts the viewer on the PC only if nothing listens on
# its port there, opens the tunnel only if nothing listens on the port here, then opens the browser. Rerunning reuses both.
#   ./view.sh                 # all of arena/logs
#   ./view.sh logs/qwen-n4    # one run, relative to arena/ (only applies when this call starts the viewer)
# HOST and PORT override the PC's ssh name and the port (7577: the PC's other Inspect viewers sit on 7575 and 7576). The tunnel
# is one `ssh -f -N` per laptop session; close it with `pkill -f "ssh -f -N -L $PORT"`. The viewer on the PC keeps running;
# stop it there with `pkill -f "inspect_ai view --log-dir $LOGS --port $PORT"`.
set -e
HOST=${HOST:-tav}
PORT=${PORT:-7577}
LOGS=${1:-logs}

ssh -o ConnectTimeout=10 "$HOST" "PORT=$PORT LOGS='$LOGS' bash -s" <<'EOF'
cd ~/wgmn/mats12_sprint/tricky_little_guys/arena
pid=$(ss -ltnp 2>/dev/null | awk -v p=":$PORT$" '$4 ~ p' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
if [ -n "$pid" ]; then
    echo "port $PORT on $(hostname) already served by: $(tr '\0' ' ' < /proc/$pid/cmdline)"
else
    nohup .venv/bin/python -m inspect_ai view --log-dir "$LOGS" --port "$PORT" > /tmp/inspect-view-$PORT.log 2>&1 < /dev/null &
    for i in $(seq 1 30); do ss -ltn | grep -q ":$PORT " && break; sleep 1; done
    ss -ltn | grep -q ":$PORT " || { echo "viewer did not start; see /tmp/inspect-view-$PORT.log on $(hostname)" >&2; exit 1; }
    echo "viewer started on $(hostname):$PORT for $LOGS"
fi
EOF

if ss -ltn | grep -q ":$PORT "; then
    echo "tunnel already up on localhost:$PORT"
else
    ssh -f -N -o ExitOnForwardFailure=yes -L "$PORT:localhost:$PORT" "$HOST"
    echo "tunnel opened: localhost:$PORT -> $HOST:$PORT"
fi
xdg-open "http://localhost:$PORT" >/dev/null 2>&1 || echo "open http://localhost:$PORT"
