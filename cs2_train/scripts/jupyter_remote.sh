#!/usr/bin/env bash
# Launch a Jupyter Lab server on the EC2 box and open an SSH tunnel locally.
#
# Usage from your laptop:
#   ./cs2_train/scripts/jupyter_remote.sh up      # start server + tunnel; print URL
#   ./cs2_train/scripts/jupyter_remote.sh status  # report what's running
#   ./cs2_train/scripts/jupyter_remote.sh down    # close tunnel + stop server
#   ./cs2_train/scripts/jupyter_remote.sh tunnel  # only re-open the tunnel (server already up)
#
# The remote server listens on 127.0.0.1, so only your laptop (via the SSH
# tunnel) can reach it. We default LOCAL_PORT=8889 so the tunnel doesn't
# collide with a local jupyter on :8888.

set -euo pipefail

PEM=${PEM:-$HOME/Downloads/csgo-test-instance.pem}
HOST=${HOST:-ubuntu@ec2-54-242-204-184.compute-1.amazonaws.com}
LOCAL_PORT=${LOCAL_PORT:-8889}
REMOTE_PORT=${REMOTE_PORT:-8888}
PROJECT_DIR=${PROJECT_DIR:-/home/ubuntu/cs2}
SOCK=${SOCK:-/tmp/cs2-jupyter-tunnel.sock}
TMUX_SESSION=${TMUX_SESSION:-cs2-jupyter}

print_url() {
  echo "Jupyter URL:"
  echo "  http://localhost:${LOCAL_PORT}/lab?token=cs2"
}

start_remote() {
  ssh -i "$PEM" "$HOST" "
    set -e
    if tmux has-session -t ${TMUX_SESSION} 2>/dev/null; then
      echo 'remote tmux session ${TMUX_SESSION} already exists'
    else
      tmux new-session -d -s ${TMUX_SESSION} -c $PROJECT_DIR \\
        \"uv run --project $PROJECT_DIR jupyter lab \\
          --ip 127.0.0.1 --port $REMOTE_PORT \\
          --no-browser \\
          --ServerApp.token=cs2 \\
          --ServerApp.password='' \\
          --ServerApp.allow_remote_access=False \\
          2>&1 | tee $PROJECT_DIR/jupyter.log\"
      sleep 4
      echo 'started:'
      tmux capture-pane -p -t ${TMUX_SESSION} -S -10 | tail -5
    fi
  "
}

open_tunnel() {
  # If the control socket exists but is stale, clean it up first.
  if [ -S "$SOCK" ]; then
    if ssh -O check -o ControlPath="$SOCK" "$HOST" 2>/dev/null; then
      echo "tunnel already up (socket: $SOCK)"
      return 0
    fi
    rm -f "$SOCK"
  fi
  # ServerAliveInterval keeps the tunnel alive across short network drops.
  # ExitOnForwardFailure makes us fail fast if the local port is busy.
  ssh -i "$PEM" -N -f \
    -o ControlMaster=yes \
    -o ControlPath="$SOCK" \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "$HOST"
  echo "tunnel: localhost:${LOCAL_PORT} -> remote 127.0.0.1:${REMOTE_PORT}"
}

case "${1:-up}" in
  up)
    echo "==> Ensuring remote Jupyter is running"
    start_remote
    echo "==> Opening tunnel"
    open_tunnel
    print_url
    ;;
  tunnel)
    echo "==> Opening tunnel only"
    open_tunnel
    print_url
    ;;
  status)
    echo "==> Tunnel:"
    if [ -S "$SOCK" ] && ssh -O check -o ControlPath="$SOCK" "$HOST" 2>/dev/null; then
      echo "  up via $SOCK -> $HOST  (local :${LOCAL_PORT})"
    else
      echo "  not connected"
    fi
    echo "==> Remote tmux:"
    ssh -i "$PEM" "$HOST" "tmux ls 2>/dev/null || echo '  (no tmux sessions)'"
    echo "==> Remote jupyter:"
    ssh -i "$PEM" "$HOST" "pgrep -af jupyter-lab | head -3 || echo '  (not running)'"
    print_url
    ;;
  down)
    echo "==> Closing tunnel"
    ssh -O exit -o ControlPath="$SOCK" "$HOST" 2>/dev/null || true
    rm -f "$SOCK"
    echo "==> Stopping remote Jupyter"
    ssh -i "$PEM" "$HOST" "tmux kill-session -t ${TMUX_SESSION} 2>/dev/null || true"
    echo "Done."
    ;;
  *)
    echo "Usage: $0 {up|tunnel|status|down}" >&2
    exit 1
    ;;
esac
