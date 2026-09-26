#!/usr/bin/env bash
set -euo pipefail
SESSION="${CORTEX_TMUX_SESSION:-cortex}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

need(){ command -v "$1" >/dev/null || { echo "Missing $1. Install with: brew install $1"; exit 1; }; }
need tmux
need nats-server
need cargo

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Cortex is already running. Attach with: tmux attach -t $SESSION"
  exit 0
fi

cargo build
tmux new-session -d -s "$SESSION" -n brain -c "$ROOT" "nats-server; exec $SHELL"
tmux split-window -t "$SESSION:brain" -h -c "$ROOT" "sleep 1; cargo run -p cortex-node -- visual; exec $SHELL"
tmux split-window -t "$SESSION:brain.0" -v -c "$ROOT" "sleep 1; cargo run -p cortex-node -- perceptual; exec $SHELL"
tmux split-window -t "$SESSION:brain.1" -v -c "$ROOT" "sleep 1; cargo run -p cortex-node -- memory; exec $SHELL"
tmux split-window -t "$SESSION:brain.2" -v -c "$ROOT" "sleep 1; cargo run -p cortex-node -- associative; exec $SHELL"
tmux split-window -t "$SESSION:brain.3" -v -c "$ROOT" "sleep 1; cargo run -p cortex-node -- workspace; exec $SHELL"
tmux select-layout -t "$SESSION:brain" tiled
tmux new-window -t "$SESSION" -n language -c "$ROOT" "sleep 1; cargo run -p cortex-node -- language; exec $SHELL"
tmux new-window -t "$SESSION" -n observer -c "$ROOT" "sleep 2; cargo run -p cortex-tui; exec $SHELL"
tmux select-window -t "$SESSION:observer"
echo "Cortex started in tmux session '$SESSION'."
echo "Detach: Ctrl-b d   Stop everything: ./scripts/stop.sh"
exec tmux attach -t "$SESSION"
