#!/usr/bin/env bash
set -euo pipefail
SESSION="${CORTEX_TMUX_SESSION:-cortex}"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
  echo "Cortex stopped: tmux session '$SESSION' and all its cortex processes were terminated."
else
  echo "Cortex is not running (no tmux session '$SESSION')."
fi
