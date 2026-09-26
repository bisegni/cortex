#!/usr/bin/env bash
set -euo pipefail
command -v nats-server >/dev/null || { echo "Install NATS first: brew install nats-server"; exit 1; }
cargo build
trap 'kill 0' EXIT
nats-server >/tmp/cortex-nats.log 2>&1 &
sleep 1
cargo run -p cortex-node -- perceptual &
cargo run -p cortex-node -- memory &
cargo run -p cortex-node -- associative &
cargo run -p cortex-node -- workspace &
echo
echo "Start these in two other terminals:"
echo "  cargo run -p cortex-node -- visual"
echo "  cargo run -p cortex-node -- language"
echo
echo "Starting observer TUI..."
cargo run -p cortex-tui
