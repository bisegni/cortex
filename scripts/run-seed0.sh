#!/usr/bin/env bash
# Build everything and open the Cortex dashboard. The dashboard starts and
# supervises nats-server plus all cortex nodes itself; quitting stops them.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

command -v cargo >/dev/null || { echo "Missing cargo (https://rustup.rs)"; exit 1; }
command -v nats-server >/dev/null || echo "note: nats-server not found; the dashboard will only work with an already-running server (brew install nats-server)"

cargo build --workspace
exec "$ROOT/target/debug/cortex-tui"
