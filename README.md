# Cortex

**Cortex** is an experiment in building a digital brain from small, independent, continuously-learning cortical processes rather than a single monolithic model.

Seed-0 tests one question: **can local symbols and cross-cortex associations emerge from asynchronous experience without programming a semantic pipeline?**

## Seed-0 architecture

```text
 visual sensor ─> Visual Cortex ─┐
                                │
 keyboard ──────> Language ─────┼──── NATS signal bus ──── TUI observer
                                │          │
                                │    ┌─────┼──────────┐
                                │    v     v          v
                                └─> Perceptual   Memory   Associative
                                                   │          │
                                                   └────┬─────┘
                                                        v
                                                Global Workspace
```

Every cortex is a **separate macOS process** with local state and a local symbol namespace. NATS transports signals only; it contains no cognition and does not orchestrate a pipeline. The TUI is an observer.

Seed-0 intentionally uses tiny deterministic latent encoders instead of pretending they are already useful neural networks. That makes the architecture, symbol formation, temporal binding, memory and association loop testable before Metal models are introduced.

## Run on macOS

Prerequisites:

```bash
brew install rust nats-server
```

Build:

```bash
cargo build
```

Terminal 1:

```bash
./scripts/run-seed0.sh
```

Terminal 2:

```bash
cargo run -p cortex-node -- visual
```

Terminal 3:

```bash
cargo run -p cortex-node -- language
```

The visual Seed-0 sensor currently accepts observation tokens such as `cup-red-left`. They are **not labels**: the visual process hashes the raw observation into a local latent signature and emits a `Vxxxx` symbol. This is a temporary sensor seam; the next step is to replace it with webcam frames + an innate Metal feature network without changing the cortical protocol.

Try entering the same or slightly recurring visual observations. In the language terminal, type a word within ~2.5 seconds of the corresponding visual observation, three or more times. Watch the TUI. Perceptual symbols (`P...`), memory recalls (`M:...`) and learned associative symbols (`C:...<->...`) should begin to appear.

The important property is that no process calls another process. Each cortex reacts independently to signals on the bus.

## Signal model

Signals carry a source cortex, local symbol, activation strength, compact embedding, timestamp, event kind and optional context. Symbols are local: `V...`, `L...`, `P...`, `M...`, `C...`. There is deliberately no universal tokenizer or globally assigned semantic symbol.

## Seed-0 success criterion

A first qualitative success is visible when repeated co-experience causes an associative activation path that we did not explicitly program. The TUI should let us say:

> We did not program this activation sequence; the system learned the association.

## Next experiment

Replace the temporary visual sensor with a webcam and a small innate Metal feature extractor (edges, motion, color/texture, boundaries), while keeping the same NATS protocol. Then test symbol persistence under movement, rotation, occlusion and lighting changes.


## Current main: webcam + tmux

`main` now runs the trainable local cortical networks and feeds Visual Cortex from the native macOS camera. Install prerequisites with `brew install rust nats-server tmux`, then run `./scripts/run-seed0.sh`. It creates a `cortex` tmux session with NATS and each cortex in its own pane/window and opens the observer TUI. Detach with `Ctrl-b d`; reattach with `tmux attach -t cortex`; stop the entire brain with `./scripts/stop.sh`.

On first camera use, macOS must grant Camera access to the terminal application that owns tmux. If camera 0 is not the intended device, launch with e.g. `CORTEX_CAMERA=1 ./scripts/run-seed0.sh`.

## Inter-cortical learning

Cortex does **not** transport gradients between processes. Sensory processes provide afferent evidence. Internal cortices exchange representations and learned top-down predictions. Perceptual cortex compares a top-down prediction with the next sensory representation, emits a `prediction_error`, and uses the resulting surprise as a local plasticity modulator. Its own backpropagation remains entirely inside its network. Associative cortex learns co-activations and can emit `prediction` signals after a relation becomes recurrent. This deliberately separates inter-area teaching/modulation from local synaptic credit assignment and gives the TUI observable prediction/error events.

## Cortex-0 developmental training

Train each local cortex independently before starting the collaborative runtime:

```bash
bash scripts/train-cortex0.sh
```

Set `CORTEX_TRAIN_STEPS` to change the default 20,000 local updates per cortex. The pipeline trains Visual, Perceptual, Memory, Associative and Language sequentially and writes geometry-checked checkpoints under `weights/cortex-0/`. These files are deliberately ignored by git: they are learned artifacts, not source code. At runtime each cortex automatically loads its matching checkpoint and then remains locally plastic. Workspace is currently a selection mechanism and has no pretrained weights.

The current trainer is a deterministic developmental bootstrap, not semantic supervised training: Visual/Perceptual/Memory learn generic signal transformations, Associative learns paired-symbol integration, and Language learns byte-pattern encoding. This validates the separate-development → checkpoint → collaborative-learning lifecycle while leaving semantic concepts to later experience.
