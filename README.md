# Cortex

**Cortex** is an experiment in building a digital brain from small, independent, continuously-learning cortical processes rather than a single monolithic model.

Seed-0 tests one question: **can local symbols and cross-cortex associations emerge from asynchronous experience without programming a semantic pipeline?**

## Seed-0 architecture

```text
 visual sensor ─> Visual Cortex ─┐
                                │
 keyboard ──────> Language ─────┼──── NATS signal bus ──── TUI dashboard
                                │          │
                                │    ┌─────┼──────────┐
                                │    v     v          v
                                └─> Perceptual   Memory   Associative
                                                   │          │
                                                   └────┬─────┘
                                                        v
                                                Global Workspace
```

Every cortex is a **separate macOS process** with local state and a local symbol namespace. NATS transports signals only; it contains no cognition and does not orchestrate a pipeline. The TUI supervises processes and observes signals; it takes no part in cognition.

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


## Current main: webcam + dashboard

`main` runs the trainable local cortical networks and feeds Visual Cortex from the native macOS camera. Install prerequisites with `brew install rust nats-server`, then run `./scripts/run-seed0.sh`. It builds the workspace and opens the `cortex-tui` dashboard, which starts and supervises `nats-server` (an already-running server on the same port is reused) and every cortex node as child processes. Quitting the dashboard stops them all.

Tabs (`1`-`6`, `Tab`, or `F1`-`F6`): **Brain** (connectome heat, per-cortex neuron activity, workspace stream), **Chat** (type text to the language cortex; brain broadcasts, recalls and associations appear as replies), **Camera** (live downsampled webcam frame from the visual cortex), **Teach** (automatic teacher with live accuracy), **Processes** (status, logs, restart/stop), **Train** (run `cortex-train` and view output; press `R` to restart the nodes and load new weights). `q` twice quits; `Ctrl-C` quits immediately.

On first camera use, macOS must grant Camera access to the terminal application running the dashboard. If camera 0 is not the intended device, launch with e.g. `CORTEX_CAMERA=1 ./scripts/run-seed0.sh`. Text can also be sent from any shell with `nats pub cortex.input.text "hello"`.

### Teaching the brain to name what it sees

Signals carry a `ttl` (remaining relay hops) so a stimulus produces a bounded cascade instead of endless echo. The visual cortex uses a fixed innate "retina" (`cortex-core/src/curriculum.rs`: shape summary, width/height profiles, stripe texture) and learns a 16-dim latent from it. The associative cortex links image latents to words Hebbian-style (nearest-prototype vote, persisted in `weights/cortex-0/links.json`). The language cortex publishes the result on `cortex.answer`.

1. **Offline pretrain** (optional but recommended): `cargo run -p cortex-node --bin cortex-train -- curriculum` (or the Train tab) trains visual, perceptual and language nets on generated shapes (`circle square triangle cross hbars vbars`).
2. **Automatic teacher**: Teach tab, press `t`. It says a class name on `cortex.input.text` while injecting generated images on `cortex.visual.inject`, and periodically tests unseen images without the word, scoring the answers.
3. **Headless check**: with nats-server and all nodes running, `cargo run -p cortex-node --bin cortex-teacher -- --headless --episodes 120 --target 0.8` prints per-class accuracy and exits non-zero below target. Set `CORTEX_CAMERA=off` on the visual node to keep the camera out of the test.
4. **Teach it yourself**: show the camera an object and type its name in Chat; later show it again and the answer appears as "I see: ...".

## Inter-cortical learning

Cortex does **not** transport gradients between processes. Sensory processes provide afferent evidence. Internal cortices exchange representations and learned top-down predictions. Perceptual cortex compares a top-down prediction with the next sensory representation, emits a `prediction_error`, and uses the resulting surprise as a local plasticity modulator. Its own backpropagation remains entirely inside its network. Associative cortex learns co-activations and can emit `prediction` signals after a relation becomes recurrent. This deliberately separates inter-area teaching/modulation from local synaptic credit assignment and gives the TUI observable prediction/error events.

## Cortex-0 developmental training

Train each local cortex independently before starting the collaborative runtime:

```bash
bash scripts/train-cortex0.sh
```

Set `CORTEX_TRAIN_STEPS` to change the default 20,000 local updates per cortex. The pipeline trains Visual, Perceptual, Memory, Associative and Language sequentially and writes geometry-checked checkpoints under `weights/cortex-0/`. These files are deliberately ignored by git: they are learned artifacts, not source code. At runtime each cortex automatically loads its matching checkpoint and then remains locally plastic. Workspace is currently a selection mechanism and has no pretrained weights.

The current trainer is a deterministic developmental bootstrap, not semantic supervised training: Visual/Perceptual/Memory learn generic signal transformations, Associative learns paired-symbol integration, and Language learns byte-pattern encoding. This validates the separate-development → checkpoint → collaborative-learning lifecycle while leaving semantic concepts to later experience.

## Seed-0 sparse connectome

Cortical communication is intentionally sparse rather than globally broadcast. Each innate projection has its own NATS subject, `cortex.<source>.to.<target>`, and carries a fixed-size cortical representation plus an explicit connection weight. The initial connectome permits only the minimum Visual→Perceptual, recurrent Perceptual/Memory/Associative loop, Language↔Associative path, and Workspace candidate/broadcast paths. Initial connection weights are neutral (1.0); later phases can learn gating and structural plasticity without changing the transport model.

The observer TUI subscribes to `cortex.>` only for telemetry. It displays the brain areas, every permitted projection, its current weight, recent message rate, and a heat bar that becomes brighter as traffic increases. The observer never participates in cognition.
