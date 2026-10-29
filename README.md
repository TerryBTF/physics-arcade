# physics-arcade

Classic physics games as benchmarks for physical prediction in world models and VLMs.

The first game is **Catapult**: a cannonball is fired from a catapult at towers of wood,
stone and ice blocks with small robots on and between them (2D rigid-body physics via
[pymunk](https://www.pymunk.org)). A model sees the first 0.4 s of the cannonball's flight,
before any contact, and has to predict what happens next.

## Leaderboard: Catapult outcome track

Which robots survive? Models see the launch frame with robot ids and 4 frames of the
cannonball's flight (images only), 200 episodes. Headline metric: balanced accuracy on
*stable* robots (fate agrees in ≥ 90 % of perturbed replays); chance is 0.50.

| Model | Flat ↑ | Cartoon ↑ | Brier ↓ (flat / cartoon) |
|---|---|---|---|
| Simulator reference† | 0.998 | 0.998 | 0.043 / 0.043 |
| Ballistic path† | 0.682 | 0.682 | 0.336 / 0.336 |
| Qwen3-VL-4B-Instruct | 0.534 | 0.579 | 0.473 / 0.431 |
| Qwen3-VL-8B-Instruct (FP8) | 0.533 | 0.566 | 0.479 / 0.440 |
| All alive / none alive | 0.500 | 0.500 | 0.48 / 0.52 |

† Not a vision model: reads the exact launch speed and angle (the simulator reference
also the exact scene) instead of images. Brier floor with perfect probabilities: 0.033.

95 % bootstrap intervals over episodes: on the flat style neither Qwen model is
distinguishable from chance (4B [0.50, 0.57], 8B [0.49, 0.57]); on the cartoon style
both are above it (4B [0.55, 0.61], 8B [0.52, 0.61]). 8B is not better than 4B on
either style. Raw model answers are in [`results/catapult/`](results/catapult/).

## Tracks

| Track | Model output | Metric |
|---|---|---|
| **Outcome** | which robot ids are still alive once everything is at rest, optionally with probabilities (`jsonl`) | balanced accuracy on stable robots (headline), Brier score on all robots |
| **Rollout** | the remaining 5.6 s of video (`mp4`, 640×360, 30 fps) | mean IoU of dynamic classes (ball, robot, wood, stone, ice) against simulator masks, ±1 frame time tolerance; final robot-count accuracy |

Collisions are chaotic (see [Predictability](#predictability)), so outcome labels are
each robot's survival probability over 50 replays with tiny launch perturbations. A robot is
*stable* when its fate agrees in ≥ 90 % of replays (85 % of robots); balanced accuracy is
computed on stable robots against the majority fate. The Brier score is the expected
squared error against an outcome drawn from the replays; its floor (perfect
probabilities) is 0.033.

Rollout videos are scored automatically: predicted frames are segmented into object
classes by color and compared with ground-truth masks rendered by the simulator.

## Render styles

Both styles share the same seeds, levels, physics and ground truth, so the score
difference isolates the effect of visual style.

- `flat`: one solid color per class. Segmented by nearest palette color.
- `cartoon`: textured, game-like artwork (procedural and original). Segmented with a
  color→class table fitted on held-out episodes (`physics-arcade fit-cartoon-lut`).

## Quick start

```bash
uv sync
D=data/catapult_v0
uv run physics-arcade generate --n 200 --out $D            # add --style cartoon for the textured version
uv run physics-arcade baseline --data $D --name ballistic --out preds/ballistic.jsonl
uv run physics-arcade eval-outcome --data $D --pred preds/ballistic.jsonl
uv run physics-arcade baseline --data $D --name oracle --out preds/rollout_oracle
uv run physics-arcade eval-rollout --data $D --pred preds/rollout_oracle
uv run physics-arcade visualize --data $D --pred preds/rollout_oracle --ep ep_00010 --out viz/ep_00010.mp4
uv run physics-arcade leaderboard --data $D "Ballistic=preds/ballistic.jsonl" "Qwen3-VL-4B=results/catapult/qwen3vl_4b_flat.jsonl"
```

Dataset layout: `tasks.jsonl` and `inputs/<id>/` (context frames, context video, first
frame with robot ids) are what a model may see; `answers/` holds the hidden ground truth.

## Reference results (200 episodes)

Outcome track (identical for both styles):

| Method | stable-robot bal. acc. | Brier ↓ |
|---|---|---|
| all alive / all dead | 0.50 | 0.50 |
| ballistic path only (ignores collapses) | 0.68 | 0.34 |
| simulator with exact scene, launch noise 1 % / 0.5° (reference) | 1.00 | 0.043 |

Rollout track (mIoU):

| Method | flat | cartoon |
|---|---|---|
| ground-truth video (upper bound) | 0.98 | 0.94 |
| ground truth resampled through the 16 fps Cosmos path (`models/cosmos/dry_run.py`) | 0.90 | 0.85 |
| freeze last context frame | 0.42 | 0.41 |

16 fps models lose ~0.08 mIoU to temporal sampling alone, mostly on the fast-moving
ball; the dry-run row above is their practical ceiling.

## Predictability

`physics-arcade predictability` re-simulates every episode 50 times from its exact
pre-launch scene under small perturbations and records how often each robot's fate
matches the recorded one. Collisions are chaotic: with launch noise of only 0.25 %
speed / 0.1°, 34 % of episodes keep the same outcome in every run and 15 % of robots
flip in 10–90 % of runs. These replays define the outcome-track labels.

## Models

### Vision-language models (outcome track)

`physics_arcade.vlm` holds the shared prompt, answer parsing and resumable loop; a
backend only maps image/text parts to a reply. `models/qwen_vl/` runs Qwen3-VL locally
(4B in bf16 or 8B in FP8 fit in 16 GB):

```bash
cd models/qwen_vl && ./setup.sh
.venv/bin/python run_qwen_vl.py --model Qwen/Qwen3-VL-8B-Instruct-FP8 \
    --data ../../data/catapult_v0 --out ../../preds/qwen3vl_8b_flat.jsonl
```

### Video world models (rollout track)

`models/cosmos/` runs NVIDIA Cosmos-Predict2-2B-Video2World (diffusers) in its own
environment (`setup.sh`, CUDA 12.8 wheels for RTX 50-series). It needs a Hugging Face
login with access to the gated `nvidia/Cosmos-Predict2-2B-Video2World`,
`nvidia/Cosmos-Guardrail1` and `nvidia/Cosmos-1.0-Guardrail` repos.

```bash
cd models/cosmos && ./setup.sh
.venv/bin/python run_cosmos.py --data ../../data/catapult_v0 --out ../../preds/rollout_cosmos2b --limit 10
```
