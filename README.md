# physics-arcade

Classic physics games as benchmarks for physical prediction in world models and VLMs.

The first game is **Slingshot**: a bird is launched from a slingshot at towers of wood,
stone and ice blocks with pigs on and between them (2D rigid-body physics via
[pymunk](https://www.pymunk.org)). A model sees the first 0.4 s of the bird's flight,
before any contact, and has to predict what happens next.

## Tracks

| Track | Model output | Metric |
|---|---|---|
| **Outcome** | which pig ids are still alive once everything is at rest, optionally with probabilities (`jsonl`) | balanced accuracy on stable pigs (headline), Brier score on all pigs |
| **Rollout** | the remaining 5.6 s of video (`mp4`, 640×360, 30 fps) | mean IoU of dynamic classes (bird, pig, wood, stone, ice) against simulator masks, ±1 frame time tolerance; final pig-count accuracy |

Collisions are chaotic (see [Predictability](#predictability)), so outcome labels are
each pig's survival probability over 50 replays with tiny launch perturbations. A pig is
*stable* when its fate agrees in ≥ 90 % of replays (83 % of pigs); balanced accuracy is
computed on stable pigs against the majority fate. The Brier score is the expected
squared error against an outcome drawn from the replays; its floor (perfect
probabilities) is 0.037.

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
D=data/slingshot_v0
uv run physics-arcade generate --n 200 --out $D            # add --style cartoon for the textured version
uv run physics-arcade baseline --data $D --name ballistic --out preds/ballistic.jsonl
uv run physics-arcade eval-outcome --data $D --pred preds/ballistic.jsonl
uv run physics-arcade baseline --data $D --name oracle --out preds/rollout_oracle
uv run physics-arcade eval-rollout --data $D --pred preds/rollout_oracle
uv run physics-arcade visualize --data $D --pred preds/rollout_oracle --ep ep_00010 --out viz/ep_00010.mp4
```

Dataset layout: `tasks.jsonl` and `inputs/<id>/` (context frames, context video, first
frame with pig ids) are what a model may see; `answers/` holds the hidden ground truth.

## Reference results (200 episodes)

Outcome track (identical for both styles):

| Method | stable-pig bal. acc. | Brier ↓ |
|---|---|---|
| all alive / all dead | 0.50 | 0.50 |
| ballistic path only (ignores collapses) | 0.66 | 0.37 |
| simulator with exact scene, launch noise 1 % / 0.5° (reference) | 1.00 | 0.047 |

Rollout track (mIoU):

| Method | flat | cartoon |
|---|---|---|
| ground-truth video (upper bound) | 0.95 | 0.93 |
| ground truth resampled through the 16 fps Cosmos path (`models/cosmos/dry_run.py`) | 0.89 | 0.85 |
| freeze last context frame | 0.41 | 0.41 |

16 fps models lose ~0.06 mIoU to temporal sampling alone, mostly on the fast-moving
bird; the dry-run row above is their practical ceiling.

## Predictability

`physics-arcade predictability` re-simulates every episode 50 times from its exact
pre-launch scene under small perturbations and records how often each pig's fate
matches the recorded one. Collisions are chaotic: with launch noise of only 0.25 %
speed / 0.1°, 35 % of episodes keep the same outcome in every run and 16 % of pigs
flip in 10–90 % of runs. These replays define the outcome-track labels.

## Models

`models/cosmos/` runs NVIDIA Cosmos-Predict2-2B-Video2World (diffusers) in its own
environment (`setup.sh`, CUDA 12.8 wheels for RTX 50-series). It needs a Hugging Face
login with access to the gated `nvidia/Cosmos-Predict2-2B-Video2World`,
`nvidia/Cosmos-Guardrail1` and `nvidia/Cosmos-1.0-Guardrail` repos.

```bash
cd models/cosmos && ./setup.sh
.venv/bin/python run_cosmos.py --data ../../data/slingshot_v0 --out ../../preds/rollout_cosmos2b --limit 10
```
