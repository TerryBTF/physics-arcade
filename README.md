# physics-arcade

Classic physics games as benchmarks for physical prediction in world models and VLMs.

The first game is an Angry Birds–style level: a bird is launched at towers of wood,
stone and ice blocks with pigs on and between them (2D rigid-body physics via
[pymunk](https://www.pymunk.org)). A model sees the first 0.4 s of the bird's flight,
before any contact, and has to predict what happens next.

## Tracks

| Track | Model output | Metric |
|---|---|---|
| **Outcome** | which pig ids are still alive once everything is at rest (`jsonl`) | per-pig balanced accuracy, episode exact match, alive-count MAE |
| **Rollout** | the remaining 5.6 s of video (`mp4`, 640×360, 30 fps) | mean IoU of dynamic classes (bird, pig, wood, stone, ice) against simulator masks, final pig-count accuracy |

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
D=data/angry_birds_v0
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

| Track | Method | flat | cartoon |
|---|---|---|---|
| Outcome | all alive / all dead | 0.50 | 0.50 |
| Outcome | ballistic path only (ignores collapses) | 0.63 | 0.63 |
| Rollout | ground-truth video (upper bound) | mIoU 0.95 | mIoU 0.93 |
| Rollout | freeze last context frame | mIoU 0.41 | mIoU 0.41 |

## Models

`models/cosmos/` runs NVIDIA Cosmos-Predict2-2B-Video2World (diffusers) in its own
environment (`setup.sh`, CUDA 12.8 wheels for RTX 50-series). It needs a Hugging Face
login with access to the gated `nvidia/Cosmos-Predict2-2B-Video2World`,
`nvidia/Cosmos-Guardrail1` and `nvidia/Cosmos-1.0-Guardrail` repos.

```bash
cd models/cosmos && ./setup.sh
.venv/bin/python run_cosmos.py --data ../../data/angry_birds_v0 --out ../../preds/rollout_cosmos2b --limit 10
```
