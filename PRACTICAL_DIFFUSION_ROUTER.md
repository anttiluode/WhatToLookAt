# Practical GPU Gate P0 — route diffusion compute where the image says it matters

Status: **pre-registered; ready for a CUDA run.**

This is the practical handoff from the Sihti AI branch.

## Why this lives in WhatToLookAt

Sihti AI-G0 and AI-G1 produced two useful negative results:

- globally refining the Sihti core was worse than refining the raw draft and did
  not save wall time;
- Sihti residue was not the best local predictor of where a 4-step teacher
  differed from a cheap 2-step refinement.

But the *general* routing signal survived. On the 9 uploaded Sihti AI-G0 cases,
the top 25% of tiles ranked by cheap **draft edge energy** contained **45.72%**
of the teacher-correction energy, versus 25% for random location selection.

That is a WhatToLookAt question:

> if a cheap observable says where the current representation is likely to need
> more work, can we spend expensive compute only there?

## P0 route

For every prompt/seed:

```text
1-step 256 draft
      ↓
2-step full 512 img2img cheap base
      ↓
rank 4 of 16 tiles using only the 256 draft
      ↓
batch-refine 192x192 context crops around those 4 tiles
      ↓
paste only the 128x128 tile centers back into the cheap base
```

Teacher:

```text
same 256 draft -> full-frame 4-step 512 img2img
```

All selectors refine exactly four tiles:

- **edge energy** — primary candidate chosen by the previous gate;
- local variance;
- matched Gaussian residual;
- Sihti residue;
- random tile selection.

The crop context means selected refinement spends more than exactly 25% of a
full frame's spatial work, but every selector pays the same crop geometry and
actual CUDA wall time is measured.

## Primary quality metric

```text
correction_recovery =
    1 - MSE(routed, teacher) / MSE(cheap_base, teacher)
```

Positive values mean the selective route moves closer to the expensive teacher.
The 4-step teacher is a compute reference, not ground truth.

## PASS rule fixed before the GPU run

Edge-selected routing passes only if all hold:

1. mean correction recovery is at least **0.05** above random;
2. edge beats random recovery in at least **6/9** cases;
3. mean edge route wall time is lower than mean teacher route wall time;
4. edge route is faster than teacher in at least **6/9** cases.

The route is allowed to fail. In particular, local crop diffusion may not
compose into the same result as a full-frame denoising trajectory.

## Run

Install the optional GPU stack:

```bash
pip install -r requirements-gpu.txt
```

Then:

```bash
python gpu_diffusion_router.py --local-only
```

If the model is not cached in Diffusers format, omit `--local-only`.

Outputs:

```text
results/practical_diffusion_router/summary.json
results/practical_diffusion_router/*_contact.png
```

The default is already the 3 prompts × 3 seeds used by the Sihti gates.
