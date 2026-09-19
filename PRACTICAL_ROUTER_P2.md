# Practical router P2 — preserve global diffusion state, mask the update

Status: **pre-registered; ready for CUDA. No speed claim is allowed in this gate.**

P1 gave us a useful separation:

```text
WHERE:
edge-selected 4/16 tiles contain 43.51% of teacher correction energy
random 4/16 tiles contain        24.31%
=> selection survives

WHAT:
independent crop diffusion has mean directional cosine only ~0.26
and an oracle-strength blend recovers only ~3.5%
=> crop actuation fails
```

So P2 changes the actuator rather than tuning the selector.

## Mechanism

Start with the same cheap 512×512 base. Define a new full-frame 4-step
img2img continuation as the teacher.

The masked route uses the **full 512×512 SDXL UNet context and the same prompt,
noise seed, scheduler and timestep sequence**, but at every denoising step it
replaces non-selected latent regions with the appropriately noised latent of the
cheap base. Only selected regions are allowed to carry the continuing denoising
state.

This is based on the Hugging Face diffusers community masked SDXL img2img
pipeline, vendored locally under its Apache-2.0 license. A separate RNG is used
for mask-background noise so it cannot perturb the main scheduler RNG.

The four masks are:

- **edge** — 4/16 tiles selected from cheap draft edge energy;
- **random** — 4/16 matched-budget attacker;
- **oracle** — top 4 teacher-correction tiles, used only as an actuator ceiling;
- **full** — all pixels, used only as a trajectory-parity check.

## Why full-mask parity matters

If the full mask cannot reproduce the ordinary full-frame teacher, then the
masked pipeline is not an honest implementation of the same diffusion
trajectory and the gate is invalid.

Pre-registered parity floor:

```text
minimum full-mask PSNR to standard teacher >= 35 dB
```

## P2 PASS rule

P2 earns **global-state masked actuation** only if all hold:

1. full-mask parity floor >= 35 dB on every case;
2. edge masked route mean correction recovery >= **20%**;
3. edge mean recovery exceeds random by >= **8 percentage points**;
4. edge beats random recovery in >= **6/9** cases.

No speed condition appears here because the masked helper still evaluates the
full-frame UNet. P2 is a mechanism-correctness gate.

If P2 passes, P3 may optimize the second denoising stage by evaluating only
selected latent regions while preserving the exact global timestep/noise state.

If P2 fails even though full-mask parity passes, the relevant diffusion update
is too globally coupled for hard spatial masking at this scale; the next
candidate should be token/feature caching or another globally coherent
acceleration mechanism rather than tiled denoising.

## Run

```bash
git pull
pip install -r requirements-gpu.txt
python gpu_diffusion_router_p2.py --local-only
```

If necessary, omit `--local-only`.

Upload:

```text
results/practical_diffusion_router_p2/
```
