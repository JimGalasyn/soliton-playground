# Regenerating the render gallery

`outputs/` is gitignored (`.gitignore:5`), so no rendered image is tracked. That is
deliberate — but only defensible if the renders regenerate, so the commands are here,
in a tracked file, rather than in a note that lives inside the ignored directory.

Everything below is CPU-only except the one run marked GPU. `soliton_playground.viz`
deliberately does not import jax, so rendering never contends for VRAM with a
relaxation that is still going.

## The run the n-field renders come from (~3 min, one GPU)

```bash
.venv/bin/python experiments/faddeev_realtime_knot.py \
    --N 64 --L 51.2 --R 14.85 --w 2.0 --relax-steps 3000 --steps 3000 \
    --checkpoints 3 --post-relax 200 --save-fields 16 --animate --anim-fps 8 \
    --out outputs/rt_wired
```

It is deterministic: repeated runs reproduce `Q_H = -1.9615`, `dH/H = -6.32e-06`,
`det settled [None, 3, 3]`. It also writes `outputs/rt_wired/launch.json` with its own
resolved flags, so a future run does not depend on this file staying accurate — read
the `launch.json` beside any output in preference to the command above.

## The depth proof

```bash
.venv/bin/python -m pytest tests/test_viz_depth.py -q
```

The numbers live in the test. The pictures come from its `_render()` helper: two
interlocked isosurface tori drawn as separate collections (the near tube wins 0.6% of
contested pixels — uniform occlusion, no weave) and merged into one (45%/55% — a real
weave). That is the whole reason `viz.add_parts` exists.

## The three n-field modes

```bash
for m in surface facets cells; do
  .venv/bin/python -m soliton_playground.viz turntable \
      --field outputs/rt_wired/fields/n_00000000.npz \
      --out outputs/viz_gallery/nfield_modes/mode_$m.gif \
      --L 51.2 --c4 4.0 --frames 12 --mode $m --size 6 --dpi 120
done
```

`surface` is the smoothed isosurface; `facets` is unsmoothed, leaving the
marching-cubes facets at lattice scale (the upstream's "no cheat" stance); `cells`
draws one cube per grid cell above the level.

## The gauged portraits and phase cycles

These need an `ehn-two-scalar` field directory — `field.npz` plus `manifest.json`. The
one used for the committed work was a sibling-checkout battery field:

```bash
FD=../null-worldtube-private/simulations/engine_dogfood/out_sbx_battery_quick/B2_trefoil_pos
V=".venv/bin/python -m soliton_playground.viz_em"
G=outputs/viz_gallery

$V --field $FD --view raw      --out $G/gauged/raw.png
$V --field $FD --view raw      --out $G/gauged/raw_second_angle.png --azim 40 --elev 55
$V --field $FD --view twist    --out $G/gauged/twist.png
$V --field $FD --view bfield   --out $G/gauged/bfield.png
$V --field $FD --view triptych --out $G/gauged/triptych.png

$V --field $FD --view cycle --cycle-fields none \
   --out $G/phase_cycle/cycle_phase_only.gif --frames 36 --cycles 3 --fps 20 --drop-frames
$V --field $FD --view cycle --cycle-fields static \
   --out $G/phase_cycle/cycle_static_fields.gif --frames 36 --cycles 3 --fps 20 --drop-frames
$V --field $FD --view cycle --cycle-fields travelling --m 3 \
   --out $G/phase_cycle/cycle_travelling_fields.gif --frames 36 --cycles 1 --fps 18 --drop-frames
```

**Not a catalog state.** `field_store/objects/` in this checkout holds only
`.manifest.json` files, not the 792 MB blobs `field_store/index.json` describes, so the
ten `ehn-two-scalar` catalog entries could not be rendered here. Any directory with the
right layout works; a catalog state would be the better subject if its blob is present
on the machine doing the rendering.

## What deliberately does not regenerate

Two images in the local gallery are pre-fix evidence and are meant to stay stale:
the E-field and triptych renders from before `viz_em.field_line_parts` gained its
magnitude floor, when `|grad s|` averaging 25x below its peak let a
direction-normalising tracer wander into a boxy cage that looked like structure.
Reproducing them means removing the floor. Don't.
