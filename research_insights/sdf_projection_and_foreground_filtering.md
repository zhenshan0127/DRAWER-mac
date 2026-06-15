# SDF surface projection: strays, outward projection, and foreground filtering

Study note for `research_insights/visualize_sdf.py`. Context: to turn a thresholded
SDF point cloud (a thick, rough lattice shell) into a clean surface, we **project**
each near-surface point onto the `sdf=0` isosurface with Newton steps along the
gradient: `x <- x - sdf(x)·∇sdf(x)` (for a true SDF `‖∇sdf‖≈1`). These notes cover
the artifacts that arise and how to get a clean foreground surface.

---

## 1. Why the few ±32 strays exist

When projecting foreground seeds (sampled in `[-1,1]³`) and then applying inverse
contraction, a tiny fraction (<0.01%) of points landed on the `±32` clip shell.
The chain:

1. **Seeds come from the foreground box `[-1,1]³`.** Some sit right at `|coord|≈1`.
2. **Projection follows the gradient to the *nearest* zero-crossing.** For a boundary
   seed whose nearest surface lies *outward*, the Newton step pushes it **past
   `mag=1`** into the contracted background shell `[1,2]`. (The original code clamped
   only to the field domain `(-2,2)`, not the seed box, so points were free to wander.)
3. **Inverse contraction explodes them.** For `mag∈[1,2]`, the inverse map is
   `1/(2-mag)`, which blows up as `mag→2`. A point at `mag≈1.97` maps to world radius
   `~33` → clipped to `±32` (the `max_range` clamp inherited from `extract_mesh.py`).

So a stray = a foreground-boundary seed that **projected onto background geometry**,
then got radially stretched to the clip shell. They are *real* surface points
(`sdf≈0`), just in the wrong place.

---

## 2. Why "outward" projection exists

The foreground box `[-1,1]³` is an **arbitrary crop for sampling — not the object's
surface.** Scene surfaces (walls, floor) straddle `mag=1`, and some sit at a radius
slightly *larger* than the seed trying to reach them.

Projection has no in/out preference — it goes to the **nearest** zero-crossing.
Direction is set by the sign of the SDF and the normal:

- `sdf>0` (free space, outside the solid): `∇sdf` points *away* from the surface, so
  `−sdf·∇sdf` steps **toward** it.
- `sdf<0` (inside the solid): the step flips and pushes out toward the surface.

**Concrete example.** A wall sits at `mag≈1.0–1.05` (just outside the unit cube). A seed
at `mag≈0.99` in the free space in front of it has `sdf≈+0.02`, and `∇sdf` points
**inward** (toward scene center, away from the wall). The step
`x − (0.02)(inward normal)` moves **outward** — across `mag=1` to reach the wall.

Only **peripheral** surfaces (the room boundary) sit near `mag≈1`, because auto-scaling
fits the cameras inside the unit ball and the enclosing walls land at the edge. So only
a thin sliver of boundary seeds escapes outward. The reverse case (surface just inside,
seed projects inward) is harmless: those points stay at `mag<1` and are never stretched.

---

## 3. Does `--project-tol` eliminate `mag>1` points? No — by itself it can't.

`--project-tol` is a **distance filter (`|sdf| ≤ tol`), not a position filter.** It has
no notion of `mag`. Evidence from identical `project-tol = 0.005` runs:

| File | clamp to box? | points with `mag>1` | at `±32` |
|---|---|---|---|
| unclamped projection | no  | **114,053** | 33 |
| clamped projection   | yes | **0** | 0 |

- **Without the clamp:** an escaped seed genuinely *reaches* `sdf≈0` in the background
  shell `[1,2]`. Its `|sdf|` is `≤ tol`, so it **passes** `project-tol`. The filter
  can't catch it — it's a valid zero-crossing, just mislocated. → 114k survive.
- **With the clamp:** projection pins that seed to the box boundary (`mag=1` face)
  instead of letting it travel outward. There it is **not** on the surface, so
  `|sdf| > tol`, and `project-tol` **drops** it. → 0 survive.

So the **clamp** turns a *position* problem (point at `mag>1`) into an *sdf* problem
(point stuck off-surface), and **`project-tol`** then removes it. To cut `mag>1` points
directly you must filter on magnitude, not on `sdf`.

---

## 4. Eliminating `mag>1` cleanly — the `--max-mag` filter

The principled way is an **explicit magnitude filter**, independent of projection/clamp
side-effects. `--max-mag T` drops any point whose **contracted** magnitude exceeds `T`,
computed in contracted space with the contraction's own norm, **before** inverse
contraction:

```python
order = pipeline.model.scene_contraction.order      # 'inf' here -> L-inf
mag = np.linalg.norm(points, ord=order, axis=1)     # = max(|x|,|y|,|z|)
keep = mag <= max_mag
```

Demonstration: sample the **full** `[-2,2]³`, project onto `sdf=0`, then `--max-mag 1.0`:

- Converged 5.07M surface points → kept **349,702** foreground, **dropped 4,724,080**
  background (`mag>1`). Result: bbox exactly `[-1,1]³`, 0 points with `mag>1`,
  mean `|sdf| = 0.00003`.

**Key insight:** most of the scene's actual surface lives in the contracted
**background** (`mag>1`) — the walls, floor, far objects. After auto-scaling only a
small central portion of geometry is genuinely foreground, so a strict `mag≤1` surface
is honest but **sparse** (~350k points here).

---

## 5. `--max-mag` corresponds to the cube `[-1,1]`, not `[0,1]`

Two different quantities, both true at once:

- **The magnitude scalar** `mag = max(|x|,|y|,|z|)` is a **norm** (absolute value built
  in), so it is always `≥ 0` and ranges `[0,1]` for the foreground. That is the
  *distance scale*, not the coordinate range.
- **The region it defines:** `mag ≤ 1` ⟺ **every coordinate lies in `[-1,1]`**. A point
  at `x=−1` has `mag=1` (kept); at `x=−1.2`, `mag=1.2` (dropped).

Verified on the kept points (per axis): `min=−1.000`, `max=+1.000`, with ~57% of points
having **negative** coordinates. So `--max-mag 1.0` keeps the symmetric `[-1,1]³` cube.
The `[0,1]` is just the magnitude scale; the coordinates span `[-1,1]`.

---

## 6. "All within `[-1,1]`" (position) ≠ "foreground" (membership)

Comparing the `[-1,1]`-clamp cloud vs the `--max-mag` cloud, their magnitude
distributions are **nearly identical** (both ~67% at `mag 0.5–0.9`; boundary pile-up
`mag>0.999` is only **1.7% vs 0.3%**). So the clamped cloud is **not** mostly
boundary-pinned background — an earlier claim to the contrary was wrong.

The real distinction:

- **"All within `[-1,1]`" is positional** — a fact about coordinates. The clamp method
  enforces it **mechanically** (sample only `[-1,1]`, clamp projection to the box). It
  makes **no decision** about whether each point's *surface* is foreground or
  background; it just never lets a point leave. The ~1.7% boundary seeds whose true
  surface is outside get **pinned and kept** as if foreground.
- **`--max-mag` is semantic** — it lets points project to their *true* surface, then
  keeps only those whose true location is `mag≤1`. That is an actual foreground/
  background *classification*.

Hence the table label "don't care about fg/bg split" for the clamp method: it confines
coordinates rather than deciding membership, and only coincides with foreground because
sampling was pre-restricted to the box.

**Why the counts differ (5M vs 350k): sampling density, not pinning.** A `256³` grid
over `[-1,1]` samples the foreground **8× denser per unit volume** than `256³` over
`[-2,2]` (half the side length ⇒ 1/8 the cell volume); times `--oversample 2` ≈ 16×.
And `350k × 16 ≈ 5.6M ≈` the clamped count. Both are genuine foreground surface points;
the clamped run just samples the same surface far more densely.

---

## Summary / how to get a clean foreground surface

| Goal | Setting | Result |
|---|---|---|
| Strict, *classified* foreground only | sample `[-2,2]`, `--project-to-surface`, `--max-mag 1.0` | guaranteed 0 `mag>1`; honest but sparse (~350k) |
| Include near-foreground / some background | `--max-mag 1.5` | more coverage, still bounded |
| Dense foreground, coordinates bounded (no classification) | `--bounding-box -1..1 --oversample 2` | ~5M pts in `[-1,1]`, ~1.7% boundary-pinned |
| Whole scene surface | no `--max-mag`, sample `[-2,2]` | full kitchen incl. background |

- `--max-mag` (position filter) is the principled way to drop `mag>1`; `--project-tol`
  (distance filter) cannot do it alone.
- `mag` is an L-∞ norm in `[0,1]`; `mag≤1` is the cube `[-1,1]³`.
- The clamp bounds *coordinates*; `--max-mag` classifies *membership*. They mostly agree
  but differ on the thin boundary sliver.

### Generated artifacts
- `cs_kitchen_sdf_surface_projected_clean.ply` — `[-1,1]` clamp, dense (~5M), normal-colored.
- `cs_kitchen_sdf_foreground_maxmag.ply` — `[-2,2]` + `--max-mag 1.0`, classified foreground (~350k).
- See also `mipnerf360_foreground_background.md` for the contraction / fg-bg definition.
