# Foreground vs. Background in Mip-NeRF 360 scene contraction

Study note: why a point with contracted magnitude `mag < 1` is **foreground** and
`1 ≤ mag ≤ 2` is **background**, and why the threshold `1` is fixed.

## The scene contraction

Mip-NeRF 360 handles *unbounded* scenes by warping all of space into a bounded
region with a piecewise map (the `SceneContraction`):

```
contract(x) = x                          if ‖x‖ < 1     # identity  (inner region)
            = (2 - 1/‖x‖) · (x/‖x‖)      if ‖x‖ ≥ 1     # warp      (outer shell)
```

where `‖·‖` is the chosen norm and `mag = ‖x‖`.

In this codebase:
- Implementation: `sdf/nerfstudio/field_components/spatial_distortions.py:73-79`
  ```python
  mask = mag >= 1
  x_new[mask] = (2 - 1/mag) * (x/mag)     # points with mag < 1 are left unchanged
  ```
- The SDF field consumes contracted coords and normalizes `[-2,2] → [0,1]` for its
  hashgrid: `sdf/nerfstudio/fields/sdf_field.py:403` → `positions = (inputs + 2.0)/4.0`.

## Why mag < 1 = foreground

For `mag < 1` the map is the **identity**: points keep their original coordinates,
so the network spends full hashgrid resolution there. This inner region is where the
scene-of-interest is placed — the **foreground**.

## Why 1 ≤ mag ≤ 2 = background

For `mag ≥ 1` the point is warped. Its magnitude *after* warping (same norm) is:

```
‖contract(x)‖ = (2 - 1/mag) · ‖x/mag‖ = (2 - 1/mag) · 1 = 2 - 1/mag
```

As the original `mag` runs `1 → ∞`, the contracted magnitude runs:

| original mag | 1 | 2 | 5 | 10 | → ∞ |
|---|---|---|---|---|---|
| contracted `2 − 1/mag` | 1.00 | 1.50 | 1.80 | 1.90 | → 2.00 |

So **everything in the world from radius 1 out to infinity is compressed into the
thin shell `1 ≤ contracted_mag < 2`** — that is the unbounded **background**. The map
is continuous at the seam (`mag = 1` gives `2 − 1 = 1`) and never reaches 2 exactly
(2 is the asymptotic limit = world infinity).

This is the whole point of the contraction: near content gets a large, undistorted
budget (the unit ball), while the infinite far field is squeezed into a bounded shell.

## Is the threshold `1` deterministic?

**Yes.** Both constants are hardcoded in the formula, not learned and not data-dependent:
- `1` — the crossover between the identity branch and the warp branch (`mask = mag >= 1`).
- `2` — the asymptotic outer bound; world infinity maps here.

### Important nuance — what *lands* inside radius 1 is set by data normalization

The contraction runs in a **pre-normalized coordinate frame**. The dataparser centers
and auto-scales the scene first, so the cameras / region-of-interest fall roughly inside
the unit ball:
- `center_poses: true`, `auto_scale_poses: true`
  (e.g. `sdf/outputs/cs_kitchen/cs_kitchen_sdf_recon/config.yml:88-89`).

Therefore:
- The boundary **1 is fixed**.
- **Which real-world points count as foreground** (land at `mag < 1`) depends on that
  upstream scaling — not on the objects themselves.

So "mag < 1 = foreground" is a convention enforced by *data preprocessing meeting a fixed
contraction threshold*, not a perfectly sharp object/background segmentation. Near content
that spills past the unit boundary occupies the lower part of the `[1,2]` shell, and how
tightly the foreground fits inside radius 1 depends on the auto-scale.

## Norm choice: sphere vs. cube

The norm `order` decides the *shape* of the level sets:
- `order = 2` (L2 / Frobenius) → the unit region is a **sphere** of radius 1, full domain a sphere of radius 2.
- `order = inf` (L-∞) → the unit region is the **cube** `[-1,1]³`, full domain the cube `[-2,2]³`.

This project uses **L-∞** (`scene_contraction_norm: inf`,
`sdf/nerfstudio/models/base_surface_model.py:131`; confirmed in the trained
`config.yml:179`). That is why:
- foreground = the cube `[-1,1]³` (`mag = max(|x|,|y|,|z|) < 1`),
- full contracted domain = the cube `[-2,2]³`,
- and why inverse-contracting a uniform grid produces *nested cubic shells* in world
  space (the background `[1,2]` ring stretches radially out toward the clip range).

## Practical takeaways

- To inspect/extract only the in-focus object, **sample the foreground cube**
  `[-1,1]³` — it is identity-mapped, so contracted coords == world coords, with no
  background warp and no cubic-shell artifacts.
- To include the far field, sample out to `[-2,2]³`, but expect the background shell to
  dominate world-space visualizations once inverse-contracted.

## References

- `sdf/nerfstudio/field_components/spatial_distortions.py:48-86` — `SceneContraction`
- `sdf/nerfstudio/models/base_surface_model.py:131,148-155` — norm = `inf` by default
- `sdf/nerfstudio/fields/sdf_field.py:399-429` — `forward_geonetwork`, `[-2,2]→[0,1]` norm
- Mip-NeRF 360: Barron et al., *Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields*, CVPR 2022.
