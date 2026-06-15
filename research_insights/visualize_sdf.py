#!/usr/bin/env python
"""visualize_sdf.py

Load a trained bakedsdf checkpoint and visualize the learned SDF as a colored
point cloud over a uniform 3D grid.

Color map (diverging blue-white-red, like matplotlib "bwr"):
    * SDF ~= 0  (the surface) -> white
    * SDF  > 0  (outside)     -> red,  the larger the value the redder
    * SDF  < 0  (inside)      -> blue, the smaller the value the bluer

The grid is sampled uniformly in the field's *contracted* input space
([-2, 2]^3 by default), which is exactly the domain `forward_geonetwork`
expects -- so the SDF values here are consistent with `extract_mesh.py`.

Example (run from <repo>/sdf):

    python scripts/visualize_sdf.py \
        --load-config outputs/cs_kitchen/cs_kitchen_sdf_recon/config.yml \
        --output-path outputs/cs_kitchen/cs_kitchen_sdf_recon/sdf_points.ply \
        --resolution 192 --sdf-clip 0.1

Open the resulting .ply in MeshLab / CloudCompare / Blender (vertex colors).
Tip: add `--abs-sdf-max 0.05` to keep only the thin shell near the surface so
you can actually see inside the volume.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import tyro
from rich.console import Console

from nerfstudio.utils.eval_utils import eval_setup

CONSOLE = Console(width=120)


def sdf_to_color(sdf: np.ndarray, clip: float) -> np.ndarray:
    """Map signed-distance values to RGB in [0, 1] (white=0, red=+, blue=-)."""
    t = np.clip(sdf / clip, -1.0, 1.0)
    colors = np.ones((sdf.shape[0], 3), dtype=np.float32)  # start white
    pos = t > 0
    neg = t < 0
    # positive -> red: fade green & blue out as t -> 1
    colors[pos, 1] = 1.0 - t[pos]
    colors[pos, 2] = 1.0 - t[pos]
    # negative -> blue: fade red & green out as t -> -1
    s = -t[neg]
    colors[neg, 0] = 1.0 - s
    colors[neg, 1] = 1.0 - s
    return colors


def normal_to_color(normals: np.ndarray) -> np.ndarray:
    """Map unit surface normals to RGB in [0, 1] (the usual (n+1)/2 normal map)."""
    return np.clip((normals + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)


def eval_sdf(field, points: np.ndarray, device, chunk: int) -> np.ndarray:
    """Evaluate the SDF (channel 0 of forward_geonetwork) at (N,3) points."""
    x = torch.from_numpy(np.ascontiguousarray(points, dtype=np.float32))
    out = torch.empty(x.shape[0], dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, x.shape[0], chunk):
            out[i : i + chunk] = field.forward_geonetwork(x[i : i + chunk].to(device))[:, 0].float().cpu()
    return out.numpy()


def _sdf_and_grad(field, x: torch.Tensor, delta: float):
    """Central-difference SDF value and gradient at x: (M,3) -> sdf (M,), grad (M,3)."""
    offs = torch.tensor(
        [[0, 0, 0], [delta, 0, 0], [-delta, 0, 0],
         [0, delta, 0], [0, -delta, 0], [0, 0, delta], [0, 0, -delta]],
        dtype=x.dtype, device=x.device,
    )  # (7,3)
    pts = (x[:, None, :] + offs[None, :, :]).reshape(-1, 3)
    s = field.forward_geonetwork(pts)[:, 0].reshape(-1, 7)
    sdf = s[:, 0]
    grad = torch.stack(
        [(s[:, 1] - s[:, 2]), (s[:, 3] - s[:, 4]), (s[:, 5] - s[:, 6])], dim=-1
    ) / (2.0 * delta)
    return sdf, grad


def project_to_surface(field, points: np.ndarray, device, iters: int, delta: float,
                       chunk: int, domain=(-2.0, 2.0)):
    """Newton-project points onto the sdf=0 isosurface using the SDF gradient.

    Step: x <- x - sdf(x) * grad / ||grad||^2  (for a true SDF ||grad||~=1, i.e.
    x <- x - sdf*grad). Returns projected points, final sdf, and unit normals.
    Operates in the field's contracted input space; points are clamped to `domain`.
    """
    x_all = torch.from_numpy(np.ascontiguousarray(points, dtype=np.float32))
    n = x_all.shape[0]
    out_pts = np.empty((n, 3), dtype=np.float32)
    out_sdf = np.empty(n, dtype=np.float32)
    out_nrm = np.empty((n, 3), dtype=np.float32)
    eps = 1e-9
    lo, hi = domain
    with torch.no_grad():
        for i in range(0, n, chunk):
            x = x_all[i : i + chunk].to(device)
            for _ in range(iters):
                sdf, grad = _sdf_and_grad(field, x, delta)
                step = (sdf[:, None] / ((grad * grad).sum(-1, keepdim=True) + eps)) * grad
                x = (x - step).clamp(lo, hi)
            sdf, grad = _sdf_and_grad(field, x, delta)
            nrm = grad / (grad.norm(dim=-1, keepdim=True) + eps)
            out_pts[i : i + chunk] = x.float().cpu().numpy()
            out_sdf[i : i + chunk] = sdf.float().cpu().numpy()
            out_nrm[i : i + chunk] = nrm.float().cpu().numpy()
    return out_pts, out_sdf, out_nrm


def write_colored_ply(path: Path, points: np.ndarray, colors01: np.ndarray) -> None:
    """Write a binary little-endian PLY point cloud with per-vertex RGB.

    points: (N, 3) float, colors01: (N, 3) float in [0, 1]. No open3d needed.
    """
    points = np.ascontiguousarray(points, dtype=np.float32)
    rgb = np.clip(colors01 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    n = points.shape[0]
    vertex = np.empty(
        n,
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
               ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertex["x"], vertex["y"], vertex["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertex.tobytes())


@dataclass
class VisualizeSDF:
    """Sample the trained SDF on a uniform grid and save it as a colored point cloud."""

    # Path to the config YAML file written during training.
    load_config: Path
    # Output .ply path for the colored point cloud.
    output_path: Path = Path("sdf_points.ply")
    # Number of samples per axis (total points = resolution^3 before filtering).
    resolution: int = 192
    # Bounding box (in the field's contracted input space; bakedsdf uses [-2, 2]).
    bounding_box_min: Tuple[float, float, float] = (-2.0, -2.0, -2.0)
    bounding_box_max: Tuple[float, float, float] = (2.0, 2.0, 2.0)
    # SDF magnitude mapped to fully saturated red/blue; |sdf| beyond this clamps.
    sdf_clip: float = 0.1
    # If set, keep only points with |sdf| <= this (thin shell near the surface).
    abs_sdf_max: Optional[float] = None
    # If set, randomly subsample to at most this many points (keeps file small).
    max_points: Optional[int] = None
    # Map points to world coordinates via inverse contraction (to overlay the mesh).
    apply_inv_contraction: bool = False
    # Clip inv-contracted world coords to +/- this (matches extract_mesh.py max_range).
    inv_contraction_max_range: float = 32.0
    # Project the kept near-surface points ONTO the sdf=0 isosurface (Newton steps).
    # This is how you get a clean "sdf == 0" surface point cloud instead of a thick shell.
    project_to_surface: bool = False
    # Number of Newton projection iterations.
    project_iters: int = 10
    # Central-difference step used for the projection gradient.
    project_delta: float = 0.003
    # After projecting, keep only points that converged to |sdf| <= this (drops strays).
    project_tol: float = 0.005
    # Densify: replace each near-surface seed with this many jittered copies before
    # projecting (multiplies surface point density). Use with --project-to-surface.
    oversample: int = 1
    # Color points by "sdf" (diverging map) or "normal" (shape-revealing normal map).
    color_by: Literal["sdf", "normal"] = "sdf"
    # If set, drop points whose contracted magnitude exceeds this (e.g. 1.0 keeps
    # only the foreground / identity-mapped region; mag>1 is the contracted background).
    # Computed in CONTRACTED space using the scene contraction's norm, before inv-contraction.
    max_mag: Optional[float] = None
    # Chunk size for SDF evaluation.
    chunk: int = 100000
    # Also dump points + raw SDF values to a .npz next to the .ply.
    save_npz: bool = False
    # Torch matmul precision.
    torch_precision: Literal["highest", "high"] = "high"

    def main(self) -> None:
        torch.set_float32_matmul_precision(self.torch_precision)
        assert str(self.output_path)[-4:] == ".ply", "--output-path must end in .ply"

        _, pipeline, _ = eval_setup(self.load_config)
        pipeline.eval()
        model = pipeline.model
        device = model.device
        CONSOLE.print(f"Loaded pipeline on device: [bold]{device}[/bold]")

        # Uniform grid in the field's input (contracted) space.
        mn, mx = self.bounding_box_min, self.bounding_box_max
        xs = torch.linspace(mn[0], mx[0], self.resolution)
        ys = torch.linspace(mn[1], mx[1], self.resolution)
        zs = torch.linspace(mn[2], mx[2], self.resolution)
        xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
        grid = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=-1)
        CONSOLE.print(f"Sampling {grid.shape[0]:,} points "
                      f"({self.resolution}^3) over {mn} -> {mx}")

        # Query the SDF (channel 0 of forward_geonetwork) over the whole grid.
        points = grid.numpy()
        sdf = eval_sdf(model.field, points, device, self.chunk)
        CONSOLE.print(
            f"SDF stats: min={sdf.min():.4f} max={sdf.max():.4f} "
            f"mean={sdf.mean():.4f}  |sdf|<{self.sdf_clip}: "
            f"{(np.abs(sdf) < self.sdf_clip).mean() * 100:.1f}% of points"
        )

        # Near-surface seed filter (these are the points we keep / project from).
        if self.abs_sdf_max is not None:
            keep = np.abs(sdf) <= self.abs_sdf_max
            points, sdf = points[keep], sdf[keep]
            CONSOLE.print(f"Kept {points.shape[0]:,} seed points with "
                          f"|sdf| <= {self.abs_sdf_max}")
            assert points.shape[0] > 0, "No points survived --abs-sdf-max filter."

        # Densify: jitter-replicate each seed within its voxel before projection.
        if self.oversample > 1:
            extent = np.array(self.bounding_box_max) - np.array(self.bounding_box_min)
            voxel = (extent / max(self.resolution - 1, 1)).astype(np.float32)
            rng = np.random.default_rng(0)
            points = np.repeat(points, self.oversample, axis=0)
            points = points + (rng.random(points.shape, dtype=np.float32) - 0.5) * voxel
            sdf = None  # stale after jitter; recomputed below
            CONSOLE.print(f"Oversampled to {points.shape[0]:,} jittered candidates")

        # Project onto the sdf=0 isosurface so every point lands ON the surface.
        normals = None
        if self.project_to_surface:
            CONSOLE.print(f"Projecting onto sdf=0 ({self.project_iters} Newton iters, "
                          f"delta={self.project_delta})...")
            proj_chunk = max(1, min(self.chunk, 50000))
            # Clamp projection to the sampling box so foreground points can't escape
            # into the background (mag>1) region and blow up under inv-contraction.
            domain = (float(min(self.bounding_box_min)), float(max(self.bounding_box_max)))
            points, sdf, normals = project_to_surface(
                model.field, points, device, self.project_iters,
                self.project_delta, proj_chunk, domain=domain,
            )
            keep = np.abs(sdf) <= self.project_tol
            points, sdf, normals = points[keep], sdf[keep], normals[keep]
            CONSOLE.print(f"Converged {points.shape[0]:,} points to "
                          f"|sdf| <= {self.project_tol} "
                          f"(mean |sdf|={np.abs(sdf).mean():.5f})")
            assert points.shape[0] > 0, "No points converged; loosen --project-tol."
        elif sdf is None:
            sdf = eval_sdf(model.field, points, device, self.chunk)

        # Explicit magnitude filter: drop background (contracted mag > max_mag).
        # Done in CONTRACTED space (points are still contracted here) with the
        # contraction's own norm, so it is exact regardless of projection/clamp.
        if self.max_mag is not None:
            order = pipeline.model.scene_contraction.order
            ord_np = order if order is not None else 2
            mag = np.linalg.norm(points, ord=ord_np, axis=1)
            keep = mag <= self.max_mag
            points, sdf = points[keep], sdf[keep]
            normals = normals[keep] if normals is not None else None
            CONSOLE.print(f"Kept {points.shape[0]:,} points with contracted "
                          f"mag <= {self.max_mag} (dropped {(~keep).sum():,} background)")
            assert points.shape[0] > 0, "No points left after --max-mag filter."

        # Optional random subsample.
        if self.max_points is not None and points.shape[0] > self.max_points:
            idx = np.random.default_rng(0).choice(
                points.shape[0], self.max_points, replace=False
            )
            points, sdf = points[idx], sdf[idx]
            normals = normals[idx] if normals is not None else None
            CONSOLE.print(f"Subsampled to {points.shape[0]:,} points")

        # Optional: map back to world coordinates (matches extract_mesh.py).
        if self.apply_inv_contraction:
            points = self._inv_contract(pipeline, points, self.inv_contraction_max_range)

        if self.color_by == "normal" and normals is not None:
            colors = normal_to_color(normals)
        else:
            if self.color_by == "normal":
                CONSOLE.print("[yellow]--color-by normal needs --project-to-surface; "
                              "falling back to sdf coloring.[/yellow]")
            colors = sdf_to_color(sdf, self.sdf_clip)
        write_colored_ply(self.output_path, points, colors)
        CONSOLE.print(f"[green]Wrote[/green] {self.output_path} "
                      f"({points.shape[0]:,} colored points)")

        if self.save_npz:
            npz_path = self.output_path.with_suffix(".npz")
            save_kw = dict(points=points, sdf=sdf)
            if normals is not None:
                save_kw["normals"] = normals
            np.savez_compressed(npz_path, **save_kw)
            CONSOLE.print(f"[green]Wrote[/green] {npz_path}")

    @staticmethod
    def _inv_contract(pipeline, points: np.ndarray, max_range: float = 32.0) -> np.ndarray:
        """Inverse of the scene contraction (same formula as extract_mesh.py).

        Points at the contraction boundary (||x|| -> 2) map to infinity, so we
        clip to +/-max_range exactly as extract_mesh.py / marching_cubes.py do.
        """
        x = torch.from_numpy(points.astype(np.float32))
        order = pipeline.model.scene_contraction.order
        mag = torch.linalg.norm(x, ord=order, dim=-1)
        mask = mag >= 1
        x_new = x.clone()
        x_new[mask] = (1 / (2 - mag[mask][..., None])) * (x[mask] / mag[mask][..., None])
        return np.clip(x_new.numpy(), -max_range, max_range)


def entrypoint():
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(tyro.conf.FlagConversionOff[VisualizeSDF]).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(VisualizeSDF)  # noqa
