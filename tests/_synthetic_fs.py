"""Minimal synthetic FreeSurfer subject: geometry only, no real anatomy.

Enough for MNE's BEM, source-space, forward, beamformer, labelling and morph
code paths to run, so the source stages can be tested without a real
reconstruction or a network download.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.freesurfer import io as fsio
import mne
from mne.surface import _get_ico_surface

N_VOX = 256
#: FreeSurfer's conformed 1 mm vox2ras.
VOX2RAS = np.array([[-1, 0, 0, 128.0], [0, 0, 1, -128.0], [0, -1, 0, 128.0], [0, 0, 0, 1]])
BRAIN_RADIUS_MM = 70.0
RIBBON_RADIUS_MM = 78.0


def _write_volumes(root: Path) -> None:
    grid = np.stack(np.meshgrid(*(np.arange(N_VOX),) * 3, indexing="ij"), -1).astype(np.float32)
    ras = nib.affines.apply_affine(VOX2RAS, grid.reshape(-1, 3)).reshape(N_VOX, N_VOX, N_VOX, 3)
    radius = np.linalg.norm(ras, axis=-1)

    t1 = np.where(radius < RIBBON_RADIUS_MM, 110, 0).astype(np.uint8)
    t1[radius < BRAIN_RADIUS_MM] = 90
    nib.save(nib.MGHImage(t1, VOX2RAS), root / "mri/T1.mgz")
    nib.save(nib.MGHImage(t1, VOX2RAS), root / "mri/brain.mgz")

    # A cortical ribbon split into two parcels that exist in the real FreeSurfer LUT.
    aseg = np.zeros((N_VOX,) * 3, np.int32)
    aseg[radius < BRAIN_RADIUS_MM] = 2                       # Left-Cerebral-White-Matter
    ribbon = (radius >= BRAIN_RADIUS_MM) & (radius < RIBBON_RADIUS_MM)
    aseg[ribbon] = np.where(ras[..., 0][ribbon] < 0, 1024, 2024)   # ctx-lh/rh-precentral
    nib.save(nib.MGHImage(aseg, VOX2RAS), root / "mri/aparc+aseg.mgz")


def _write_hemisphere(root: Path, hemi: str, shift_mm: float,
                      unit: np.ndarray, tris: np.ndarray) -> None:
    for name, radius in (("white", 60.0), ("pial", 64.0), ("inflated", 70.0)):
        fsio.write_geometry(str(root / f"surf/{hemi}.{name}"),
                            unit * radius + np.array([shift_mm, 0.0, 0.0]), tris)
    # FreeSurfer's spherical registration surfaces have radius 100.
    for name in ("sphere", "sphere.reg"):
        fsio.write_geometry(str(root / f"surf/{hemi}.{name}"), unit * 100.0, tris)
    for name in ("curv", "sulc"):
        fsio.write_morph_data(str(root / f"surf/{hemi}.{name}"), np.zeros(len(unit), np.float32))

    labels = (unit[:, 2] > 0).astype(np.int32)
    ctab = np.array([[25, 100, 40, 0, 25 + 100 * 256 + 40 * 65536],
                     [125, 100, 160, 0, 125 + 100 * 256 + 160 * 65536]], np.int32)
    fsio.write_annot(str(root / f"label/{hemi}.aparc.annot"), labels, ctab,
                     ["precentral", "postcentral"], fill_ctab=False)


def make_synthetic_freesurfer(subjects_dir, subject: str = "sub-01",
                              with_trans: bool = True) -> Path:
    """Write a synthetic FreeSurfer subject and return its directory."""
    mne.set_log_level("ERROR")
    root = Path(subjects_dir) / subject
    if root.exists():
        shutil.rmtree(root)
    for folder in ("mri/transforms", "surf", "bem", "label"):
        (root / folder).mkdir(parents=True, exist_ok=True)

    _write_volumes(root)
    (root / "mri/transforms/talairach.xfm").write_text(
        "MNI Transform File\n% synthetic\n\nTransform_Type = Linear;\n"
        "Linear_Transform =\n1.0 0.0 0.0 0.0\n0.0 1.0 0.0 0.0\n0.0 0.0 1.0 0.0;\n"
    )

    ico5 = _get_ico_surface(5)
    unit = ico5["rr"] / np.linalg.norm(ico5["rr"], axis=1, keepdims=True)
    _write_hemisphere(root, "lh", -5.0, unit, ico5["tris"])
    _write_hemisphere(root, "rh", +5.0, unit, ico5["tris"])

    ico4 = _get_ico_surface(4)
    unit4 = ico4["rr"] / np.linalg.norm(ico4["rr"], axis=1, keepdims=True)
    for name, radius in (("inner_skull", 80.0), ("outer_skull", 85.0), ("outer_skin", 90.0)):
        mne.write_surface(root / f"bem/{name}.surf", unit4 * radius, ico4["tris"],
                          overwrite=True)

    if with_trans:
        # Identity coregistration: the synthetic sensors are already in head space.
        mne.write_trans(root / f"bem/{subject}-trans.fif",
                        mne.transforms.Transform("mri", "head", np.eye(4)), overwrite=True)
    return root


if __name__ == "__main__":
    print(make_synthetic_freesurfer(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "sub-01"))
