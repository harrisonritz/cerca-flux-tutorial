import json
from pathlib import Path

code = r'''import os
os.environ["MNE_DONTWRITE_HOME"] = "true"
import mne, numpy as np
mne.set_log_level("WARNING")
src = mne.read_source_spaces("data/princeton_sub-038/freesurfer/bem/sub-038_ses-01-oct6-src.fif")
vertices = [s["vertno"] for s in src]
data = np.zeros((sum(map(len, vertices)), 1))
data[100:400, 0] = np.linspace(0, 1, 300)
stc = mne.SourceEstimate(data, vertices, 0, 1, subject="freesurfer")
mne.viz.set_3d_backend("notebook")
brain = stc.plot(
    subject="freesurfer",
    subjects_dir="data/princeton_sub-038",
    surface="inflated",
    hemi="split",
    views=["lat", "med"],
    view_layout="horizontal",
    colormap="RdBu_r",
    clim=dict(kind="value", lims=[0.1, 0.5, 1.0]),
    smoothing_steps=5,
    colorbar=True,
    cortex="low_contrast",
    background="white",
    foreground="black",
    initial_time=0.0,
    time_viewer=False,
    show_traces=False,
    backend="notebook",
    size=(1000, 600),
    title="MNE notebook backend test",
)
brain
'''
nb = {
    "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(True), "id": "backend-test"}],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
Path("surface_backend_test.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
