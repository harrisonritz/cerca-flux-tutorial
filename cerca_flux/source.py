"""Forward modelling and beamformer source reconstruction.

    10. :func:`build_forward`        - BEM, source spaces, lead fields (ForwardModel)
    11. :func:`reconstruct_sources`  - LCMV and DICS beamformers (DICSbeamforming)
    12. :func:`morph_sources`        - move estimates onto a template for group work

Coregistration is *not* re-fitted here.  The MRI/head transform is treated as
experimental metadata produced by the Cerca 3D scan and the MNE coregistration
GUI; it should be inspected visually before any inferential source analysis.
"""

from __future__ import annotations

import numpy as np
import mne
from mne.beamformer import apply_dics_csd, apply_lcmv, make_dics, make_lcmv
from mne.time_frequency import csd_multitaper

from .context import SubjectContext
from .paths import check_freesurfer, find_bem, find_trans, sanitize
from .utils import resolve_picks, save_figure, timed


# --------------------------------------------------------------------------- #
# 10. Forward model
# --------------------------------------------------------------------------- #


def build_forward(ctx: SubjectContext) -> dict[str, mne.Forward]:
    """Build the BEM, source spaces and lead-field matrices for one subject.

    A single-shell BEM (conductivity 0.3 S/m) is adequate for MEG: unlike EEG,
    magnetic fields are barely distorted by the skull and scalp.  Two source
    spaces are built when both are configured - a volumetric grid bounded by the
    inner skull, and a cortical-surface grid - because they answer different
    questions: the volume grid localises subcortical and deep sources and
    overlays directly on the T1, while the surface grid respects cortical
    geometry and supports parcel-wise group analysis.
    """
    cfg = ctx.cfg
    fs_subject = ctx.fs_subject
    subjects_dir = cfg.fs_subjects_dir
    check_freesurfer(cfg, fs_subject)

    trans = find_trans(cfg, ctx.rec, fs_subject, ctx.paths)
    if trans is None:
        raise FileNotFoundError(
            f"{ctx.rec.key}: no MRI/head transform found for FreeSurfer subject "
            f"{fs_subject!r}. Produce one with `mne coreg` (or `mne.gui.coregistration`) "
            "and point forward.trans at it."
        )
    ctx.state["trans"] = str(trans)
    ctx.logger.info("forward: using transform %s", trans.name)

    bem_sol = _resolve_bem(ctx, fs_subject, subjects_dir)
    info = _forward_info(ctx)

    forwards: dict[str, mne.Forward] = {}
    for space in cfg.source.spaces:
        space_cfg = getattr(cfg.forward, space)
        if not space_cfg.enabled:
            continue
        src = _resolve_src(ctx, space, fs_subject, subjects_dir, bem_sol)
        fwd_file = ctx.paths.fwd(space)
        if ctx.needs(fwd_file):
            with timed(f"forward solution ({space})", ctx.logger):
                fwd = mne.make_forward_solution(
                    info, trans=trans, src=src, bem=bem_sol, meg=True, eeg=False,
                    mindist=space_cfg.mindist, n_jobs=1, verbose="ERROR",
                )
            mne.write_forward_solution(fwd_file, fwd, overwrite=True, verbose="ERROR")
        else:
            ctx.logger.info("forward: reusing %s", fwd_file.name)
            fwd = mne.read_forward_solution(fwd_file, verbose="ERROR")
        forwards[space] = fwd
        ctx.record(**{
            f"fwd_{space}_sources": int(fwd["nsource"]),
            f"fwd_{space}_channels": int(fwd["nchan"]),
        })
        ctx.logger.info(
            "forward (%s): %d sources x %d channels", space, fwd["nsource"], fwd["nchan"]
        )

    if not forwards:
        raise RuntimeError(f"{ctx.rec.key}: no source space is enabled in the config")
    return forwards


def _forward_info(ctx: SubjectContext) -> mne.Info:
    """Sensor geometry for the lead field, taken from the preprocessed data.

    The forward model is built for every channel that survived preprocessing;
    :func:`reconstruct_sources` narrows it to the analysis picks afterwards, so
    one forward solution serves both the radial-only and all-axis analyses.
    """
    for candidate in (ctx.paths.ica_raw, ctx.paths.ann_raw, ctx.paths.hfc_raw):
        if candidate.exists():
            return mne.io.read_raw_fif(candidate, preload=False, verbose="ERROR").info
    if ctx.paths.epochs.exists():
        return mne.read_epochs(ctx.paths.epochs, preload=False, verbose="ERROR").info
    from .preprocess import load_raw
    return load_raw(ctx).info


def _resolve_bem(ctx: SubjectContext, fs_subject: str, subjects_dir) -> mne.bem.ConductorModel:
    """Reuse the study's BEM solution if it has one, otherwise build and cache it."""
    existing = find_bem(ctx.cfg, ctx.rec, fs_subject, ctx.paths)
    if existing is not None and not ctx.cfg.output.overwrite:
        ctx.logger.info("forward: reusing BEM %s", existing.name)
        return mne.read_bem_solution(existing, verbose="ERROR")

    bem_file = ctx.paths.bem
    if not ctx.needs(bem_file):
        return mne.read_bem_solution(bem_file, verbose="ERROR")

    with timed("BEM solution", ctx.logger):
        model = mne.make_bem_model(
            fs_subject, ico=ctx.cfg.forward.bem_ico,
            conductivity=tuple(ctx.cfg.forward.bem_conductivity),
            subjects_dir=subjects_dir, verbose="ERROR",
        )
        bem = mne.make_bem_solution(model, verbose="ERROR")
    mne.write_bem_solution(bem_file, bem, overwrite=True, verbose="ERROR")
    return bem


def _resolve_src(ctx: SubjectContext, space: str, fs_subject: str, subjects_dir, bem):
    src_file = ctx.paths.src(space)
    if not ctx.needs(src_file):
        ctx.logger.info("forward: reusing source space %s", src_file.name)
        return mne.read_source_spaces(src_file, verbose="ERROR")

    cfg = getattr(ctx.cfg.forward, space)
    with timed(f"{space} source space", ctx.logger):
        if space == "volume":
            # Bound the grid by the inner-skull surface when FreeSurfer provides
            # it, which is what the FLUX tutorial does; fall back to the BEM.
            inner_skull = subjects_dir / fs_subject / "bem" / "inner_skull.surf"
            kwargs = ({"surface": str(inner_skull)} if inner_skull.exists() else {"bem": bem})
            src = mne.setup_volume_source_space(
                subject=fs_subject, pos=cfg.pos, subjects_dir=subjects_dir,
                mri=str(subjects_dir / fs_subject / "mri" / "T1.mgz"),
                add_interpolator=cfg.add_interpolator, verbose="ERROR", **kwargs,
            )
        else:
            src = mne.setup_source_space(
                subject=fs_subject, spacing=cfg.spacing, surface=cfg.surface,
                subjects_dir=subjects_dir, add_dist=cfg.add_dist, n_jobs=1, verbose="ERROR",
            )
    mne.write_source_spaces(src_file, src, overwrite=True, verbose="ERROR")
    return src


# --------------------------------------------------------------------------- #
# 11. Beamformers
# --------------------------------------------------------------------------- #


def reconstruct_sources(ctx: SubjectContext, epochs: mne.Epochs | None = None,
                        forwards: dict[str, mne.Forward] | None = None) -> dict:
    """Run the LCMV and DICS beamformers on every configured source space.

    Both beamformers use a **common** spatial filter, built from data pooled
    across the conditions and intervals that are later contrasted.  A separate
    filter per condition would differ between the things being compared, and the
    contrast would then partly reflect the filters rather than the brain.

    Results are always expressed as contrasts (dB change for LCMV, relative
    change for DICS) because beamformer output carries a depth bias: absolute
    power grows towards the centre of the head regardless of the data.
    """
    cfg = ctx.cfg
    if epochs is None:
        from .preprocess import make_epochs
        epochs = make_epochs(ctx)
    if forwards is None:
        forwards = build_forward(ctx)

    picks = resolve_picks(epochs, cfg.source.picks, cfg.channels)
    epochs = epochs.copy().pick(picks)
    epochs.info.normalize_proj()

    rank = _estimate_rank(ctx, epochs)
    results: dict[str, dict] = {}

    for space, fwd in forwards.items():
        fwd = mne.pick_channels_forward(
            fwd, include=epochs.ch_names, ordered=True, copy=True, verbose="ERROR"
        )
        if fwd["nchan"] != len(epochs.ch_names):
            raise RuntimeError(
                f"{ctx.rec.key}: the {space} forward model covers {fwd['nchan']} of the "
                f"{len(epochs.ch_names)} analysis channels. Rebuild it with "
                "output.overwrite=true after changing the channel selection."
            )
        space_results: dict[str, mne.SourceEstimate] = {}
        if cfg.source.lcmv.enabled:
            space_results.update(_run_lcmv(ctx, epochs, fwd, space, rank))
        if cfg.source.dics.enabled and cfg.source.dics.bands:
            space_results.update(_run_dics(ctx, epochs, fwd, space, rank))
        results[space] = space_results
        for name, stc in space_results.items():
            _save_stc(ctx, stc, name, space)
        _label_and_plot(ctx, space, fwd, space_results)

    return results


def _estimate_rank(ctx: SubjectContext, epochs: mne.Epochs):
    """Data rank, which HFC and ICA have reduced below the channel count.

    The beamformer inverts a covariance/CSD matrix, so it must be told how many
    dimensions actually carry data; otherwise the inversion amplifies numerical
    noise in the null space that HFC created.
    """
    setting = ctx.cfg.source.rank
    if setting == "none":
        return None
    if setting == "relative":
        rank = mne.compute_rank(epochs, tol=1e-5, tol_kind="relative", proj=True, verbose="ERROR")
    else:
        rank = mne.compute_rank(epochs, rank="info", verbose="ERROR")
    ctx.record(source_rank=dict(rank), n_source_channels=len(epochs.ch_names))
    ctx.logger.info("source: rank %s over %d channels", dict(rank), len(epochs.ch_names))
    return rank


def _select(epochs: mne.Epochs, conditions: list[str]) -> mne.Epochs:
    """Pool the named conditions, or keep everything when none are named."""
    if not conditions:
        return epochs
    present = []
    for name in conditions:
        try:
            if len(epochs[name]):
                present.append(name)
        except KeyError:
            continue
    if not present:
        raise RuntimeError(f"none of the conditions {conditions} have epochs")
    return epochs[present]


def _run_lcmv(ctx: SubjectContext, epochs, fwd, space: str, rank) -> dict:
    """Time-domain beamformer plus a response-versus-baseline power map."""
    cfg = ctx.cfg.source
    lcmv_cfg = cfg.lcmv
    selected = _select(epochs, lcmv_cfg.conditions)

    with timed(f"LCMV ({space})", ctx.logger):
        data_cov = mne.compute_covariance(
            selected, tmin=lcmv_cfg.cov_tmin, tmax=lcmv_cfg.cov_tmax,
            method=lcmv_cfg.cov_method, rank=rank, verbose="ERROR",
        )
        filters = make_lcmv(
            selected.info, fwd, data_cov, reg=cfg.reg, noise_cov=None,
            pick_ori=lcmv_cfg.pick_ori, weight_norm=lcmv_cfg.weight_norm,
            reduce_rank=cfg.reduce_rank, depth=cfg.depth, rank=rank, verbose="ERROR",
        )
        stc = apply_lcmv(selected.average(), filters, verbose="ERROR")

    power_map = _power_change_db(stc, lcmv_cfg.baseline_window, lcmv_cfg.active_window)
    ctx.record(**{
        f"lcmv_{space}_peak_db": float(np.nanmax(power_map.data)),
        f"lcmv_{space}_n_trials": int(len(selected)),
    })
    ctx.cache[f"lcmv_timecourse_{space}"] = stc
    return {"lcmv": stc, "lcmv-powerdb": power_map}


def _power_change_db(stc, baseline_window, active_window):
    """Per-source dB change between an active and a baseline interval."""
    baseline = (stc.times >= baseline_window[0]) & (stc.times <= baseline_window[1])
    active = (stc.times >= active_window[0]) & (stc.times <= active_window[1])
    if not baseline.any() or not active.any():
        raise ValueError(
            f"LCMV baseline {baseline_window} or active {active_window} window falls outside "
            f"the epoch ({stc.times[0]:.2f} to {stc.times[-1]:.2f} s)"
        )
    baseline_power = np.mean(np.abs(stc.data[:, baseline]) ** 2, axis=1)
    active_power = np.mean(np.abs(stc.data[:, active]) ** 2, axis=1)
    floor = np.finfo(float).eps * max(float(np.nanmax(baseline_power)), np.finfo(float).tiny)
    db = 10 * np.log10(np.maximum(active_power, floor) / np.maximum(baseline_power, floor))
    out = stc.copy().crop(stc.times[0], stc.times[0])
    out._data = db[:, np.newaxis]
    return out


def _run_dics(ctx: SubjectContext, epochs, fwd, space: str, rank) -> dict:
    """Frequency-domain beamformer, one common filter per band."""
    cfg = ctx.cfg.source
    out: dict[str, mne.SourceEstimate] = {}

    for band in cfg.dics.bands:
        selected = _select(epochs, band.conditions)
        if not band.windows:
            ctx.logger.warning("DICS band %r defines no time windows; skipping", band.name)
            continue

        csds: dict[str, object] = {}
        with timed(f"DICS {band.name} CSDs ({space})", ctx.logger):
            for window, (tmin, tmax) in band.windows.items():
                # csd_multitaper already restricts to [fmin, fmax], so mean()
                # with no arguments averages exactly that band. Passing the
                # edges instead would require them to fall on FFT bin centres.
                csds[window] = csd_multitaper(
                    selected, fmin=band.fmin, fmax=band.fmax, tmin=tmin, tmax=tmax,
                    bandwidth=band.bandwidth, adaptive=band.adaptive,
                    low_bias=band.low_bias, n_jobs=1, verbose="ERROR",
                ).mean()

        # One filter for every window, so a later contrast reflects the data and
        # not a difference between filters.
        common = _average_csds(list(csds.values()))
        filters = make_dics(
            selected.info, fwd, common, reg=cfg.reg, noise_csd=None,
            pick_ori="max-power", reduce_rank=cfg.reduce_rank,
            real_filter=cfg.real_filter, depth=cfg.depth, rank=rank, verbose="ERROR",
        )
        window_stcs = {
            window: apply_dics_csd(csd, filters, verbose="ERROR")[0]
            for window, csd in csds.items()
        }
        for window, stc in window_stcs.items():
            out[f"dics-{sanitize(band.name)}-{sanitize(window)}"] = stc

        for contrast in band.contrasts:
            a, b = window_stcs[contrast.a], window_stcs[contrast.b]
            name = f"dics-{sanitize(band.name)}-{sanitize(contrast.name)}"
            out[name] = _stc_contrast(a, b, contrast.kind)
            ctx.record(**{
                f"{name}_{space}_peak": float(np.nanmax(np.real(out[name].data))),
                f"{name}_{space}_trough": float(np.nanmin(np.real(out[name].data))),
            })
    return out


def _average_csds(csds: list):
    """Mean of several cross-spectral density matrices.

    ``CrossSpectralDensity`` has no public arithmetic, so the underlying arrays
    are averaged directly, exactly as the FLUX DICS tutorial does.
    """
    if len(csds) == 1:
        return csds[0].copy()
    shapes = {c._data.shape for c in csds}
    if len(shapes) != 1:
        raise ValueError(f"cannot average CSDs with differing shapes: {shapes}")
    common = csds[0].copy()
    common._data = np.mean([c._data for c in csds], axis=0)
    return common


def _stc_contrast(a, b, kind: str):
    """Relative contrasts; they cancel most of the beamformer's depth bias."""
    tiny = np.finfo(float).eps * float(np.nanmax(np.abs(a.data) + np.abs(b.data)) or 1.0)
    if kind == "relative_change":
        return (a - b) / (b + tiny)
    if kind == "normalised_difference":
        return (a - b) / (a + b + tiny)
    if kind == "difference":
        return a - b
    raise ValueError(f"unknown source contrast kind {kind!r}")


def _save_stc(ctx: SubjectContext, stc, name: str, space: str) -> None:
    stc.save(ctx.paths.stc(name, space), ftype="h5", overwrite=True, verbose="ERROR")


# --------------------------------------------------------------------------- #
# Peak labelling and figures
# --------------------------------------------------------------------------- #


def _label_and_plot(ctx: SubjectContext, space: str, fwd, results: dict) -> None:
    peaks = {}
    for name, stc in results.items():
        if name == "lcmv":  # the time course itself has no single peak map
            continue
        try:
            if space == "volume":
                peaks[name] = _volume_peak(ctx, stc, fwd)
                _plot_volume(ctx, stc, fwd, name)
            else:
                peaks[name] = _surface_peak(ctx, stc)
                _plot_surface(ctx, stc, name)
        except (RuntimeError, ValueError, FileNotFoundError, KeyError) as exc:
            ctx.logger.warning("source: could not label/plot %s (%s): %s", name, space, exc)
    if peaks:
        ctx.record(**{f"source_peaks_{space}": peaks})


def _volume_peak(ctx: SubjectContext, stc, fwd) -> dict:
    """Peak voxel: its FreeSurfer parcel and its MNI coordinate."""
    import nibabel as nib
    from nibabel.processing import resample_from_to
    from nilearn import image as nli
    from scipy.ndimage import distance_transform_edt

    subjects_dir = ctx.cfg.fs_subjects_dir
    fs_subject = ctx.fs_subject
    values = np.real(stc.data[:, 0])
    index = int(np.nanargmax(values))

    vertices = np.asarray(stc.vertices[0], int)
    head_coords = fwd["src"][0]["rr"][vertices]
    mni = mne.head_to_mni(
        head_coords, subject=fs_subject, mri_head_t=fwd["mri_head_t"],
        subjects_dir=subjects_dir, verbose="ERROR",
    )

    peak = {
        "value": float(values[index]),
        "mni_x_mm": float(mni[index, 0]),
        "mni_y_mm": float(mni[index, 1]),
        "mni_z_mm": float(mni[index, 2]),
    }

    aseg_path = subjects_dir / fs_subject / "mri" / "aparc+aseg.mgz"
    if not aseg_path.exists():
        peak["parcel"] = "aparc+aseg.mgz not available"
        return peak

    img = nli.index_img(stc.as_volume(fwd["src"], dest="mri", mri_resolution=False), 0)
    atlas = resample_from_to(nib.load(str(aseg_path)), img, order=0).get_fdata()
    data = img.get_fdata()
    voxel = np.unravel_index(int(np.nanargmax(data)), data.shape)

    lut, _ = mne.read_freesurfer_lut()
    names = {value: key for key, value in lut.items()}
    label_id = int(np.rint(atlas[voxel]))
    peak["parcel"] = names.get(label_id, f"unknown label {label_id}")
    peak["parcel_assignment"] = "peak voxel"
    peak["parcel_distance_mm"] = 0.0

    if label_id == 0:
        # A 5 mm grid point can fall just outside the thin cortical ribbon; in
        # that case report the nearest cortical parcel and how far away it is.
        cortical = np.array([v for k, v in lut.items() if k.startswith("ctx-")], dtype=int)
        mask = np.isin(np.rint(atlas).astype(int), cortical)
        if mask.any():
            sizes = nib.affines.voxel_sizes(img.affine)
            distances, nearest = distance_transform_edt(~mask, sampling=sizes, return_indices=True)
            nearest_voxel = tuple(nearest[:, voxel[0], voxel[1], voxel[2]])
            label_id = int(np.rint(atlas[nearest_voxel]))
            peak["parcel"] = names.get(label_id, f"unknown label {label_id}")
            peak["parcel_assignment"] = "nearest cortical parcel"
            peak["parcel_distance_mm"] = float(distances[voxel])
    peak["atlas_label_id"] = label_id
    return peak


def _surface_peak(ctx: SubjectContext, stc) -> dict:
    values = np.real(stc.data[:, 0])
    index = int(np.nanargmax(values))
    n_lh = len(stc.vertices[0])
    hemi = "lh" if index < n_lh else "rh"
    vertex = int(stc.vertices[0 if hemi == "lh" else 1][index if hemi == "lh" else index - n_lh])
    labels = mne.read_labels_from_annot(
        ctx.fs_subject, parc="aparc", hemi=hemi,
        subjects_dir=ctx.cfg.fs_subjects_dir, verbose="ERROR",
    )
    matches = [label.name for label in labels if vertex in label.vertices]
    return {
        "value": float(values[index]),
        "hemisphere": hemi,
        "vertex": vertex,
        "parcel": matches[0] if matches else f"unlabelled-{hemi}",
    }


def _plot_volume(ctx: SubjectContext, stc, fwd, name: str) -> None:
    from nilearn import image as nli, plotting as nlp
    import matplotlib.pyplot as plt

    if not ctx.cfg.output.figures:
        return
    t1 = ctx.cfg.fs_subjects_dir / ctx.fs_subject / "mri" / "T1.mgz"
    img = nli.index_img(stc.as_volume(fwd["src"], dest="mri", mri_resolution=False), 0)
    values = np.abs(np.real(stc.data[:, 0]))
    # An exploratory display threshold, not a statistical one: the 95th
    # percentile suppresses the diffuse low-amplitude background.
    threshold = float(np.nanpercentile(values, 95))
    vmax = max(float(np.nanpercentile(values, 99)), threshold + np.finfo(float).eps)

    fig = plt.figure(figsize=(11, 3.6))
    nlp.plot_stat_map(
        img, bg_img=str(t1) if t1.exists() else None, display_mode="ortho",
        threshold=threshold, vmax=vmax, symmetric_cbar=True, cmap="RdBu_r",
        colorbar=True, black_bg=False, dim=-0.25, figure=fig,
        title=f"{ctx.rec.key}: {name}",
    )
    path = save_figure(fig, ctx.figure_path(f"10_source_volume_{sanitize(name)}"), ctx.cfg)
    ctx.add_figure("Source (volume)", name, path)


def _plot_surface(ctx: SubjectContext, stc, name: str) -> None:
    import warnings

    if not ctx.cfg.output.figures:
        return
    values = np.abs(np.real(stc.data[:, 0]))
    threshold = float(np.nanpercentile(values, 95))
    vmax = max(float(np.nanpercentile(values, 99)), threshold + np.finfo(float).eps)
    clim = dict(kind="value", pos_lims=[threshold, (threshold + vmax) / 2, vmax])
    for hemi in ("lh", "rh"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                # The Matplotlib backend renders one hemisphere and view at a
                # time but needs no 3D/OpenGL stack, so it works on a cluster.
                fig = stc.plot(
                    subject=ctx.fs_subject, subjects_dir=ctx.cfg.fs_subjects_dir,
                    surface="inflated", hemi=hemi, views="lat", colormap="RdBu_r",
                    clim=clim, transparent=True, smoothing_steps=5, colorbar=True,
                    background="white", foreground="black", initial_time=stc.times[0],
                    time_viewer=False, show_traces=False, backend="matplotlib",
                )
            path = save_figure(
                fig, ctx.figure_path(f"10_source_surface_{sanitize(name)}_{hemi}"), ctx.cfg
            )
            ctx.add_figure("Source (surface)", f"{name} ({hemi})", path)
        except (RuntimeError, ValueError, TypeError) as exc:
            ctx.logger.warning("source: surface plot failed for %s %s (%s)", name, hemi, exc)


# --------------------------------------------------------------------------- #
# 12. Morphing to a template
# --------------------------------------------------------------------------- #


def morph_sources(ctx: SubjectContext, results: dict | None = None) -> dict:
    """Move each source estimate onto ``fsaverage`` for group analysis.

    Individual source spaces have different grids and different numbers of
    vertices, so subjects can only be averaged after a common template has been
    imposed.  Morphs are cached per source space, since computing one is far
    more expensive than applying it.
    """
    cfg = ctx.cfg.morph
    if not cfg.enabled:
        return {}
    subjects_dir = ctx.cfg.fs_subjects_dir
    template = cfg.subject_to
    if not (subjects_dir / template).is_dir():
        if cfg.fetch_fsaverage:
            try:
                mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir, verbose="ERROR")
            except Exception as exc:  # network, permissions, disk - all non-fatal
                ctx.logger.warning(
                    "morph skipped: template %r is missing from %s and could not be "
                    "downloaded (%s). Install it once and re-run this stage.",
                    template, subjects_dir, exc,
                )
                return {}
        else:
            ctx.logger.warning("morph skipped: template %r not in %s", template, subjects_dir)
            return {}

    if results is None:
        results = _load_native_stcs(ctx)

    morphed: dict[str, dict] = {}
    for space, stcs in results.items():
        if not stcs:
            continue
        src = mne.read_source_spaces(ctx.paths.src(space), verbose="ERROR")
        try:
            with timed(f"morph to {template} ({space})", ctx.logger):
                morph = mne.compute_source_morph(
                    src, subject_from=ctx.fs_subject, subject_to=template,
                    subjects_dir=subjects_dir,
                    zooms=cfg.volume_zooms if space == "volume" else "auto",
                    spacing=cfg.surface_spacing if space == "surface" else 5,
                    verbose="ERROR",
                )
        except (RuntimeError, ValueError, ImportError) as exc:
            ctx.logger.warning("morph failed for the %s source space (%s)", space, exc)
            continue
        morphed[space] = {}
        for name, stc in stcs.items():
            if name == "lcmv":
                continue  # the full time course is large and not needed at group level
            out = morph.apply(stc, verbose="ERROR")
            out.save(ctx.paths.stc(name, space, morphed=True), ftype="h5",
                     overwrite=True, verbose="ERROR")
            morphed[space][name] = out
        ctx.logger.info("morph: wrote %d %s estimates on %s", len(morphed[space]), space, template)
    ctx.state["morphed"] = {space: sorted(v) for space, v in morphed.items()}
    return morphed


def _load_native_stcs(ctx: SubjectContext) -> dict:
    """Reload the source estimates written by a previous run of stage 11."""
    out: dict[str, dict] = {}
    for space in ctx.cfg.source.spaces:
        found = {}
        for path in sorted(ctx.paths.analysis_dir.glob(f"{ctx.rec.key}_*_{space}_native-*.h5")):
            name = path.name[len(ctx.rec.key) + 1:].rsplit(f"_{space}_native", 1)[0]
            found[name] = mne.read_source_estimate(path, subject=ctx.fs_subject)
        if found:
            out[space] = found
    return out
