# Data

We cannot redistribute any data: FLAME and the HAAR hairstyles are
MPI-licensed, and BIWI and FaMoS come with their own licenses. The recipes
below regenerate everything from the original sources.

## 1. Synthetic training corpus (fixed-FoV, 25,000 FLAME renders)

The corpus is rendered procedurally with Blender and FLAME. Each of
the 250 identities is a FLAME 2023 shape with a sampled skin albedo texture,
200 identities additionally wear a HAAR hairstyle (identity/hairstyle
selection seed 42) and 50 are bald. Every identity is rendered under 2 HDRI
lighting environments x 10 expressions x 5 viewpoints with Blender 3.6.23
(Cycles), exporting per-view camera intrinsics and extrinsics alongside each
image. The renders are then converted to a shared fixed focal length (the
corpus maximum, fx ~= 2630.8) by rescaling every view accordingly,
hardlinking unchanged files, and writing a
`focal_normalization_manifest.json` per corpus. Finally, precompute the
per-view face boxes with `evaluation/precompute_face_bbox.py --root_dir
<corpus>` for each corpus — the training loader requires the resulting
`face_bbox.npy` files.

External assets (own licenses): FLAME 2023 files from
[MPI FLAME](https://flame.is.tue.mpg.de/), HAAR hairstyle assets from the
[HAAR](https://haar.is.tue.mpg.de/) authors, 665 PolyHaven HDRIs (CC0), 54
albedo textures.

Note: the FaMoS evaluator and the enrollment FLAME-fit stage load FLAME
fitting/rendering helpers at runtime from an external directory (point
`BLENDED_IN_FLAMES_ROOT` at it). That directory must provide
`trial_photo_to_5views.py` — the FLAME build/fit/color-sampling helpers —
together with the FLAME assets and modules it imports. The helpers are loaded
lazily, so everything else in this repository runs without them.

## 2. Fixed pair manifests

Point the script at your corpora first (pass `--root_sets` with a JSON mapping,
or edit `FREE_FOV_ROOTS`/`FIXED_FOV_ROOTS` at the top of the script, both the
free-FoV and the fixed-FoV corpora are required, with matching root names
across the two variants):

```bash
ROOTS='{"free_fov":  [{"path": "<corpora>/dataset_20k_hair_v2",  "name": "hair"},
                      {"path": "<corpora>/ar_pose_poc_5k",       "name": "nohair"}],
        "fixed_fov": [{"path": "<corpora>/dataset_20k_hair_v2_fixed_focal_max", "name": "hair"},
                      {"path": "<corpora>/ar_pose_poc_5k_fixed_focal_max",      "name": "nohair"}]}'
```

Use distinct output directories — each run writes a train and a test CSV
under the same names:

```bash
python evaluation/build_flame_fixed_pair_sets.py --root_sets "$ROOTS" --num_train_pairs 1000000 --pair_seed 20260531 --out_dir pair_manifests/large1m   # training manifest
python evaluation/build_flame_fixed_pair_sets.py --root_sets "$ROOTS" --pair_seed 20260529 --out_dir pair_manifests/step1                              # validation manifest
```

With the same seeds the builder selects the same pairs, so a regeneration
under your roots trains on the identical pair sequence (the CSVs embed
absolute image paths, which is why the manifests themselves are not shipped).
The shipped builder also writes a few extra provenance columns (per-side
root/identity/lighting names and a two-sided `group_id`) that our original
manifests did not have, the training loader reads only the shared columns, so
the pairs and their order are unchanged.
The consensus-threshold calibration of the supplementary runs on held-out
synthetic identities via `evaluation/eval_synthetic_heldout_anchor_fusion.py`,
its per-hypothesis outputs are scored into the threshold grid with
`reva/experiments/sweep_synthetic_ransac_thresholds.py` (clean and corrupted
sweeps) and plotted with
`reva/experiments/plot_corrupted_synthetic_criterion_heatmap.py`.

## 3. BIWI (evaluation)

Obtain from the original authors (Fanelli et al., IJCV 2013, ETH license),
the 24 recordings (of 20 individuals) are the per-directory subjects below.
`BIWI_ROOT` layout: `faces_0/01..24/` with `rgb.cal`, `frame_*_rgb.png`,
`frame_*_pose.txt`, **plus the per-subject `NN.obj` meshes** (required by the
PnP branch). The evaluation protocol is frozen under `protocols/`:
`biwi_table1_protocol.json` summarizes it, the 15,654 query keys are in
`biwi_table1_common_query_keys.csv`, the first-frame enrollments (with
per-image SHA-256 to verify your BIWI copy) in
`biwi_table1_standard_enrollments.csv`, and the detector-control exact-common
set of 14,980 frames detected by both MTCNN and RetinaFace (supplementary
Table 3) in `biwi_exact_common_pairs.csv`. The evaluators re-derive the same
protocol from the data, these files are the reference to check a run against.

## 4. FaMoS (evaluation)

Obtain FaMoS from [MPI:IS](https://tempeh.is.tue.mpg.de/) under their license.
The paper protocol uses camera view 00, with frame-level 6DoF labels obtained
by Procrustes-aligning each subject's FLAME-topology template to the provided
registrations (head-no-neck vertex mask). The exact 188 evaluated query
identifiers are frozen (without labels) in
`protocols/famos_188_query_keys.csv`, we share the derived labels and the
prepared evaluation CSV on request with researchers holding a FaMoS license. The protocol subset was derived with
`reva/experiments/filter_famos_frames.py` followed by
`filter_famos_face_visible_mtcnn.py` (the latter with `--views 0
--exclude-subjects FaMoS_subject_091`), enrollment manifests (per-group and
per-subject first-frame selections) come from
`prepare_famos_enrollment_manifest.py`, and the per-subject avatar banks from
`build_famos_facelift_manifest.py` + `run_famos_flame_enrollment_entry.py`
followed by the avatar-pipeline stages.

Building the evaluator's driver CSV (`FAMOS_READY_ROOT`), in run order: expose
FaMoS in the BIWI layout with `prepare_biwi_compatible_dataset.py` and
`build_biwi_compatible_pair_csv.py`, produce GT-anchor reference predictions
with `eval_famos_gt_anchor.py` and reshape them with
`convert_famos_gt_anchor_predictions.py`, then assemble everything with
`build_cross_dataset_refinement_inputs.py --protocol
famos_view00_mtcnn_visible_no091`.

Note: on FaMoS the enrollment EPnP runs 200 iterations with a 6 px
reprojection threshold and falls back to a least-squares refit over all
correspondences when fewer than six inliers survive (BIWI uses 300 iterations
/ 5 px / at least six inliers). The published FaMoS translations use these
settings.

## 5. Avatar banks

Banks render dataset subjects and are therefore never distributed. Rebuild per
subject from the single enrollment image with `reva/avatar_pipeline/`:
FaceLift reconstruction (`run_facelift_no_video.py`) -> multiview landmarks ->
triangulation -> FLAME fit -> similarity -> `generate_metric_anchor_bank.py`
(20 front-hemisphere directions, radius scales 0.85-1.15, the five banks
together give the 100 Direct-5r candidates).
