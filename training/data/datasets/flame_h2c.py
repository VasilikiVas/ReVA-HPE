import os
import os.path as osp
import logging
import random
import glob
import pickle
import csv
from contextlib import contextmanager
from collections import defaultdict

import cv2
import numpy as np
import albumentations as A

from data.base_dataset import BaseDataset
from data.dataset_util import read_image_cv2


@contextmanager
def temporary_numpy_python_seed(seed):
    py_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


class FlameH2CDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        root_dirs: list = None,
        len_train: int = 200000,
        len_test: int = 5000,
        num_views: int = 2,
        same_identity: bool = True,
        same_lighting: bool = True,
        same_expression: bool = False,
        depth_max: float = 10.0,
        generate_tracks: bool = False,
        num_tracks: int = 1024,
        track_occlusion_eps: float = 1e-2,
        max_track_sample_tries: int = 20,
        n_val_identities_per_root: int = 5,
        split_seed: int = 42,
        face_crop: bool = True,
        face_crop_mode: str = "per_view",
        ad_min: float = 1.0,
        ad_max: float = 1.0,
        shift_ratio: float = 0.0,
        bbox_filename: str = "face_bbox.npy",
        color_aug: bool = True,
        extra_cojitter_ratio: float = 1.0,  
        max_h2c_rot_deg: float = None,
        max_h2c_trans: float = None,
        nearest_view_pool_size: int = None,
        strict_view_constraints: bool = False,
        view_constraint_max_tries: int = 20,
        pair_csv: str = None,
        color_aug_seed_base: int = 20260529,
    ):
        super().__init__(common_conf=common_conf)
        if root_dirs is None:
            root_dirs = [
                {
                    "path": "data/blended_in_flames/sample_outputs/dataset_20k_hair_v2",
                    "name": "hair",
                },
                {
                    "path": "data/blended_in_flames/sample_outputs/ar_pose_poc_5k",
                    "name": "nohair",
                },
            ]

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.root_dirs = root_dirs
        self.split = split
        self.num_views = num_views
        self.same_identity = same_identity
        self.same_lighting = same_lighting
        self.same_expression = same_expression
        self.depth_max = depth_max
        self.generate_tracks = generate_tracks
        self.num_tracks = num_tracks
        self.track_occlusion_eps = track_occlusion_eps
        self.max_track_sample_tries = max_track_sample_tries
        self.max_h2c_rot_deg = None if max_h2c_rot_deg is None else float(max_h2c_rot_deg)
        self.max_h2c_trans = None if max_h2c_trans is None else float(max_h2c_trans)
        self.nearest_view_pool_size = (
            None if nearest_view_pool_size in (None, 0) else int(nearest_view_pool_size)
        )
        self.strict_view_constraints = bool(strict_view_constraints)
        self.view_constraint_max_tries = max(1, int(view_constraint_max_tries))
        self.pair_csv = pair_csv
        self.fixed_pair_rows = []
        self.color_aug_seed_base = int(color_aug_seed_base)

        self.n_val_per_root = n_val_identities_per_root
        self.split_seed = split_seed

        self.face_crop = face_crop
        self.face_crop_mode = face_crop_mode
        self.ad_min = ad_min
        self.ad_max = ad_max
        self.shift_ratio = shift_ratio
        self.bbox_filename = bbox_filename
        valid_crop_modes = {"per_view", "pair_union"}
        if self.face_crop_mode not in valid_crop_modes:
            raise ValueError(
                f"face_crop_mode must be one of {sorted(valid_crop_modes)}, got {self.face_crop_mode}"
            )

        self.extra_color_aug = color_aug and self.training
        self.extra_cojitter_ratio = extra_cojitter_ratio
        if self.extra_color_aug:
            self.extra_color_transform = A.ReplayCompose([
                A.RandomGamma(p=0.5),
                A.CLAHE(p=0.25),
                A.RGBShift(p=0.25),
                A.Blur(p=0.1),
                A.GaussNoise(p=0.5),
            ])

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        self._build_index()
        if self.pair_csv:
            self._load_pair_csv(self.pair_csv)

        status = "Training" if self.training else "Testing"
        logging.info(
            f"{status}: FlameH2C split={split}, "
            f"identities={len(self.identity_list)}, "
            f"views={len(self.all_views)}, "
            f"epoch_length={len(self)}, "
            f"pair_csv={self.pair_csv or 'random-sampling'}"
        )

    # ------------------------------------------------------------------
    # Identity-based split
    # ------------------------------------------------------------------
    def _split_identities_for_root(self, root_path, root_name):
        identity_dirs = sorted(glob.glob(osp.join(root_path, "identity_*")))
        identity_names = [osp.basename(d) for d in identity_dirs]

        rng = np.random.RandomState(self.split_seed)
        rng.shuffle(identity_names)

        n_val = min(self.n_val_per_root, len(identity_names))
        val_names = set(identity_names[:n_val])
        train_names = set(identity_names[n_val:])

        logging.info(
            f"  {root_name} ({root_path}): "
            f"{len(train_names)} train + {n_val} val identities"
        )

        train_ids = {(root_name, name) for name in train_names}
        val_ids = {(root_name, name) for name in val_names}
        return train_ids, val_ids

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------
    def _build_index(self):
        self.identities = {}
        self.all_views = []

        # Compute split for each root
        all_train_ids = set()
        all_val_ids = set()

        for root_info in self.root_dirs:
            root_path = root_info["path"]
            root_name = root_info["name"]
            train_ids, val_ids = self._split_identities_for_root(root_path, root_name)
            all_train_ids |= train_ids
            all_val_ids |= val_ids

        self.train_identities = all_train_ids
        self.val_identities = all_val_ids
        split_ids = all_val_ids if self.split == "test" else all_train_ids

        # Build view index for this split
        for root_info in self.root_dirs:
            root_path = root_info["path"]
            root_name = root_info["name"]

            identity_dirs = sorted(glob.glob(osp.join(root_path, "identity_*")))
            for identity_dir in identity_dirs:
                id_name = osp.basename(identity_dir)
                id_key = (root_name, id_name)

                if id_key not in split_ids:
                    continue

                # Use id_key as the identity key to avoid name collisions across roots
                self.identities[id_key] = {"lightings": {}}

                for lighting_dir in sorted(glob.glob(osp.join(identity_dir, "lighting_*"))):
                    lt_name = osp.basename(lighting_dir)
                    self.identities[id_key]["lightings"][lt_name] = {"expressions": {}}

                    for expr_dir in sorted(glob.glob(osp.join(lighting_dir, "expr_*"))):
                        ex_name = osp.basename(expr_dir)
                        self.identities[id_key]["lightings"][lt_name]["expressions"][ex_name] = {"views": []}

                        for view_dir in sorted(glob.glob(osp.join(expr_dir, "view_*"))):
                            view_info = {
                                "path": view_dir,
                                "root_name": root_name,
                                "identity": id_key,  # (root_name, identity_name)
                                "lighting": lt_name,
                                "expression": ex_name,
                                "view": osp.basename(view_dir),
                            }
                            self.identities[id_key]["lightings"][lt_name]["expressions"][ex_name]["views"].append(view_info)
                            self.all_views.append(view_info)

        self.identity_list = list(self.identities.keys())

        self.views_by_id_light = defaultdict(list)
        for v in self.all_views:
            key = (v["identity"], v["lighting"])
            self.views_by_id_light[key].append(v)

    # ------------------------------------------------------------------
    # Optional fixed pair CSV
    # ------------------------------------------------------------------
    def _load_pair_csv(self, pair_csv):
        required = {
            "root_name",
            "identity_name",
            "lighting",
            "anchor_expression",
            "anchor_view",
            "anchor_path",
            "query_expression",
            "query_view",
            "query_path",
        }
        with open(pair_csv, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            missing = required.difference(fieldnames)
            if missing:
                raise ValueError(f"pair_csv {pair_csv} is missing columns: {sorted(missing)}")
            rows = list(reader)

        if not rows:
            raise ValueError(f"pair_csv {pair_csv} is empty")

        self.fixed_pair_rows = rows
        self.len_train = len(rows)
        logging.info(f"Loaded {len(rows)} fixed FLAME pairs from {pair_csv}")

    def _view_info_from_pair_row(self, row, prefix):
        root_name = row.get(f"{prefix}_root_name") or row["root_name"]
        identity_name = row.get(f"{prefix}_identity_name") or row["identity_name"]
        lighting = row.get(f"{prefix}_lighting") or row["lighting"]
        return {
            "path": row[f"{prefix}_path"],
            "root_name": root_name,
            "identity": (root_name, identity_name),
            "lighting": lighting,
            "expression": row[f"{prefix}_expression"],
            "view": row[f"{prefix}_view"],
        }

    def _views_from_pair_csv(self, seq_index, requested_views):
        if requested_views not in (1, 2):
            raise ValueError(
                f"pair_csv mode supports one or two requested views, got {requested_views}"
            )
        if seq_index is None:
            row_idx = random.randrange(len(self.fixed_pair_rows))
        else:
            row_idx = int(seq_index) % len(self.fixed_pair_rows)
        row = self.fixed_pair_rows[row_idx]

        query_view = self._view_info_from_pair_row(row, "query")
        if requested_views == 1:
            return [query_view], row, row_idx
        anchor_view = self._view_info_from_pair_row(row, "anchor")
        return [anchor_view, query_view], row, row_idx

    def _augmentation_seed_from_pair_row(self, row, row_idx):
        pair_index = int(row.get("pair_index", row_idx))
        split_offset = 1000000 if row.get("split") == "test" else 0
        return self.color_aug_seed_base + split_offset + pair_index

    # ------------------------------------------------------------------
    # H2C and intrinsics loading
    # ------------------------------------------------------------------
    def _load_h2c(self, view_dir):
        cam_path = osp.join(view_dir, "output_opencv_camera.pkl")
        obj_path = osp.join(view_dir, "output_object_transform_post_render.pkl")

        with open(cam_path, "rb") as f:
            cam = pickle.load(f)
        with open(obj_path, "rb") as f:
            obj = pickle.load(f)

        R_w2c = np.array(cam["R"], dtype=np.float64)
        t_w2c = np.array(cam["t"], dtype=np.float64).flatten()

        R_h2w = np.array(obj["R_world"], dtype=np.float64)
        t_h2w = np.array(obj["t_world"], dtype=np.float64).flatten()

        R_h2c = R_w2c @ R_h2w
        t_h2c = R_w2c @ t_h2w + t_w2c
        return R_h2c, t_h2c

    def _load_intrinsics(self, view_dir):
        cam_path = osp.join(view_dir, "output_opencv_camera.pkl")
        with open(cam_path, "rb") as f:
            cam = pickle.load(f)
        K = np.array(cam["K"], dtype=np.float64).reshape(3, 3)
        return K

    def _get_cached_h2c(self, view_info):
        pose = view_info.get("_cached_h2c")
        if pose is None:
            pose = self._load_h2c(view_info["path"])
            view_info["_cached_h2c"] = pose
        return pose

    def _relative_h2c_delta(self, anchor_view, candidate_view):
        R_anchor, t_anchor = self._get_cached_h2c(anchor_view)
        R_cand, t_cand = self._get_cached_h2c(candidate_view)

        R_rel = R_cand @ R_anchor.T
        trace = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
        rot_deg = float(np.degrees(np.arccos(trace)))
        trans = float(np.linalg.norm(t_cand - t_anchor))
        return rot_deg, trans

    def _apply_view_constraints(self, anchor_view, candidates):
        use_constraints = any(
            value is not None
            for value in (
                self.max_h2c_rot_deg,
                self.max_h2c_trans,
                self.nearest_view_pool_size,
            )
        )
        if not use_constraints:
            return candidates

        scored_candidates = []
        for candidate in candidates:
            rot_deg, trans = self._relative_h2c_delta(anchor_view, candidate)
            if self.max_h2c_rot_deg is not None and rot_deg > self.max_h2c_rot_deg:
                continue
            if self.max_h2c_trans is not None and trans > self.max_h2c_trans:
                continue
            scored_candidates.append((rot_deg, trans, candidate))

        if not scored_candidates:
            return []

        scored_candidates.sort(key=lambda item: (item[0], item[1]))
        if self.nearest_view_pool_size is not None:
            scored_candidates = scored_candidates[: self.nearest_view_pool_size]

        return [candidate for _, _, candidate in scored_candidates]

    def _load_mesh_geometry(self, view_path):
        mesh_path = osp.join(view_path, "output_mesh_post_render.obj")
        if not osp.exists(mesh_path):
            mesh_path = osp.join(view_path, "output_mesh.obj")
        if not osp.exists(mesh_path):
            return None, None

        vertices = []
        normals = []
        faces = []
        with open(mesh_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("v "):
                    parts = line.strip().split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith("vn "):
                    parts = line.strip().split()
                    normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith("f "):
                    parts = line.strip().split()[1:]
                    face = []
                    for token in parts:
                        vidx = token.split("/")[0]
                        if vidx:
                            face.append(int(vidx) - 1)
                    if len(face) >= 3:
                        for face_idx in range(1, len(face) - 1):
                            faces.append((face[0], face[face_idx], face[face_idx + 1]))

        if not vertices:
            return None, None

        verts = np.asarray(vertices, dtype=np.float32)
        norms = None
        if len(normals) == len(vertices):
            norms = np.asarray(normals, dtype=np.float32)
            norms /= np.maximum(np.linalg.norm(norms, axis=1, keepdims=True), 1e-8)
            center = verts.mean(axis=0, keepdims=True)
            radial = verts - center
            flip = np.sum(norms * radial, axis=1) < 0
            norms[flip] *= -1.0
        elif len(normals) > 0:
            logging.warning(
                f"Mesh normals count ({len(normals)}) != vertices count ({len(vertices)}) in {mesh_path}, "
                "recomputing normals from faces."
            )
        if norms is None and len(faces) > 0:
            norms = np.zeros_like(verts, dtype=np.float32)
            for i0, i1, i2 in faces:
                v0, v1, v2 = verts[i0], verts[i1], verts[i2]
                face_normal = np.cross(v1 - v0, v2 - v0)
                norms[i0] += face_normal
                norms[i1] += face_normal
                norms[i2] += face_normal
            norms /= np.maximum(np.linalg.norm(norms, axis=1, keepdims=True), 1e-8)
            center = verts.mean(axis=0, keepdims=True)
            radial = verts - center
            flip = np.sum(norms * radial, axis=1) < 0
            norms[flip] *= -1.0

        return verts, norms

    def _generate_tracks(self, view_infos, extrinsics_list, intrinsics_list, image_hw):
        seq_len = len(view_infos)
        height, width = image_hw

        verts_per_view = []
        norms_per_view = []
        vertex_count = None
        for view_info in view_infos:
            verts, norms = self._load_mesh_geometry(view_info["path"])
            if verts is None:
                logging.warning(f"No mesh file found in {view_info['path']}, skipping GT tracks.")
                return None, None
            if vertex_count is None:
                vertex_count = verts.shape[0]
            elif verts.shape[0] != vertex_count:
                logging.warning(
                    f"Mesh vertex count mismatch across views in sequence {view_infos[0]['path']}: "
                    f"expected {vertex_count}, got {verts.shape[0]}"
                )
                return None, None
            verts_per_view.append(verts)
            norms_per_view.append(norms)

        all_uv = []
        all_vis = []
        for verts, norms, extri, intri in zip(verts_per_view, norms_per_view, extrinsics_list, intrinsics_list):
            rotation = extri[:3, :3]
            translation = extri[:3, 3]
            cam_points = (verts @ rotation.T) + translation
            depth = cam_points[:, 2]

            uv_h = (intri @ cam_points.T).T
            uv = uv_h[:, :2] / np.maximum(uv_h[:, 2:3], 1e-6)

            vis = (
                (depth > 0)
                & (uv[:, 0] >= 0) & (uv[:, 0] < width)
                & (uv[:, 1] >= 0) & (uv[:, 1] < height)
            )

            if np.any(vis):
                px = np.rint(uv[:, 0]).astype(np.int32)
                py = np.rint(uv[:, 1]).astype(np.int32)
                in_pix = (px >= 0) & (px < width) & (py >= 0) & (py < height)
                vis_pix = vis & in_pix

                zbuf = np.full((height, width), np.inf, dtype=np.float32)
                np.minimum.at(zbuf, (py[vis_pix], px[vis_pix]), depth[vis_pix].astype(np.float32))

                zmin = np.full_like(depth, np.inf, dtype=np.float32)
                zmin[vis_pix] = zbuf[py[vis_pix], px[vis_pix]]
                vis = vis & (depth <= (zmin + self.track_occlusion_eps))

            if norms is not None:
                norms_cam = norms @ rotation.T
                view_dir = -cam_points / np.maximum(np.linalg.norm(cam_points, axis=1, keepdims=True), 1e-8)
                facing_score = np.sum(norms_cam * view_dir, axis=1)
                if np.any(depth > 0) and np.median(facing_score[depth > 0]) < 0:
                    facing_score = -facing_score
                vis = vis & (facing_score > 0.4)

            all_uv.append(uv.astype(np.float32))
            all_vis.append(vis)

        all_uv = np.stack(all_uv, axis=0)
        all_vis = np.stack(all_vis, axis=0)

        valid = all_vis[0] & all_vis[1:].any(axis=0)
        valid_idx = np.where(valid)[0]

        if len(valid_idx) < 10:
            logging.warning(
                f"Too few visible mesh vertices ({len(valid_idx)}) for GT tracks in {view_infos[0]['path']}"
            )
            return None, None

        track_count = min(self.num_tracks, len(valid_idx))
        sampled = np.random.choice(valid_idx, size=track_count, replace=False)

        tracks = all_uv[:, sampled, :]
        masks = all_vis[:, sampled]

        if track_count < self.num_tracks:
            pad = self.num_tracks - track_count
            tracks = np.concatenate([tracks, np.zeros((seq_len, pad, 2), dtype=np.float32)], axis=1)
            masks = np.concatenate([masks, np.zeros((seq_len, pad), dtype=bool)], axis=1)

        return tracks, masks

    # ------------------------------------------------------------------
    # Face crop from precomputed bbox
    # ------------------------------------------------------------------
    def _load_face_bbox(self, view_dir):
        bbox_path = osp.join(view_dir, self.bbox_filename)
        bbox = np.load(bbox_path)
        return bbox[0], bbox[1], bbox[2], bbox[3]

    def _pair_union_bbox(self, view_infos):
        bboxes = [self._load_face_bbox(view_info["path"]) for view_info in view_infos]
        return (
            min(float(b[0]) for b in bboxes),
            max(float(b[1]) for b in bboxes),
            min(float(b[2]) for b in bboxes),
            max(float(b[3]) for b in bboxes),
        )

    @staticmethod
    def _identity_crop_params(image_hw):
        return np.array([0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0], dtype=np.float32)

    @staticmethod
    def _crop_params_from_bounds(col_start, row_start, col_end, row_end, image_hw):
        H, W = image_hw
        x0 = float(col_start) / max(float(W), 1.0)
        y0 = float(row_start) / max(float(H), 1.0)
        x1 = float(col_end + 1) / max(float(W), 1.0)
        y1 = float(row_end + 1) / max(float(H), 1.0)
        w = x1 - x0
        h = y1 - y0
        return np.array(
            [x0, y0, x1, y1, x0 + 0.5 * w, y0 + 0.5 * h, w, h],
            dtype=np.float32,
        )

    def _face_crop(self, image, depth_map, K, xmin, xmax, ymin, ymax):
        H, W = image.shape[:2]
        augment = self.training

        bw = xmax - xmin
        bh = ymax - ymin

        ad = (
            np.random.uniform(self.ad_min, self.ad_max)
            if augment
            else (self.ad_min + self.ad_max) / 2.0
        )

        x1 = xmin - ad * bw
        y1 = ymin - ad * bh
        x2 = xmax + ad * bw
        y2 = ymax + ad * bh

        # Force square
        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w < crop_h:
            diff = crop_h - crop_w
            x1 -= diff / 2
            x2 += diff / 2
        elif crop_h < crop_w:
            diff = crop_w - crop_h
            y1 -= diff / 2
            y2 += diff / 2

        # Random shift
        if augment:
            side = max(x2 - x1, y2 - y1)
            shift_x = np.random.uniform(-self.shift_ratio, self.shift_ratio) * side
            shift_y = np.random.uniform(-self.shift_ratio, self.shift_ratio) * side
            x1 += shift_x
            x2 += shift_x
            y1 += shift_y
            y2 += shift_y

        # Clamp to image bounds
        col_start = max(int(x1), 0)
        row_start = max(int(y1), 0)
        col_end = min(int(x2), W - 1)
        row_end = min(int(y2), H - 1)

        # Re-equalize after clamping to keep square
        crop_w = col_end - col_start + 1
        crop_h = row_end - row_start + 1
        if crop_w > crop_h:
            excess = crop_w - crop_h
            col_start += excess // 2
            col_end = col_start + crop_h - 1
        elif crop_h > crop_w:
            excess = crop_h - crop_w
            row_start += excess // 2
            row_end = row_start + crop_w - 1

        crop_params = self._crop_params_from_bounds(
            col_start, row_start, col_end, row_end, (H, W)
        )

        image = image[row_start : row_end + 1, col_start : col_end + 1]
        if depth_map is not None:
            depth_map = depth_map[row_start : row_end + 1, col_start : col_end + 1]

        K = K.copy()
        K[0, 2] -= col_start
        K[1, 2] -= row_start

        return image, depth_map, K, crop_params

    # ------------------------------------------------------------------
    # View sampling
    # ------------------------------------------------------------------
    def _get_candidate_views(self, anchor_view):
        if self.same_identity and self.same_lighting:
            key = (anchor_view["identity"], anchor_view["lighting"])
            candidates = list(self.views_by_id_light[key])
        elif self.same_identity:
            candidates = []
            for lk, views in self.views_by_id_light.items():
                if lk[0] == anchor_view["identity"]:
                    candidates.extend(views)
        else:
            candidates = list(self.all_views)

        if self.same_expression:
            candidates = [v for v in candidates if v["expression"] == anchor_view["expression"]]

        candidates = [v for v in candidates if v["path"] != anchor_view["path"]]

        if len(candidates) == 0:
            expr_views = (
                self.identities[anchor_view["identity"]]["lightings"][anchor_view["lighting"]]
                ["expressions"][anchor_view["expression"]]["views"]
            )
            candidates = [v for v in expr_views if v["path"] != anchor_view["path"]]

        if len(candidates) == 0:
            candidates = [anchor_view]

        return candidates

    def _sample_views(self, num_views=None, seed=None):
        if num_views is None:
            num_views = self.num_views
        num_views = int(num_views)
        if num_views < 1:
            raise ValueError(f"num_views must be >= 1, got {num_views}")

        rng = np.random.RandomState(seed) if seed is not None else None
        use_constraints = any(
            value is not None
            for value in (
                self.max_h2c_rot_deg,
                self.max_h2c_trans,
                self.nearest_view_pool_size,
            )
        )
        max_tries = self.view_constraint_max_tries if use_constraints else 1
        fallback = None

        for _ in range(max_tries):
            if rng is not None:
                view1 = self.all_views[int(rng.randint(len(self.all_views)))]
            else:
                view1 = random.choice(self.all_views)

            views = [view1]
            if num_views == 1:
                return views

            candidates = self._get_candidate_views(view1)
            if fallback is None:
                fallback = (view1, candidates)

            constrained_candidates = self._apply_view_constraints(view1, candidates)
            if constrained_candidates:
                replace = len(constrained_candidates) < (num_views - 1)
                if rng is not None:
                    sampled_idxs = rng.choice(
                        len(constrained_candidates),
                        size=num_views - 1,
                        replace=replace,
                    )
                    extra_views = [
                        constrained_candidates[int(i)] for i in np.atleast_1d(sampled_idxs)
                    ]
                else:
                    if replace:
                        extra_views = [
                            random.choice(constrained_candidates)
                            for _ in range(num_views - 1)
                        ]
                    else:
                        extra_views = random.sample(constrained_candidates, num_views - 1)
                views.extend(extra_views)
                return views

            if not use_constraints:
                break

        if fallback is not None and not self.strict_view_constraints:
            view1, candidates = fallback
            replace = len(candidates) < (num_views - 1)
            if rng is not None:
                sampled_idxs = rng.choice(len(candidates), size=num_views - 1, replace=replace)
                extra_views = [candidates[int(i)] for i in np.atleast_1d(sampled_idxs)]
            else:
                if replace:
                    extra_views = [random.choice(candidates) for _ in range(num_views - 1)]
                else:
                    extra_views = random.sample(candidates, num_views - 1)
            return [view1] + extra_views

        raise RuntimeError(
            "Failed to sample views that satisfy the configured relative H2C constraints. "
            "Try relaxing max_h2c_rot_deg/max_h2c_trans, increasing nearest_view_pool_size, "
            "or disabling strict_view_constraints."
        )

    # ------------------------------------------------------------------
    # Main data loading
    # ------------------------------------------------------------------
    def get_data(
        self,
        seq_index=None,
        img_per_seq=None,
        seq_name=None,
        ids=None,
        aspect_ratio=1.0,
    ):
        requested_views = self.num_views if img_per_seq is None else img_per_seq
        target_image_shape = self.get_target_shape(aspect_ratio)
        max_attempts = self.max_track_sample_tries if self.generate_tracks else 1
        last_failure = "unknown track generation failure"

        for attempt_idx in range(max_attempts):
            pair_row = None
            pair_row_idx = None
            if self.fixed_pair_rows:
                view_infos, pair_row, pair_row_idx = self._views_from_pair_csv(seq_index, requested_views)
            else:
                attempt_seed = None
                if self.split == "test" and seq_index is not None:
                    attempt_seed = seq_index + attempt_idx

                view_infos = self._sample_views(num_views=requested_views, seed=attempt_seed)

            shared_crop_bbox = None
            skip_face_crop_for_pair = False
            if self.face_crop and self.face_crop_mode == "pair_union":
                try:
                    shared_crop_bbox = self._pair_union_bbox(view_infos)
                except FileNotFoundError:
                    logging.warning(
                        f"No {self.bbox_filename} for pair-union crop in {view_infos[0]['path']}, "
                        "falling back to uncropped pair"
                    )
                    skip_face_crop_for_pair = True

            images, depths, cam_points, world_points = [], [], [], []
            point_masks, extrinsics, intrinsics, original_sizes, crop_params = [], [], [], [], []

            for view_info in view_infos:
                view_dir = view_info["path"]

                image_path = osp.join(view_dir, "output.png")
                image = read_image_cv2(image_path)
                if image is None:
                    logging.warning(f"Could not read {image_path}")
                    continue

                view_crop_params = self._identity_crop_params(image.shape[:2])
                R_h2c, t_h2c = self._get_cached_h2c(view_info)
                intri = self._load_intrinsics(view_dir)

                depth_map = np.ones(image.shape[:2], dtype=np.float32)

                if self.face_crop and not skip_face_crop_for_pair:
                    try:
                        if shared_crop_bbox is None:
                            xmin, xmax, ymin, ymax = self._load_face_bbox(view_dir)
                        else:
                            xmin, xmax, ymin, ymax = shared_crop_bbox
                        image, depth_map, intri, view_crop_params = self._face_crop(
                            image, depth_map, intri, xmin, xmax, ymin, ymax,
                        )
                    except FileNotFoundError:
                        logging.warning(
                            f"No {self.bbox_filename} for {view_dir}, skipping face crop"
                        )
                    except Exception as e:
                        logging.warning(f"Face crop failed for {view_dir}: {e}")

                original_size = np.array(image.shape[:2])

                extri_h2c = np.zeros((3, 4), dtype=np.float64)
                extri_h2c[:3, :3] = R_h2c
                extri_h2c[:3, 3] = t_h2c

                (
                    image, depth_map, extri_h2c, intri,
                    world_coords_points, cam_coords_points, point_mask, _,
                ) = self.process_one_image(
                    image, depth_map, extri_h2c, intri,
                    original_size, target_image_shape,
                    filepath=image_path,
                )

                if (image.shape[:2] != target_image_shape).any():
                    logging.warning(
                        f"Wrong shape for {view_dir}: "
                        f"expected {target_image_shape}, got {image.shape[:2]}"
                    )
                    continue

                images.append(image)
                depths.append(depth_map)
                extrinsics.append(extri_h2c)
                intrinsics.append(intri)
                cam_points.append(cam_coords_points)
                world_points.append(world_coords_points)
                point_masks.append(point_mask)
                original_sizes.append(original_size)
                crop_params.append(view_crop_params)

            if len(extrinsics) != len(view_infos):
                last_failure = (
                    f"incomplete processed view set ({len(extrinsics)}/{len(view_infos)}) "
                    f"for {view_infos[0]['path']}"
                )
                if self.generate_tracks:
                    continue

            augmentation_seed = None
            if pair_row is not None:
                augmentation_seed = self._augmentation_seed_from_pair_row(pair_row, pair_row_idx)

            def _apply_extra_color_aug():
                def _set_transform_seed(seed):
                    if hasattr(self.extra_color_transform, "set_random_seed"):
                        self.extra_color_transform.set_random_seed(int(seed))

                def _augment_one_view(view_idx, seed=None):
                    if seed is None:
                        images[view_idx] = self.extra_color_transform(image=images[view_idx])["image"]
                        return
                    with temporary_numpy_python_seed(seed):
                        _set_transform_seed(seed)
                        images[view_idx] = self.extra_color_transform(image=images[view_idx])["image"]

                if self.extra_color_aug and len(images) >= 2:
                    use_shared = random.random() < self.extra_cojitter_ratio
                    if use_shared:
                        if augmentation_seed is not None:
                            _set_transform_seed(augmentation_seed)
                        result0 = self.extra_color_transform(image=images[0])
                        images[0] = result0["image"]
                        replay = result0["replay"]
                        images[1] = A.ReplayCompose.replay(replay, image=images[1])["image"]
                    else:
                        for i in range(len(images)):
                            view_seed = (
                                None
                                if augmentation_seed is None
                                else int(augmentation_seed) + 1009 * (i + 1)
                            )
                            _augment_one_view(i, view_seed)
                elif self.extra_color_aug and len(images) == 1:
                    view_seed = None if augmentation_seed is None else int(augmentation_seed) + 1009
                    _augment_one_view(0, view_seed)

            if augmentation_seed is not None:
                with temporary_numpy_python_seed(augmentation_seed):
                    _apply_extra_color_aug()
            else:
                _apply_extra_color_aug()

            id_key = view_infos[0]["identity"]
            seq_name = f"flame_h2c_{id_key[0]}_{id_key[1]}_{view_infos[0]['lighting']}"
            ids_array = np.arange(len(view_infos))

            tracks, track_masks = None, None
            if self.generate_tracks and len(extrinsics) == len(view_infos):
                image_hw = (target_image_shape[0], target_image_shape[1])
                tracks, track_masks = self._generate_tracks(
                    view_infos, extrinsics, intrinsics, image_hw
                )
                if tracks is None or track_masks is None:
                    last_failure = f"no valid GT tracks for {view_infos[0]['path']}"
                    continue

            return {
                "seq_name": seq_name,
                "ids": ids_array,
                "frame_num": len(extrinsics),
                "images": images,
                "depths": depths,
                "extrinsics": extrinsics,
                "intrinsics": intrinsics,
                "cam_points": cam_points,
                "world_points": world_points,
                "point_masks": point_masks,
                "original_sizes": original_sizes,
                "crop_params": crop_params,
                "tracks": tracks,
                "track_masks": track_masks,
                "view_paths": [v["path"] for v in view_infos],
                "mesh_paths": [osp.join(v["path"], "output_mesh.obj") for v in view_infos],
                "augmentation_seed": augmentation_seed,
            }

        raise RuntimeError(
            f"Failed to sample a valid FlameH2C example after {max_attempts} attempts: {last_failure}"
        )
