"""Shared BIWI evaluation library: dataset parsing, MTCNN crops, the BIWI
Euler/wrap-around metrics, VGGT+LoRA model loading, pose decoding, and the
mesh-overlay renderer used by the evaluators."""

import os, sys, io, gc, glob, re, csv, logging
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import PolyCollection
from PIL import Image
from torchvision import transforms as TF

# ── Paths / defaults ──────────────────────────────────────────────────────────
SCRIPT_DIR            = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEST_SUBJECTS = [5, 6, 9, 14, 16, 17, 20, 24]
RELATIVE_VARIANT_NAMES = ["B_no_flip", "E_h2c_correct"]
ANCHOR_ABS_RELATIVE_VARIANT_NAMES = ["REL_pred_anchor", "REL_gt_anchor", "ABS_query"]
ABSOLUTE_SECOND_VARIANT_NAMES = ["ABS_second"]
ABSOLUTE_SINGLE_VARIANT_NAMES = ["ABS_single"]

S_CV2P3D = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_variant_names(vggt_pose_mode):
    if vggt_pose_mode == "relative_h2c":
        return RELATIVE_VARIANT_NAMES
    if vggt_pose_mode == "anchor_abs_relative_h2c":
        return ANCHOR_ABS_RELATIVE_VARIANT_NAMES
    if vggt_pose_mode == "absolute_second":
        return ABSOLUTE_SECOND_VARIANT_NAMES
    if vggt_pose_mode == "absolute_single":
        return ABSOLUTE_SINGLE_VARIANT_NAMES
    raise ValueError(f"Unknown vggt_pose_mode: {vggt_pose_mode}")


def get_primary_variant_name(vggt_pose_mode):
    if vggt_pose_mode == "relative_h2c":
        return "E_h2c_correct"
    if vggt_pose_mode == "anchor_abs_relative_h2c":
        return "REL_pred_anchor"
    if vggt_pose_mode == "absolute_second":
        return "ABS_second"
    if vggt_pose_mode == "absolute_single":
        return "ABS_single"
    raise ValueError(f"Unknown vggt_pose_mode: {vggt_pose_mode}")


# ═════════════════════════════════════════════════════════════════════════════
# 1.  BIWI DATA HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def get_subject_frames(subject_dir):
    """Return sorted [(frame_num, rgb_path, pose_path), ...] for one subject."""
    frames = []
    for rgb_path in sorted(glob.glob(os.path.join(subject_dir, "frame_*_rgb.png"))):
        m = re.search(r"frame_(\d+)_rgb\.png", os.path.basename(rgb_path))
        pose_path = rgb_path.replace("_rgb.png", "_pose.txt")
        if m and os.path.exists(pose_path):
            frames.append((int(m.group(1)), rgb_path, pose_path))
    return frames


def parse_pose_txt(path):
    """Parse BIWI frame_XXXXX_pose.txt -> R (3×3), t (3,) in mm."""
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    R = np.array([[float(v) for v in lines[i].split()] for i in range(3)], dtype=np.float64)
    t = np.array([float(v) for v in lines[3].split()], dtype=np.float64)
    return R, t


def load_rgb_intrinsics(path):
    """Parse rgb.cal -> K (3×3), R_cam (3×3), t_cam (3,)."""
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    K     = np.array([[float(x) for x in lines[i].split()] for i in range(3)])
    R_cam = np.array([[float(x) for x in lines[i].split()] for i in range(4, 7)])
    t_cam = np.array([float(x) for x in lines[7].split()])
    return K, R_cam, t_cam


def load_obj(path):
    """Parse per-subject .obj mesh -> vertices (V,3), faces (F,3)."""
    vertices, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                toks = [int(x.split("/")[0]) - 1 for x in line.split()[1:]]
                if len(toks) == 3:
                    faces.append(toks)
                elif len(toks) == 4:
                    faces += [toks[:3], [toks[0], toks[2], toks[3]]]
    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int64)


def load_img_rgb(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_img_bgr(path):
    """Load as BGR (for MTCNN which expects BGR->RGB internally, but
       mtcnn.detect_faces actually wants RGB, so we convert)."""
    return cv2.imread(path)


def crop_face(img_rgb, t, K, crop_size=256):
    """Crop face region using projected 3D head-centre (for VGGT)."""
    h, w = img_rgb.shape[:2]
    u = int(round(K[0, 0] * t[0] / t[2] + K[0, 2]))
    v = int(round(K[1, 1] * t[1] / t[2] + K[1, 2]))
    half = crop_size // 2
    x1, y1 = u - half, v - half
    x2, y2 = x1 + crop_size, y1 + crop_size
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    patch = img_rgb[y1c:y2c, x1c:x2c]
    if patch.shape[0] < crop_size or patch.shape[1] < crop_size:
        canvas = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        canvas[y1c - y1:y1c - y1 + patch.shape[0],
               x1c - x1:x1c - x1 + patch.shape[1]] = patch
        return canvas
    return patch


def crop_face_mtcnn(img_bgr, detector, prev_box, crop_size=256, ad=0.4):
    """
    MTCNN-based face crop matching the npz creation script.
    Uses facenet-pytorch MTCNN (pure PyTorch, no TensorFlow).

    Returns (crop_rgb, new_box).  new_box is [xw1,xw2,yw1,yw2] for fallback.
    If detection fails or jumps too far, uses prev_box.
    """
    img_h, img_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # facenet-pytorch returns boxes (N,4) as [x1,y1,x2,y2] and probs (N,)
    boxes, probs = detector.detect(pil_img)

    chosen_box = None

    if boxes is not None and len(boxes) > 0:
        dis_list = []
        XY = []
        for i_d in range(len(boxes)):
            if probs[i_d] > 0.95:
                x1, y1, x2, y2 = boxes[i_d]
                w = x2 - x1
                h = y2 - y1
                xw1 = max(int(x1 - ad * w), 0)
                yw1 = max(int(y1 - ad * h), 0)
                xw2 = min(int(x2 + ad * w), img_w - 1)
                yw2 = min(int(y2 + ad * h), img_h - 1)
                XY.append([xw1, xw2, yw1, yw2])
                dis = abs(xw1 - img_w * 2 / 3) + abs(yw1 - img_h * 2 / 3)
                dis_list.append(dis)

        if len(dis_list) > 0:
            min_id = np.argmin(dis_list)
            candidate = XY[min_id]
            # Check jump distance against previous frame
            if prev_box is None or abs(candidate[0] - prev_box[0]) < 80:
                chosen_box = candidate

    # Fallback to previous box
    if chosen_box is None:
        if prev_box is not None:
            chosen_box = prev_box
        else:
            # No detection ever — fall back to centre crop
            cx, cy = img_w // 2, img_h // 2
            half = crop_size // 2
            chosen_box = [cx - half, cx + half, cy - half, cy + half]

    xw1, xw2, yw1, yw2 = chosen_box
    crop = img_bgr[yw1:yw2 + 1, xw1:xw2 + 1, :]
    # Keep BGR (no conversion) so PIL.fromarray interprets channels as RGB,
    # matching how the BIWI npz training data was created (cv2 BGR stored as-is).
    crop_bgr = cv2.resize(crop, (crop_size, crop_size))
    return crop_bgr, chosen_box


# ═════════════════════════════════════════════════════════════════════════════
# 2.  EULER ANGLES  (BIWI convention)
# ═════════════════════════════════════════════════════════════════════════════

def biwi_euler_deg_np(R):
    """Single 3×3 numpy R -> (yaw, pitch, roll) degrees."""
    R_t   = R.T
    yaw   = -np.arctan2(-R_t[2, 0], np.sqrt(R_t[2, 1]**2 + R_t[2, 2]**2)) * (180 / np.pi)
    pitch =  np.arctan2(R_t[2, 1], R_t[2, 2]) * (180 / np.pi)
    roll  = -np.arctan2(R_t[1, 0], R_t[0, 0]) * (180 / np.pi)
    return yaw, pitch, roll


def biwi_euler_deg_batch(R_batch):
    """Batched torch (N,3,3) -> yaw, pitch, roll each (N,) degrees."""
    R_t   = R_batch.transpose(-1, -2)
    yaw   = -torch.atan2(-R_t[..., 2, 0],
                          torch.sqrt(R_t[..., 2, 1]**2 + R_t[..., 2, 2]**2)) * (180 / np.pi)
    pitch =  torch.atan2(R_t[..., 2, 1], R_t[..., 2, 2]) * (180 / np.pi)
    roll  = -torch.atan2(R_t[..., 1, 0], R_t[..., 0, 0]) * (180 / np.pi)
    return yaw, pitch, roll


# ═════════════════════════════════════════════════════════════════════════════
# 3.  METRICS
# ═════════════════════════════════════════════════════════════════════════════

def angular_error(gt_deg, pred_deg):
    """Min angular error with 360/180 wrapping (matches 6DRepNet test.py)."""
    return torch.min(torch.stack([
        torch.abs(gt_deg - pred_deg),
        torch.abs(pred_deg + 360 - gt_deg),
        torch.abs(pred_deg - 360 - gt_deg),
        torch.abs(pred_deg + 180 - gt_deg),
        torch.abs(pred_deg - 180 - gt_deg),
    ]), dim=0)[0]


def _empty_variant_acc(variant_names):
    return {vname: {"y": 0.0, "p": 0.0, "r": 0.0} for vname in variant_names}


# ═════════════════════════════════════════════════════════════════════════════
# 4.  MODEL LOADING
# ═════════════════════════════════════════════════════════════════════════════

LORA_TARGET_PATTERNS = [
    "frame_blocks.*.attn.qkv",  "frame_blocks.*.attn.proj",
    "frame_blocks.*.mlp.fc1",   "frame_blocks.*.mlp.fc2",
    "global_blocks.*.attn.qkv", "global_blocks.*.attn.proj",
    "global_blocks.*.mlp.fc1",  "global_blocks.*.mlp.fc2",
]


def load_vggt_model(
    base_ckpt_path,
    lora_ckpt_path=None,
    device="cuda",
    enable_track=False,
    pose_encoding_type="absT_quaR_FoV",
    crop_posenc_dim=0,
    enable_absolute_camera=False,
):
    from vggt.models.vggt import VGGT

    def load_compatible_state(model, state, label):
        model_state = model.state_dict()
        compatible_state = {}
        skipped = []
        for key, value in state.items():
            target_value = model_state.get(key)
            if target_value is not None and target_value.shape != value.shape:
                skipped.append((key, tuple(value.shape), tuple(target_value.shape)))
                continue
            compatible_state[key] = value

        miss, unex = model.load_state_dict(compatible_state, strict=False)
        logging.info(f"  {label} -- missing: {len(miss)}, unexpected: {len(unex)}")
        if skipped:
            skipped_keys = ", ".join(key for key, _, _ in skipped[:4])
            logging.info(
                f"  {label} -- skipped {len(skipped)} shape-mismatched keys: {skipped_keys}"
            )
        return miss, unex

    lora_cfg = None
    if lora_ckpt_path is not None:
        lora_cfg = dict(enabled=True, rank=8, alpha=16, dropout=0.0,
                        freeze_base=True, target_patterns=LORA_TARGET_PATTERNS)

    model = VGGT(enable_camera=True, enable_depth=False,
                 enable_point=False, enable_track=enable_track,
                 pose_encoding_type=pose_encoding_type,
                 enable_absolute_camera=enable_absolute_camera,
                 absolute_pose_encoding_type=pose_encoding_type,
                 crop_posenc_dim=crop_posenc_dim,
                 lora=lora_cfg)

    logging.info(f"Loading VGGT base: {base_ckpt_path}")
    ckpt  = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    load_compatible_state(model, state, "Base")

    if lora_ckpt_path is not None:
        logging.info(f"Loading LoRA: {lora_ckpt_path}")
        lckpt  = torch.load(lora_ckpt_path, map_location="cpu", weights_only=False)
        lstate = lckpt["model"] if "model" in lckpt else lckpt
        load_compatible_state(model, lstate, "LoRA")
        if "epoch" in lckpt:
            logging.info(f"  Checkpoint epoch: {lckpt['epoch']}")

    model.eval().to(device)
    return model


def load_sixdrepnet_model(ckpt_path, sixdrepnet_dir, device="cuda"):
    sys.path.insert(0, sixdrepnet_dir)
    from model import SixDRepNet

    model = SixDRepNet(backbone_name="RepVGG-B1g2",
                       backbone_file="", deploy=True, pretrained=False)
    logging.info(f"Loading 6DRepNet: {ckpt_path}")
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(saved.get("model_state_dict", saved))
    model.eval().to(device)
    return model


# ═════════════════════════════════════════════════════════════════════════════
# 5.  POSE DECODING
# ═════════════════════════════════════════════════════════════════════════════

def decode_vggt_pose(
    pose_enc,
    R_anchor_t,
    R_cam_t,
    pose_mode="relative_h2c",
    pose_encoding_type="absT_quaR_FoV",
):
    """
    pose_enc  : (B, S, D)  absT_quaR_FoV
    R_anchor_t: (3, 3) R_pose of anchor frame from pose.txt, only used in relative mode
    R_cam_t   : (3, 3) depth->RGB rotation from rgb.cal

    Returns   : dict of {variant_name: R_pose (B,3,3)} in pose.txt space.
    """
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    extrinsics, _ = pose_encoding_to_extri_intri(
        pose_enc.float(),
        image_size_hw=(518, 518),
        pose_encoding_type=pose_encoding_type,
        build_intrinsics=False,
    )
    RcT = R_cam_t.T.unsqueeze(0)    # (1, 3, 3) R_cam.T

    if pose_mode == "relative_h2c":
        # (B, 3, 3) OpenCV relative H2C rotation.
        R_rel = extrinsics[:, 1, :3, :3].cpu()
        Ra = R_anchor_t.unsqueeze(0)   # (1, 3, 3) anchor pose in pose.txt space
        Rc = R_cam_t.unsqueeze(0)      # (1, 3, 3) R_cam (depth->RGB)
        return {
            "B_no_flip": R_rel @ Ra,
            "E_h2c_correct": RcT @ R_rel @ Rc @ Ra,
        }

    if pose_mode == "anchor_abs_relative_h2c":
        # Frame 0 predicts absolute anchor H2C; frame 1 predicts relative anchor->query H2C.
        F = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
        R_abs_anchor_h2c = extrinsics[:, 0, :3, :3].cpu() @ F
        R_anchor_pred = RcT @ R_abs_anchor_h2c
        R_rel = extrinsics[:, 1, :3, :3].cpu()
        Rc = R_cam_t.unsqueeze(0)
        variants = {
            "REL_pred_anchor": RcT @ R_rel @ Rc @ R_anchor_pred,
        }
        if R_anchor_t is not None:
            Ra = R_anchor_t.unsqueeze(0)
            variants["REL_gt_anchor"] = RcT @ R_rel @ Rc @ Ra
        return variants

    if pose_mode == "absolute_second":
        # The target prediction is absolute H2C for frame 2.
        # Training convention: R_h2c = R_cam @ R_pose, so R_pose = R_cam.T @ R_h2c.
        R_abs_h2c = extrinsics[:, 1, :3, :3].cpu()
        return {
            "ABS_second": RcT @ R_abs_h2c,
        }

    if pose_mode == "absolute_single":
        # Single-view absolute H2C prediction for the only frame in the batch item.
        R_abs_h2c = extrinsics[:, 0, :3, :3].cpu()
        return {
            "ABS_single": RcT @ R_abs_h2c,
        }

    raise ValueError(f"Unknown pose_mode: {pose_mode}")


def decode_sixdrepnet_pose(R_pred_sixd):
    """
    R_pred is in get_R(pitch, yaw, roll) space.
    compute_euler gives [x,y,z] = [pitch_biwi, yaw_biwi, roll_biwi].
    Also reconstruct R in pose.txt space for mesh visualisation.
    """
    sy       = torch.sqrt(R_pred_sixd[:, 0, 0]**2 + R_pred_sixd[:, 1, 0]**2)
    singular = (sy < 1e-6).float()
    x  = torch.atan2( R_pred_sixd[:, 2, 1],  R_pred_sixd[:, 2, 2])
    y  = torch.atan2(-R_pred_sixd[:, 2, 0],  sy)
    z  = torch.atan2( R_pred_sixd[:, 1, 0],  R_pred_sixd[:, 0, 0])
    xs = torch.atan2(-R_pred_sixd[:, 1, 2],  R_pred_sixd[:, 1, 1])
    ys = torch.atan2(-R_pred_sixd[:, 2, 0],  sy)
    pitch = (x * (1 - singular) + xs * singular) * (180 / np.pi)
    yaw   = (y * (1 - singular) + ys * singular) * (180 / np.pi)
    roll  = (z * (1 - singular)               ) * (180 / np.pi)

    N = R_pred_sixd.shape[0]
    R_biwi_batch = np.stack([
        euler_to_R_biwi(yaw[i].item(), pitch[i].item(), roll[i].item())
        for i in range(N)
    ])
    return yaw, pitch, roll, R_biwi_batch


def euler_to_R_biwi(yaw_deg, pitch_deg, roll_deg):
    """Invert BIWI euler convention -> R in pose.txt space (for mesh vis)."""
    r = -roll_deg  * (np.pi / 180.0)
    y = -yaw_deg   * (np.pi / 180.0)
    p =  pitch_deg * (np.pi / 180.0)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(p), -np.sin(p)],
                   [0, np.sin(p),  np.cos(p)]])
    Ry = np.array([[ np.cos(y), 0, np.sin(y)],
                   [0,          1, 0         ],
                   [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0],
                   [np.sin(r),  np.cos(r), 0],
                   [0,          0,         1]])
    return (Rz @ Ry @ Rx).T


# ═════════════════════════════════════════════════════════════════════════════
# 6.  PERSPECTIVE-PROJECTION MESH RENDERER
# ═════════════════════════════════════════════════════════════════════════════

def _project_vertices(vertices, R, t, R_cam, t_cam, K):
    v_world = (R @ vertices.T).T + t
    v_cam   = (R_cam @ v_world.T).T + t_cam
    valid   = v_cam[:, 2] > 0
    v_img   = (K @ v_cam.T).T
    v_img  /= v_img[:, 2:3]
    return v_cam, v_img[:, :2], valid


def render_mesh_on_image(img_rgb, vertices, faces, R, t, R_cam, t_cam, K,
                          color="#4a90d9", alpha=0.5):
    h, w = img_rgb.shape[:2]
    v_cam, v_img, valid = _project_vertices(vertices, R, t, R_cam, t_cam, K)

    face_list, face_depths, face_normals_z = [], [], []
    for fi in faces:
        if not all(valid[fi]):
            continue
        pts = v_img[fi]
        e1, e2 = pts[1] - pts[0], pts[2] - pts[0]
        if (e1[0]*e2[1] - e1[1]*e2[0]) > 0:
            continue
        v0, v1, v2 = v_cam[fi]
        n  = np.cross(v1 - v0, v2 - v0)
        nm = np.linalg.norm(n)
        nz = n[2] / nm if nm > 1e-8 else 0
        face_list.append(pts)
        face_depths.append(v_cam[fi, 2].mean())
        face_normals_z.append(abs(nz))

    if not face_list:
        return img_rgb

    order    = np.argsort(face_depths)[::-1]
    base_rgb = np.array(mcolors.to_rgb(color))

    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    fig.patch.set_alpha(0)
    ax.set_facecolor((0, 0, 0, 0))

    polys  = [face_list[i] for i in order]
    colors = [(*base_rgb * max(0.3, face_normals_z[i]), alpha) for i in order]
    pc = PolyCollection(polys, facecolors=colors, edgecolors="none",
                        linewidths=0, antialiased=False)
    ax.add_collection(pc)
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_aspect("equal"); ax.axis("off")
    plt.subplots_adjust(0, 0, 1, 1)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, transparent=True,
                bbox_inches="tight", pad_inches=0)
    plt.close(fig); buf.seek(0)
    overlay = np.array(Image.open(buf).resize((w, h), Image.BICUBIC))

    fa   = overlay[..., 3:].astype(np.float32) / 255.0
    comp = (img_rgb.astype(np.float32) * (1 - fa) +
            overlay[..., :3].astype(np.float32) * fa)
    return comp.clip(0, 255).astype(np.uint8)


# ═════════════════════════════════════════════════════════════════════════════
# 7.  VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def save_vis_frame(
    img_rgb, vertices, faces, K_rgb, R_cam, t_cam,
    R_gt, t_gt, euler_gt,
    vggt_variants,
    R_sixd, euler_sixd,
    save_path,
):
    def _title(name, euler):
        y, p, r = euler
        return f"{name}\nY:{y:.1f}  P:{p:.1f}  R:{r:.1f}"

    VARIANT_COLORS = {
        "A_full_flip" : "#e8744f",
        "B_no_flip"   : "#c0392b",
        "C_flip_Y"    : "#8e44ad",
        "D_order_flip": "#795548",
    }

    panel_gt = render_mesh_on_image(img_rgb.copy(), vertices, faces,
                                     R_gt, t_gt, R_cam, t_cam, K_rgb,
                                     color="#4a90d9", alpha=0.5)
    panels = [img_rgb, panel_gt]
    titles = ["RGB", _title("GT", euler_gt)]

    for vname, (R_v, euler_v) in vggt_variants.items():
        p = render_mesh_on_image(img_rgb.copy(), vertices, faces,
                                  R_v, t_gt, R_cam, t_cam, K_rgb,
                                  color=VARIANT_COLORS.get(vname, "#e8744f"),
                                  alpha=0.5)
        panels.append(p)
        titles.append(_title(vname, euler_v))

    if R_sixd is not None:
        panel_sixd = render_mesh_on_image(img_rgb.copy(), vertices, faces,
                                           R_sixd, t_gt, R_cam, t_cam, K_rgb,
                                           color="#50b86c", alpha=0.5)
        panels.append(panel_sixd)
        titles.append(_title("6DRepNet", euler_sixd))

    n_cols = len(panels)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════

SIXD_TRANSFORM = TF.Compose([
    TF.Resize(256), TF.CenterCrop(224), TF.ToTensor(),
    TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def vggt_transform(crop_np, target_size=518):
    pil = Image.fromarray(crop_np).resize((target_size, target_size), Image.BICUBIC)
    return TF.ToTensor()(pil)




# ═════════════════════════════════════════════════════════════════════════════



