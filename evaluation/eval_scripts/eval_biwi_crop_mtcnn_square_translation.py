import argparse
import csv
import gc
import glob
import logging
import os

import cv2
import torch
from PIL import Image

import eval_biwi_crop as base


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _empty_translation_acc():
    return {"tx": 0.0, "ty": 0.0, "tz": 0.0, "l2": 0.0}


def _empty_normalized_translation_acc():
    return {"x_over_z": 0.0, "y_over_z": 0.0, "log_z": 0.0, "l2": 0.0}


def accumulate_translation_errors(acc, pred_t, gt_t):
    err = (pred_t - gt_t).abs()
    acc["tx"] += err[:, 0].sum().item()
    acc["ty"] += err[:, 1].sum().item()
    acc["tz"] += err[:, 2].sum().item()
    acc["l2"] += (pred_t - gt_t).norm(dim=-1).sum().item()


def _normalized_translation_encoding(t, eps=1e-6):
    z = t[:, 2].clamp(min=eps)
    return torch.stack([t[:, 0] / z, t[:, 1] / z, torch.log(z)], dim=-1)


def _relative_projective_translation_encoding(t, anchor_t, eps=1e-6):
    z = t[:, 2].clamp(min=eps)
    z0 = anchor_t[:, 2].clamp(min=eps)
    return torch.stack(
        [
            t[:, 0] / z - anchor_t[:, 0] / z0,
            t[:, 1] / z - anchor_t[:, 1] / z0,
            torch.log(z / z0),
        ],
        dim=-1,
    )

def accumulate_normalized_translation_errors(acc, pred_t, gt_t):
    pred = _normalized_translation_encoding(pred_t)
    gt = _normalized_translation_encoding(gt_t)
    err = (pred - gt).abs()
    acc["x_over_z"] += err[:, 0].sum().item()
    acc["y_over_z"] += err[:, 1].sum().item()
    acc["log_z"] += err[:, 2].sum().item()
    acc["l2"] += (pred - gt).norm(dim=-1).sum().item()


def accumulate_relative_projective_translation_errors(acc, pred_t, gt_t, anchor_t):
    pred = _relative_projective_translation_encoding(pred_t, anchor_t)
    gt = _relative_projective_translation_encoding(gt_t, anchor_t)
    err = (pred - gt).abs()
    acc["x_over_z"] += err[:, 0].sum().item()
    acc["y_over_z"] += err[:, 1].sum().item()
    acc["log_z"] += err[:, 2].sum().item()
    acc["l2"] += (pred - gt).norm(dim=-1).sum().item()


def translation_metrics_from_acc(acc, count):
    return {
        "tx": acc["tx"] / count,
        "ty": acc["ty"] / count,
        "tz": acc["tz"] / count,
        "l2": acc["l2"] / count,
    }


def normalized_translation_metrics_from_acc(acc, count):
    return {
        "x_over_z": acc["x_over_z"] / count,
        "y_over_z": acc["y_over_z"] / count,
        "log_z": acc["log_z"] / count,
        "l2": acc["l2"] / count,
    }


def accumulate_translation_diagnostics(accs, diagnostics, gt_t):
    for name, pred_t in diagnostics.items():
        if name not in accs:
            accs[name] = _empty_translation_acc()
        accumulate_translation_errors(accs[name], pred_t, gt_t)


def decode_vggt_predictions(
    pose_enc,
    R_anchor_t,
    t_anchor_t,
    R_cam_t,
    t_cam_t,
    R_target_t=None,
    pose_enc_abs=None,
    pose_mode="relative_h2c",
    translation_scale=1000.0,
    pose_encoding_type="absT_quaR_FoV",
    translation_rotation_mode="predicted",
    translation_target_type="relative_se3",
    return_translation_diagnostics=False,
    return_translation_variants=False,
):
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    extrinsics, _ = pose_encoding_to_extri_intri(
        pose_enc.float(),
        image_size_hw=(518, 518),
        pose_encoding_type=pose_encoding_type,
        build_intrinsics=False,
    )
    abs_extrinsics = None
    if pose_enc_abs is not None:
        abs_extrinsics, _ = pose_encoding_to_extri_intri(
            pose_enc_abs.float(),
            image_size_hw=(518, 518),
            pose_encoding_type=pose_encoding_type,
            build_intrinsics=False,
        )

    Rc = R_cam_t.unsqueeze(0).cpu()
    RcT = R_cam_t.T.unsqueeze(0).cpu()
    tc = t_cam_t.unsqueeze(0).cpu()
    F = torch.diag(torch.tensor([1.0, -1.0, -1.0]))

    def _depth_to_rgb_current(t_depth):
        return (Rc @ t_depth[..., None]).squeeze(-1) + tc

    def _rgb_to_depth_current(t_rgb):
        return (RcT @ (t_rgb - tc)[..., None]).squeeze(-1)

    def _depth_to_rgb_inverse_rgbcal(t_depth):
        return (RcT @ (t_depth - tc)[..., None]).squeeze(-1)

    def _rgb_to_depth_inverse_rgbcal(t_rgb):
        return (Rc @ t_rgb[..., None]).squeeze(-1) + tc

    def _decode_translation_slot_rgb(anchor_rgb, slot_rgb, target_type, mode, R_rel_for_translation=None):
        if target_type == "relative_se3":
            delta_rgb = slot_rgb * translation_scale
            if mode == "delta_flip_x":
                delta_rgb = torch.stack([-delta_rgb[:, 0], delta_rgb[:, 1], delta_rgb[:, 2]], dim=-1)
            elif mode == "delta_flip_y":
                delta_rgb = torch.stack([delta_rgb[:, 0], -delta_rgb[:, 1], delta_rgb[:, 2]], dim=-1)
            elif mode == "delta_flip_xy":
                delta_rgb = torch.stack([-delta_rgb[:, 0], -delta_rgb[:, 1], delta_rgb[:, 2]], dim=-1)
            elif mode == "delta_swap_xy":
                delta_rgb = torch.stack([delta_rgb[:, 1], delta_rgb[:, 0], delta_rgb[:, 2]], dim=-1)
            elif mode == "inverse_delta":
                delta_rgb = -delta_rgb
            elif mode != "normal":
                raise ValueError(f"Unknown translation diagnostic mode: {mode}")
            if R_rel_for_translation is None:
                raise ValueError("R_rel_for_translation is required for relative_se3 translation diagnostics")
            return (R_rel_for_translation @ anchor_rgb[..., None]).squeeze(-1) + delta_rgb

        if target_type == "delta_t":
            delta_rgb = slot_rgb * translation_scale
            if mode == "delta_flip_x":
                delta_rgb = torch.stack([-delta_rgb[:, 0], delta_rgb[:, 1], delta_rgb[:, 2]], dim=-1)
            elif mode == "delta_flip_y":
                delta_rgb = torch.stack([delta_rgb[:, 0], -delta_rgb[:, 1], delta_rgb[:, 2]], dim=-1)
            elif mode == "delta_flip_xy":
                delta_rgb = torch.stack([-delta_rgb[:, 0], -delta_rgb[:, 1], delta_rgb[:, 2]], dim=-1)
            elif mode == "delta_swap_xy":
                delta_rgb = torch.stack([delta_rgb[:, 1], delta_rgb[:, 0], delta_rgb[:, 2]], dim=-1)
            elif mode == "inverse_delta":
                delta_rgb = -delta_rgb
            elif mode != "normal":
                raise ValueError(f"Unknown translation diagnostic mode: {mode}")
            return anchor_rgb + delta_rgb

        if target_type in {"depth_norm_delta_t", "projective_delta_t"}:
            z0 = anchor_rgb[:, 2].clamp(min=1e-6)
            dx = slot_rgb[:, 0]
            dy = slot_rgb[:, 1]
            log_z_ratio = slot_rgb[:, 2].clamp(min=-20.0, max=20.0)
            u0 = anchor_rgb[:, 0] / z0
            v0 = anchor_rgb[:, 1] / z0

            if target_type == "depth_norm_delta_t":
                if mode == "normal":
                    z = z0 * torch.exp(log_z_ratio)
                    x = anchor_rgb[:, 0] + dx * z0
                    y = anchor_rgb[:, 1] + dy * z0
                elif mode == "delta_flip_x":
                    z = z0 * torch.exp(log_z_ratio)
                    x = anchor_rgb[:, 0] - dx * z0
                    y = anchor_rgb[:, 1] + dy * z0
                elif mode == "delta_flip_y":
                    z = z0 * torch.exp(log_z_ratio)
                    x = anchor_rgb[:, 0] + dx * z0
                    y = anchor_rgb[:, 1] - dy * z0
                elif mode == "delta_flip_xy":
                    z = z0 * torch.exp(log_z_ratio)
                    x = anchor_rgb[:, 0] - dx * z0
                    y = anchor_rgb[:, 1] - dy * z0
                elif mode == "delta_swap_xy":
                    z = z0 * torch.exp(log_z_ratio)
                    x = anchor_rgb[:, 0] + dy * z0
                    y = anchor_rgb[:, 1] + dx * z0
                elif mode == "inverse_delta":
                    z = z0 * torch.exp(-log_z_ratio)
                    x = anchor_rgb[:, 0] - dx * z
                    y = anchor_rgb[:, 1] - dy * z
                else:
                    raise ValueError(f"Unknown translation diagnostic mode: {mode}")
            else:
                if mode == "normal":
                    z = z0 * torch.exp(log_z_ratio)
                    u = u0 + dx
                    v = v0 + dy
                elif mode == "delta_flip_x":
                    z = z0 * torch.exp(log_z_ratio)
                    u = u0 - dx
                    v = v0 + dy
                elif mode == "delta_flip_y":
                    z = z0 * torch.exp(log_z_ratio)
                    u = u0 + dx
                    v = v0 - dy
                elif mode == "delta_flip_xy":
                    z = z0 * torch.exp(log_z_ratio)
                    u = u0 - dx
                    v = v0 - dy
                elif mode == "delta_swap_xy":
                    z = z0 * torch.exp(log_z_ratio)
                    u = u0 + dy
                    v = v0 + dx
                elif mode == "inverse_delta":
                    z = z0 * torch.exp(-log_z_ratio)
                    u = u0 - dx
                    v = v0 - dy
                else:
                    raise ValueError(f"Unknown translation diagnostic mode: {mode}")
                x = u * z
                y = v * z
            return torch.stack([x, y, z], dim=-1)

        raise ValueError(
            "translation_target_type must be 'relative_se3', 'delta_t', 'depth_norm_delta_t', "
            "or 'projective_delta_t', "
            f"got {target_type!r}"
        )

    def _decode_absolute_norm_translation_rgb(slot_rgb):
        log_z = slot_rgb[:, 2].clamp(min=-20.0, max=20.0)
        z = torch.exp(log_z) * translation_scale
        x = slot_rgb[:, 0] * z
        y = slot_rgb[:, 1] * z
        return torch.stack([x, y, z], dim=-1)

    if pose_mode == "relative_h2c":
        R_rel_rgb = extrinsics[:, 1, :3, :3].cpu()
        t_slot_rgb = extrinsics[:, 1, :3, 3].cpu()

        Ra = R_anchor_t.cpu()
        ta = t_anchor_t.cpu()
        if Ra.ndim == 2:
            Ra = Ra.unsqueeze(0)
        if ta.ndim == 1:
            ta = ta.unsqueeze(0)

        variants = {
            "B_no_flip": R_rel_rgb @ Ra,
            "E_h2c_correct": RcT @ R_rel_rgb @ Rc @ Ra,
        }

        # Convert BIWI anchor pose.txt translation (depth camera) to RGB-camera H2C.
        t_anchor_rgb = _depth_to_rgb_current(ta)
        R_rel_for_translation = None
        if translation_target_type == "relative_se3":
            t_rel_rgb = t_slot_rgb * translation_scale
            if translation_rotation_mode == "predicted":
                R_rel_for_translation = R_rel_rgb
            elif translation_rotation_mode == "gt":
                if R_target_t is None:
                    raise ValueError("R_target_t is required when translation_rotation_mode='gt'")
                Rt = R_target_t.cpu()
                R_rel_for_translation = Rc @ Rt @ Ra.transpose(-1, -2) @ RcT
            else:
                raise ValueError(
                    "translation_rotation_mode must be either 'predicted' or 'gt', "
                    f"got {translation_rotation_mode!r}"
                )
            # Undo the training-time first-frame SE(3) normalization in RGB-camera space.
            t_abs_rgb = (R_rel_for_translation @ t_anchor_rgb[..., None]).squeeze(-1) + t_rel_rgb
        elif translation_target_type == "delta_t":
            t_abs_rgb = _decode_translation_slot_rgb(
                t_anchor_rgb,
                t_slot_rgb,
                translation_target_type,
                "normal",
            )
        elif translation_target_type in {"depth_norm_delta_t", "projective_delta_t"}:
            t_abs_rgb = _decode_translation_slot_rgb(
                t_anchor_rgb,
                t_slot_rgb,
                translation_target_type,
                "normal",
            )
        else:
            raise ValueError(
                "translation_target_type must be 'relative_se3', 'delta_t', 'depth_norm_delta_t', "
                "or 'projective_delta_t', "
                f"got {translation_target_type!r}"
            )
        # Convert back to BIWI pose.txt depth-camera translation.
        t_primary = _rgb_to_depth_current(t_abs_rgb)
        if return_translation_diagnostics:
            diagnostics = {}
            for mode in [
                "normal",
                "delta_flip_x",
                "delta_flip_y",
                "delta_flip_xy",
                "delta_swap_xy",
                "inverse_delta",
            ]:
                diag_abs_rgb = _decode_translation_slot_rgb(
                    t_anchor_rgb,
                    t_slot_rgb,
                    translation_target_type,
                    mode,
                    R_rel_for_translation=R_rel_for_translation,
                )
                diagnostics[mode] = _rgb_to_depth_current(diag_abs_rgb)

            t_anchor_rgb_inv = _depth_to_rgb_inverse_rgbcal(ta)
            diag_abs_rgb_inv = _decode_translation_slot_rgb(
                t_anchor_rgb_inv,
                t_slot_rgb,
                translation_target_type,
                "normal",
                R_rel_for_translation=R_rel_for_translation,
            )
            diagnostics["calib_inverse_rgbcal"] = _rgb_to_depth_inverse_rgbcal(diag_abs_rgb_inv)

            diagnostics["depth_abs_flip_x"] = torch.stack(
                [-t_primary[:, 0], t_primary[:, 1], t_primary[:, 2]], dim=-1
            )
            diagnostics["depth_abs_flip_y"] = torch.stack(
                [t_primary[:, 0], -t_primary[:, 1], t_primary[:, 2]], dim=-1
            )
            diagnostics["depth_abs_flip_xy"] = torch.stack(
                [-t_primary[:, 0], -t_primary[:, 1], t_primary[:, 2]], dim=-1
            )
            diagnostics["depth_abs_swap_xy"] = torch.stack(
                [t_primary[:, 1], t_primary[:, 0], t_primary[:, 2]], dim=-1
            )
            return variants, t_primary, diagnostics
        return variants, t_primary

    if pose_mode == "anchor_abs_relative_h2c":
        if abs_extrinsics is None:
            raise ValueError("pose_mode='anchor_abs_relative_h2c' requires pose_enc_abs from the absolute camera head")

        R_abs_anchor_rgb = abs_extrinsics[:, 0, :3, :3].cpu() @ F
        R_abs_query_rgb = abs_extrinsics[:, 1, :3, :3].cpu() @ F
        t_anchor_rgb = _decode_absolute_norm_translation_rgb(abs_extrinsics[:, 0, :3, 3].cpu())
        t_abs_query_rgb = _decode_absolute_norm_translation_rgb(abs_extrinsics[:, 1, :3, 3].cpu())
        R_rel_rgb = extrinsics[:, 1, :3, :3].cpu()
        t_slot_rgb = extrinsics[:, 1, :3, 3].cpu()

        R_anchor_pred = RcT @ R_abs_anchor_rgb
        variants = {
            "REL_pred_anchor": RcT @ R_rel_rgb @ Rc @ R_anchor_pred,
            "ABS_query": RcT @ R_abs_query_rgb,
        }
        if R_anchor_t is not None:
            Ra_gt = R_anchor_t.unsqueeze(0).cpu()
            variants["REL_gt_anchor"] = RcT @ R_rel_rgb @ Rc @ Ra_gt

        R_rel_for_translation = None
        t_abs_rgb_gt_anchor = None
        if translation_target_type == "relative_se3":
            t_rel_rgb = t_slot_rgb * translation_scale
            if translation_rotation_mode == "predicted":
                R_rel_for_translation = R_rel_rgb
            elif translation_rotation_mode == "gt":
                if R_target_t is None:
                    raise ValueError("R_target_t is required when translation_rotation_mode='gt'")
                Rt = R_target_t.cpu()
                R_rel_for_translation = Rc @ Rt @ R_anchor_pred.transpose(-1, -2) @ RcT
            else:
                raise ValueError(
                    "translation_rotation_mode must be either 'predicted' or 'gt', "
                    f"got {translation_rotation_mode!r}"
                )
            t_abs_rgb = (R_rel_for_translation @ t_anchor_rgb[..., None]).squeeze(-1) + t_rel_rgb
            if t_anchor_t is not None:
                ta_gt = t_anchor_t.unsqueeze(0).cpu()
                t_anchor_rgb_gt = _depth_to_rgb_current(ta_gt)
                t_abs_rgb_gt_anchor = (R_rel_for_translation @ t_anchor_rgb_gt[..., None]).squeeze(-1) + t_rel_rgb
        elif translation_target_type in {"delta_t", "depth_norm_delta_t", "projective_delta_t"}:
            t_abs_rgb = _decode_translation_slot_rgb(
                t_anchor_rgb,
                t_slot_rgb,
                translation_target_type,
                "normal",
            )
            if t_anchor_t is not None:
                ta_gt = t_anchor_t.unsqueeze(0).cpu()
                t_anchor_rgb_gt = _depth_to_rgb_current(ta_gt)
                t_abs_rgb_gt_anchor = _decode_translation_slot_rgb(
                    t_anchor_rgb_gt,
                    t_slot_rgb,
                    translation_target_type,
                    "normal",
                )
        else:
            raise ValueError(
                "translation_target_type must be 'relative_se3', 'delta_t', 'depth_norm_delta_t', "
                "or 'projective_delta_t', "
                f"got {translation_target_type!r}"
            )

        t_primary = _rgb_to_depth_current(t_abs_rgb)
        translation_variants = {
            "REL_pred_anchor": t_primary,
            "ABS_query": _rgb_to_depth_current(t_abs_query_rgb),
        }
        if t_abs_rgb_gt_anchor is not None:
            translation_variants["REL_gt_anchor"] = _rgb_to_depth_current(t_abs_rgb_gt_anchor)
        if return_translation_diagnostics:
            diagnostics = {}
            for mode in [
                "normal",
                "delta_flip_x",
                "delta_flip_y",
                "delta_flip_xy",
                "delta_swap_xy",
                "inverse_delta",
            ]:
                diag_abs_rgb = _decode_translation_slot_rgb(
                    t_anchor_rgb,
                    t_slot_rgb,
                    translation_target_type,
                    mode,
                    R_rel_for_translation=R_rel_for_translation,
                )
                diagnostics[mode] = _rgb_to_depth_current(diag_abs_rgb)
            if return_translation_variants:
                return variants, t_primary, diagnostics, translation_variants
            return variants, t_primary, diagnostics
        if return_translation_variants:
            return variants, t_primary, translation_variants
        return variants, t_primary

        '''if pose_mode == "absolute_second":
        R_abs_rgb = extrinsics[:, 1, :3, :3].cpu()
        t_abs_rgb = extrinsics[:, 1, :3, 3].cpu() * translation_scale
        variants = {"ABS_second": RcT @ R_abs_rgb}
        t_primary = (RcT @ (t_abs_rgb - tc)[..., None]).squeeze(-1)
        return variants, t_primary

    if pose_mode == "absolute_single":
        R_abs_rgb = extrinsics[:, 0, :3, :3].cpu()
        t_abs_rgb = extrinsics[:, 0, :3, 3].cpu() * translation_scale
        variants = {"ABS_single": RcT @ R_abs_rgb}
        t_primary = (RcT @ (t_abs_rgb - tc)[..., None]).squeeze(-1)
        return variants, t_primary'''
    if pose_mode == "absolute_second":
        R_abs_rgb = extrinsics[:, 1, :3, :3].cpu()
        t_abs_rgb = extrinsics[:, 1, :3, 3].cpu() * translation_scale
        # VGGT R maps OpenGL-mesh -> OpenCV-camera;
        # right-multiply by F so it maps OpenCV-mesh -> OpenCV-camera,
        # matching BIWI pose.txt convention.
        R_abs_rgb = R_abs_rgb @ F
        variants = {"ABS_second": RcT @ R_abs_rgb}
        t_primary = (RcT @ (t_abs_rgb - tc)[..., None]).squeeze(-1)
        return variants, t_primary

    if pose_mode == "absolute_single":
        R_abs_rgb = extrinsics[:, 0, :3, :3].cpu()
        t_abs_rgb = extrinsics[:, 0, :3, 3].cpu() * translation_scale
        R_abs_rgb = R_abs_rgb @ F
        variants = {"ABS_single": RcT @ R_abs_rgb}
        t_primary = (RcT @ (t_abs_rgb - tc)[..., None]).squeeze(-1)
        return variants, t_primary

    raise ValueError(f"Unknown pose_mode: {pose_mode}")






