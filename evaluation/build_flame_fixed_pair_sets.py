#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


FREE_FOV_ROOTS = [
    {
        "path": "data/blended_in_flames/sample_outputs/dataset_20k_hair_v2",
        "name": "hair",
    },
    {
        "path": "data/blended_in_flames/sample_outputs/ar_pose_poc_5k",
        "name": "nohair",
    },
]

FIXED_FOV_ROOTS = [
    {
        "path": "data/blended_in_flames/sample_outputs/dataset_20k_hair_v2_fixed_focal_max",
        "name": "hair",
    },
    {
        "path": "data/blended_in_flames/sample_outputs/ar_pose_poc_5k_fixed_focal_max",
        "name": "nohair",
    },
]

DEFAULT_ROOT_SETS = {
    "free_fov": FREE_FOV_ROOTS,
    "fixed_fov": FIXED_FOV_ROOTS,
}

DEFAULT_OUT_DIR = Path("pair_manifests")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root_sets",
        type=str,
        default=None,
        help=(
            "Optional JSON mapping variant names to lists of "
            '{"path": ..., "name": ...} roots. Defaults to free_fov/fixed_fov.'
        ),
    )
    p.add_argument(
        "--canonical_variant",
        default="free_fov",
        help="Variant whose tree is used for deterministic pair sampling.",
    )
    p.add_argument("--n_val_identities_per_root", type=int, default=5)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--pair_seed", type=int, default=20260529)
    p.add_argument("--num_train_pairs", type=int, default=200000)
    p.add_argument("--num_test_pairs", type=int, default=5000)
    p.add_argument("--same_identity", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--different_identity",
        action="store_true",
        help="Require anchor/query to come from different identities. Implies --no-same_identity.",
    )
    p.add_argument("--same_lighting", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--same_expression", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--allow_replacement",
        action="store_true",
        help="Allow duplicate anchor/query pairs. By default, rows are unique within each split.",
    )
    p.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    return p.parse_args()


def parse_root_sets(args: argparse.Namespace) -> dict[str, list[dict[str, str]]]:
    if args.root_sets:
        root_sets = json.loads(args.root_sets)
    else:
        root_sets = DEFAULT_ROOT_SETS

    if args.canonical_variant not in root_sets:
        raise ValueError(
            f"canonical_variant={args.canonical_variant!r} not found in root sets: "
            f"{sorted(root_sets)}"
        )

    canonical_names = {root["name"] for root in root_sets[args.canonical_variant]}
    for variant, roots in root_sets.items():
        names = {root["name"] for root in roots}
        if names != canonical_names:
            raise ValueError(
                f"Variant {variant!r} root names {sorted(names)} do not match "
                f"canonical names {sorted(canonical_names)}"
            )
    return root_sets


def split_identities(root_path: Path, n_val: int, split_seed: int) -> tuple[set[str], set[str]]:
    identity_names = sorted(name for name in os.listdir(root_path) if name.startswith("identity_"))
    rng = np.random.RandomState(split_seed)
    shuffled = identity_names.copy()
    rng.shuffle(shuffled)
    n_val = min(n_val, len(shuffled))
    val = set(shuffled[:n_val])
    train = set(shuffled[n_val:])
    return train, val


def iter_view_dirs(identity_dir: Path) -> Iterable[tuple[str, str, Path]]:
    for lighting_dir in sorted(identity_dir.glob("lighting_*")):
        if not lighting_dir.is_dir():
            continue
        for expr_dir in sorted(lighting_dir.glob("expr_*")):
            if not expr_dir.is_dir():
                continue
            for view_dir in sorted(expr_dir.glob("view_*")):
                if view_dir.is_dir():
                    yield lighting_dir.name, expr_dir.name, view_dir


def build_index(
    roots: list[dict[str, str]],
    split: str,
    n_val_identities_per_root: int,
    split_seed: int,
) -> dict[str, object]:
    all_views: list[dict[str, str]] = []
    views_by_id_light: dict[tuple[tuple[str, str], str], list[dict[str, str]]] = defaultdict(list)
    views_by_identity: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    views_by_expr: dict[tuple[tuple[str, str], str, str], list[dict[str, str]]] = defaultdict(list)
    views_by_lighting: dict[str, list[dict[str, str]]] = defaultdict(list)
    split_counts: dict[str, dict[str, int]] = {}

    for root in roots:
        root_path = Path(root["path"])
        root_name = root["name"]
        if not root_path.is_dir():
            raise FileNotFoundError(f"Missing root for {root_name}: {root_path}")

        train_ids, val_ids = split_identities(root_path, n_val_identities_per_root, split_seed)
        selected_ids = train_ids if split == "train" else val_ids
        split_counts[root_name] = {
            "train_identities": len(train_ids),
            "test_identities": len(val_ids),
            "selected_identities": len(selected_ids),
        }

        for identity_name in sorted(selected_ids):
            identity_dir = root_path / identity_name
            for lighting, expression, view_dir in iter_view_dirs(identity_dir):
                rel_path = view_dir.relative_to(root_path).as_posix()
                view = {
                    "root_name": root_name,
                    "identity_name": identity_name,
                    "identity_key": (root_name, identity_name),
                    "lighting": lighting,
                    "expression": expression,
                    "view": view_dir.name,
                    "rel_path": rel_path,
                    "path": str(view_dir),
                }
                all_views.append(view)
                views_by_id_light[((root_name, identity_name), lighting)].append(view)
                views_by_identity[(root_name, identity_name)].append(view)
                views_by_expr[((root_name, identity_name), lighting, expression)].append(view)
                views_by_lighting[lighting].append(view)

    if not all_views:
        raise RuntimeError(f"No views found for split={split}")

    return {
        "all_views": all_views,
        "views_by_id_light": views_by_id_light,
        "views_by_identity": views_by_identity,
        "views_by_expr": views_by_expr,
        "views_by_lighting": views_by_lighting,
        "split_counts": split_counts,
    }


def candidate_views(
    index: dict[str, object],
    anchor: dict[str, str],
    same_identity: bool,
    different_identity: bool,
    same_lighting: bool,
    same_expression: bool,
) -> list[dict[str, str]]:
    all_views = index["all_views"]
    views_by_id_light = index["views_by_id_light"]
    views_by_identity = index["views_by_identity"]
    views_by_expr = index["views_by_expr"]

    identity_key = anchor["identity_key"]
    if same_identity and different_identity:
        raise ValueError("same_identity and different_identity cannot both be true")

    if same_identity and same_lighting:
        candidates = list(views_by_id_light[(identity_key, anchor["lighting"])])
    elif same_identity:
        candidates = list(views_by_identity[identity_key])
    else:
        if same_lighting:
            candidates = list(index["views_by_lighting"][anchor["lighting"]])
        else:
            candidates = list(all_views)
        if same_lighting:
            candidates = [v for v in candidates if v["lighting"] == anchor["lighting"]]
        if different_identity:
            candidates = [v for v in candidates if v["identity_key"] != identity_key]

    if same_expression:
        candidates = [v for v in candidates if v["expression"] == anchor["expression"]]

    candidates = [
        v
        for v in candidates
        if (v["root_name"], v["rel_path"]) != (anchor["root_name"], anchor["rel_path"])
    ]

    if not candidates and different_identity:
        raise RuntimeError(
            f"No different-identity candidates for {anchor['root_name']}/"
            f"{anchor['identity_name']}/{anchor['lighting']}/{anchor['expression']}"
        )

    if not candidates:
        candidates = [
            v
            for v in views_by_expr[(identity_key, anchor["lighting"], anchor["expression"])]
            if v["rel_path"] != anchor["rel_path"]
        ]

    if not candidates:
        candidates = [anchor]

    return candidates


def sample_query_view(
    index: dict[str, object],
    anchor: dict[str, str],
    rng: np.random.RandomState,
    same_identity: bool,
    different_identity: bool,
    same_lighting: bool,
    same_expression: bool,
) -> dict[str, str]:
    if not same_identity and different_identity:
        if same_lighting:
            pool = index["views_by_lighting"][anchor["lighting"]]
        else:
            pool = index["all_views"]

        for _ in range(100):
            query = pool[int(rng.randint(len(pool)))]
            if query["identity_key"] == anchor["identity_key"]:
                continue
            if same_expression and query["expression"] != anchor["expression"]:
                continue
            if (query["root_name"], query["rel_path"]) == (anchor["root_name"], anchor["rel_path"]):
                continue
            return query

    candidates = candidate_views(
        index, anchor, same_identity, different_identity, same_lighting, same_expression
    )
    return candidates[int(rng.randint(len(candidates)))]


def sample_pair_specs(
    index: dict[str, object],
    split: str,
    num_pairs: int,
    rng: np.random.RandomState,
    same_identity: bool,
    different_identity: bool,
    same_lighting: bool,
    same_expression: bool,
    allow_replacement: bool,
) -> list[dict[str, object]]:
    all_views = index["all_views"]
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    attempts = 0
    max_attempts = max(num_pairs * 100, 1000)

    while len(rows) < num_pairs:
        if attempts > max_attempts:
            raise RuntimeError(
                f"Could only sample {len(rows)} unique {split} pairs after {attempts} attempts. "
                "Use --allow_replacement or reduce the requested pair count."
            )
        attempts += 1

        anchor = all_views[int(rng.randint(len(all_views)))]
        query = sample_query_view(
            index=index,
            anchor=anchor,
            rng=rng,
            same_identity=same_identity,
            different_identity=different_identity,
            same_lighting=same_lighting,
            same_expression=same_expression,
        )
        key = (anchor["root_name"], anchor["rel_path"], query["root_name"], query["rel_path"])
        if not allow_replacement and key in seen:
            continue
        seen.add(key)

        pair_index = len(rows)
        rows.append(
            {
                "pair_index": pair_index,
                "split": split,
                "group_id": (
                    f"{anchor['root_name']}/{anchor['identity_name']}/{anchor['lighting']}"
                    f"__to__{query['root_name']}/{query['identity_name']}/{query['lighting']}"
                ),
                "root_name": anchor["root_name"],
                "identity_name": anchor["identity_name"],
                "lighting": anchor["lighting"],
                "anchor_root_name": anchor["root_name"],
                "anchor_identity_name": anchor["identity_name"],
                "anchor_lighting": anchor["lighting"],
                "anchor_expression": anchor["expression"],
                "anchor_view": anchor["view"],
                "anchor_rel_path": anchor["rel_path"],
                "query_root_name": query["root_name"],
                "query_identity_name": query["identity_name"],
                "query_lighting": query["lighting"],
                "query_expression": query["expression"],
                "query_view": query["view"],
                "query_rel_path": query["rel_path"],
            }
        )

    return rows


def map_specs_to_variant(
    specs: list[dict[str, object]],
    variant: str,
    roots: list[dict[str, str]],
) -> list[dict[str, object]]:
    root_by_name = {root["name"]: Path(root["path"]) for root in roots}
    rows: list[dict[str, object]] = []
    for spec in specs:
        anchor_root_name = str(spec.get("anchor_root_name", spec["root_name"]))
        query_root_name = str(spec.get("query_root_name", spec["root_name"]))
        anchor_root_path = root_by_name[anchor_root_name]
        query_root_path = root_by_name[query_root_name]
        anchor_path = anchor_root_path / str(spec["anchor_rel_path"])
        query_path = query_root_path / str(spec["query_rel_path"])
        if not anchor_path.is_dir():
            raise FileNotFoundError(f"Missing mapped anchor path for {variant}: {anchor_path}")
        if not query_path.is_dir():
            raise FileNotFoundError(f"Missing mapped query path for {variant}: {query_path}")

        row = dict(spec)
        row["dataset_variant"] = variant
        row["anchor_path"] = str(anchor_path)
        row["query_path"] = str(query_path)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair_index",
        "split",
        "dataset_variant",
        "group_id",
        "root_name",
        "identity_name",
        "lighting",
        "anchor_root_name",
        "anchor_identity_name",
        "anchor_lighting",
        "anchor_expression",
        "anchor_view",
        "anchor_rel_path",
        "anchor_path",
        "query_root_name",
        "query_identity_name",
        "query_lighting",
        "query_expression",
        "query_view",
        "query_rel_path",
        "query_path",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    root_counts = Counter(str(row["root_name"]) for row in rows)
    group_counts = Counter(str(row["group_id"]) for row in rows)
    cross_identity = sum(
        1
        for row in rows
        if (
            str(row.get("anchor_root_name", row["root_name"])),
            str(row.get("anchor_identity_name", row["identity_name"])),
        )
        != (
            str(row.get("query_root_name", row["root_name"])),
            str(row.get("query_identity_name", row["identity_name"])),
        )
    )
    return {
        "pairs": len(rows),
        "cross_identity_pairs": cross_identity,
        "roots": dict(sorted(root_counts.items())),
        "groups": len(group_counts),
        "pairs_per_group_min": min(group_counts.values()) if group_counts else 0,
        "pairs_per_group_max": max(group_counts.values()) if group_counts else 0,
    }


def main() -> None:
    args = parse_args()
    if args.different_identity:
        args.same_identity = False
    root_sets = parse_root_sets(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    canonical_roots = root_sets[args.canonical_variant]
    split_specs: dict[str, list[dict[str, object]]] = {}
    split_summaries: dict[str, object] = {}

    for split, count in (("train", args.num_train_pairs), ("test", args.num_test_pairs)):
        index = build_index(
            canonical_roots,
            split=split,
            n_val_identities_per_root=args.n_val_identities_per_root,
            split_seed=args.split_seed,
        )
        rng = np.random.RandomState(args.pair_seed + (0 if split == "train" else 1))
        specs = sample_pair_specs(
            index=index,
            split=split,
            num_pairs=count,
            rng=rng,
            same_identity=args.same_identity,
            different_identity=args.different_identity,
            same_lighting=args.same_lighting,
            same_expression=args.same_expression,
            allow_replacement=args.allow_replacement,
        )
        split_specs[split] = specs
        split_summaries[split] = {
            "split_counts": index["split_counts"],
            "sampled": summarize_rows(specs),
        }

    manifest: dict[str, object] = {
        "canonical_variant": args.canonical_variant,
        "n_val_identities_per_root": args.n_val_identities_per_root,
        "split_seed": args.split_seed,
        "pair_seed": args.pair_seed,
        "same_identity": args.same_identity,
        "different_identity": args.different_identity,
        "same_lighting": args.same_lighting,
        "same_expression": args.same_expression,
        "allow_replacement": args.allow_replacement,
        "root_sets": root_sets,
        "splits": split_summaries,
        "outputs": {},
    }

    for variant, roots in sorted(root_sets.items()):
        manifest["outputs"][variant] = {}
        for split, specs in split_specs.items():
            rows = map_specs_to_variant(specs, variant, roots)
            csv_path = out_dir / f"flame_step1_{variant}_{split}_pairs.csv"
            write_csv(csv_path, rows)
            manifest["outputs"][variant][split] = str(csv_path)
            print(f"Wrote {variant:9s} {split:5s}: {csv_path} ({len(rows)} pairs)")

    manifest_path = out_dir / "flame_step1_fixed_pair_sets_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
