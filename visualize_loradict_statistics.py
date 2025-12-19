#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot LoRA metrics (A/B only; ignore C) + base-model parameter mean L2 norms.

What this script does
---------------------
1) Reads:
     <visualize_dir>/lora_val_metrics_per_key.json
   Expected keys look like:
     "34.attention.q.A", "34.attention.q.B", "34.mlp.gate.A", ...
   Each key maps to metrics, ideally:
     {"mean_l2": ..., "mean_erank": ..., "mean_ratio": ...}
   (This script is tolerant to a few alternate field names like "l2"/"erank"/"ratio".)

2) Produces 3 heatmaps (ONE per metric) with architecture-like x-axis:
     Attn.q.A  Attn.q.B  Attn.k.A  Attn.k.B  Attn.v.A  Attn.v.B  Attn.o.A  Attn.o.B  |  MLP.gate.A  MLP.gate.B  MLP.up.A  MLP.up.B  MLP.down.A  MLP.down.B

   Files:
     mean_l2_AB_heatmap.png
     mean_erank_AB_heatmap.png
     mean_ratio_AB_heatmap.png

3) Produces 1 heatmap for base-model parameter MEAN L2 (RMS) norms (weights only):
     Attn.q  Attn.k  Attn.v  Attn.o | MLP.gate  MLP.up  MLP.down

   File:
     base_weight_mean_l2_heatmap.png

Base-model norm source (choose one)
-----------------------------------
Option A (recommended): provide --base_ckpt (HF model directory or checkpoint file)
  -> loads tensors and computes mean L2 = RMS = sqrt(mean(w^2)).

Option B: provide --base_l2_json (JSON: param_name -> mean_l2)
  -> plots provided values.

Usage examples
--------------
A) Compute base mean L2 directly:
  python plot_lora_and_base_meanl2.py --visualize_dir /path/to/.../visualize --base_ckpt /path/to/models/Qwen3-8B

B) Use precomputed base mean L2 JSON:
  python plot_lora_and_base_meanl2.py --visualize_dir /path/to/.../visualize --base_l2_json /path/to/base_mean_l2.json
"""

import os
import re
import json
import math
import argparse
from typing import Dict, Tuple, Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Parsing patterns
# -------------------------

# LoRA per-key JSON keys like: "34.attention.q.A"
LORA_KEY_RE = re.compile(
    r"^(?P<layer>\d+)\.(?P<block>attention|mlp)\.(?P<sub>q|k|v|o|gate|up|down)\.(?P<param>A|B|C)$"
)

# Base model state_dict keys like: "model.layers.34.self_attn.q_proj.weight"
BASE_KEY_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.(?P<section>self_attn|mlp)\.(?P<name>q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.(?P<wb>weight|bias)$"
)


# -------------------------
# Column layout (architecture-like)
# -------------------------

BASE_MODULES: List[Tuple[str, str]] = [
    ("attention", "q"),
    ("attention", "k"),
    ("attention", "v"),
    ("attention", "o"),
    ("mlp", "gate"),
    ("mlp", "up"),
    ("mlp", "down"),
]

# Expand each module into A and B columns
PARAMS = ["A", "B"]  # ignore C

LORA_COLS: List[Tuple[str, str, str]] = []
LORA_COL_NAMES: List[str] = []
for block, sub in BASE_MODULES:
    prefix = "Attn" if block == "attention" else "MLP"
    for p in PARAMS:
        LORA_COLS.append((block, sub, p))
        LORA_COL_NAMES.append(f"{prefix}.{sub}.{p}")

# Base columns (weights only)
BASE_COLS: List[Tuple[str, str]] = [
    ("self_attn", "q_proj"),
    ("self_attn", "k_proj"),
    ("self_attn", "v_proj"),
    ("self_attn", "o_proj"),
    ("mlp", "gate_proj"),
    ("mlp", "up_proj"),
    ("mlp", "down_proj"),
]
BASE_COL_NAMES = ["Attn.q", "Attn.k", "Attn.v", "Attn.o", "MLP.gate", "MLP.up", "MLP.down"]


# -------------------------
# Utils
# -------------------------

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def nanmean_list(xs: List[float]) -> float:
    xs2 = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs2:
        return float("nan")
    return float(np.mean(xs2))


def get_metric(d: Dict[str, Any], keys: List[str], default: float = float("nan")) -> float:
    """
    Robust metric getter: try multiple field names.
    """
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except Exception:
                pass
    return default


def save_heatmap(
    grid: np.ndarray,
    title: str,
    out_path: str,
    xlabels: List[str],
    split_x: Optional[float] = None,
    per_module_separators: Optional[List[float]] = None,
):
    fig, ax = plt.subplots(figsize=(max(12, len(xlabels) * 0.65), max(6, grid.shape[0] * 0.15)))

    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(masked, aspect="auto")  # default colormap

    ax.set_title(title)
    ax.set_xlabel("Module position in layer")
    ax.set_ylabel("Layer index")

    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")

    # y ticks: keep readable
    if grid.shape[0] <= 64:
        ax.set_yticks(np.arange(grid.shape[0]))
    else:
        step = max(1, grid.shape[0] // 32)
        ax.set_yticks(np.arange(0, grid.shape[0], step))

    if split_x is not None:
        ax.axvline(x=split_x, linewidth=2)

    if per_module_separators:
        for x in per_module_separators:
            ax.axvline(x=x, linewidth=1)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Value", rotation=90)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -------------------------
# LoRA: build A/B grids (ignore C)
# -------------------------

def lora_json_to_grids(per_key: Dict[str, Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns 3 grids of shape [num_layers, 14] for:
      mean_l2, mean_erank, mean_ratio
    with columns in LORA_COLS order, using A/B only.

    If duplicates exist for a cell, nan-mean them.
    """
    # bucket[(layer, block, sub, param)] -> lists of values
    bucket: Dict[Tuple[int, str, str, str], Dict[str, List[float]]] = {}
    max_layer = -1
    matched = 0

    for key, metrics in per_key.items():
        m = LORA_KEY_RE.match(key)
        if not m:
            continue
        layer = int(m.group("layer"))
        block = m.group("block")
        sub = m.group("sub")
        param = m.group("param")

        if param not in ("A", "B"):
            continue  # ignore C entirely

        matched += 1
        max_layer = max(max_layer, layer)

        cell = (layer, block, sub, param)
        if cell not in bucket:
            bucket[cell] = {"l2": [], "erank": [], "ratio": []}

        # tolerant to field naming
        l2 = get_metric(metrics, ["mean_l2", "l2"], default=float("nan"))
        er = get_metric(metrics, ["mean_erank", "erank"], default=float("nan"))
        ra = get_metric(metrics, ["mean_ratio", "ratio"], default=float("nan"))

        bucket[cell]["l2"].append(l2)
        bucket[cell]["erank"].append(er)
        bucket[cell]["ratio"].append(ra)

    if matched == 0:
        raise ValueError(
            "No keys matched expected LoRA pattern like '34.attention.q.A' (A/B only). "
            "Please check the keys in lora_val_metrics_per_key.json."
        )

    num_layers = max_layer + 1
    num_cols = len(LORA_COLS)

    grid_l2 = np.full((num_layers, num_cols), np.nan, dtype=np.float64)
    grid_er = np.full((num_layers, num_cols), np.nan, dtype=np.float64)
    grid_ra = np.full((num_layers, num_cols), np.nan, dtype=np.float64)

    for layer in range(num_layers):
        for j, (block, sub, param) in enumerate(LORA_COLS):
            cell = (layer, block, sub, param)
            if cell not in bucket:
                continue
            grid_l2[layer, j] = nanmean_list(bucket[cell]["l2"])
            grid_er[layer, j] = nanmean_list(bucket[cell]["erank"])
            grid_ra[layer, j] = nanmean_list(bucket[cell]["ratio"])

    return grid_l2, grid_er, grid_ra


# -------------------------
# Base model: compute / load mean L2 (RMS) norms and map to grid
# -------------------------

def compute_base_mean_l2_from_hf(model_path: str) -> Dict[str, float]:
    """
    Compute mean L2 norm (RMS) per parameter:
      mean_l2(W) = sqrt(mean(W^2)) = ||W||_F / sqrt(numel(W))

    Returns dict: param_name -> mean_l2
    """
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    sd = model.state_dict()

    out: Dict[str, float] = {}
    for name, t in sd.items():
        if not torch.is_tensor(t):
            continue
        tt = t.detach().float()
        if tt.numel() == 0:
            out[name] = 0.0
        else:
            out[name] = float(torch.sqrt(torch.mean(tt * tt)).item())
    return out


def base_mean_l2_to_grid(base_mean_l2: Dict[str, Any]) -> np.ndarray:
    """
    Build grid [num_layers, 7] from dict param_name -> mean_l2
    Uses WEIGHTS ONLY, matching:
      model.layers.{i}.self_attn.{q/k/v/o}_proj.weight
      model.layers.{i}.mlp.{gate/up/down}_proj.weight
    """
    bucket: Dict[Tuple[int, str, str], List[float]] = {}
    max_layer = -1
    matched = 0

    for name, val in base_mean_l2.items():
        m = BASE_KEY_RE.match(name)
        if not m:
            continue
        if m.group("wb") != "weight":
            continue  # plot weights only

        layer = int(m.group("layer"))
        section = m.group("section")
        proj = m.group("name")

        try:
            v = float(val)
        except Exception:
            continue

        matched += 1
        max_layer = max(max_layer, layer)
        bucket.setdefault((layer, section, proj), []).append(v)

    if matched == 0:
        raise ValueError(
            "No base keys matched expected pattern like 'model.layers.34.self_attn.q_proj.weight'. "
            "If your base keys are prefixed (e.g. 'module.model.layers...'), you can strip/normalize before plotting."
        )

    num_layers = max_layer + 1
    grid = np.full((num_layers, len(BASE_COLS)), np.nan, dtype=np.float64)

    for layer in range(num_layers):
        for j, (section, proj) in enumerate(BASE_COLS):
            cell = (layer, section, proj)
            if cell not in bucket:
                continue
            grid[layer, j] = nanmean_list(bucket[cell])

    return grid


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visualize_dir", type=str, required=True)
    ap.add_argument("--lora_json", type=str, default="lora_val_metrics_per_key.json")

    # Provide one of:
    ap.add_argument("--base_ckpt", type=str, default=None, help="HF model dir or checkpoint to compute base mean L2")
    ap.add_argument("--base_l2_json", type=str, default=None, help="JSON of base param -> mean_l2 (RMS)")

    args = ap.parse_args()

    vis_dir = args.visualize_dir
    lora_path = os.path.join(vis_dir, args.lora_json)
    if not os.path.isfile(lora_path):
        raise FileNotFoundError(f"Missing LoRA metrics JSON: {lora_path}")

    per_key = load_json(lora_path)
    if not isinstance(per_key, dict):
        raise ValueError(f"{lora_path} must be a dict, got {type(per_key)}")

    # --- LoRA grids
    grid_l2, grid_er, grid_ra = lora_json_to_grids(per_key)

    # separators for LoRA layout:
    # - attention (4 modules *2 cols) = 8 cols -> split at 7.5
    lora_split = 7.5
    # - module separators every 2 cols (after each module), except at the attention/mlp split handled by lora_split
    lora_module_seps = [1.5, 3.5, 5.5, 9.5, 11.5]  # visual only

    save_heatmap(
        grid_l2,
        "LoRA mean L2 norm (RMS) by layer & module (A/B separate; C omitted)",
        os.path.join(vis_dir, "mean_l2_AB_heatmap.png"),
        LORA_COL_NAMES,
        split_x=lora_split,
        per_module_separators=lora_module_seps,
    )
    save_heatmap(
        grid_er,
        "LoRA mean effective rank by layer & module (A/B separate; C omitted)",
        os.path.join(vis_dir, "mean_erank_AB_heatmap.png"),
        LORA_COL_NAMES,
        split_x=lora_split,
        per_module_separators=lora_module_seps,
    )
    save_heatmap(
        grid_ra,
        "LoRA/Base mean L2 ratio (RMS/RMS) by layer & module (A/B separate; C omitted)",
        os.path.join(vis_dir, "mean_ratio_AB_heatmap.png"),
        LORA_COL_NAMES,
        split_x=lora_split,
        per_module_separators=lora_module_seps,
    )

    # --- Base mean L2 (RMS) norms
    if args.base_l2_json is not None and args.base_ckpt is not None:
        raise ValueError("Please provide only one of --base_l2_json or --base_ckpt.")

    if args.base_l2_json is None and args.base_ckpt is None:
        raise ValueError("Provide --base_ckpt (recommended) OR --base_l2_json to plot base parameter mean L2 norms.")

    if args.base_l2_json is not None:
        base_mean_l2 = load_json(args.base_l2_json)
        if not isinstance(base_mean_l2, dict):
            raise ValueError(f"--base_l2_json must be dict name->mean_l2, got {type(base_mean_l2)}")
    else:
        base_mean_l2 = compute_base_mean_l2_from_hf(args.base_ckpt)

    base_grid = base_mean_l2_to_grid(base_mean_l2)

    save_heatmap(
        base_grid,
        "Base model weight mean L2 norm (RMS) by layer & module (weights only)",
        os.path.join(vis_dir, "base_weight_mean_l2_heatmap.png"),
        BASE_COL_NAMES,
        split_x=3.5,  # between attention(4) and mlp(3)
        per_module_separators=None,
    )

    print("Saved:")
    print(" ", os.path.join(vis_dir, "mean_l2_AB_heatmap.png"))
    print(" ", os.path.join(vis_dir, "mean_erank_AB_heatmap.png"))
    print(" ", os.path.join(vis_dir, "mean_ratio_AB_heatmap.png"))
    print(" ", os.path.join(vis_dir, "base_weight_mean_l2_heatmap.png"))


if __name__ == "__main__":
    main()
