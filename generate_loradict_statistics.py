#!/usr/bin/env python
# -*- coding: utf-8 -*-

from csv import writer
import os
import math
import json
from typing import Dict, List, Tuple, Any, Optional, Union, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from omegaconf import DictConfig, OmegaConf
import hydra
from datasets import load_dataset
from tqdm import tqdm

from metanetwork_family import Metanetwork
from utils.mydataset import (
    TextDataset, SquadCollator, PretrainCollator,
    GroupedSquadDataset, GroupTextDataset, GroupPretrainCollator,
    IFTCollator, IFTDataset, IFTC1QADataset
)
from utils.myseed import set_seed
from utils.mylogging import get_logger
from utils.mysaveload import (
    load_checkpoint,
    load_training_state,
    get_latest_checkpoint,
)
from utils.myfreeze import freeze
from utils.myinit import _resolve_device, _import_class

logger = get_logger("metalora")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ---------------------------
# Metrics helpers (Mean L2 = RMS)
# ---------------------------

def _to_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    Convert tensor to 2D for SVD-based effective rank.
    - 2D stays 2D
    - 1D becomes (1, N)
    - >2D flattens all but last dim: (prod(dims[:-1]), dims[-1])
    """
    if x.ndim == 2:
        return x
    if x.ndim == 1:
        return x.unsqueeze(0)
    return x.reshape(-1, x.shape[-1])


@torch.no_grad()
def mean_l2_norm(t: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Mean L2 norm (RMS) over all elements:
      mean_l2(T) = sqrt(mean(T^2)) = ||T||_F / sqrt(numel(T))
    """
    tt = t.detach().float()
    if tt.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(tt * tt) + eps).item())


@torch.no_grad()
def effective_rank(x: torch.Tensor, eps: float = 1e-12, max_svd_dim: int = 4096) -> float:
    """
    Effective rank (erank) = exp( H(p) ), where p_i = s_i / sum(s).
    Uses singular values of the 2D view.
    NOTE: SVD can be expensive; we optionally downsample very large matrices.
    """
    x = x.detach()
    m = _to_matrix(x).float()

    # Optional: reduce huge matrices for speed/memory
    if max(m.shape) > max_svd_dim:
        r = min(m.shape[0], max_svd_dim)
        c = min(m.shape[1], max_svd_dim)
        m = m[:r, :c]

    try:
        s = torch.linalg.svdvals(m)
    except RuntimeError:
        s = torch.linalg.svd(m, full_matrices=False).S

    s_sum = s.sum()
    if s_sum <= eps:
        return 0.0
    p = s / (s_sum + eps)
    h = -(p * (p + eps).log()).sum()
    return float(torch.exp(h).item())


def _flatten_loradict(loradict: Any) -> Dict[str, torch.Tensor]:
    """
    Recursively flatten nested loradict structures into {path: tensor}.

    Works for structures like:
      {34: {"attention": {"q": {"A": T, "B": T, "C": T}, ...}, "mlp": {...}}}
    Produces keys like:
      "34.attention.q.A", "34.mlp.gate.B", ...
    """
    out: Dict[str, torch.Tensor] = {}

    def rec(obj: Any, prefix: str):
        if torch.is_tensor(obj):
            out[prefix] = obj
            return

        if isinstance(obj, Mapping):
            for k, v in obj.items():
                key = str(k)
                new_prefix = f"{prefix}.{key}" if prefix else key
                rec(v, new_prefix)
            return

        if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
            for i, v in enumerate(obj):
                new_prefix = f"{prefix}[{i}]" if prefix else f"[{i}]"
                rec(v, new_prefix)
            return

        # ignore everything else (None, scalars, etc.)

    rec(loradict, "")
    return out


def _candidate_base_keys(lora_key: str) -> List[str]:
    """
    Tailored for Qwen3-8B state_dict keys like:
      model.layers.{i}.self_attn.q_proj.weight
      model.layers.{i}.mlp.gate_proj.weight

    Expects flattened loradict keys like:
      "{layer}.attention.q.A" / "{layer}.attention.q.B" / "{layer}.attention.q.C"
      "{layer}.mlp.gate.A" / "{layer}.mlp.up.B" / "{layer}.mlp.down.C"
    """
    k = str(lora_key).strip(".").replace("..", ".")
    parts = k.split(".")

    if len(parts) < 4 or not parts[0].isdigit():
        cands = [k]
        if not k.endswith(".weight"):
            cands.append(k + ".weight")
        return cands

    layer = int(parts[0])
    block = parts[1]
    sub = parts[2]
    abct = parts[3]

    if block == "attention":
        proj_map = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj"}
        if sub not in proj_map:
            base_prefix = f"model.layers.{layer}.self_attn.{sub}"
        else:
            base_prefix = f"model.layers.{layer}.self_attn.{proj_map[sub]}"
    elif block == "mlp":
        proj_map = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}
        if sub not in proj_map:
            base_prefix = f"model.layers.{layer}.mlp.{sub}"
        else:
            base_prefix = f"model.layers.{layer}.mlp.{proj_map[sub]}"
    else:
        base_prefix = f"model.layers.{layer}.{block}.{sub}"

    base_param = "bias" if abct == "C" else "weight"
    core = f"{base_prefix}.{base_param}"

    candidates = [
        core,
        f"module.{core}",
        f"metamodel.{core}",
        f"model.{core}",
    ]

    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


@torch.no_grad()
def compute_batch_metrics(
    loradict: Any,
    base_param_map: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, Dict[str, float]], float, float]:
    """
    Returns:
      per_key_metrics[name] = {"l2": ..., "erank": ..., "base_l2": ..., "ratio": ...}
        - l2 and base_l2 are MEAN L2 (RMS): sqrt(mean(t^2))
      mean_l2_lora (mean over keys in this batch)
      mean_ratio (mean over keys where base matched)
    """
    flat = _flatten_loradict(loradict)

    per_key: Dict[str, Dict[str, float]] = {}
    l2_vals: List[float] = []
    ratio_vals: List[float] = []

    for name, t in flat.items():
        if not torch.is_tensor(t):
            continue
        tt = t.detach()

        # ✅ Mean L2 norm (RMS), not torch.norm
        l2 = mean_l2_norm(tt)
        er = effective_rank(tt)

        base_l2 = None
        ratio = None

        for cand in _candidate_base_keys(name):
            if cand in base_param_map:
                bw = base_param_map[cand].detach()

                # ✅ Mean L2 norm (RMS) for base weights too
                base_l2 = mean_l2_norm(bw)
                ratio = (l2 / base_l2) if base_l2 > 0 else 0.0
                break

        per_key[name] = {
            "l2": l2,
            "erank": er,
            "base_l2": float(base_l2) if base_l2 is not None else float("nan"),
            "ratio": float(ratio) if ratio is not None else float("nan"),
        }

        l2_vals.append(l2)
        if ratio is not None and not math.isnan(ratio):
            ratio_vals.append(ratio)

    mean_l2 = float(np.mean(l2_vals)) if l2_vals else 0.0
    mean_ratio = float(np.mean(ratio_vals)) if ratio_vals else float("nan")
    return per_key, mean_l2, mean_ratio


def _accumulate_running(
    running: Dict[str, Dict[str, float]],
    counts: Dict[str, int],
    batch_metrics: Dict[str, Dict[str, float]],
):
    """
    running[name] holds sums of metrics over batches.
    counts[name] counts how many batches contributed for that key.
    """
    for name, m in batch_metrics.items():
        if name not in running:
            running[name] = {"l2_sum": 0.0, "erank_sum": 0.0, "ratio_sum": 0.0, "ratio_count": 0.0}
            counts[name] = 0

        running[name]["l2_sum"] += float(m["l2"])
        running[name]["erank_sum"] += float(m["erank"])
        counts[name] += 1

        if not math.isnan(float(m["ratio"])):
            running[name]["ratio_sum"] += float(m["ratio"])
            running[name]["ratio_count"] += 1.0


def _finalize_running(
    running: Dict[str, Dict[str, float]],
    counts: Dict[str, int],
) -> Dict[str, Dict[str, float]]:
    """
    Convert sums to means across val batches.
    """
    out: Dict[str, Dict[str, float]] = {}
    for name, sums in running.items():
        c = max(counts.get(name, 0), 1)
        ratio_c = int(sums.get("ratio_count", 0))

        out[name] = {
            "mean_l2": sums["l2_sum"] / c,
            "mean_erank": sums["erank_sum"] / c,
            "mean_ratio": (sums["ratio_sum"] / ratio_c) if ratio_c > 0 else float("nan"),
        }
    return out


# ---------------------------
# Main
# ---------------------------

@hydra.main(version_base=None, config_path="configs")
@torch.no_grad()
def main(cfg: DictConfig):
    assert cfg.mode == "visualize", "Only visualize mode is supported in this script."
    torch.set_float32_matmul_precision("high")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    logger.info("Resolved config:")
    logger.info(f"\n\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    set_seed(int(cfg.run.seed))
    device = _resolve_device(cfg.run.device)
    torch.backends.cudnn.benchmark = True

    # Load model/tokenizer
    logger.info("Loading model & tokenizer...")
    MetaModelCls = _import_class(cfg.model.metamodel_class_path)
    ConfigCls = _import_class(cfg.model.config_class_path)
    config = ConfigCls.from_pretrained(cfg.model.model_from)

    config.num_mem_token = -1
    cfg.hidden_size = config.hidden_size
    cfg.num_layers = config.num_hidden_layers

    # Auto-set num_mem_token for certain metanetwork types
    if cfg.metanetwork.type in ["transformer", "linear", "lineargate"]:
        tmp_model = MetaModelCls.from_pretrained(cfg.model.model_from, config=config)
        lora_numel = tmp_model.lora_params_numel(cfg.model.lora_r)
        assert lora_numel % (cfg.hidden_size * cfg.num_layers) == 0, \
            "For transformer metanetwork, num_mem_token must be set to model.lora_params_numel(lora_r) * mean_pool_size / (hidden_size * num_layers)"
        config.num_mem_token = (
            tmp_model.lora_params_numel(cfg.model.lora_r)
            * cfg.metanetwork.transformer_cfg.mean_pool_size
            // (cfg.hidden_size * cfg.num_layers)
        )
        cfg.num_mem_token = config.num_mem_token
        del tmp_model
        logger.info(f"Using {cfg.metanetwork.type} metanetwork, automatically set num_mem_token to {config.num_mem_token}")
    elif cfg.metanetwork.type in []:
        config.num_mem_token = cfg.num_mem_token
        logger.info(f"Using {cfg.metanetwork.type} metanetwork, set num_mem_token to {config.num_mem_token} as configured")
    else:
        raise ValueError(f"Unknown metanetwork type: {cfg.metanetwork.type}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_from, padding_side="left", use_fast=True)
    tokenizer.add_tokens(["<RECON>", "<COMP>", "<NOTHING>"])

    # Keep your original template (you had it as a placeholder in the pasted snippet)
    if isinstance(getattr(tokenizer, "chat_template", None), str):
        pass
    else:
        tokenizer.chat_template = "...(unchanged, keep your template here)..."

    metamodel = MetaModelCls.from_pretrained(cfg.model.model_from, config=config)
    metamodel.reset_mem_tokens()
    metamodel.resize_token_embeddings(len(tokenizer))

    metanetwork = Metanetwork(metamodel, cfg, metamodel.lora_params_numel(cfg.model.lora_r))
    metanetwork.train()
    metanetwork.to(device)
    freeze(metamodel)

    logger.info(f"Metanetwork type: {cfg.metanetwork.type}, Transform method: {cfg.metanetwork.method}")

    # Resolve checkpoint dir
    ckpt_root = os.path.join("checkpoints", f"{cfg.name}", f"{cfg.visualize.visualize_mode}")
    if cfg.resume_global_step == -1:
        raise ValueError(f"when visualize resume_global_step must be specified, got {cfg.resume_global_step}")
    elif cfg.resume_global_step == "latest":
        resume_dir = get_latest_checkpoint(ckpt_root)
    elif isinstance(cfg.resume_global_step, int) and cfg.resume_global_step > 0:
        resume_dir = os.path.join(ckpt_root, f"checkpoint-{cfg.resume_global_step}")
        if not os.path.isdir(resume_dir):
            raise ValueError(f"Requested resume dir {resume_dir} does not exist.")
    elif isinstance(cfg.resume_global_step, str) and cfg.resume_global_step.startswith("checkpoint-epoch-"):
        resume_dir = os.path.join(ckpt_root, cfg.resume_global_step)
        if not os.path.isdir(resume_dir):
            raise ValueError(f"Requested resume dir {resume_dir} does not exist.")
    else:
        raise ValueError(f"when visualize resume_global_step must be specified, got {cfg.resume_global_step}")

    logger.info(f"Resume mode, loading from {resume_dir}...")
    metanetwork, metalora = load_checkpoint(metanetwork, resume_dir, device)
    _ = load_training_state(resume_dir)  # not required for metrics, but kept for compatibility

    metanetwork.metamodel.config.use_cache = False
    metanetwork.to(device)

    # Build base param map ONCE (for ratio computation)
    base_param_map: Dict[str, torch.Tensor] = {n: p for n, p in metamodel.named_parameters()}

    # Data
    logger.info("Preparing data...")
    if cfg.data.source == "transmla":
        dataset = load_dataset(os.path.join("data", "transmla_pretrain_6B_tokens"), split="train")
        split_dataset = dataset.train_test_split(test_size=0.0001, seed=42)
        train_texts = split_dataset["train"]
        val_texts = split_dataset["test"]
        train_ds = TextDataset(train_texts["text"], tokenizer)
        val_ds = TextDataset(val_texts["text"], tokenizer)
        train_collator = PretrainCollator(
            tokenizer=tokenizer, metatrain=True, cfg=cfg,
            conversation_max_length=cfg.data.conversation_max_length,
            context_max_length=cfg.data.context_max_length
        )
        val_collator = PretrainCollator(
            tokenizer=tokenizer, metatrain=True, cfg=cfg,
            conversation_max_length=cfg.data.conversation_max_length,
            context_max_length=cfg.data.context_max_length
        )
    elif cfg.data.source == "grouptransmla":
        dataset = load_dataset(os.path.join("data", "transmla_pretrain_6B_tokens"), split="train")
        split_dataset = dataset.train_test_split(test_size=0.0001, seed=42)
        train_texts = split_dataset["train"]
        val_texts = split_dataset["test"]
        train_ds = GroupTextDataset(
            train_texts["text"], tokenizer, cfg.data.conversation_max_length,
            os.path.join("data", "transmla_pretrain_6B_tokens"), "train"
        )
        val_ds = GroupTextDataset(
            val_texts["text"], tokenizer, cfg.data.conversation_max_length,
            os.path.join("data", "transmla_pretrain_6B_tokens"), "val"
        )
        train_collator = GroupPretrainCollator(
            tokenizer, cfg,
            conversation_max_length=cfg.data.conversation_max_length,
            context_max_length=cfg.data.context_max_length,
            metatrain=True
        )
        val_collator = GroupPretrainCollator(
            tokenizer, cfg,
            conversation_max_length=cfg.data.conversation_max_length,
            context_max_length=cfg.data.context_max_length,
            metatrain=True
        )
    elif cfg.data.source == "squad":
        train_dataset = load_dataset(os.path.join("data", "squad"), split="train")
        val_dataset = load_dataset(os.path.join("data", "squad"), split="validation")
        val_dataset = val_dataset.shuffle(seed=42).select(range(1000))
        train_ds = GroupedSquadDataset(train_dataset, tokenizer, 512, name="Train", sep="\n\n")
        val_ds = GroupedSquadDataset(val_dataset, tokenizer, 512, name="Validation", sep="\n\n")
        train_collator = SquadCollator(
            tokenizer=tokenizer,
            conversation_max_length=cfg.data.conversation_max_length,
            context_max_length=cfg.data.context_max_length,
            metatrain=True, cfg=cfg
        )
        val_collator = SquadCollator(
            tokenizer=tokenizer,
            conversation_max_length=cfg.data.conversation_max_length,
            context_max_length=cfg.data.context_max_length,
            metatrain=True, cfg=cfg
        )
    elif cfg.data.source == "ift":
        data_path = os.path.join("data", "ift_cqa.json")
        group_idx_path = os.path.join(
            "data",
            f"ift_cqa_group_idxs_context{cfg.data.context_max_length}_conversation{cfg.data.conversation_max_length}.json"
        )
        train_ds = IFTDataset(data_path, group_idx_path, use_exceed=True)
        val_dataset = load_dataset(os.path.join("data", "squad"), split="validation")
        val_dataset = val_dataset.shuffle(seed=42).select(range(1000))
        val_ds = GroupedSquadDataset(val_dataset, tokenizer, 512, name="Validation", sep="<|endoftext|>")
        train_collator = IFTCollator(tokenizer, cfg.data.context_max_length, cfg.data.conversation_max_length, cfg=cfg)
        val_collator = SquadCollator(
            tokenizer=tokenizer,
            conversation_max_length=cfg.data.conversation_max_length,
            context_max_length=cfg.data.context_max_length,
            metatrain=True, cfg=cfg
        )
    elif cfg.data.source == "ift-c1qa":
        data_path = os.path.join("data", "ift_c1qa.json")
        train_ds = IFTC1QADataset(
            data_path, use_exceed=False,
            max_context_len=cfg.data.context_max_length,
            max_conversation_len=cfg.data.conversation_max_length
        )
        val_dataset = load_dataset(os.path.join("data", "squad"), split="validation")
        val_dataset = val_dataset.shuffle(seed=42).select(range(1000))
        val_ds = GroupedSquadDataset(val_dataset, tokenizer, 512, name="Validation", sep="\n\n")
        train_collator = IFTCollator(tokenizer, cfg.data.context_max_length, cfg.data.conversation_max_length, cfg=cfg)
        val_collator = SquadCollator(
            tokenizer=tokenizer,
            conversation_max_length=cfg.data.conversation_max_length,
            context_max_length=cfg.data.context_max_length,
            metatrain=True, cfg=cfg
        )
    else:
        raise ValueError(f"Unknown data source: {cfg.data.source}")

    pin = (device.type == "cuda")
    num_workers_default = 2 if device.type == "cuda" else 0

    # Only val_loader is needed for metrics; train_loader left out intentionally
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.eval_batch_size,
        shuffle=False,
        collate_fn=val_collator,
        pin_memory=pin,
        num_workers=getattr(cfg.data, "num_workers", num_workers_default),
        persistent_workers=pin and getattr(cfg.data, "num_workers", num_workers_default) > 0,
    )

    visualize_dir = os.path.join(resume_dir, "visualize")
    os.makedirs(visualize_dir, exist_ok=True)

    running: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}
    batch_mean_l2_list: List[float] = []
    batch_mean_ratio_list: List[float] = []

    logger.info("Iterating over val_loader to compute LoRA stats (mean L2 = RMS)...")
    for _, batch in enumerate(tqdm(val_loader, desc="Val batches"), start=1):
        evidence_ids = batch["evidence_ids"].to(device, non_blocking=True)
        evidence_attention_mask = batch["evidence_attention_mask"].to(device, non_blocking=True)

        loradict = metanetwork.generate_lora_dict(
            evidence_ids,
            evidence_attention_mask,
            metalora,
            use_gradient_checkpoint=False,
            return_plain=False,
        )

        per_key_metrics, mean_l2_lora, mean_ratio = compute_batch_metrics(
            loradict=loradict,
            base_param_map=base_param_map,
        )

        _accumulate_running(running, counts, per_key_metrics)
        batch_mean_l2_list.append(mean_l2_lora)
        if not math.isnan(mean_ratio):
            batch_mean_ratio_list.append(mean_ratio)

    final = _finalize_running(running, counts)

    overall_mean_l2 = float(np.mean(batch_mean_l2_list)) if batch_mean_l2_list else 0.0
    overall_mean_ratio = float(np.mean(batch_mean_ratio_list)) if batch_mean_ratio_list else float("nan")

    summary = {
        "num_val_batches": int(len(batch_mean_l2_list)),
        "overall_mean_l2_lora_per_key": overall_mean_l2,
        "overall_mean_ratio_lora_to_base_per_key": overall_mean_ratio,
    }

    # Save results
    summary_path = os.path.join(visualize_dir, "lora_val_metrics_summary.json")
    perkey_path = os.path.join(visualize_dir, "lora_val_metrics_per_key.json")
    csv_path = os.path.join(visualize_dir, "lora_val_metrics_per_key.csv")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(perkey_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = writer(f)
        w.writerow(["key", "mean_l2", "mean_erank", "mean_ratio_lora_to_base"])
        for k in sorted(final.keys()):
            w.writerow([k, final[k]["mean_l2"], final[k]["mean_erank"], final[k]["mean_ratio"]])

    logger.info(f"Saved summary to: {summary_path}")
    logger.info(f"Saved per-key metrics to: {perkey_path}")
    logger.info(f"Saved per-key metrics CSV to: {csv_path}")
    logger.info(f"Summary: {summary}")


if __name__ == "__main__":
    main()
