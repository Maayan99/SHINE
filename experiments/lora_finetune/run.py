#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run.py – Test-time LoRA fine-tuning experiment for SHINE.

For each test sample we:
  1. Generate a LoRA from the hypernetwork  (SHINE, 0 grad steps)
  2. Clone it into trainable leaf tensors
  3. Run K Adam steps optimising those tensors against the RECON LM loss
     (reconstruct the evidence text – same objective as SHINE pretraining)
  4. Snapshot the LoRA at each step in `lora_finetune.eval_at`
  5. Run QA generation + F1 evaluation at each snapshot
  6. Repeat steps 2-5 from a scratch-init LoRA baseline (A∼N(0,√scale), B=0)

Usage (run from SHINE/SHINE/):
    python experiments/lora_finetune/run.py lora_finetune.num_samples=5

    # Override checkpoint stage:
    python experiments/lora_finetune/run.py \
        checkpoint_stage=train \
        test_global_step=epoch-2
"""

import sys
import os

# ── path fix: ensure repo root and this script's dir are on sys.path ─────────
_this_dir  = os.path.abspath(os.path.dirname(__file__))
_repo_root = os.path.abspath(os.path.join(_this_dir, "..", ".."))
for _p in (_repo_root, _this_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json
import math
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from omegaconf import DictConfig, OmegaConf
import hydra
from transformers import AutoTokenizer

from metanetwork_family import Metanetwork
from utils.mydataset import (
    SquadCollator, GroupedSquadDataset,
    HotpotqaDataset, MsmarcoDataset, MusiqueDataset,
)
from utils.mysaveload import load_checkpoint, get_latest_checkpoint
from utils.myfreeze import freeze
from utils.myloradict import merge_loradicts
from utils.myinit import _resolve_device, _import_class
from utils.myseed import set_seed
from utils.mylogging import get_logger
from calculate_f1 import compute_f1
from evaluation.hotpotqa import f1_score as hotpotqa_compute_f1
from test import extract_think_and_answer, compute_sample_f1

from finetune_lora import (
    clone_loradict_to_params,
    zero_init_loradict,
    build_clm_inputs,
    build_recon_inputs,
    finetune_lora_on_evidence,
)

logger = get_logger("run_lora_finetune")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper: run generation + decode answer for a single sample
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_lora(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    input_mask: torch.Tensor,
    loradict: dict,
    max_new_tokens: int,
    ground_truth: Any,
    f1_metric,
    sample_id: int,
    step: int,
    condition: str,
    question: str = "",
) -> Tuple[str, str, float]:
    """
    Run generation with the given loradict, extract the answer, compute F1.

    Returns: (full_generated_text, extracted_answer, f1_score)
    """
    model.eval()
    gen_out = model.generate(
        input_ids=input_ids,
        attention_mask=input_mask,
        loradict=loradict,
        ignore_mem_token=True,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    # Decode only the newly generated tokens (everything after the input)
    new_tokens = gen_out[0, input_ids.shape[1]:]
    answer_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    full_text = answer_text

    _, answer = extract_think_and_answer(answer_text)

    # Detect truncated thinking: if no </think> was found, the model hit max_new_tokens
    # mid-think and extract_think_and_answer returned the raw "<think>..." text.
    # In this case return empty string so F1=0 (accurate signal, not misleading garbage).
    if answer.lower().startswith("<think>"):
        logger.warning(
            f"[sample {sample_id}][{condition}] step={step}: "
            f"Thinking was truncated (no </think> found). "
            f"Returning empty answer (F1=0). Consider increasing max_new_tokens."
        )
        answer = ""

    f1 = compute_sample_f1(ground_truth, answer, f1_metric)

    logger.info(
        f"[sample {sample_id}][{condition}] step={step}: f1={f1:.4f}\n"
        f"  question   : {question}\n"
        f"  ground truth: {ground_truth}\n"
        f"  raw output : {answer_text.strip()[:300]}\n"
        f"  answer     : {answer}"
    )
    return full_text, answer, f1


# ─────────────────────────────────────────────────────────────────────────────
# Load dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_test_data(cfg, tokenizer, num_samples: int):
    """
    Returns (dataset, collator, f1_metric, split_name).
    Mirrors the data-loading logic in test.py.
    """
    source = cfg.test.source
    seed = 42
    N = num_samples if num_samples > 0 else 1_000_000

    logger.info(f"Loading dataset: source={source}, max_samples={N}")

    if source == "squad":
        f1_metric = compute_f1
        squad_path = os.path.join("data", "squad")
        if os.path.isfile(os.path.join(squad_path, "dataset_dict.json")):
            data = load_from_disk(squad_path)["validation"]
        else:
            data = load_dataset(squad_path, split="validation")
        data = data.shuffle(seed=seed)
        subset = data.select(range(min(N, len(data))))
        ds = GroupedSquadDataset(subset, tokenizer, cfg.test.context_avg_len)
        split_name = f"squad_{cfg.test.context_avg_len}"

    elif source == "hotpotqa":
        f1_metric = hotpotqa_compute_f1
        data = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
        data = data.shuffle(seed=seed)
        subset = data.select(range(min(N, len(data))))
        ds = HotpotqaDataset(subset)
        split_name = "hotpotqa"

    elif source == "musique":
        f1_metric = compute_f1
        data = load_dataset("dgslibisey/MuSiQue", split="validation")
        data = data.shuffle(seed=seed)
        subset = data.select(range(min(N, len(data))))
        ds = MusiqueDataset(subset)
        split_name = "musique"

    elif source == "2wikimultihopqa":
        f1_metric = hotpotqa_compute_f1
        data = load_dataset("framolfese/2WikiMultihopQA", split="validation")
        data = data.shuffle(seed=seed)
        subset = data.select(range(min(N, len(data))))
        ds = HotpotqaDataset(subset)
        split_name = "2wikimultihopqa"

    elif source == "msmarco_v1":
        f1_metric = compute_f1
        data = load_dataset("microsoft/ms_marco", "v1.1", split="test")
        data = data.shuffle(seed=seed)
        subset = data.select(range(min(N, len(data))))
        ds = MsmarcoDataset(subset)
        split_name = "msmarco_v1"

    elif source == "msmarco_v2":
        f1_metric = compute_f1
        data = load_dataset("microsoft/ms_marco", "v2.1", split="validation")
        data = data.shuffle(seed=seed)
        subset = data.select(range(min(N, len(data))))
        ds = MsmarcoDataset(subset)
        split_name = "msmarco_v2"

    else:
        raise ValueError(f"Unknown test source: {source}")

    collator = SquadCollator(
        tokenizer=tokenizer,
        context_max_length=cfg.test.context_max_length,
        conversation_max_length=cfg.test.conversation_max_length,
        cfg=cfg,
    )

    logger.info(f"Dataset loaded: {len(ds)} samples → split_name='{split_name}'")
    return ds, collator, f1_metric, split_name


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_results(records: List[dict], eval_at: List[int]) -> dict:
    """
    Given a list of per-sample records (each with "shine"/{step: {f1}} and
    "scratch"/{step: {f1}}), compute per-step average F1 and std.
    """
    agg = {
        "eval_at_steps":   eval_at,
        "shine_finetune":  {},
        "scratch_finetune": {},
        "num_samples": len(records),
    }

    for condition in ("shine", "scratch"):
        for step in eval_at:
            step_str = str(step)
            f1_vals = []
            for rec in records:
                entry = rec.get(condition, {}).get(step_str)
                if entry is not None and isinstance(entry.get("f1"), (int, float)):
                    f1_vals.append(float(entry["f1"]))
            if f1_vals:
                avg = sum(f1_vals) / len(f1_vals)
                std = math.sqrt(sum((x - avg) ** 2 for x in f1_vals) / max(1, len(f1_vals)))
            else:
                avg, std = float("nan"), float("nan")
            agg[f"{condition}_finetune"][step_str] = {
                "avg_f1": avg,
                "std_f1": std,
                "n":      len(f1_vals),
            }
            logger.info(
                f"[aggregate] {condition} step={step}: "
                f"avg_f1={avg:.4f} std={std:.4f} n={len(f1_vals)}"
            )

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    logger.info("=" * 70)
    logger.info("SHINE LoRA fine-tuning experiment")
    logger.info("=" * 70)
    logger.info(f"\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    # ── seed + device ─────────────────────────────────────────────────────────
    set_seed(int(cfg.run.seed))
    if cfg.run.device == "mps":
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        logger.info(f"MPS requested; using device={device}")
    else:
        device = _resolve_device(cfg.run.device)
    logger.info(f"device={device}  seed={cfg.run.seed}")

    # ── load model + tokenizer ────────────────────────────────────────────────
    logger.info("Loading model and tokenizer...")
    MetaModelCls = _import_class(cfg.model.metamodel_class_path)
    ConfigCls    = _import_class(cfg.model.config_class_path)

    config = ConfigCls.from_pretrained(cfg.model.model_from)
    config.num_mem_token = -1
    cfg.hidden_size = config.hidden_size
    cfg.num_layers  = config.num_hidden_layers
    logger.info(f"Base model: hidden_size={config.hidden_size}, num_layers={config.num_hidden_layers}")

    # Compute num_mem_token for transformer metanetwork
    if cfg.metanetwork.type == "transformer":
        tmp_model = MetaModelCls.from_pretrained(cfg.model.model_from, config=config)
        lora_numel = tmp_model.lora_params_numel(cfg.model.lora_r)
        assert lora_numel % (cfg.hidden_size * cfg.num_layers) == 0, (
            "lora_params_numel must be divisible by hidden_size * num_layers "
            "for transformer metanetwork"
        )
        config.num_mem_token = lora_numel // (cfg.hidden_size * cfg.num_layers)
        cfg.num_mem_token = config.num_mem_token
        del tmp_model
        logger.info(f"Transformer metanetwork: num_mem_token={config.num_mem_token}")
    else:
        config.num_mem_token = cfg.num_mem_token

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.tokenizer_from, padding_side="left", use_fast=True
    )
    # Add RECON/COMP special tokens – same as test_pretrain.py
    tokenizer.add_tokens(["<RECON>", "<COMP>"])
    logger.info(f"Tokenizer vocab size after adding RECON/COMP: {len(tokenizer)}")

    metamodel = MetaModelCls.from_pretrained(cfg.model.model_from, config=config)
    metamodel.reset_mem_tokens()
    metamodel.resize_token_embeddings(len(tokenizer))

    metanetwork = Metanetwork(metamodel, cfg, metamodel.lora_params_numel(cfg.model.lora_r))
    metanetwork.to(device)
    freeze(metamodel)
    logger.info("Model loaded and frozen.")

    # ── load checkpoint ───────────────────────────────────────────────────────
    stage = cfg.get("checkpoint_stage", "train")  # "pretrain" or "train"
    ckpt_root = os.path.join("checkpoints", cfg.name, stage)
    logger.info(f"Looking for checkpoint in: {ckpt_root}  stage={stage}")

    step_spec = cfg.test_global_step
    if step_spec == "latest":
        resume_dir = get_latest_checkpoint(ckpt_root)
        if resume_dir is None:
            raise FileNotFoundError(f"No checkpoint found under {ckpt_root}")
    elif step_spec == "final":
        resume_dir = os.path.join(ckpt_root, "final")
    elif isinstance(step_spec, str) and step_spec.startswith("epoch-"):
        resume_dir = os.path.join(ckpt_root, f"checkpoint-{step_spec}")
    elif isinstance(step_spec, str) and step_spec.startswith("checkpoint-epoch-"):
        resume_dir = os.path.join(ckpt_root, step_spec)
    elif isinstance(step_spec, int) and step_spec > 0:
        resume_dir = os.path.join(ckpt_root, f"checkpoint-{step_spec}")
    else:
        raise ValueError(f"Invalid test_global_step: {step_spec!r}")

    if not os.path.isdir(resume_dir):
        raise FileNotFoundError(f"Checkpoint dir not found: {resume_dir}")

    logger.info(f"Loading checkpoint from: {resume_dir}")
    USE_ADDITIONAL_METALORA = bool(cfg.model.ift_additional_metalora_r >= 0)
    metanetwork, metalora, ift_additional_metalora = load_checkpoint(
        metanetwork,
        resume_dir,
        device,
        load_ift_additional_metalora=USE_ADDITIONAL_METALORA,
        zero_ift_additional_metalora=(cfg.model.ift_additional_metalora_r == 0),
    )
    if USE_ADDITIONAL_METALORA and ift_additional_metalora is not None:
        metalora = merge_loradicts(metalora, ift_additional_metalora)
        logger.info("Merged ift_additional_metalora into metalora.")

    metanetwork.eval()
    freeze(metamodel)
    logger.info("Checkpoint loaded. Model and metamodel are fully frozen.")

    # ── dataset ───────────────────────────────────────────────────────────────
    num_samples = int(cfg.lora_finetune.num_samples)
    ds, collator, f1_metric, split_name = load_test_data(cfg, tokenizer, num_samples)

    loader = DataLoader(
        ds,
        batch_size=1,          # must be 1 for per-sample fine-tuning
        shuffle=False,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
        num_workers=0,         # keep simple; no workers needed for bs=1
    )

    # ── output paths ─────────────────────────────────────────────────────────
    out_dir = cfg.lora_finetune.out_dir
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path   = os.path.join(out_dir, f"{split_name}_per_sample.jsonl")
    summary_path = os.path.join(out_dir, f"{split_name}_summary.json")
    logger.info(f"Output dir: {out_dir}")
    logger.info(f"Per-sample JSONL: {jsonl_path}")
    logger.info(f"Summary JSON:     {summary_path}")

    eval_at = list(cfg.lora_finetune.eval_at)
    lr      = float(cfg.lora_finetune.lr)
    max_new_tokens = int(cfg.test.max_new_tokens)
    recon_mode = cfg.lora_finetune.recon_mode
    logger.info(
        f"Experiment config: eval_at={eval_at}, lr={lr}, "
        f"recon_mode={recon_mode}, max_new_tokens={max_new_tokens}"
    )

    # ── resume support: figure out last completed sample ─────────────────────
    completed_ids: set = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    completed_ids.add(int(rec["sample_idx"]))
                except Exception:
                    pass
        logger.info(f"Resuming: found {len(completed_ids)} completed samples in {jsonl_path}")

    all_records: List[dict] = []
    # pre-load already-completed records for aggregation
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    all_records.append(json.loads(line))
                except Exception:
                    pass

    jsonl_f = open(jsonl_path, "a", encoding="utf-8")

    # ── main per-sample loop ─────────────────────────────────────────────────
    samples_processed = 0
    t_start = time.time()

    for batch_idx, batch in enumerate(loader):
        sample_idx = batch_idx  # loader is shuffled=False, bs=1

        if num_samples > 0 and samples_processed >= num_samples:
            logger.info(f"Reached num_samples={num_samples}, stopping.")
            break

        if sample_idx in completed_ids:
            logger.info(f"[sample {sample_idx}] Already processed, skipping.")
            samples_processed += 1
            continue

        logger.info(
            f"\n{'='*60}\n"
            f"[sample {sample_idx}] batch {batch_idx+1}  "
            f"({samples_processed+1}/{num_samples if num_samples > 0 else '?'})\n"
            f"{'='*60}"
        )

        # ── unpack batch (bs=1) ───────────────────────────────────────────────
        # GroupedSquadDataset may return multiple QA pairs per context group.
        # We always work with exactly the first example to keep batch dim=1
        # throughout (generate_lora_dict, finetune, and evaluate all expect [1, *]).
        evidence_text         = batch["evidence"][0]          # raw string
        evidence_ids          = batch["evidence_ids"][0:1].to(device)
        evidence_mask         = batch["evidence_attention_mask"][0:1].to(device)
        input_ids             = batch["input_ids"][0:1].to(device)
        input_mask            = batch["input_attention_mask"][0:1].to(device)
        ground_truth          = batch["full_answers"][0]
        question              = batch["questions"][0]

        logger.info(
            f"[sample {sample_idx}] evidence_len={evidence_ids.shape[1]} tokens, "
            f"question='{question[:100]}', ground_truth='{str(ground_truth)[:80]}'"
        )

        # ── build fine-tuning supervision sequence ────────────────────────────
        logger.info(f"[sample {sample_idx}] Building inputs (recon_mode={recon_mode})...")
        if recon_mode == "clm":
            recon_input_ids, recon_labels, recon_mask = build_clm_inputs(
                tokenizer=tokenizer,
                evidence_text=evidence_text,
                max_length=int(cfg.lora_finetune.recon_context_max_length),
                device=device,
            )
        else:
            recon_input_ids, recon_labels, recon_mask = build_recon_inputs(
                tokenizer=tokenizer,
                evidence_text=evidence_text,
                recon_context_max_length=int(cfg.lora_finetune.recon_context_max_length),
                recon_conversation_max_length=int(cfg.lora_finetune.recon_conversation_max_length),
                device=device,
            )
        supervised_count = (recon_labels[0] != -100).sum().item()
        logger.info(
            f"[sample {sample_idx}] seq_len={recon_input_ids.shape[1]}, "
            f"supervised_tokens={supervised_count}"
        )
        if supervised_count == 0:
            logger.warning(
                f"[sample {sample_idx}] WARNING: 0 supervised tokens in RECON labels! "
                f"Loss will be NaN. Skipping sample."
            )
            samples_processed += 1
            continue

        # ── generate SHINE LoRA (no grad) ─────────────────────────────────────
        logger.info(f"[sample {sample_idx}] Generating SHINE LoRA via hypernetwork...")
        with torch.no_grad():
            shine_lora = metanetwork.generate_lora_dict(
                evidence_ids=evidence_ids,
                evidence_attention_mask=evidence_mask,
                metalora=metalora,
            )
        logger.info(f"[sample {sample_idx}] SHINE LoRA generated.")

        record: Dict[str, Any] = {
            "sample_idx":    sample_idx,
            "question":      question,
            "ground_truth":  ground_truth,
            "evidence":      evidence_text[:200] + "..." if len(evidence_text) > 200 else evidence_text,
            "shine":         {},
            "scratch":       {},
        }

        # ── condition 1: fine-tune from SHINE warm start ──────────────────────
        logger.info(f"[sample {sample_idx}] === Condition: SHINE warm start ===")
        shine_params = clone_loradict_to_params(shine_lora)

        def on_shine_snapshot(step: int, lora: dict) -> None:
            _, answer, f1 = evaluate_lora(
                model=metamodel,
                tokenizer=tokenizer,
                input_ids=input_ids,
                input_mask=input_mask,
                loradict=lora,
                max_new_tokens=max_new_tokens,
                ground_truth=ground_truth,
                f1_metric=f1_metric,
                sample_id=sample_idx,
                step=step,
                condition="shine",
                question=question,
            )
            record["shine"][str(step)] = {"answer": answer, "f1": f1}

        finetune_lora_on_evidence(
            model=metamodel,
            lora_params=shine_params,
            recon_input_ids=recon_input_ids,
            recon_labels=recon_labels,
            recon_mask=recon_mask,
            lr=lr,
            eval_at=eval_at,
            sample_id=sample_idx,
            condition_name="shine",
            on_snapshot=on_shine_snapshot,
        )

        # ── condition 2: fine-tune from scratch (random A, zero B) ───────────
        logger.info(f"[sample {sample_idx}] === Condition: scratch-init LoRA ===")
        scratch_params = zero_init_loradict(shine_lora)

        def on_scratch_snapshot(step: int, lora: dict) -> None:
            _, answer, f1 = evaluate_lora(
                model=metamodel,
                tokenizer=tokenizer,
                input_ids=input_ids,
                input_mask=input_mask,
                loradict=lora,
                max_new_tokens=max_new_tokens,
                ground_truth=ground_truth,
                f1_metric=f1_metric,
                sample_id=sample_idx,
                step=step,
                condition="scratch",
                question=question,
            )
            record["scratch"][str(step)] = {"answer": answer, "f1": f1}

        finetune_lora_on_evidence(
            model=metamodel,
            lora_params=scratch_params,
            recon_input_ids=recon_input_ids,
            recon_labels=recon_labels,
            recon_mask=recon_mask,
            lr=lr,
            eval_at=eval_at,
            sample_id=sample_idx,
            condition_name="scratch",
            on_snapshot=on_scratch_snapshot,
        )

        # ── stream to JSONL ───────────────────────────────────────────────────
        jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        jsonl_f.flush()
        all_records.append(record)
        samples_processed += 1

        # ── quick per-sample summary to console ──────────────────────────────
        shine_f1_by_step   = {s: record["shine"].get(str(s), {}).get("f1", float("nan")) for s in eval_at}
        scratch_f1_by_step = {s: record["scratch"].get(str(s), {}).get("f1", float("nan")) for s in eval_at}
        logger.info(
            f"[sample {sample_idx}] SUMMARY:\n"
            f"  SHINE warm-start F1 by step: {shine_f1_by_step}\n"
            f"  Scratch-init     F1 by step: {scratch_f1_by_step}"
        )

        elapsed = time.time() - t_start
        logger.info(
            f"[sample {sample_idx}] Elapsed: {elapsed:.1f}s  "
            f"({elapsed / samples_processed:.1f}s/sample)"
        )

    jsonl_f.close()
    logger.info(f"\nAll {samples_processed} samples processed. Building aggregate summary...")

    # ── aggregate and save summary ────────────────────────────────────────────
    summary = aggregate_results(all_records, eval_at)
    summary["split_name"]   = split_name
    summary["checkpoint"]   = resume_dir
    summary["lr"]           = lr
    summary["recon_mode"]   = recon_mode
    summary["num_samples_requested"] = num_samples

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"Summary saved to {summary_path}")

    # ── print final table ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    header = f"{'Step':>6}  {'SHINE F1':>10}  {'Scratch F1':>12}"
    logger.info(header)
    logger.info("-" * len(header))
    for step in eval_at:
        s_info = summary["shine_finetune"].get(str(step), {})
        z_info = summary["scratch_finetune"].get(str(step), {})
        s_f1 = s_info.get("avg_f1", float("nan"))
        z_f1 = z_info.get("avg_f1", float("nan"))
        logger.info(f"{step:>6}  {s_f1:>10.4f}  {z_f1:>12.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
