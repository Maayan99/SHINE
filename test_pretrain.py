#!/usr/bin/env python
# -*- coding: utf-8 -*-

from csv import writer
import os
import math
import time
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
from functools import partial
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import (
    AutoTokenizer,
    Qwen3ForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from datasets import Dataset as HFDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import importlib
from omegaconf import DictConfig, OmegaConf
import hydra
from datasets import load_dataset
import logging
from torch.utils.tensorboard import SummaryWriter
from metanetwork_family import Metanetwork

from utils.mydataset import SquadDataset, SquadCollator, GroupedSquadDataset, TextDataset, TestPretrainCollator
from utils.myseed import set_seed
from utils.mylogging import get_logger
from utils.mysaveload import (
    save_checkpoint,
    load_checkpoint,
    save_training_state,
    load_training_state,
    get_latest_checkpoint,
)
from utils.myfreeze import freeze
from utils.myoptmize import init_optimize
from utils.myddp import (
    should_use_ddp,
    ddp_is_active,
    get_world_size,
    get_rank,
    get_local_rank,
    is_main_process,
    ddp_init_if_needed,
    ddp_cleanup_if_needed,
    distributed_mean,
    barrier,
)
from utils.myinit import _resolve_device, _import_class
import re
from collections import OrderedDict, Counter
from utils.mydebug import debug_print_ids

# ===================== (matplotlib for visualization) =====================
import matplotlib
matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt
# ==========================================================================

logger = get_logger("test")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True


def exact_prefix_match_ratio(ref: List[int], hyp: List[int]) -> float:
    """
    If both have m tokens and first mismatch is at position n (0-based),
    exact match ratio = n / m. If identical, = 1.0
    """
    if len(ref) < len(hyp):
        raise ValueError(f"ref length must be >= hyp length, got {len(ref)} < {len(hyp)}")
    m = len(ref)
    if m == 0:
        return 1.0
    n = 0
    for x, y in zip(ref, hyp):
        if x != y:
            break
        n += 1
    return n / m


def extract_think_and_answer(text: str) -> Tuple[str, str]:
    """
    Splits model output into (think_part, answer_part).
    If no valid <think>...</think> block exists, think = "".
    """
    lower = text.lower()
    start_tag = "<think>"
    end_tag = "</think>"

    think = ""
    answer = text.strip()

    # ---- Case 1: Proper <think>...</think> block exists ----
    start = lower.find(start_tag)
    end = lower.find(end_tag)
    if start != -1 and end != -1 and end > start:
        think = text[start + len(start_tag): end].strip()
        answer = text[end + len(end_tag):].strip()
    else:
        # ---- Case 2: No valid think block → think = "" ----
        # Remove any malformed or inline think tags from final answer
        answer = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
        think = ""

    # ---- Clean common prefixes like "Answer:" or "Final answer:" ----
    answer = re.sub(r"^(final answer|answer)\s*:\s*", "", answer, flags=re.IGNORECASE).strip()

    # ---- Take only the first non-empty line as final answer ----
    if "\n" in answer:
        for line in answer.splitlines():
            if line.strip():
                answer = line.strip()
                break

    return think, answer


@torch.no_grad()
def test_and_save(
    cfg,
    metanetwork_ddp_or_module,
    tokenizer,
    testloader,
    split_name: str,
    use_metanet: bool = True,
    metalora: Any = None,
    device: torch.device = "cuda",
    output_suffix: str = ".json",
):
    """
    Run inference on `testloader`, stream results to disk (per-rank JSONL),
    support resuming from partial output, and finally gather & save a merged
    JSON file on rank 0.

    Metrics retained:
      - mean_loss/std_loss  (per-sample loss statistics)
      - PPL mean/std        (per-sample exp(loss) statistics)
      - EM mean/std         (exact_prefix_match_ratio averaged over samples)
    """

    if use_metanet:
        assert metalora is not None, "metalora cannot be None when use_metanet is True"

    rank = get_rank()
    world_size = get_world_size()

    # Handle both wrapped and unwrapped metanetwork
    metanet = (
        metanetwork_ddp_or_module.module
        if isinstance(metanetwork_ddp_or_module, DDP)
        else metanetwork_ddp_or_module
    )
    metanet.eval()

    # ---------- Paths ----------
    out_dir = os.path.join(cfg.test.save_path, cfg.test.source)
    final_out_path = os.path.join(out_dir, f"{split_name}{output_suffix}")
    rank_tmp_path = os.path.join(out_dir, f"{split_name}.rank{rank}.jsonl")

    # Make sure directory exists on all ranks
    if is_main_process():
        os.makedirs(out_dir, exist_ok=True)
    if ddp_is_active():
        dist.barrier()

    # ===================== NEW: split-level resume =====================
    # If final merged json already exists, skip generating this split entirely.
    if os.path.exists(final_out_path):
        if is_main_process():
            logger.info(f"[SKIP] Found existing {final_out_path}, skipping split '{split_name}'.")
        if ddp_is_active():
            dist.barrier()
        return
    # ================================================================

    # ---------- Figure out where to resume (per-rank JSONL) ----------
    start_sample_idx = 0
    if os.path.exists(rank_tmp_path):
        with open(rank_tmp_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "sample_idx" in rec:
                    start_sample_idx = max(start_sample_idx, rec["sample_idx"] + 1)
        if is_main_process():
            logger.info(
                f"[Rank {rank}] Resuming from sample_idx={start_sample_idx} for split '{split_name}'"
            )

    # Open rank tmp file for appending
    tmp_f = open(rank_tmp_path, "a", encoding="utf-8")

    sample_idx = 0  # global (per-rank) index of samples seen by this rank

    for batch_idx, batch in enumerate(testloader):
        batch_size = len(batch["questions"])

        # If this entire batch is already processed, skip without running the model
        if sample_idx + batch_size <= start_sample_idx:
            sample_idx += batch_size
            continue

        print(f"[Rank {rank}] Processing batch {batch_idx + 1}/{len(testloader)}...")

        evidences = batch["evidence"]
        evidence_ids = batch["evidence_ids"].to(device, non_blocking=True)
        evidence_attention_mask = batch["evidence_attention_mask"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        input_attention_mask = batch["input_attention_mask"].to(device, non_blocking=True)
        ground_truths = batch["answers"]
        ground_truths_ids = batch["answer_ids"]
        questions = batch["questions"]
        labels = batch["labels"].to(device, non_blocking=True)
        full_input_ids = batch["full_input_ids"].to(device, non_blocking=True)
        full_input_attention_mask = batch["full_input_attention_mask"].to(device, non_blocking=True)

        outputs = metanet(
            input_ids=full_input_ids,
            input_attention_mask=full_input_attention_mask,
            evidence_ids=evidence_ids,
            evidence_attention_mask=evidence_attention_mask,
            labels=labels,
            use_metanet=use_metanet,
            metalora=metalora,
        )
        logits = outputs.logits  # (B, T, V)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none"
        )
        loss_per_token = loss_per_token.view(shift_labels.size())
        mask = (shift_labels != -100)
        loss_per_token = loss_per_token * mask
        token_nums = mask[:, :-2].sum(dim=1)
        loss_per_sample = loss_per_token[:, :-2].sum(dim=1) / token_nums

        loradict = None
        if use_metanet:
            loradict = metanet.generate_lora_dict(
                evidence_ids=evidence_ids,
                evidence_attention_mask=evidence_attention_mask,
                metalora=metalora,
            )
        gen_out = metanet.metamodel.generate(
            input_ids=input_ids,
            attention_mask=input_attention_mask,
            loradict=loradict,
            ignore_mem_token=True,
            max_new_tokens=cfg.test.max_new_tokens,
            do_sample=False,
        )

        gen_out = gen_out[:, 9:].to("cpu")

        input_lens = input_attention_mask.sum(dim=1).tolist()
        input_ids_cpu = input_ids.to("cpu")

        for i in range(gen_out.size(0)):
            # If this particular sample was already written in previous run, skip it
            if sample_idx < start_sample_idx:
                sample_idx += 1
                continue

            full_text = tokenizer.decode(gen_out[i], skip_special_tokens=True)
            input_text = tokenizer.decode(
                input_ids_cpu[i][-input_lens[i]:], skip_special_tokens=True
            )
            think, answer = extract_think_and_answer(full_text)

            t = int(token_nums[i].item())  # python int

            ref = ground_truths_ids[i][-t:]
            hyp = gen_out[i][:t]

            # convert to list[int]
            if torch.is_tensor(ref):
                ref = ref.tolist()
            else:
                ref = list(map(int, ref))

            if torch.is_tensor(hyp):
                hyp = hyp.tolist()
            else:
                hyp = list(map(int, hyp))

            em = exact_prefix_match_ratio(ref, hyp)

            statistics = {
                "length": len(ref),
                "em": em,
                "exact_prefix_match": em,  # keep old key for compatibility if needed
            }

            record = {
                "sample_idx": sample_idx,  # used for resuming and sorting
                "statistics": statistics,
                "loss": loss_per_sample[i].item(),
                "evidence": evidences[i],
                "input": input_text,
                "question": questions[i],
                "think": think,
                "answer": answer,
                "answer_ids": gen_out[i].tolist(),
                "ground_truth": ground_truths[i],
                "ground_truth_ids": ground_truths_ids[i].tolist(),
            }

            tmp_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            tmp_f.flush()

            sample_idx += 1

    tmp_f.close()
    metanet.train()

    # ---------- Final gather & merged save ----------
    local_results = []
    if os.path.exists(rank_tmp_path):
        with open(rank_tmp_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                local_results.append(rec)

    if ddp_is_active():
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_results)

        if is_main_process():
            merged = []
            for part in gathered:
                if part:
                    merged.extend(part)
    else:
        merged = local_results

    if is_main_process():
        # Sort and strip resume-only fields
        merged.sort(key=lambda x: x.get("sample_idx", 0))
        for rec in merged:
            rec.pop("sample_idx", None)

        # ----- Compute summary stats (mean + std) -----
        def _finite_float(x):
            return isinstance(x, (int, float)) and (not math.isnan(x)) and (not math.isinf(x))

        losses = [r.get("loss", None) for r in merged]
        losses = [float(x) for x in losses if _finite_float(x)]

        loss_mean = float(np.mean(losses)) if losses else float("nan")
        loss_std = float(np.std(losses, ddof=0)) if losses else float("nan")

        # Per-sample PPL stats: ppl_i = exp(loss_i)
        ppls = [math.exp(x) for x in losses] if losses else []
        ppl_mean = float(np.mean(ppls)) if ppls else float("nan")
        ppl_std = float(np.std(ppls, ddof=0)) if ppls else float("nan")

        exacts = []
        lens_ = []

        for r in merged:
            st = r.get("statistics", {}) or {}
            v_em = st.get("em", st.get("exact_prefix_match", None))
            if _finite_float(v_em):
                exacts.append(float(v_em))
            v_len = st.get("length", None)
            if _finite_float(v_len):
                lens_.append(float(v_len))

        em_mean = float(np.mean(exacts)) if exacts else float("nan")
        em_std = float(np.std(exacts, ddof=0)) if exacts else float("nan")

        length_mean = float(np.mean(lens_)) if lens_ else float("nan")
        length_std = float(np.std(lens_, ddof=0)) if lens_ else float("nan")

        summary = {
            "num_samples": len(merged),
            "mean_loss": loss_mean,  # keep existing key for compatibility
            "std_loss": loss_std,    # NEW
            "mean_statistics": {
                "length": length_mean,
                "em": em_mean,
                "exact_prefix_match": em_mean,  # compatibility
                "ppl": ppl_mean,                # mean(exp(loss_i))
            },
            "std_statistics": {                 # NEW
                "length": length_std,
                "em": em_std,
                "exact_prefix_match": em_std,
                "ppl": ppl_std,
            },
        }

        final_payload = {
            "summary": summary,
            "predictions": merged,
        }

        with open(final_out_path, "w", encoding="utf-8") as f:
            json.dump(final_payload, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(merged)} predictions (+summary) to {final_out_path}")


# ===================== visualize 4 separate figures (PPL + Loss) =====================
def visualize_recon_comp_curves_separate(
    cfg,
    lens: List[int],
    out_dir: str,
    recon_ppl_name: str = "recon_ppl.png",
    comp_ppl_name: str = "comp_ppl.png",
    recon_loss_name: str = "recon_loss.png",
    comp_loss_name: str = "comp_loss.png",
):
    """
    Create 4 separate images:
      1) Recon PPL (mean ± std)
      2) Comp  PPL (mean ± std)
      3) Recon Loss (mean ± std)
      4) Comp  Loss (mean ± std)

    x-axis: 100..1000 (len*100)

    y-axis:
      - ppl plots: mean(exp(loss_i)) and std(exp(loss_i))
      - loss plots: mean(loss_i) and std(loss_i)

    Requirements:
      - recon/comp PPL images share same y-limits (include std bands)
      - recon/comp Loss images share same y-limits (include std bands)
    """

    xs = [l * 100 for l in lens]

    def read_mean_std(path: str, kind: str) -> Tuple[float, float]:
        """
        kind: 'loss' or 'ppl'
        """
        if not os.path.exists(path):
            return float("nan"), float("nan")
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        s = obj.get("summary", {}) or {}

        if kind == "loss":
            return float(s.get("mean_loss", float("nan"))), float(s.get("std_loss", float("nan")))
        if kind == "ppl":
            ms = s.get("mean_statistics", {}) or {}
            ss = s.get("std_statistics", {}) or {}
            return float(ms.get("ppl", float("nan"))), float(ss.get("ppl", float("nan")))
        return float("nan"), float("nan")

    recon_ppl_mean, recon_ppl_std = [], []
    comp_ppl_mean, comp_ppl_std = [], []
    recon_loss_mean, recon_loss_std = [], []
    comp_loss_mean, comp_loss_std = [], []

    for l in lens:
        recon_path = os.path.join(out_dir, f"{l}_recon.json")
        comp_path = os.path.join(out_dir, f"{l}_comp.json")

        m, s = read_mean_std(recon_path, "ppl")
        recon_ppl_mean.append(m); recon_ppl_std.append(s)

        m, s = read_mean_std(comp_path, "ppl")
        comp_ppl_mean.append(m); comp_ppl_std.append(s)

        m, s = read_mean_std(recon_path, "loss")
        recon_loss_mean.append(m); recon_loss_std.append(s)

        m, s = read_mean_std(comp_path, "loss")
        comp_loss_mean.append(m); comp_loss_std.append(s)

    def _finite(x):
        return isinstance(x, (int, float)) and (not math.isnan(x)) and (not math.isinf(x))

    def finite_band(means, stds):
        vals = []
        for m, s in zip(means, stds):
            if _finite(m) and _finite(s):
                vals.append(m - s)
                vals.append(m + s)
        return vals

    ppl_all = finite_band(recon_ppl_mean, recon_ppl_std) + finite_band(comp_ppl_mean, comp_ppl_std)
    loss_all = finite_band(recon_loss_mean, recon_loss_std) + finite_band(comp_loss_mean, comp_loss_std)

    # def padded_ylim(all_vals):
    #     if not all_vals:
    #         return None
    #     ymin, ymax = min(all_vals), max(all_vals)
    #     pad = (ymax - ymin) * 0.05 if ymax > ymin else (abs(ymax) * 0.05 + 1e-6)
    #     return (ymin - pad, ymax + pad)

    # ppl_ylim = padded_ylim(ppl_all)
    # loss_ylim = padded_ylim(loss_all)

    def plot_mean_std(xs, mean, std, title, xlabel, ylabel, xticks, ylim, save_path):
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(xs, mean, marker="o")

        lower = []
        upper = []
        for m, s in zip(mean, std):
            if _finite(m) and _finite(s):
                lower.append(m - s)
                upper.append(m + s)
            else:
                lower.append(float("nan"))
                upper.append(float("nan"))

        ax.fill_between(xs, lower, upper, alpha=0.2)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(xticks)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=200)
        plt.close(fig)

    # ---- PPL figures ----
    ppl_ylim = (1.0, 2.0)
    plot_mean_std(
        xs, recon_ppl_mean, recon_ppl_std,
        title="Recon PPL (mean ± std)",
        xlabel="Context length",
        ylabel="PPL",
        xticks=xs,
        ylim=ppl_ylim,
        save_path=os.path.join(out_dir, recon_ppl_name),
    )
    plot_mean_std(
        xs, comp_ppl_mean, comp_ppl_std,
        title="Comp PPL (mean ± std)",
        xlabel="Context length",
        ylabel="PPL",
        xticks=xs,
        ylim=ppl_ylim,
        save_path=os.path.join(out_dir, comp_ppl_name),
    )

    # ---- Loss figures ----
    loss_ylim = (0.0, 1.0)
    plot_mean_std(
        xs, recon_loss_mean, recon_loss_std,
        title="Recon Loss (mean ± std)",
        xlabel="Context length",
        ylabel="Loss",
        xticks=xs,
        ylim=loss_ylim,
        save_path=os.path.join(out_dir, recon_loss_name),
    )
    plot_mean_std(
        xs, comp_loss_mean, comp_loss_std,
        title="Comp Loss (mean ± std)",
        xlabel="Context length",
        ylabel="Loss",
        xticks=xs,
        ylim=loss_ylim,
        save_path=os.path.join(out_dir, comp_loss_name),
    )

    if is_main_process():
        logger.info(
            "Saved visualizations: "
            f"{os.path.join(out_dir, recon_ppl_name)}, "
            f"{os.path.join(out_dir, comp_ppl_name)}, "
            f"{os.path.join(out_dir, recon_loss_name)}, "
            f"{os.path.join(out_dir, comp_loss_name)}"
        )
# ==============================================================================


@hydra.main(version_base=None, config_path="configs")
def main(cfg: DictConfig):
    # ========= DDP init (safe for single-process) =========
    ddp_init_if_needed()

    if is_main_process():
        logger.info("Resolved config:")
        logger.info(f"\n\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    # Seed & device
    set_seed(int(cfg.run.seed) + get_rank())
    device = _resolve_device(cfg.run.device)
    torch.backends.cudnn.benchmark = True

    # Load model/tokenizer (supports your local LoRA-wrapped Qwen class)
    if is_main_process():
        logger.info("Loading model & tokenizer...")
    MetaModelCls = _import_class(cfg.model.metamodel_class_path)
    ConfigCls = _import_class(cfg.model.config_class_path)
    config = ConfigCls.from_pretrained(cfg.model.model_from)
    config.num_mem_token = -1
    cfg.hidden_size = config.hidden_size
    cfg.num_layers = config.num_hidden_layers

    if cfg.metanetwork.type == "transformer":
        tmp_model = MetaModelCls.from_pretrained(cfg.model.model_from, config=config)
        assert tmp_model.lora_params_numel(cfg.model.lora_r) % (
            cfg.hidden_size * cfg.num_layers
        ) == 0, (
            "For transformer metanetwork, num_mem_token must be set to "
            "model.lora_params_numel(lora_r) / (hidden_size * num_layers)"
        )
        config.num_mem_token = (
            tmp_model.lora_params_numel(cfg.model.lora_r)
            // (cfg.hidden_size * cfg.num_layers)
        )
        cfg.num_mem_token = config.num_mem_token
        del tmp_model
        if is_main_process():
            logger.info(
                f"Using transformer metanetwork, set num_mem_token to {config.num_mem_token}"
            )
    else:
        config.num_mem_token = cfg.num_mem_token

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.tokenizer_from, padding_side="left", use_fast=True
    )
    tokenizer.add_tokens(['<RECON>', '<COMP>'])
    tokenizer.chat_template = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if (loop.last or (not loop.last and reasoning_content)) and (enable_thinking is not defined or enable_thinking != false) %}\n                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n            {%- else %}\n                {{- '<|im_start|>' + message.role + '\\n' + content }}\n            {%- endif %}\n        {%- else %}\n            {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is not defined or enable_thinking != false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"
    metamodel = MetaModelCls.from_pretrained(cfg.model.model_from, config=config)
    metamodel.reset_mem_tokens()
    metamodel.resize_token_embeddings(len(tokenizer))
    metanetwork = Metanetwork(metamodel, cfg, metamodel.lora_params_numel(cfg.model.lora_r))
    metanetwork.train()
    metanetwork.to(device)
    freeze(metamodel)

    # Training loop scaffolding
    hydra_run_dir = os.getcwd()
    ckpt_root = os.path.join("checkpoints", f"{cfg.name}", "pretrain")

    if cfg.test_global_step == "latest":
        resume_dir = get_latest_checkpoint(ckpt_root)
    elif cfg.test_global_step == "final":
        resume_dir = os.path.join(ckpt_root, "final")
        if not os.path.isdir(resume_dir):
            raise ValueError(f"Requested resume dir {resume_dir} does not exist.")
    elif isinstance(cfg.test_global_step, int) and cfg.test_global_step > 0:
        resume_dir = os.path.join(ckpt_root, f"checkpoint-{cfg.test_global_step}")
        if not os.path.isdir(resume_dir):
            raise ValueError(f"Requested resume dir {resume_dir} does not exist.")
    elif isinstance(cfg.test_global_step, str) and cfg.test_global_step.startswith("epoch-"):
        resume_dir = os.path.join(ckpt_root, f"checkpoint-{cfg.test_global_step}")
        if not os.path.isdir(resume_dir):
            raise ValueError(f"Requested resume dir {resume_dir} does not exist.")
    elif isinstance(cfg.test_global_step, str) and cfg.test_global_step.startswith("checkpoint-epoch-"):
        resume_dir = os.path.join(ckpt_root, cfg.test_global_step)
        if not os.path.isdir(resume_dir):
            raise ValueError(f"Requested resume dir {resume_dir} does not exist.")
    else:
        raise ValueError(f"Invalid test_global_step: {cfg.test_global_step}")

    # Load model
    if is_main_process():
        logger.info(f"Resume mode, loading from {resume_dir}...")
    metanetwork, metalora, _ = load_checkpoint(metanetwork, resume_dir, device)

    # Data
    if is_main_process():
        logger.info("Preparing data...")
    if cfg.test.source == "wikitext":
        def build_wikitext2_raw_articles(ds):
            title_re = re.compile(r"^\s*(=+)\s*([^=].*?)\s*\1\s*$")

            articles = []
            cur_title = None
            cur_lines = []

            def flush():
                nonlocal cur_title, cur_lines
                if cur_title is None:
                    cur_lines = []
                    return
                text = "\n".join(cur_lines).strip()
                if text:
                    articles.append({"title": cur_title, "text": text})
                cur_lines = []

            for line in ds["text"]:
                m = title_re.match(line)
                if m:
                    flush()
                    cur_title = m.group(2)
                    continue
                cur_lines.append(line)

            flush()
            return articles

        lens = [i for i in range(1, 12)]
        datasets = []
        data_dir = os.path.join("data", "wikitext", "wikitext-2-raw-v1")
        ds = load_dataset(data_dir, split='train')
        data = build_wikitext2_raw_articles(ds)
        idx_dict = json.load(open(os.path.join(data_dir, "idx_dict.json")))
        for l in lens:
            datasets.append(TextDataset([data[i]['text'].strip() for i in idx_dict[str(l)]]))
            if is_main_process():
                print(f"{l}: datasets num: {len(datasets[l-1])}")
        collators = []
        for l in lens:
            collators.append((
                TestPretrainCollator(tokenizer, cfg, context_max_length=l*100 + 20, conversation_max_length=l*100 + 31, mode="recon"),
                TestPretrainCollator(tokenizer, cfg, context_max_length=l*100 + 20, conversation_max_length=l*100 + 31, mode="comp")
            ))
    else:
        raise ValueError(f"Unknown data source: {cfg.test.source}")

    pin = device.type == "cuda"
    out_dir = os.path.join(cfg.test.save_path, cfg.test.source)

    for i, (ds, collator) in enumerate(zip(datasets, collators), start=1):
        # ===================== NEW: split-level skip in main =====================
        recon_final = os.path.join(out_dir, f"{i}_recon.json")
        comp_final = os.path.join(out_dir, f"{i}_comp.json")
        need_recon = not os.path.exists(recon_final)
        need_comp = not os.path.exists(comp_final)

        if not (need_recon or need_comp):
            if is_main_process():
                logger.info(f"[SKIP] Both exist: {recon_final} and {comp_final}. Skipping i={i}.")
            continue
        # =======================================================================

        test_sampler = (
            DistributedSampler(ds, num_replicas=get_world_size(), rank=get_rank(), shuffle=False)
            if get_world_size() > 1
            else None
        )
        num_workers_default = 2 if device.type == "cuda" else 0

        test_loader_0 = DataLoader(
            ds,
            batch_size=cfg.test.batch_size,
            shuffle=False,
            sampler=test_sampler,
            collate_fn=collator[0],
            pin_memory=pin,
            num_workers=getattr(cfg.test, "num_workers", num_workers_default),
            persistent_workers=pin and getattr(cfg.test, "num_workers", num_workers_default) > 0,
        )
        test_loader_1 = DataLoader(
            ds,
            batch_size=cfg.test.batch_size,
            shuffle=False,
            sampler=test_sampler,
            collate_fn=collator[1],
            pin_memory=pin,
            num_workers=getattr(cfg.test, "num_workers", num_workers_default),
            persistent_workers=pin and getattr(cfg.test, "num_workers", num_workers_default) > 0,
        )

        if ddp_is_active():
            dist.barrier()

        if need_recon:
            test_and_save(
                cfg=cfg,
                metanetwork_ddp_or_module=metanetwork,
                tokenizer=tokenizer,
                testloader=test_loader_0,
                split_name=f"{i}_recon",
                use_metanet=True,
                metalora=metalora,
                device=device,
                output_suffix=".json",
            )
        else:
            if is_main_process():
                logger.info(f"[SKIP] Found existing {recon_final}, not regenerating.")

        if need_comp:
            test_and_save(
                cfg=cfg,
                metanetwork_ddp_or_module=metanetwork,
                tokenizer=tokenizer,
                testloader=test_loader_1,
                split_name=f"{i}_comp",
                use_metanet=True,
                metalora=metalora,
                device=device,
                output_suffix=".json",
            )
        else:
            if is_main_process():
                logger.info(f"[SKIP] Found existing {comp_final}, not regenerating.")

    # ===================== visualize after all json exist (PPL + Loss) =====================
    if ddp_is_active():
        dist.barrier()  # ensure rank0 can see all outputs

    if is_main_process():
        out_dir = os.path.join(cfg.test.save_path, cfg.test.source)
        lens = [i for i in range(1, 12)]  # 1..10 => x=100..1000

        visualize_recon_comp_curves_separate(
            cfg=cfg,
            lens=lens,
            out_dir=out_dir,
            recon_ppl_name="recon_ppl.png",
            comp_ppl_name="comp_ppl.png",
            recon_loss_name="recon_loss.png",
            comp_loss_name="comp_loss.png",
        )
    # ==============================================================================

    ddp_cleanup_if_needed()


if __name__ == "__main__":
    main()
