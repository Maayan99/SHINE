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
from collections import OrderedDict
import time
import re

logger = get_logger("test")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True


def extract_think_and_answer(text: str) -> Tuple[str, str]:
    """
    Splits model output into (think_part, answer_part).
    If no valid <think>...</think> block exists, think = "".
    """

    # Normalize for searching
    lower = text.lower()
    start_tag = "<think>"
    end_tag = "</think>"

    think = ""
    answer = text.strip()

    # ---- Case 1: Proper <think>...</think> block exists ----
    start = lower.find(start_tag)
    end = lower.find(end_tag)
    if start != -1 and end != -1 and end > start:
        think = text[start + len(start_tag) : end].strip()
        answer = text[end + len(end_tag) :].strip()

    else:
        # ---- Case 2: No valid think block → think = "" ----
        # Remove any malformed or inline think tags from final answer
        answer = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
        think = ""  # force empty

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
    use_amp: bool = False,
    device: torch.device = "cuda",
    amp_dtype=None,
    output_suffix: str = ".json",
):
    """
    Run inference on `testloader`, stream results to disk (per-rank JSONL),
    support resuming from partial output, and finally gather & save a merged
    JSON file on rank 0.

    Resumability:
      - Per rank we keep an intermediate file:
          {cfg.test.save_path}/{cfg.test.source}/{split_name}.rank{rank}.jsonl
      - Every written record has a monotonically increasing `sample_idx`.
      - On resume, we read this file, find the max existing `sample_idx`,
        and skip earlier samples in the DataLoader.
      - Final merged file {split_name}{output_suffix} does NOT contain
        `sample_idx`; it’s only used for resuming and ordering.
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

    # ---------- Figure out where to resume ----------
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
        evidence_attention_mask = batch["evidence_attention_mask"].to(
            device, non_blocking=True
        )
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        input_attention_mask = batch["input_attention_mask"].to(
            device, non_blocking=True
        )
        ground_truths = batch["full_answers"]
        questions = batch["questions"]
        labels = (
            None
            if batch["labels"] is None
            else batch["labels"].to(device, non_blocking=True)
        )

        loradict = None
        if use_metanet:
            loradict = metanet.generate_lora_dict(
                evidence_ids=evidence_ids,
                evidence_attention_mask=evidence_attention_mask,
                metalora=metalora,
            )

        if use_amp:
            if amp_dtype is None:
                amp_dtype = (
                    torch.bfloat16
                    if torch.cuda.is_available()
                    and torch.cuda.is_bf16_supported()
                    else torch.float16
                )
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                gen_out = metanet.metamodel.generate(
                    input_ids=input_ids,
                    attention_mask=input_attention_mask,
                    loradict=loradict,
                    ignore_mem_token=True,
                    max_new_tokens=cfg.test.max_new_tokens,
                    do_sample=False,
                )
        else:
            gen_out = metanet.metamodel.generate(
                input_ids=input_ids,
                attention_mask=input_attention_mask,
                loradict=loradict,
                ignore_mem_token=True,
                max_new_tokens=cfg.test.max_new_tokens,
                do_sample=False,
            )

        input_lens = input_attention_mask.sum(dim=1).tolist()

        gen_out = gen_out.to("cpu")
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

            if full_text.startswith(input_text):
                answer_text = full_text[len(input_text) :]
            else:
                answer_text = full_text

            think, answer = extract_think_and_answer(answer_text)

            record = {
                "sample_idx": sample_idx,  # used for resuming and sorting
                "evidence": evidences[i],
                "input": input_text,
                "question": questions[i],
                "think": think,
                "answer": answer,
                "ground_truth": ground_truths[i],
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
        merged.sort(key=lambda x: x.get("sample_idx", 0))
        for rec in merged:
            rec.pop("sample_idx", None)

        with open(final_out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(merged)} predictions to {final_out_path}")


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
    elif isinstance(cfg.test_global_step, str) and cfg.test_global_step.startswith(
        "checkpoint-epoch-"
    ):
        resume_dir = os.path.join(ckpt_root, cfg.test_global_step)
        if not os.path.isdir(resume_dir):
            raise ValueError(f"Requested resume dir {resume_dir} does not exist.")
    else:
        raise ValueError(f"Invalid test_global_step: {cfg.test_global_step}")

    # Load model
    if is_main_process():
        logger.info(f"Resume mode, loading from {resume_dir}...")
    metanetwork, metalora = load_checkpoint(metanetwork, resume_dir, device)

    # Data
    if is_main_process():
        logger.info("Preparing data...")
    if cfg.test.source == "wikitext":
        lens = [i for i in range(1, 11)]
        datasets = []
        data_dir = os.path.join("data", "wikitext", "wikitext-103-raw-v1")
        ds = load_dataset(data_dir)
        data = list(ds["train"])
        idx_dict = json.load(open(os.path.join(data_dir, "idx_dict.json")))
        for l in lens:
            datasets.append(TextDataset([data[i]['text'] for i in idx_dict[str(l)]]))
            if is_main_process():
                print(f"{l}: datasets num: {len(datasets[l-1])}")
        collator = TestPretrainCollator(tokenizer, cfg, context_max_length=1020, conversation_max_length=1030, mode="recon")
    else:
        raise ValueError(f"Unknown data source: {cfg.test.source}")

    pin = device.type == "cuda"
    for i, ds in enumerate(datasets):
        test_sampler = (
            DistributedSampler(
                ds, num_replicas=get_world_size(), rank=get_rank(), shuffle=False
            )
            if get_world_size() > 1
            else None
        )
        num_workers_default = 2 if device.type == "cuda" else 0

        test_loader = DataLoader(
            ds,
            batch_size=cfg.test.batch_size,
            shuffle=False,
            sampler=test_sampler,
            collate_fn=collator,
            pin_memory=pin,
            num_workers=getattr(cfg.test, "num_workers", num_workers_default),
            persistent_workers=pin
            and getattr(cfg.test, "num_workers", num_workers_default) > 0,
        )

        ckpt_root = os.path.join(hydra_run_dir, "checkpoints")

        if ddp_is_active():
            dist.barrier()

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        test_and_save(
            cfg=cfg,
            metanetwork_ddp_or_module=metanetwork,
            tokenizer=tokenizer,
            testloader=test_loader,
            split_name=f"{i}",  # e.g. "squad"
            use_metanet=True,
            metalora=metalora,
            use_amp=cfg.run.use_amp,
            device=device,
            amp_dtype=amp_dtype,
            output_suffix=".json",
        )

    ddp_cleanup_if_needed()


if __name__ == "__main__":
    main()
