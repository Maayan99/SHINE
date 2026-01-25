#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from csv import writer
import gc
import os
import math
from pyexpat.errors import messages
import time
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from functools import partial
import numpy as np

import random
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

from utils.mydataset import SquadDataset, SquadCollator, GroupedSquadDataset, TextDataset, TestPretrainCollator, MQADataset, MQACollator, SFTCollator, SFTDataset
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
from utils.myloradict import iter_learnable_tensors
from utils.myinit import _resolve_device, _import_class
import re
from collections import OrderedDict, Counter
from utils.mydebug import debug_print_ids
from calculate_f1 import compute_f1

logger = get_logger("test")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True

import re
from typing import Tuple

def extract_think_and_answer(text: str) -> Tuple[str, str]:
    """
    Splits model output into (think_part, answer_part) with the following rules:
    
    - If both <think> and </think> are present AND the first </think> comes after the first <think>,
      treat as valid: extract think content and process the answer.
    - If neither tag appears, treat as normal: think = "", and process the whole text as answer.
    - If only one of the tags appears (or they appear but in wrong order), 
      return think="[error]" and answer = original input text (unchanged).
    """
    lower = text.lower()
    has_start = "<think>" in lower
    has_end = "</think>" in lower

    # Case 1: Only one tag exists → error
    if has_start != has_end:  # XOR: exactly one is True
        if text.startswith("<think>\n") or text.startswith("<think>\n\n"):
            return "", text[len("<think>\n"):].strip()
        return "[error]", text

    # Case 2: Neither tag exists → normal no-think case
    if not has_start and not has_end:
        answer = text.strip()
        # Clean common prefixes
        answer = re.sub(r"^(final answer|answer)\s*:\s*", "", answer, flags=re.IGNORECASE).strip()
        # Take first non-empty line
        if "\n" in answer:
            for line in answer.splitlines():
                if line.strip():
                    answer = line.strip()
                    break
        return "", answer

    # Case 3: Both tags exist → check order
    start = lower.find("<think>")
    end = lower.find("</think>")

    # If the first </think> is before the first <think>, consider malformed
    if end < start:
        return "[error]", text

    # Valid structure: extract think and process answer
    think = text[start + len("<think>"): end].strip()
    answer = text[end + len("</think>"):].strip()

    # Clean answer prefix
    answer = re.sub(r"^(final answer|answer)\s*:\s*", "", answer, flags=re.IGNORECASE).strip()
    # Take first non-empty line
    if "\n" in answer:
        for line in answer.splitlines():
            if line.strip():
                answer = line.strip()
                break

    return think, answer

def sft_lora(model, sftdataloader, r, scale, device, tokenizer, num_sample, num_epoch=10):
    model.eval()
    results = []
    for i, batch in enumerate(sftdataloader):
        result = {}
        t0 = time.perf_counter()
        evidence_ids = batch["evidence_ids"][0].unsqueeze(0).to(device, non_blocking=True)
        evidence_attention_mask = batch["evidence_attention_mask"][0].unsqueeze(0).to(device, non_blocking=True)
        orig_input_ids = batch["orig_input_ids"][0].unsqueeze(0).to(device, non_blocking=True)
        orig_input_attention_mask = batch["orig_input_attention_mask"][0].unsqueeze(0).to(device, non_blocking=True)
        orig_labels = batch["orig_labels"][0].unsqueeze(0).to(device, non_blocking=True)

        loradict = model.init_lora_dict(r, scale, device)

        grouped_params = [
            {
                "params": list(iter_learnable_tensors(loradict)),
                "weight_decay": 0.01,
            }
        ]
        optimizer = torch.optim.AdamW(grouped_params, lr=1e-4)

        total_steps = num_epoch * num_sample
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps,
        )

        # (optional) if you want faster and correct timing on GPU, consider torch.cuda.synchronize()
        # before/after timing; see note below.
            
        for epoch in range(1, num_epoch + 1):
            for j in range(1, num_sample + 1):
                input_ids = batch["input_ids"][j][0].unsqueeze(0).to(device, non_blocking=True)
                input_attention_mask = batch["input_attention_mask"][j][0].unsqueeze(0).to(device, non_blocking=True)
                labels = batch["labels"][j][0].unsqueeze(0).to(device, non_blocking=True)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=input_attention_mask,
                    labels=labels,
                    ignore_mem_token=True,
                    loradict=loradict,
                )

                train_loss = outputs.loss
                optimizer.zero_grad(set_to_none=True)
                train_loss.backward()
                optimizer.step()
                lr_scheduler.step()

        
        
        # If on CUDA, synchronize before stopping timer for accurate measurement
        if device is not None and str(device).startswith("cuda"):
            torch.cuda.synchronize()
        train_elapsed = time.perf_counter() - t0
        result["train loss"] = train_loss.item()
        result["train time"] = train_elapsed

        with torch.no_grad():
            eval_outputs = model(
                input_ids=orig_input_ids,
                attention_mask=orig_input_attention_mask,
                labels=orig_labels,
                ignore_mem_token=True,
                loradict=loradict,
            )
            ev_loss = eval_outputs.loss
            result["eval loss"] = ev_loss.item()
            ppl = math.exp(ev_loss.item()) if ev_loss.item() < 20 else float("inf")
            result["eval ppl"] = ppl

        t1 = time.perf_counter()
        messages = [batch["initial_messages"][0]] if batch["initial_messages"][0] is not {} else []
        conversation_log = [{"initial message": deepcopy(messages)}]
        f1_scores = []
        error_count_local = 0
        orig_questions = batch["orig_questions"][0]
        orig_answers = batch["orig_answers"][0]
        for q_idx, question in enumerate(orig_questions):
            messages.append({"role": "user", "content": question})
            
            input_enc = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                max_length=3000,
                truncation=True,
                return_dict=True,
                padding="max_length",
                enable_thinking=False,
            )
            
            input_ids = input_enc["input_ids"].to(device)
            attention_mask = input_enc["attention_mask"].to(device)
            
            gen_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": 500,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "do_sample": False,
                "ignore_mem_token": True,
                "loradict": loradict,
            }
            
            outputs = model.generate(**gen_kwargs)
            new_tokens = outputs[0, input_ids.shape[1]:]
            think_answer_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            think_text, answer_text = extract_think_and_answer(think_answer_text)
            
            if think_text == "[error]":
                error_count_local += 1
            
            messages.append({"role": "assistant", "content": answer_text})
            f1 = compute_f1(orig_answers[q_idx], answer_text)
            f1_scores.append(f1)
            
            conversation_log.append({
                "turn": q_idx + 1,
                "question": question,
                "think": think_text,
                "answer": answer_text,
                "ground_truth": orig_answers[q_idx],
                "f1": f1,
            })
        
        conversation_log[0]["avg_f1"] = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        conversation_log[0]["error_count"] = error_count_local
        generation_elapsed = time.perf_counter() - t1
        result["generation time"] = generation_elapsed
        result["conversation_log"] = conversation_log
        results.append(result)
    
    total_samples = len(results)
    num_turns_expected = 15  # per your specification
    assert total_samples > 0, "No samples processed in SFT evaluation."

    # Initialize per-turn accumulators
    turn_f1_sums = [0.0] * num_turns_expected
    turn_error_counts = [0] * num_turns_expected
    total_error_count = 0
    all_f1s = []
    train_times = []
    generation_times = []

    for result in results:
        conv_log = result["conversation_log"]
        train_times.append(result["train time"])
        generation_times.append(result["generation time"])
        turns = conv_log[1:]  # skip initial metadata dict
        if len(turns) != num_turns_expected:
            # Optionally: log warning or skip; here we skip incomplete samples
            continue

        for turn_idx in range(num_turns_expected):
            turn = turns[turn_idx]
            f1 = turn["f1"]
            think = turn["think"]

            all_f1s.append(f1)
            turn_f1_sums[turn_idx] += f1

            if think == "[error]":
                turn_error_counts[turn_idx] += 1
                total_error_count += 1

    overall_avg_f1 = sum(all_f1s) / len(all_f1s) if all_f1s else 0.0
    per_turn_avg_f1 = [
        turn_f1_sums[i] / total_samples for i in range(num_turns_expected)
    ]

    stats = {
        "total_samples": total_samples,
        "total_turns": len(all_f1s),
        "error_count": total_error_count,
        "overall_avg_f1": overall_avg_f1,
        "per_turn_avg_f1": per_turn_avg_f1,
        "per_turn_error_count": turn_error_counts,
        "avg_train_time": sum(train_times) / len(train_times) if train_times else 0.0,
        "avg_generation_time": sum(generation_times) / len(generation_times) if generation_times else 0.0,
    }
    
    return results, stats

                
    


@torch.no_grad()
def evaluate(metanetwork_ddp_or_module, dataloader, device, use_metanet: bool = True, metalora: Optional[torch.Tensor] = None) -> Dict[str, float]:
    # Handle both wrapped and unwrapped metanetwork
    metanet = metanetwork_ddp_or_module.module if isinstance(metanetwork_ddp_or_module, DDP) else metanetwork_ddp_or_module
    metanet.eval()

    if use_metanet:
        assert metalora is not None, "metalora cannot be None when use_metanet is True"

    total_loss = 0.0
    n_tokens = 0
    
    for i, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        input_attention_mask = batch["input_attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        evidence_ids = batch["evidence_ids"].to(device, non_blocking=True)
        evidence_attention_mask = batch["evidence_attention_mask"].to(device, non_blocking=True)
        
        outputs = metanet(
            input_ids=input_ids,
            input_attention_mask=input_attention_mask,
            evidence_ids=evidence_ids,
            evidence_attention_mask=evidence_attention_mask,
            labels=labels,
            use_metanet=use_metanet,
            metalora=metalora,
        )
        loss = outputs.loss

        valid_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * valid_tokens
        n_tokens += valid_tokens

    # Reduce across ranks
    if ddp_is_active():
        t = torch.tensor([total_loss, n_tokens], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_loss = float(t[0].item())
        n_tokens = int(t[1].item())

    avg_loss = total_loss / max(n_tokens, 1)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")

    metanet.train()
    return {"loss": avg_loss, "perplexity": ppl}


@torch.no_grad()
def generate_multiturn(
    metanetwork,
    dataloader,
    tokenizer,
    device,
    use_metanet: bool = True,
    metalora: Optional[torch.Tensor] = None,
    max_new_tokens: int = 500,
    max_conversation_length: int = 3000,
):
    metanetwork.eval()
    results = []
    
    assert dataloader.batch_size == 1, "generate_multiturn only supports batch_size=1 for simplicity"
    
    for i, batch in enumerate(dataloader):
        questions = batch['questions'][0]
        ground_truths = batch['answers'][0]
        messages = [batch["initial_messages"][0]] if batch["initial_messages"][0] is not {} else []
        evidence = batch["evidence"][0]
        evidence_ids = batch["evidence_ids"].to(device, non_blocking=True)
        evidence_attention_mask = batch["evidence_attention_mask"].to(device, non_blocking=True)
        
        lora_dict = None
        if use_metanet:
            lora_dict = metanetwork.generate_lora_dict(evidence_ids, evidence_attention_mask, metalora)
        
        conversation_log = [{"initial message": deepcopy(messages)}]
        f1_scores = []
        error_count_local = 0
        
        for q_idx, question in enumerate(questions):
            messages.append({"role": "user", "content": question})
            
            input_enc = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                max_length=max_conversation_length,
                truncation=True,
                return_dict=True,
                padding="max_length",
                enable_thinking=False,
            )
            
            input_ids = input_enc["input_ids"].to(device)
            attention_mask = input_enc["attention_mask"].to(device)
            
            gen_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "do_sample": False,
                "ignore_mem_token": True,
                "loradict": lora_dict,
            }
            
            outputs = metanetwork.metamodel.generate(**gen_kwargs)
            new_tokens = outputs[0, input_ids.shape[1]:]
            think_answer_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            think_text, answer_text = extract_think_and_answer(think_answer_text)
            
            if think_text == "[error]":
                error_count_local += 1
            
            messages.append({"role": "assistant", "content": answer_text})
            f1 = compute_f1(ground_truths[q_idx], answer_text)
            f1_scores.append(f1)
            
            conversation_log.append({
                "turn": q_idx + 1,
                "question": question,
                "think": think_text,
                "answer": answer_text,
                "ground_truth": ground_truths[q_idx],
                "f1": f1,
            })
        
        conversation_log[0]["avg_f1"] = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        conversation_log[0]["error_count"] = error_count_local
        results.append(conversation_log)
    
    # ====== Gather across DDP processes ======
    if ddp_is_active():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        gathered_data = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(results, gathered_data, dst=0)
        
        if rank == 0:
            final_results = []
            for process_results in gathered_data:
                if process_results is not None:
                    final_results.extend(process_results)
        else:
            # Non-root ranks return dummy to avoid misuse
            return [], []
    else:
        final_results = results

    # ====== Global & Per-Turn Statistics (only on rank 0 or non-DDP) ======
    if ddp_is_active() and dist.get_rank() != 0:
        return [], []

    total_samples = len(final_results)
    num_turns_expected = 15  # per your specification

    if total_samples == 0:
        stats = {
            "total_samples": 0,
            "total_turns": 0,
            "error_count": 0,
            "overall_avg_f1": 0.0,
            "per_turn_avg_f1": [0.0] * num_turns_expected,
            "per_turn_error_count": [0] * num_turns_expected,
        }
        return final_results, stats

    # Initialize per-turn accumulators
    turn_f1_sums = [0.0] * num_turns_expected
    turn_error_counts = [0] * num_turns_expected
    total_error_count = 0
    all_f1s = []

    for conv_log in final_results:
        turns = conv_log[1:]  # skip initial metadata dict
        if len(turns) != num_turns_expected:
            # Optionally: log warning or skip; here we skip incomplete samples
            continue

        for turn_idx in range(num_turns_expected):
            turn = turns[turn_idx]
            f1 = turn["f1"]
            think = turn["think"]

            all_f1s.append(f1)
            turn_f1_sums[turn_idx] += f1

            if think == "[error]":
                turn_error_counts[turn_idx] += 1
                total_error_count += 1

    overall_avg_f1 = sum(all_f1s) / len(all_f1s) if all_f1s else 0.0
    per_turn_avg_f1 = [
        turn_f1_sums[i] / total_samples for i in range(num_turns_expected)
    ]

    stats = {
        "total_samples": total_samples,
        "total_turns": len(all_f1s),
        "error_count": total_error_count,
        "overall_avg_f1": overall_avg_f1,
        "per_turn_avg_f1": per_turn_avg_f1,
        "per_turn_error_count": turn_error_counts,
    }

    return final_results, stats

@torch.no_grad()
def benchmark_generate_multiturn(
    metanetwork,
    tokenizer,
    device: torch.device,
    *,
    # Conversation constraints (match generate_multiturn behavior)
    max_conversation_length: int = 3000,
    max_new_tokens: int = 500,

    # Synthetic shape
    context_length: int = 2048,  # only used when use_metanet=False
    num_turns: int = 10,
    question_length: int = 32,
    answer_length: int = 128,    # forced fixed decode length (EOS disabled)

    # Metanet
    use_metanet: bool = True,
    metalora: Optional[torch.Tensor] = None,
    evidence_length: int = 32,

    # Benchmark controls
    warmup: int = 2,
    reset_cuda_cache: bool = True,
) -> Dict[str, float]:
    """
    Synthetic speed benchmark for generate_multiturn-style multi-turn generation.

    Rules enforced:
      - If use_metanet=True: initial conversation length = 0
      - If use_metanet=False: initial conversation length = context_length

    Improvements:
      - Left-truncation to max_conversation_length (prevents OOM, matches real truncation)
      - Incremental attention_mask (avoids rebuilding ones_like each time)
      - CUDA sync + reset peak stats, optional empty_cache to reduce 2nd-run OOM
      - del outputs each turn to reduce peak / fragmentation
      - Separate metanet_time and decode_time, report tokens/sec

    EOS is disabled (eos_token_id=None) to force fixed-length generation.
    """

    metanetwork.eval()
    model = metanetwork.metamodel

    if use_metanet:
        assert metalora is not None, "metalora must be provided when use_metanet=True"

    # stable token id
    base_token_id = tokenizer.encode("hello", add_special_tokens=False)[0]

    # ---- Optional allocator cleanup (helps reduce 2nd-run OOM) ----
    if device.type == "cuda" and reset_cuda_cache:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

    # ======================================================
    # Initial prompt (ENFORCE YOUR RULE)
    # ======================================================
    if use_metanet:
        init_len = 0
    else:
        init_len = context_length

    if init_len > 0:
        cur_input_ids = torch.full(
            (1, init_len),
            base_token_id,
            dtype=torch.long,
            device=device,
        )
        cur_attention_mask = torch.ones_like(cur_input_ids, dtype=torch.long)
    else:
        cur_input_ids = torch.empty((1, 0), dtype=torch.long, device=device)
        cur_attention_mask = torch.empty((1, 0), dtype=torch.long, device=device)

    # defensive truncate
    if cur_input_ids.shape[1] > max_conversation_length:
        cur_input_ids = cur_input_ids[:, -max_conversation_length:]
        cur_attention_mask = cur_attention_mask[:, -max_conversation_length:]

    # ======================================================
    # Fake evidence for metanet
    # ======================================================
    fake_evidence_ids = torch.full(
        (1, evidence_length),
        base_token_id,
        dtype=torch.long,
        device=device,
    )
    fake_evidence_mask = torch.ones_like(fake_evidence_ids, dtype=torch.long)

    # ======================================================
    # Warmup: metanet + model
    # ======================================================
    warmup_prompt_len = max(1, min(32, max_conversation_length))
    warmup_input_ids = torch.full(
        (1, warmup_prompt_len),
        base_token_id,
        dtype=torch.long,
        device=device,
    )
    warmup_attention_mask = torch.ones_like(warmup_input_ids, dtype=torch.long)

    for _ in range(warmup):
        if use_metanet:
            _ = metanetwork.generate_lora_dict(fake_evidence_ids, fake_evidence_mask, metalora)

        _ = model.generate(
            input_ids=warmup_input_ids,
            attention_mask=warmup_attention_mask,
            max_new_tokens=min(8, max_new_tokens),
            do_sample=False,
            eos_token_id=None,
            pad_token_id=tokenizer.pad_token_id,
            loradict=None,
            ignore_mem_token=True,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

    # Memory baseline after warmup
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        mem_start = torch.cuda.memory_allocated()
    else:
        mem_start = 0

    # ======================================================
    # Benchmark
    # ======================================================
    t_start = time.perf_counter()

    metanet_time = 0.0
    decode_time = 0.0
    total_generated_tokens = 0

    # --- metanet once (like generate_multiturn per sample) ---
    lora_dict = None
    if use_metanet:
        t0 = time.perf_counter()
        lora_dict = metanetwork.generate_lora_dict(
            fake_evidence_ids,
            fake_evidence_mask,
            metalora,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        metanet_time = time.perf_counter() - t0

    # Force fixed decode length but respect max_new_tokens
    decode_len = min(answer_length, max_new_tokens)

    for _turn in range(num_turns):
        # ---- append synthetic question ----
        q_ids = torch.full(
            (1, question_length),
            base_token_id,
            dtype=torch.long,
            device=device,
        )
        q_mask = torch.ones_like(q_ids, dtype=torch.long)

        cur_input_ids = torch.cat([cur_input_ids, q_ids], dim=1)
        cur_attention_mask = torch.cat([cur_attention_mask, q_mask], dim=1)

        # left-truncate BEFORE decode
        if cur_input_ids.shape[1] > max_conversation_length:
            cur_input_ids = cur_input_ids[:, -max_conversation_length:]
            cur_attention_mask = cur_attention_mask[:, -max_conversation_length:]

        # ---- decode ----
        t0 = time.perf_counter()
        outputs = model.generate(
            input_ids=cur_input_ids,
            attention_mask=cur_attention_mask,
            max_new_tokens=decode_len,
            do_sample=False,
            eos_token_id=None,  # disable EOS to force fixed length
            pad_token_id=tokenizer.pad_token_id,
            loradict=None,
            ignore_mem_token=True,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        decode_time += time.perf_counter() - t0

        gen_tokens = outputs.shape[1] - cur_input_ids.shape[1]
        if gen_tokens < 0:
            gen_tokens = 0

        total_generated_tokens += gen_tokens

        # ---- append answer ----
        if gen_tokens > 0:
            new_tokens = outputs[:, -gen_tokens:]
            new_mask = torch.ones_like(new_tokens, dtype=torch.long)

            cur_input_ids = torch.cat([cur_input_ids, new_tokens], dim=1)
            cur_attention_mask = torch.cat([cur_attention_mask, new_mask], dim=1)

            # truncate after appending
            if cur_input_ids.shape[1] > max_conversation_length:
                cur_input_ids = cur_input_ids[:, -max_conversation_length:]
                cur_attention_mask = cur_attention_mask[:, -max_conversation_length:]

        # free large tensor ASAP
        del outputs

    total_time = time.perf_counter() - t_start

    if device.type == "cuda":
        mem_peak = torch.cuda.max_memory_allocated()
    else:
        mem_peak = 0

    tokens_per_sec = (total_generated_tokens / decode_time) if decode_time > 0 else float("inf")

    return {
        # Config
        "use_metanet": float(1.0 if use_metanet else 0.0),
        "initial_context_length": float(init_len),
        "context_length_arg": float(context_length),
        "max_conversation_length": float(max_conversation_length),
        "max_new_tokens": float(max_new_tokens),
        "num_turns": float(num_turns),
        "question_length": float(question_length),
        "answer_length": float(answer_length),
        "decode_len_used": float(decode_len),
        "evidence_length": float(evidence_length),

        # Time
        "total_time_sec": float(total_time),
        "metanet_time_sec": float(metanet_time),
        "decode_time_sec": float(decode_time),

        # Tokens
        "total_generated_tokens": float(total_generated_tokens),
        "tokens_per_sec": float(tokens_per_sec),

        # Memory
        "gpu_mem_start_mb": float(mem_start / 1024**2),
        "gpu_mem_peak_mb": float(mem_peak / 1024**2),
    }
        
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
    ckpt_root = os.path.join("checkpoints", f"{cfg.name}", "iftpwc")

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
    test_sources = cfg.test.source.split(",")
    test_name = []
    test_datasets = []
    test_collator = MQACollator(tokenizer, context_max_length=cfg.test.context_max_length, conversation_max_length=cfg.test.conversation_max_length, cfg=cfg)
    test_prompt_collator = MQACollator(tokenizer, context_max_length=cfg.test.context_max_length, conversation_max_length=cfg.test.conversation_max_length, cfg=cfg, sys_msg=True)
    test_only_question_collator = MQACollator(tokenizer, context_max_length=cfg.test.context_max_length, conversation_max_length=cfg.test.conversation_max_length, cfg=cfg, sys_msg=True, no_evidence=True)
    sft_collator = SFTCollator(tokenizer, context_max_length=cfg.test.context_max_length, conversation_max_length=cfg.test.conversation_max_length, cfg=cfg, sys_msg=True, no_evidence=True)
    if is_main_process():
        logger.info("Preparing data...")
    if "msmacro-mqa" in test_sources:
        with open("data/msmacro-mqa/test.jsonl", "r") as f:
            data = [json.loads(line) for line in f.readlines()]
        random.seed(42)
        random.shuffle(data)
        data = data[:1000]
        mqa_dataset = MQADataset(data)
        test_name.append("msmacro-mqa")
        test_datasets.append(mqa_dataset)
        # with open("data/msmacro-mqa/test_sft.jsonl", "r") as f:
        #     sft_data = [json.loads(line) for line in f.readlines()]
        # random.shuffle(sft_data)
        # mqa_sft_dataset = SFTDataset(sft_data)
        # test_name.append("msmacro-mqa_sft")
        # test_datasets.append(mqa_sft_dataset)
    else:
        raise ValueError(f"Unknown data source: {cfg.test.source}")

    pin = device.type == "cuda"

    for i, (ds, name) in enumerate(zip(test_datasets, test_name), start=1):
        if is_main_process():
            logger.info(f"Testing on {name}...")
        out_dir = os.path.join(cfg.test.save_path, name)
        os.makedirs(out_dir, exist_ok=True)
        
        test_sampler = (
            DistributedSampler(ds, num_replicas=get_world_size(), rank=get_rank(), shuffle=False)
            if get_world_size() > 1
            else None
        )
        num_workers_default = 2 if device.type == "cuda" else 0
        # test_loader = DataLoader(
        #     ds,
        #     batch_size=cfg.test.batch_size,
        #     shuffle=False,
        #     sampler=test_sampler,
        #     collate_fn=test_collator,
        #     pin_memory=pin,
        #     num_workers=getattr(cfg.test, "num_workers", num_workers_default),
        #     persistent_workers=pin and getattr(cfg.test, "num_workers", num_workers_default) > 0,
        # )
        # results = evaluate(
        #     metanetwork,
        #     test_loader,
        #     device,
        #     use_metanet=True,
        #     metalora=metalora,
        # )
        # if is_main_process():
        #     logger.info(f"With Hypernetwork, Test results on {name}: {results}")
            
        # results = evaluate(
        #     metanetwork,
        #     test_loader,
        #     device,
        #     use_metanet=False,
        # )
        # if is_main_process():
        #     logger.info(f"Only question, Test results on {name}: {results}")
        
        # prompt_test_loader = DataLoader(
        #     ds,
        #     batch_size=cfg.test.batch_size,
        #     shuffle=False,
        #     sampler=test_sampler,
        #     collate_fn=test_prompt_collator,
        #     pin_memory=pin,
        #     num_workers=getattr(cfg.test, "num_workers", num_workers_default),
        #     persistent_workers=pin and getattr(cfg.test, "num_workers", num_workers_default) > 0,
        # )
        # results = evaluate(
        #     metanetwork,
        #     prompt_test_loader,
        #     device,
        #     use_metanet=False,
        # )
        # if is_main_process():
        #     logger.info(f"In-Context Prompt, Test results on {name}: {results}")
        
        # generate_test_loader = DataLoader(
        #     ds,
        #     batch_size=1,
        #     shuffle=False,
        #     sampler=test_sampler,
        #     collate_fn=test_collator,
        #     pin_memory=pin,
        #     num_workers=getattr(cfg.test, "num_workers", num_workers_default),
        #     persistent_workers=pin and getattr(cfg.test, "num_workers", num_workers_default) > 0,
        # )
        # results, stats = generate_multiturn(
        #     metanetwork,
        #     generate_test_loader,
        #     tokenizer,
        #     device,
        #     use_metanet=True,
        #     metalora=metalora,
        #     max_new_tokens=500,
        #     max_conversation_length=3000,
        # )
        # if is_main_process():
        #     output_path = os.path.join(out_dir, "generated_results.jsonl")
        #     with open(output_path, "w") as f:
        #         for res in results:
        #             f.write(json.dumps(res, ensure_ascii=False) + "\n")
        #     stats_path = os.path.join(out_dir, "generation_stats.json")
        #     with open(stats_path, "w") as f:
        #         json.dump(stats, f, indent=2)
        #     logger.info(f"Generation results saved to {output_path}")
        #     logger.info(f"Generation stats saved to {stats_path}")
            
        # generate_prompt_test_loader = DataLoader(
        #     ds,
        #     batch_size=1,
        #     shuffle=False,
        #     sampler=test_sampler,
        #     collate_fn=test_prompt_collator,
        #     pin_memory=pin,
        #     num_workers=getattr(cfg.test, "num_workers", num_workers_default),
        #     persistent_workers=pin and getattr(cfg.test, "num_workers", num_workers_default) > 0,
        # )
        # results, stats = generate_multiturn(
        #     metanetwork,
        #     generate_prompt_test_loader,
        #     tokenizer,
        #     device,
        #     use_metanet=False,
        #     max_new_tokens=500,
        #     max_conversation_length=3000,
        # )
        # if is_main_process():
        #     output_path = os.path.join(out_dir, "generated_results_prompt.jsonl")
        #     with open(output_path, "w") as f:
        #         for res in results:
        #             f.write(json.dumps(res, ensure_ascii=False) + "\n")
        #     stats_path = os.path.join(out_dir, "generation_stats_prompt.json")
        #     with open(stats_path, "w") as f:
        #         json.dump(stats, f, indent=2)
        #     logger.info(f"Generation results saved to {output_path}")
        #     logger.info(f"Generation stats saved to {stats_path}")
        
        # generate_only_question_test_loader = DataLoader(
        #     ds,
        #     batch_size=1,
        #     shuffle=False,
        #     sampler=test_sampler,
        #     collate_fn=test_only_question_collator,
        #     pin_memory=pin,
        #     num_workers=getattr(cfg.test, "num_workers", num_workers_default),
        #     persistent_workers=pin and getattr(cfg.test, "num_workers", num_workers_default) > 0,
        # )
            
        # results, stats = generate_multiturn(
        #     metanetwork,
        #     generate_only_question_test_loader,
        #     tokenizer,
        #     device,
        #     use_metanet=False,
        #     max_new_tokens=500,
        #     max_conversation_length=3000,
        # )
        # if is_main_process():
        #     output_path = os.path.join(out_dir, "generated_results_only_question.jsonl")
        #     with open(output_path, "w") as f:
        #         for res in results:
        #             f.write(json.dumps(res, ensure_ascii=False) + "\n")
        #     stats_path = os.path.join(out_dir, "generation_stats_only_question.json")
        #     with open(stats_path, "w") as f:
        #         json.dump(stats, f, indent=2)
        #     logger.info(f"Generation results saved to {output_path}")
        #     logger.info(f"Generation stats saved to {stats_path}")
        
        result = benchmark_generate_multiturn(
            metanetwork,
            tokenizer,
            device,
            max_conversation_length=3000,
            max_new_tokens=500,
            context_length=0,
            num_turns=15,
            question_length=20,
            answer_length=20,
            use_metanet=True,
            metalora=metalora,
            evidence_length=800,
            warmup=2,
        )
        if is_main_process():
            benchmark_path = os.path.join(out_dir, f"metanet_reproduce.json")
            with open(benchmark_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Generation benchmark with metanet saved to {benchmark_path}")
        result = benchmark_generate_multiturn(
            metanetwork,
            tokenizer,
            device,
            max_conversation_length=3000,
            max_new_tokens=500,
            context_length=820,
            num_turns=15,
            question_length=20,
            answer_length=20,
            use_metanet=False,
            evidence_length=0,
            warmup=2,
        )
        if is_main_process():
            benchmark_path = os.path.join(out_dir, f"incontext_reproduce.json")
            with open(benchmark_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Generation benchmark without metanet saved to {benchmark_path}")
        result = benchmark_generate_multiturn(
            metanetwork,
            tokenizer,
            device,
            max_conversation_length=3000,
            max_new_tokens=500,
            context_length=0,
            num_turns=15,
            question_length=20,
            answer_length=20,
            use_metanet=False,
            evidence_length=0,
            warmup=2,
        )
        if is_main_process():
            benchmark_path = os.path.join(out_dir, f"naive_reproduce.json")
            with open(benchmark_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Generation benchmark naive saved to {benchmark_path}")
        
        ################################  SFT ################################
        # sft_loader = DataLoader(
        #     mqa_sft_dataset,
        #     batch_size=1,
        #     shuffle=False,
        #     sampler=test_sampler,
        #     collate_fn=sft_collator,
        #     pin_memory=pin,
        #     num_workers=getattr(cfg.test, "num_workers", num_workers_default),
        #     persistent_workers=pin and getattr(cfg.test, "num_workers", num_workers_default) > 0,
        # )
        
        # num_sample=1  # 10, 5, 3
        # num_epoch=10 # 5 as default
        # results, stats = sft_lora(metanetwork.metamodel, sft_loader, 8, 0.001, device, tokenizer, num_sample=num_sample, num_epoch=num_epoch)
        # suffix = f"{num_sample}sample_{num_epoch}epoch"
        # with open(os.path.join(out_dir, f"sft_results_{suffix}.json"), "w") as f:
        #     json.dump(results, f, indent=2)
        # with open(os.path.join(out_dir, f"sft_stats_{suffix}.json"), "w") as f:
        #     json.dump(stats, f, indent=2)
        # logger.info(f"SFT results saved to {os.path.join(out_dir, f'sft_results_{suffix}.json')}")
        # logger.info(f"SFT stats saved to {os.path.join(out_dir, f'sft_stats_{suffix}.json')}")
        
    

    ddp_cleanup_if_needed()


if __name__ == "__main__":
    main()
