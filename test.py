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

from utils.mydataset import SquadDataset, SquadCollator, TextDataset, CausalLMDataCollator, create_mock_dataset, LoogleDataset, LoogleCollator
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
from meta_train_parallel import generate_stepwise

logger = get_logger("test")


def extract_think_and_answer(text: str) -> Tuple[str, str]:
    """
    Splits model output into (think_part, answer_part).
    Returns empty strings if either is missing.
    """
    # Make search case-insensitive
    lower = text.lower()
    start_tag = "<think>"
    end_tag = "</think>"

    think = ""
    answer = text.strip()

    # Case 1: has explicit <think>...</think>
    start = lower.find(start_tag)
    end = lower.find(end_tag)
    if start != -1 and end != -1 and end > start:
        think = text[start + len(start_tag):end].strip()
        answer = text[end + len(end_tag):].strip()
    else:
        # Case 2: remove any inline think tags (just in case)
        think_match = re.search(r"<think>(.*?)</think>", text, flags=re.IGNORECASE | re.DOTALL)
        if think_match:
            think = think_match.group(1).strip()
        answer = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.IGNORECASE | re.DOTALL).strip()

    # Clean common prefixes like "Answer:" or "Final answer:"
    answer = re.sub(r"^(final answer|answer)\s*:\s*", "", answer, flags=re.IGNORECASE).strip()

    # Optionally: take only the first non-empty line as the final answer
    if "\n" in answer:
        for line in answer.splitlines():
            line = line.strip()
            if line:
                answer = line
                break

    return think, answer


@torch.no_grad()
def test(cfg, metanetwork_ddp_or_module, tokenizer, testloader, use_metanet: bool = True, metalora: Any = None, use_amp: bool = False, device: torch.device = 'cuda') -> Dict[str, float]:
    if use_metanet:
        assert metalora is not None, "metalora cannot be None when use_metanet is True"
    
    # Handle both wrapped and unwrapped metanetwork
    metanet = metanetwork_ddp_or_module.module if isinstance(metanetwork_ddp_or_module, DDP) else metanetwork_ddp_or_module
    metanet.eval()
    if use_amp and device.type == "cuda":
        scaler_ctx = partial(torch.amp.autocast, device_type=str(device))
    else:
        from contextlib import nullcontext
        scaler_ctx = nullcontext
    
    results = []
    # gen_kwargs = {
    #     "max_new_tokens": cfg.test.max_new_tokens,
    #     "do_sample": False,
    #     "pad_token_id": getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id,
    #     "num_beams": 4,
    #     "length_penalty": 2.0,
    #     "no_repeat_ngram_size": 3,
    #     "early_stopping": True,
    # }

    for batch_idx, batch in enumerate(testloader):
        print(f"Processing batch {batch_idx + 1}/{len(testloader)}...")
        evidences = batch["evidence"]
        evidence_ids = batch["evidence_ids"].to(device, non_blocking=True)
        evidence_attention_mask = batch["evidence_attention_mask"].to(device, non_blocking=True)
        # question_ids = batch["question_ids"].to(device, non_blocking=True)
        # question_attention_mask = batch["question_attention_mask"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        input_attention_mask = batch["input_attention_mask"].to(device, non_blocking=True)
        ground_truths = batch["answers"]
        questions = batch["questions"]
        labels = None if batch["labels"] is None else batch["labels"].to(device, non_blocking=True)

        with scaler_ctx():
            loradict = None
            if use_metanet:
                # Produce LoRA dict from the MetaNetwork
                loradict = metanet.generate_lora_dict(evidence_ids=evidence_ids, evidence_attention_mask=evidence_attention_mask, metalora=metalora)
            gen_out = metanet.metamodel.generate(
                input_ids=input_ids,
                attention_mask=input_attention_mask,
                loradict=loradict,
                ignore_mem_token=True,
                max_new_tokens=1000,
                do_sample=False,
                # return_dict_in_generate=True,
                # output_scores=True
            )
            input_lens = input_attention_mask.sum(dim=1).tolist()
            
            # gen_out = generate_stepwise(
            #     model=metanet.metamodel,
            #     input_ids=input_ids,
            #     attention_mask=input_attention_mask,
            #     loradict=loradict,
            #     ignore_mem_token=True,
            #     max_new_tokens=1000,
            #     do_sample=False,
            #     tokenizer=tokenizer,
            #     device=device,
            # )
            # input_lens = input_attention_mask.sum(dim=1).tolist()

        # scores = torch.stack(gen_out.scores)  # [seq_len, batch, vocab]
        # probs = F.softmax(scores, dim=-1)

        # for step, step_probs in enumerate(probs):
        #     topk = torch.topk(step_probs[0], 5)
        #     print(f"\nStep {step+1}")
        #     for token_id, prob in zip(topk.indices, topk.values):
        #         print(f"{tokenizer.decode(token_id)}\t{prob.item():.4f}")
        # exit()
        
        # Decode per item: strip the prompt portion to keep only the generated continuation
        gen_out = gen_out.to("cpu")
        tokens = tokenizer.convert_ids_to_tokens(gen_out[0])
        input_ids_cpu = input_ids.to("cpu")

        for i in range(gen_out.size(0)):
            full_text = tokenizer.decode(gen_out[i], skip_special_tokens=True)
            input_text = tokenizer.decode(input_ids_cpu[i][-input_lens[i] :], skip_special_tokens=True)
            # Keep only the continuation after the prompt (best-effort split)
            if full_text.startswith(input_text):
                answer_text = full_text[len(input_text):]
            else:
                # Fallback if tokenization/spacing prevents a clean prefix match
                answer_text = full_text

            if not use_metanet:
                think, answer = extract_think_and_answer(answer_text)
            else:
                think, answer = "I know the answer because I have read something about this.", answer_text
            
            results.append({
                "evidence": evidences[i],
                "input": input_text,
                "question": questions[i],
                "think": think,
                "answer": answer,
                "ground_truth": ground_truths[i],      
            })
            
    metanet.train()
    return results


@hydra.main(version_base=None, config_path="configs")
def main(cfg: DictConfig):  
    # ========= DDP init (safe for single-process) =========
    ddp_init_if_needed()

    if is_main_process():
        logger.info("Resolved config:")
        logger.info(f"\n\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    # Seed & device
    # Make seed rank-dependent to vary shuffles but keep reproducibility per rank
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
        assert tmp_model.lora_params_numel(cfg.model.lora_r) % (cfg.hidden_size * cfg.num_layers) == 0, \
            "For transformer metanetwork, num_mem_token must be set to model.lora_params_numel(lora_r) / (hidden_size * num_layers)"
        config.num_mem_token = tmp_model.lora_params_numel(cfg.model.lora_r) // (cfg.hidden_size * cfg.num_layers)
        cfg.num_mem_token = config.num_mem_token
        del tmp_model
        if is_main_process():
            logger.info(f"Using transformer metanetwork, set num_mem_token to {config.num_mem_token}")
    else:
        config.num_mem_token = cfg.num_mem_token

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_from, padding_side="left")
    metamodel = MetaModelCls.from_pretrained(cfg.model.model_from, config=config)
    metamodel.reset_mem_tokens()
    metanetwork = Metanetwork(metamodel, cfg, metamodel.lora_params_numel(cfg.model.lora_r))
    metanetwork.train()
    metanetwork.to(device)
    freeze(metamodel) 
    
    
    # Training loop scaffolding
    hydra_run_dir = os.getcwd()
    ckpt_root = os.path.join(hydra_run_dir, "checkpoints")
    
    if cfg.test_global_step == "latest":
        resume_dir = get_latest_checkpoint(ckpt_root)
    elif cfg.test_global_step == "final":
        resume_dir = os.path.join(ckpt_root, "final")
        if not os.path.isdir(resume_dir):
            raise ValueError(f"Requested resume dir {resume_dir} does not exist.")
    # elif cfg.test_global_step == "best":
    #     resume_dir = os.path.join(ckpt_root, "best")
    #     if not os.path.isdir(resume_dir):
    #         raise ValueError(f"Requested resume dir {resume_dir} does not exist.")
    elif isinstance(cfg.test_global_step, int) and cfg.test_global_step > 0:
        resume_dir = os.path.join(ckpt_root, f"checkpoint-{cfg.test_global_step}")
        if not os.path.isdir(resume_dir):
            raise ValueError(f"Requested resume dir {resume_dir} does not exist.")
    else:
        raise ValueError(f"Invalid test_global_step: {cfg.test_global_step}")

    # Load model & tokenizer
    if is_main_process():
        logger.info(f"Resume mode, loading from {resume_dir}...")
    metanetwork, metalora = load_checkpoint(metanetwork, resume_dir, device)

    # Data
    if is_main_process():
        logger.info("Preparing data...")
    # if cfg.test.source == "loogle":
    #     # names = ["shortdep_qa", "shortdep_cloze", "longdep_qa", "summarization"]
    #     names = ["shortdep_qa", "shortdep_cloze", "longdep_qa"]
    #     datasets = []
    #     for testset in names:
    #         data = load_dataset(
    #             os.path.join("bigai-nlco/LooGLE", testset),
    #             split="test",
    #             cache_dir=os.path.join('data', 'loogle', testset),
    #         )
    #         datasets.append(LoogleDataset(data, tokenizer, max_length=cfg.data.max_length))
    #         if is_main_process():
    #             logger.info(f"Loaded loogle/{testset} with {len(data)} samples")
    #     collator = LoogleCollator(tokenizer=tokenizer, max_length=cfg.data.max_length, use_reference=False)
    #     collator_no_metanet = LoogleCollator(tokenizer=tokenizer, max_length=cfg.data.max_length, use_reference=True)
    # elif cfg.test.source == "easy":
    #     names = ["0"]
    #     for testset in names:
    #         texts = [
    #             "What does lewis eat every evening ?",
    #             "What does lewis eat every morning ?",
    #             "What is most expensive in Beijing ?",
    #             "Why Jack refuse to see Alice ?",
    #             "Does Jack love Alice ?",
    #             "What's the meaning when I say I don't like it ?",
    #         ]
    #         answers = [
    #             "Rice",
    #             "Dumplings",
    #             "Housing prices",
    #             "Because he hates her",
    #             "No",
    #             "I hate it very much",
    #         ]
    #         evidences = [
    #             "Lewis eats dumplings every morning and rice every evening.",
    #             "Lewis eats dumplings every morning and rice every evening.",
    #             "The most expensive thing in Beijing is housing prices.",
    #             "Jack refuse to see Alice because he hates her.",
    #             "Jack refuse to see Alice because he hates her.",
    #             "When I say I don't like it I mean I hate it very much."
    #         ]
    #         data = [{"question": q, "evidence": e, "answer": a} for q, e, a in zip(texts, evidences, answers)]
    #         datasets = [LoogleDataset(data, tokenizer, max_length=cfg.data.max_length)]
    #         if is_main_process():
    #             logger.info(f"Loaded easy testset with {len(data)} samples")    
    #         collator = LoogleCollator(tokenizer=tokenizer, max_length=cfg.data.max_length, use_reference=False)
    #         collator_no_metanet = LoogleCollator(tokenizer=tokenizer, max_length=cfg.data.max_length, use_reference=True)
    if cfg.test.source == "squad":
        names = ["squad"]
        datasets = []
        for testset in names:
            data = load_dataset(
                os.path.join("data", "squad"),
                split="validation",
            )
            datasets.append(SquadDataset(data, tokenizer, max_length=cfg.data.max_length))
            if is_main_process():
                logger.info(f"Loaded {cfg.test.source}/{testset} with {len(data)} samples")
        collator = SquadCollator(tokenizer=tokenizer, max_length=cfg.data.max_length)
        collator_no_metanet = SquadCollator(tokenizer=tokenizer, max_length=cfg.data.max_length, use_reference=True)
        collator_only_question = SquadCollator(tokenizer=tokenizer, max_length=cfg.data.max_length, only_question=True)
    else:
        raise ValueError(f"Unknown data source: {cfg.test.source}")

    

    pin = (device.type == "cuda")
    for i, ds in enumerate(datasets):
        # Distributed samplers (only if world_size > 1)
        test_sampler = DistributedSampler(ds, num_replicas=get_world_size(), rank=get_rank(), shuffle=False) if get_world_size() > 1 else None
        # Use a few workers by default when on GPU
        num_workers_default = 2 if device.type == "cuda" else 0
        test_loader = DataLoader(
            ds,
            batch_size=cfg.test.batch_size,
            shuffle=False,
            sampler=test_sampler,
            collate_fn=collator,
            pin_memory=pin,
            num_workers=getattr(cfg.test, "num_workers", num_workers_default),
            persistent_workers=pin and getattr(cfg.test, "num_workers", num_workers_default) > 0,
        )
        test_loader_no_metanet = DataLoader(
            ds,
            batch_size=cfg.test.batch_size,
            shuffle=False,
            sampler=test_sampler,
            collate_fn=collator_no_metanet,
            pin_memory=pin,
            num_workers=getattr(cfg.test, "num_workers", num_workers_default),
            persistent_workers=pin and getattr(cfg.test, "num_workers", num_workers_default) > 0,
        )
        test_loader_only_question = DataLoader(
            ds,
            batch_size=cfg.test.batch_size,
            shuffle=False,
            sampler=test_sampler,
            collate_fn=collator_only_question,
            pin_memory=pin,
            num_workers=getattr(cfg.test, "num_workers", num_workers_default),
            persistent_workers=pin and getattr(cfg.test, "num_workers", num_workers_default) > 0,
        )

        # Checkpoint root
        ckpt_root = os.path.join(hydra_run_dir, "checkpoints")

        # Make sure all ranks see the directory
        if ddp_is_active():
            dist.barrier()

        # ===== Generate answers on this split and save to JSON (DDP-safe) =====
        # local_results = test(cfg, metanetwork, tokenizer, test_loader, use_metanet=True, use_amp=cfg.run.use_fp16, device=device, metalora=metalora)
        # gather_and_save(local_results, ".json")
        # local_results_no_metanet = test(cfg, metanetwork, tokenizer, test_loader_no_metanet, use_metanet=False, use_amp=cfg.run.use_fp16, device=device)
        # gather_and_save(local_results_no_metanet, "_no_metanet.json")
        local_results_only_question = test(cfg, metanetwork, tokenizer, test_loader_only_question, use_metanet=False, use_amp=cfg.run.use_fp16, device=device)
        gather_and_save(local_results_only_question, "_only_question.json")
        
        def gather_and_save(local_results, output_suffix):
            # Gather results across ranks (if distributed), then write once on rank 0
            if ddp_is_active():
                gathered = [None for _ in range(get_world_size())]
                dist.all_gather_object(gathered, local_results)
                if is_main_process():
                    merged = []
                    for part in gathered:
                        if part:
                            merged.extend(part)
            else:
                merged = local_results

            if is_main_process():
                out_dir = os.path.join(cfg.test.save_path, cfg.test.source)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{names[i]}", f"{output_suffix}")
                with open(out_path.replace(".json", f"{output_suffix}"), "w", encoding="utf-8") as f:
                    json.dump(merged, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved {len(merged)} predictions to {out_path}")
        

        # # Gather results across ranks (if distributed), then write once on rank 0
        # if ddp_is_active():
        #     gathered = [None for _ in range(get_world_size())]
        #     gathered_no_metanet = [None for _ in range(get_world_size())]
        #     gathered_only_question = [None for  _ in range(get_world_size())]
        #     dist.all_gather_object(gathered, local_results)
        #     dist.all_gather_object(gathered_no_metanet, local_results_no_metanet)
        #     dist.all_gather_object(gathered_only_question, local_results_only_question)
        #     if is_main_process():
        #         merged = []
        #         for part in gathered:
        #             if part:
        #                 merged.extend(part)
        #         merged_no_metanet = []
        #         for part in gathered_no_metanet:
        #             if part:
        #                 merged_no_metanet.extend(part)
        #         for part in gathered_only_question:
        #             if part:
        #                 merged_no_metanet.extend(part)
        # else:
        #     merged = local_results
        #     merged_no_metanet = local_results_no_metanet

        # if is_main_process():
        #     out_dir = os.path.join(cfg.test.save_path, cfg.test.source)
        #     os.makedirs(out_dir, exist_ok=True)
        #     out_path = os.path.join(out_dir, f"{names[i]}.json")
        #     with open(out_path, "w", encoding="utf-8") as f:
        #         json.dump(merged, f, ensure_ascii=False, indent=2)
        #     logger.info(f"Saved {len(merged)} predictions to {out_path}")
        #     with open(out_path.replace(".json", "_no_metanet.json"), "w", encoding="utf-8") as f:
        #         json.dump(merged_no_metanet, f, ensure_ascii=False, indent=2)
        #     logger.info(f"Saved {len(merged_no_metanet)} predictions (no metanet) to {out_path.replace('.json', '_no_metanet.json')}")
        

    # Cleanup DDP
    ddp_cleanup_if_needed()


if __name__ == "__main__":
    main()
