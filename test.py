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

from utils.mydataset import TextDataset, CausalLMDataCollator, create_mock_dataset, LoogleDataset, LoogleCollator
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

logger = get_logger("test")


@torch.no_grad()
def test(cfg, model, metanetwork_ddp_or_module, tokenizer, testloader, use_amp: bool = False, device: torch.device = 'cuda') -> Dict[str, float]:
    # Handle both wrapped and unwrapped metanetwork
    if metanetwork_ddp_or_module is None:
        use_metanet = False
    else:
        use_metanet = True
        metanet = metanetwork_ddp_or_module.module if isinstance(metanetwork_ddp_or_module, DDP) else metanetwork_ddp_or_module

    model.eval()
    if use_metanet:
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
        evidences = batch["evidence"]
        evidence_ids = batch["evidence_ids"].to(device, non_blocking=True)
        evidence_attention_mask = batch["evidence_attention_mask"].to(device, non_blocking=True)
        # question_ids = batch["question_ids"].to(device, non_blocking=True)
        # question_attention_mask = batch["question_attention_mask"].to(device, non_blocking=True)
        prompt_ids = batch["prompt_ids"].to(device, non_blocking=True)
        prompt_attention_mask = batch["prompt_attention_mask"].to(device, non_blocking=True)
        ground_truths = batch["answers"]

        with scaler_ctx():
            if use_metanet:
                # Produce LoRA dict from the MetaNetwork
                loradict = metanet(input_ids=evidence_ids, attention_mask=evidence_attention_mask)
                gen_out = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_attention_mask,
                    loradict=loradict,
                    # **gen_kwargs,
                )
                input_lens = prompt_attention_mask.sum(dim=1).tolist()
                input_ids = prompt_ids
            else:
                gen_out = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_attention_mask,
                    # **gen_kwargs,
                )
                input_lens = prompt_attention_mask.sum(dim=1).tolist()
                input_ids = prompt_ids

        # Decode per item: strip the prompt portion to keep only the generated continuation
        gen_out = gen_out.to("cpu")
        input_ids_cpu = input_ids.to("cpu")

        for i in range(gen_out.size(0)):
            full_text = tokenizer.decode(gen_out[i], skip_special_tokens=True)
            input_text = tokenizer.decode(input_ids_cpu[i][-input_lens[i] :], skip_special_tokens=True)
            # Keep only the continuation after the prompt (best-effort split)
            if full_text.startswith(input_text):
                answer_text = full_text[len(input_text):].strip()
            else:
                # Fallback if tokenization/spacing prevents a clean prefix match
                answer_text = full_text

            results.append({
                "evidence": evidences[i],
                "input": input_text,
                "answer": answer_text,
                "ground_truth": ground_truths[i],
            })
        
    return results


@hydra.main(version_base=None, config_path="configs", config_name="base")
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

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_from)
    metamodel = MetaModelCls.from_pretrained(cfg.model.model_from, config=config)
    metamodel.reset_mem_tokens()
    metanetwork = Metanetwork(metamodel, cfg, metamodel.lora_params_numel(cfg.model.lora_r))
    metanetwork.train()
    metanetwork.to(device)
    freeze(metamodel) 
    
    
    # Training loop scaffolding
    hydra_run_dir = os.getcwd()
    ckpt_root = os.path.join(hydra_run_dir, "checkpoints")
    if cfg.resume_global_step == "latest":
        resume_dir = get_latest_checkpoint(ckpt_root)
    elif isinstance(cfg.resume_global_step, int) and cfg.resume_global_step > 0:
        resume_dir = os.path.join(ckpt_root, f"checkpoint-{cfg.resume_global_step}")
        if not os.path.isdir(resume_dir):
            raise ValueError(f"Requested resume dir {resume_dir} does not exist.")
    else:
        raise ValueError(f"Invalid resume_global_step: {cfg.resume_global_step}")
    
    # Load model & tokenizer
    if is_main_process():
        logger.info(f"Resume mode, loading from {resume_dir}...")
    metanetwork = load_checkpoint(metanetwork, resume_dir, device)

    # Data
    if is_main_process():
        logger.info("Preparing data...")
    if cfg.test.source == "loogle":
        # 1) Main process downloads
        # names = ["shortdep_qa", "shortdep_cloze", "longdep_qa", "summarization"]
        names = ["shortdep_qa", "shortdep_cloze", "longdep_qa"]
        if ddp_is_active() and is_main_process():
            logger.info("Preparing data (downloading to cache if needed)...")
            for testset in names:
                _ = load_dataset(
                    "bigai-nlco/LooGLE",
                    testset,
                    split="test",
                    cache_dir=os.path.join('data', 'loogle', testset),
                )
                logger.info(f"Cached loogle/{testset}")
        # 2) Sync
        barrier()
        # 3) Everyone loads from cache only
        datasets = []
        for testset in names:
            data = load_dataset(
                "bigai-nlco/LooGLE",
                testset,
                split="test",
                cache_dir=os.path.join('data', 'loogle', testset),
            )
            datasets.append(LoogleDataset(data, tokenizer, max_length=cfg.data.max_length))
            if is_main_process():
                logger.info(f"Loaded loogle/{testset} with {len(data)} samples")

        PROMPT_TEMPLATE = " Please answer the question as short as you can. {question}"
        PROMPT_TEMPLATE_NO_METANETWORK = "You know that \"{evidence}\" Please answer the question as short as you can \"{question}\""
        collator = LoogleCollator(tokenizer=tokenizer, max_length=cfg.data.max_length, PROMPT_TEMPLATE=PROMPT_TEMPLATE)
        collator_no_metanet = LoogleCollator(tokenizer=tokenizer, max_length=cfg.data.max_length, PROMPT_TEMPLATE=PROMPT_TEMPLATE_NO_METANETWORK)
    elif cfg.test.source == "easy":
        names = ["0"]
        for testset in names:
            texts = [
                "What does lewis eat every evening ?",
                "What does lewis eat every morning ?",
                "What is most expensive in Beijing ?",
                "Why Jack refuse to see Alice ?",
                "Does Jack love Alice ?",
                "What's the meaning when I say I don't like it ?",
            ]
            answers = [
                "Rice",
                "Dumplings",
                "Housing prices",
                "Because he hates her",
                "No",
                "I hate it very much",
            ]
            evidences = [
                "Lewis eats dumplings every morning and rice every evening.",
                "Lewis eats dumplings every morning and rice every evening.",
                "The most expensive thing in Beijing is housing prices.",
                "Jack refuse to see Alice because he hates her.",
                "Jack refuse to see Alice because he hates her.",
                "When I say I don't like it I mean I hate it very much."
            ]
            data = [{"question": q, "evidence": e, "answer": a} for q, e, a in zip(texts, evidences, answers)]
            datasets = [LoogleDataset(data, tokenizer, max_length=cfg.data.max_length)]
            if is_main_process():
                logger.info(f"Loaded easy testset with {len(data)} samples")    
            
            PROMPT_TEMPLATE = " Please answer the question as short as you can. {question}"
            PROMPT_TEMPLATE_NO_METANETWORK = "You know that \"{evidence}\" Please answer the question as short as you can \"{question}\""
            collator = LoogleCollator(tokenizer=tokenizer, max_length=cfg.data.max_length, PROMPT_TEMPLATE=PROMPT_TEMPLATE)
            collator_no_metanet = LoogleCollator(tokenizer=tokenizer, max_length=cfg.data.max_length, PROMPT_TEMPLATE=PROMPT_TEMPLATE_NO_METANETWORK)
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

        # Checkpoint root
        ckpt_root = os.path.join(hydra_run_dir, "checkpoints")

        # Make sure all ranks see the directory
        if ddp_is_active():
            dist.barrier()

        # ===== Generate answers on this split and save to JSON (DDP-safe) =====
        local_results = test(cfg, model, metanetwork, tokenizer, test_loader, use_amp=cfg.run.use_fp16, device=device)
        local_results_no_metanet = test(cfg, model, None, tokenizer, test_loader_no_metanet, use_amp=cfg.run.use_fp16, device=device)

        # Gather results across ranks (if distributed), then write once on rank 0
        if ddp_is_active():
            gathered = [None for _ in range(get_world_size())]
            gathered_no_metanet = [None for _ in range(get_world_size())]
            dist.all_gather_object(gathered, local_results)
            dist.all_gather_object(gathered_no_metanet, local_results_no_metanet)
            if is_main_process():
                merged = []
                for part in gathered:
                    if part:
                        merged.extend(part)
                merged_no_metanet = []
                for part in gathered_no_metanet:
                    if part:
                        merged_no_metanet.extend(part)
        else:
            merged = local_results
            merged_no_metanet = local_results_no_metanet

        if is_main_process():
            out_dir = os.path.join(cfg.test.save_path, cfg.test.source)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{names[i]}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(merged)} predictions to {out_path}")
            with open(out_path.replace(".json", "_no_metanet.json"), "w", encoding="utf-8") as f:
                json.dump(merged_no_metanet, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(merged_no_metanet)} predictions (no metanet) to {out_path.replace('.json', '_no_metanet.json')}")
        

    # Cleanup DDP
    ddp_cleanup_if_needed()


if __name__ == "__main__":
    main()
