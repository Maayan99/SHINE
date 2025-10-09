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

from utils.mydataset import TextDataset, CausalLMDataCollator, create_mock_dataset
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

logger = get_logger("metalora")


@torch.no_grad()
def evaluate(metanetwork_ddp_or_module, dataloader, device, use_amp: bool = False, use_metanet: bool = True) -> Dict[str, float]:
    # Handle both wrapped and unwrapped metanetwork
    metanet = metanetwork_ddp_or_module.module if isinstance(metanetwork_ddp_or_module, DDP) else metanetwork_ddp_or_module
    metanet.eval()
    
    total_loss = 0.0
    n_tokens = 0
    if use_amp and device.type == "cuda":
        scaler_ctx = partial(torch.amp.autocast, device_type=str(device))
    else:
        from contextlib import nullcontext
        scaler_ctx = nullcontext

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with scaler_ctx():
            outputs = metanet(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_metanet=use_metanet)
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
    return {"eval_loss": avg_loss, "perplexity": ppl}


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
    freeze(metamodel)  # base model frozen; metanetwork has trainable params
    
    # Training loop scaffolding
    hydra_run_dir = os.getcwd()
    ckpt_root = os.path.join(hydra_run_dir, "checkpoints")
    os.makedirs(ckpt_root, exist_ok=True)
    if cfg.resume_global_step == -1:
        resume_dir = None
    elif cfg.resume_global_step == "latest":
        resume_dir = get_latest_checkpoint(ckpt_root)
    elif isinstance(cfg.resume_global_step, int) and cfg.resume_global_step > 0:
        resume_dir = os.path.join(ckpt_root, f"checkpoint-{cfg.resume_global_step}")
        if not os.path.isdir(resume_dir):
            raise ValueError(f"Requested resume dir {resume_dir} does not exist.")
    else:
        raise ValueError(f"Invalid resume_global_step: {cfg.resume_global_step}")
    
    resume_state = None
    if resume_dir is not None:
        # Load model & tokenizer
        if is_main_process():
            logger.info(f"Resume mode, loading from {resume_dir}...")
        metanetwork, tokenizer = load_checkpoint(metanetwork, tokenizer, resume_dir, device)
        resume_state = load_training_state(resume_dir)

    # ====== Wrap ONLY the trainable module in DDP when applicable ======
    if should_use_ddp():
        ddp_metanet = DDP(
            metanetwork,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
    else:
        ddp_metanet = metanetwork  # no wrapping in single-process run

    # Optimizer & Scheduler
    if is_main_process():
        logger.info("Setting up optimizer & scheduler...")
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight", "norm1", "norm2"]
    grouped_params = [
        {
            "params": [p for n, p in ddp_metanet.named_parameters() if (not any(nd in n for nd in no_decay) and not n.startswith("module.metamodel"))],
            "weight_decay": cfg.optim.weight_decay,
        },
        {
            "params": [p for n, p in ddp_metanet.named_parameters() if (any(nd in n for nd in no_decay) and not n.startswith("module.metamodel"))],
            "weight_decay": 0.0,
        },
        # mem_tokens are already part of metanetwork's parameters
    ]  
    
    # Data
    if is_main_process():
        logger.info("Preparing data...")
    if cfg.data.source == "mock":
        train_texts, val_texts = create_mock_dataset()
    elif cfg.data.source == "transmla":
        dataset = load_dataset(os.path.join("data", "transmla_pretrain_6B_tokens"), split="train")
        split_dataset = dataset.train_test_split(test_size=0.001, seed=42)
        train_texts = split_dataset["train"]
        val_texts = split_dataset["test"]
        if is_main_process():
            logger.info(f"Train len: {len(train_texts)}")
            logger.info(f"Val len: {len(val_texts)}")
    else:
        raise ValueError(f"Unknown data source: {cfg.data.source}")

    train_ds = TextDataset(train_texts, tokenizer, max_length=cfg.data.max_length)
    val_ds = TextDataset(val_texts, tokenizer, max_length=cfg.data.max_length)

    collator = CausalLMDataCollator(tokenizer=tokenizer, max_length=cfg.data.max_length)

    pin = (device.type == "cuda")

    # Distributed samplers (only if world_size > 1)
    train_sampler = DistributedSampler(train_ds, num_replicas=get_world_size(), rank=get_rank(), shuffle=True, seed=cfg.run.seed) if get_world_size() > 1 else None
    val_sampler = DistributedSampler(val_ds, num_replicas=get_world_size(), rank=get_rank(), shuffle=False) if get_world_size() > 1 else None

    # Use a few workers by default when on GPU
    num_workers_default = 2 if device.type == "cuda" else 0

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.train_batch_size,
        shuffle=False,
        sampler=train_sampler,
        collate_fn=collator,
        pin_memory=pin,
        num_workers=getattr(cfg.data, "num_workers", num_workers_default),
        persistent_workers=pin and getattr(cfg.data, "num_workers", num_workers_default) > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.eval_batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collator,
        pin_memory=pin,
        num_workers=getattr(cfg.data, "num_workers", num_workers_default),
        persistent_workers=pin and getattr(cfg.data, "num_workers", num_workers_default) > 0,
    )

    
    optimizer, lr_scheduler, scaler = init_optimize(grouped_params, train_loader, cfg, device)

    # Only main process writes TB logs
    tb_log_dir = os.path.join(hydra_run_dir, "tensorboard")
    writer = SummaryWriter(log_dir=tb_log_dir) if is_main_process() else None
    if is_main_process():
        logger.info(f"TensorBoard logs will be written to: {tb_log_dir}")
        logger.info("Starting training loop...")

    # Checkpoint root
    ckpt_root = os.path.join(hydra_run_dir, "checkpoints")
    if is_main_process():
        os.makedirs(ckpt_root, exist_ok=True)

    # Make sure all ranks see the directory
    if ddp_is_active():
        dist.barrier()

    global_step = 0
    best_eval_loss = float("inf")
    start_epoch = 0
    start_step_in_epoch = 0
    if resume_state is not None:
        global_step = resume_state["global_step"]
        best_eval_loss = resume_state["best_eval_loss"]
        start_epoch = resume_state["epoch"]
        start_step_in_epoch = resume_state["step_in_epoch"]

    def one_train_epoch(epoch, start_epoch=1, start_step_in_epoch=0):
        nonlocal global_step, best_eval_loss
        epoch_loss = 0.0
        epoch_tokens = 0
        tmp_loss = 0.0
        tmp_tokens = 0
        # need to change
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        
        if epoch < start_epoch:
            for step, batch in enumerate(train_loader, start=1):
                if step % max(1, cfg.run.gradient_accumulation_steps) == 0:
                    lr_scheduler.step()
            return 

        pbar = train_loader
        if is_main_process():
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.optim.num_epochs}")

        for step, batch in enumerate(pbar, start=1):
            if epoch == start_epoch and step <= start_step_in_epoch:
                lr_scheduler.step()
                continue
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast(enabled=(cfg.run.use_fp16 and device.type == "cuda"), device_type=str(device)):
                # Forward through possibly DDP-wrapped metanetwork
                outputs = ddp_metanet(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / max(1, cfg.run.gradient_accumulation_steps)

            if writer is not None:
                writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], global_step)

            scaler.scale(loss).backward()

            valid_tokens = (labels != -100).sum().item()
            # Track per-rank; we’ll reduce for logging only
            epoch_loss += loss.item() * valid_tokens * max(1, cfg.run.gradient_accumulation_steps)
            tmp_loss += loss.item() * valid_tokens * max(1, cfg.run.gradient_accumulation_steps)
            epoch_tokens += valid_tokens
            tmp_tokens += valid_tokens

            if step % max(1, cfg.run.gradient_accumulation_steps) == 0:
                if cfg.optim.grad_clip_norm and cfg.optim.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    for group in optimizer.param_groups:
                        torch.nn.utils.clip_grad_norm_(group["params"], cfg.optim.grad_clip_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                global_step += 1

                # Periodic logging (only on rank 0, with distributed averages)
                if cfg.logging.logging_steps and global_step % cfg.logging.logging_steps == 0:
                    # everyone computes + participates in the reduction
                    avg_loss_local = (epoch_loss / max(epoch_tokens, 1))
                    tmp_loss_local = (tmp_loss / max(tmp_tokens, 1))
                    avg_loss_world = distributed_mean(avg_loss_local, device)
                    tmp_loss_world = distributed_mean(tmp_loss_local, device)
                    if is_main_process():
                        avg_ppl = math.exp(avg_loss_world) if avg_loss_world < 20 else float("inf")
                        tmp_ppl = math.exp(tmp_loss_world) if tmp_loss_world < 20 else float("inf")
                        if writer is not None:
                            writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], global_step)
                            writer.add_scalar("train/epoch_avg_loss", avg_loss_world, global_step)
                            writer.add_scalar("train/epoch_avg_ppl", avg_ppl, global_step)
                            writer.add_scalar("train/tmp_loss", tmp_loss_world, global_step)
                            writer.add_scalar("train/tmp_ppl", tmp_ppl, global_step)
                        if isinstance(pbar, tqdm):
                            pbar.set_postfix({"lr": lr_scheduler.get_last_lr()[0],
                                            "epoch_avg_loss": f"{avg_loss_world:.4f}", "epoch_avg_ppl": f"{avg_ppl:.2f}",
                                            "tmp_loss": f"{tmp_loss_world:.4f}", "tmp_ppl": f"{tmp_ppl:.2f}"})
                    tmp_loss = 0.0
                    tmp_tokens = 0

                # ---- Periodic checkpoint (rank 0 only) ----
                if getattr(cfg.save, "save_steps", 0) and global_step % cfg.save.save_steps == 0:
                    if ddp_is_active():
                        dist.barrier()
                    if is_main_process():
                        ckpt_dir = os.path.join(ckpt_root, f"checkpoint-{global_step}")
                        logger.info(f"Saving checkpoint to {ckpt_dir}")
                        # Save unwrapped metanetwork (state is in ddp_metanet.module when DDP)
                        save_checkpoint(
                            ddp_metanet.module if isinstance(ddp_metanet, DDP) else ddp_metanet,
                            tokenizer,
                            ckpt_dir,
                            extra_state={"global_step": global_step},
                        )
                        save_training_state(
                            ckpt_dir,
                            global_step,
                            epoch,
                            step,
                            best_eval_loss,
                        )
                    if ddp_is_active():
                        dist.barrier()

                # ---- Eval + best checkpoint ----
                if getattr(cfg.eval, "eval_steps", 0) and global_step % cfg.eval.eval_steps == 0:
                    eval_metrics = evaluate(ddp_metanet, val_loader, device, use_amp=cfg.run.use_fp16)
                    if writer is not None:
                        writer.add_scalar("eval/loss", eval_metrics["eval_loss"], global_step)
                        writer.add_scalar("eval/ppl", eval_metrics["perplexity"], global_step)
                    if is_main_process():
                        logger.info(f"[Eval @ step {global_step}] loss={eval_metrics['eval_loss']:.4f} ppl={eval_metrics['perplexity']:.2f}")

                    # Best checkpoint saving on rank 0
                    if getattr(cfg.save, "save_best", True) and is_main_process():
                        if eval_metrics["eval_loss"] < best_eval_loss:
                            best_eval_loss = eval_metrics["eval_loss"]
                            best_dir = os.path.join(ckpt_root, "best")
                            logger.info(f"New best model! Saving to {best_dir}")
                            save_checkpoint(
                                ddp_metanet.module if isinstance(ddp_metanet, DDP) else ddp_metanet,
                                tokenizer,
                                best_dir,
                                extra_state={"global_step": global_step, "best_eval_loss": best_eval_loss},
                            )
                            save_training_state(
                                best_dir,
                                global_step,
                                epoch,
                                step,
                                best_eval_loss,
                            )
                    if ddp_is_active():
                        dist.barrier()

        # Epoch-end eval/log (averaged)
        avg_epoch_loss_local = (epoch_loss / max(epoch_tokens, 1))
        avg_epoch_loss_world = distributed_mean(avg_epoch_loss_local, device)
        epoch_ppl = math.exp(avg_epoch_loss_world) if avg_epoch_loss_world < 20 else float("inf")
        if is_main_process():
            logger.info(f"Epoch {epoch} done. train_loss={avg_epoch_loss_world:.4f} train_ppl={epoch_ppl:.2f}")

        eval_metrics = evaluate(ddp_metanet, val_loader, device, use_amp=cfg.run.use_fp16)
        if writer is not None:
            writer.add_scalar("eval/loss", eval_metrics["eval_loss"], global_step)
            writer.add_scalar("eval/ppl", eval_metrics["perplexity"], global_step)
        if is_main_process():
            logger.info(f"[Epoch {epoch} Eval] loss={eval_metrics['eval_loss']:.4f} ppl={eval_metrics['perplexity']:.2f}")

        if getattr(cfg.save, "save_best", True) and is_main_process():
            if eval_metrics["eval_loss"] < best_eval_loss:
                best_eval_loss = eval_metrics["eval_loss"]
                best_dir = os.path.join(ckpt_root, "best")
                logger.info(f"New best model! Saving to {best_dir}")
                save_checkpoint(
                    ddp_metanet.module if isinstance(ddp_metanet, DDP) else ddp_metanet,
                    tokenizer,
                    best_dir,
                    extra_state={"global_step": global_step, "best_eval_loss": best_eval_loss},
                )
                save_training_state(
                    best_dir,
                    global_step,
                    epoch,
                    step,
                    best_eval_loss,
                )
        if ddp_is_active():
            dist.barrier()
    
    # Initial eval
    init_eval_without_metanetwork = evaluate(ddp_metanet, val_loader, device, use_amp=cfg.run.use_fp16, use_metanet=False)
    if is_main_process():
        logger.info(f"[without lora] loss={init_eval_without_metanetwork['eval_loss']:.4f} ppl={init_eval_without_metanetwork['perplexity']:.2f}")
    init_eval = evaluate(ddp_metanet, val_loader, device, use_amp=cfg.run.use_fp16)
    if writer is not None:
        writer.add_scalar("eval/loss", init_eval["eval_loss"], global_step)
        writer.add_scalar("eval/ppl", init_eval["perplexity"], global_step)
    if is_main_process():
        logger.info(f"[Eval @ step {global_step}] loss={init_eval['eval_loss']:.4f} ppl={init_eval['perplexity']:.2f}")

    # Main training epochs
    for epoch in range(1, cfg.optim.num_epochs + 1):
        one_train_epoch(epoch, start_epoch, start_step_in_epoch)

    # Final save (rank 0 only)
    if is_main_process():
        logger.info("Saving final model...")
        final_dir = os.path.join(ckpt_root, "final")
        save_checkpoint(
            ddp_metanet.module if isinstance(ddp_metanet, DDP) else ddp_metanet,
            tokenizer,
            final_dir,
            extra_state={"global_step": global_step},
        )

        if cfg.paths.output_dir:
            stable_out = cfg.paths.output_dir
            os.makedirs(stable_out, exist_ok=True)
            save_checkpoint(
                ddp_metanet.module if isinstance(ddp_metanet, DDP) else ddp_metanet,
                tokenizer,
                stable_out,
                extra_state={"global_step": global_step},
            )
            logger.info(f"Model saved to {stable_out}")

        logger.info(f"All artifacts in Hydra run dir: {hydra_run_dir}")

    if writer is not None:
        writer.close()

    # Cleanup DDP
    ddp_cleanup_if_needed()


if __name__ == "__main__":
    main()
