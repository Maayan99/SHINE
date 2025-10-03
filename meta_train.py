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

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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

# Configure logging once at the entrypoint
logging.basicConfig(
    level=logging.INFO,  # You can set to DEBUG for more verbosity
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("metalora")

# ---------------------------
# Mock dataset for demo
# ---------------------------
def create_mock_dataset() -> Tuple[List[str], List[str]]:
    texts = [
        "1231",
        "2342",
        "3453",
        "4564",
        "5675",
        "6786",
        "7897",
        "8908",
        "9019",
        "0120",
    ] * 100
    df = pd.DataFrame({'text': texts})
    train_texts, val_texts = train_test_split(df['text'], test_size=0.1, random_state=42)
    return train_texts.tolist(), val_texts.tolist()


# ---------------------------
# Dataset
# ---------------------------
class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx) -> Dict[str, Any]:
        return {"text": str(self.texts[idx])}


# ---------------------------
# Collator with dynamic padding and label masking
# ---------------------------
@dataclass
class CausalLMDataCollator:
    tokenizer: Any
    max_length: int = 512

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [ex["text"] for ex in batch]
        enc = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        labels = input_ids.clone()

        # Ensure a pad token exists
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        pad_id = self.tokenizer.pad_token_id
        labels[labels == pad_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, metanetwork, tokenizer, out_dir: str, step: int, extra_state: Dict[str, Any] = None):
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(os.path.join(out_dir, "model"))
    metanetwork.lora_model.save_pretrained(os.path.join(out_dir, "metamodel"))
    torch.save(metanetwork.metanetwork.state_dict(), os.path.join(out_dir, "metanetwork.pth"))
    tokenizer.save_pretrained(os.path.join(out_dir, "tokenizer"))
    if extra_state is not None:
        with open(os.path.join(out_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
            json.dump(extra_state, f, ensure_ascii=False, indent=2)

def load_checkpoint(model, metamodel, metanetwork, tokenizer, in_dir: str):
    model.from_pretrained(os.path.join(in_dir, "model"))
    metamodel.from_pretrained(os.path.join(in_dir, "metamodel"))
    metanetwork.lora_model = metamodel
    metanetwork.metanetwork.load_state_dict(torch.load(os.path.join(in_dir, "metanetwork.pth")))
    tokenizer.from_pretrained(os.path.join(in_dir, "tokenizer"))
    return model, metamodel, metanetwork, tokenizer


@torch.no_grad()
def evaluate(model, metamodel, metanetwork, dataloader, device, use_amp: bool = True) -> Dict[str, float]:
    model.eval()
    metamodel.eval()
    metanetwork.eval()
    total_loss = 0.0
    n_tokens = 0
    if use_amp and device.type == "cuda":
        scaler_ctx = partial(torch.amp.autocast, device_type=str(device))
    else:
        # On CPU autocast is a no-op on many PyTorch versions; using a dummy ctx manager
        from contextlib import nullcontext
        scaler_ctx = nullcontext
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with scaler_ctx():
            outputs = metamodel(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            memory_states = outputs.memory_states
            loradict = metanetwork(memory_states)
            new_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, loradict=loradict)
            loss = new_outputs.loss 

        valid_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * valid_tokens
        n_tokens += valid_tokens

    avg_loss = total_loss / max(n_tokens, 1)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    model.train()
    metamodel.train()
    metanetwork.train()
    return {"eval_loss": avg_loss, "perplexity": ppl}


def _resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_cfg in ("cuda", "cpu"):
        return torch.device(device_cfg)
    raise ValueError(f"Unsupported device setting: {device_cfg}")


def _import_class(path: str):
    """
    Import a class from a 'module.ClassName' path.
    """
    if "." not in path:
        raise ValueError("model.class_path must be 'module.ClassName'")
    mod_name, cls_name = path.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


# ---------------------------
# Hydra entrypoint
# ---------------------------
@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig):
    logger.info("Resolved config:")
    logger.info(f"\n\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    # Seed & device
    set_seed(cfg.run.seed)
    device = _resolve_device(cfg.run.device)

    # Load model/tokenizer (supports your local LoRA-wrapped Qwen class)
    logger.info("Loading model & tokenizer...")
    ModelCls = _import_class(cfg.model.model_class_path)
    MetaModelCls = _import_class(cfg.model.meta_model_class_path) 
    ConfigCls = _import_class(cfg.model.config_class_path)
    config = ConfigCls.from_pretrained(cfg.model.model_from)
    cfg.hidden_size = config.hidden_size
    cfg.num_layers = config.num_hidden_layers
    model = ModelCls.from_pretrained(cfg.model.model_from, config=config)
    model.train()
    model.to(device)
    if cfg.metanetwork.type == "transformer":
        assert model.lora_params_numel(cfg.model.lora_r) % (cfg.hidden_size * cfg.num_layers) == 0, "For transformer metanetwork, num_mem_token must be set to model.lora_params_numel(lora_r) / (hidden_size * num_layers)"
        config.num_mem_token = model.lora_params_numel(cfg.model.lora_r) // (cfg.hidden_size * cfg.num_layers)
        cfg.num_mem_token = config.num_mem_token
        logger.info(f"Using transformer metanetwork, set num_mem_token to {config.num_mem_token}")
    else:
        config.num_mem_token = cfg.model.num_mem_token
    metamodel = MetaModelCls.from_pretrained(cfg.model.model_from, config=config)
    metamodel.train()
    metamodel.to(device)
    metamodel.reset_mem_tokens()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_from)
    metanetwork = Metanetwork(metamodel, cfg)
    metanetwork.train()
    metanetwork.to(device)
    for param in model.parameters():
        param.requires_grad = False  # freeze the base model
    for param in metamodel.parameters():
        param.requires_grad = False  # freeze the meta model except mem_tokens
    metamodel.mem_tokens.requires_grad = True
    
    # Data
    logger.info("Preparing data...")
    if cfg.data.source == "mock":
        train_texts, val_texts = create_mock_dataset()
    elif cfg.data.source == "transmla":
        dataset = load_dataset(os.path.join("data", "transmla_pretrain_6B_tokens"), split="train")
        split_dataset = dataset.train_test_split(test_size=0.001, seed=42)
        train_texts = split_dataset["train"]
        val_texts = split_dataset["test"]
        logger.info(f"Train len: {len(train_texts)}")
        logger.info(f"Val len: {len(val_texts)}")
    else:
        raise ValueError(f"Unknown data source: {cfg.data.source}")

    train_ds = TextDataset(train_texts, tokenizer, max_length=cfg.data.max_length)
    val_ds = TextDataset(val_texts, tokenizer, max_length=cfg.data.max_length)

    collator = CausalLMDataCollator(tokenizer=tokenizer, max_length=cfg.data.max_length)

    pin = (device.type == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.train_batch_size,
        shuffle=True,
        collate_fn=collator,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        pin_memory=pin,
    )

    # Optimizer & Scheduler
    logger.info("Setting up optimizer & scheduler...")
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight", "norm1", "norm2"]
    grouped_params = [
        {
            "params": [p for n, p in metanetwork.named_parameters() if (not any(nd in n for nd in no_decay) and not n.startswith("lora_model"))],
            "weight_decay": cfg.optim.weight_decay,
        },
        {
            "params": [p for n, p in metanetwork.named_parameters() if (any(nd in n for nd in no_decay) and not n.startswith("lora_model"))],
            "weight_decay": 0.0,
        },
        {
            "params": [metamodel.mem_tokens],
            "weight_decay": cfg.optim.weight_decay,
        }
    ]
    optimizer = torch.optim.AdamW(grouped_params, lr=cfg.optim.learning_rate)

    total_steps = cfg.optim.num_epochs * math.ceil(len(train_loader) / max(1, cfg.run.gradient_accumulation_steps))
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.optim.warmup_steps,
        num_training_steps=total_steps,
    )

    # AMP scaler
    scaler = torch.amp.GradScaler(enabled=(cfg.run.use_fp16 and device.type == "cuda"))

    # Training loop
    hydra_run_dir = os.getcwd()
    tb_log_dir = os.path.join(hydra_run_dir, "tensorboard")
    writer = SummaryWriter(log_dir=tb_log_dir)
    logger.info(f"TensorBoard logs will be written to: {tb_log_dir}")
    logger.info("Starting training loop...")
    global_step = 0
    best_eval_loss = float("inf")

    # Use Hydra's run dir for checkpoints; also mirror to a stable final output_dir if desired  
    ckpt_root = os.path.join(hydra_run_dir, "checkpoints")
    os.makedirs(ckpt_root, exist_ok=True)

    def one_train_epoch(epoch):
        nonlocal global_step, best_eval_loss
        epoch_loss = 0.0
        epoch_tokens = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.optim.num_epochs}")

        for step, batch in enumerate(pbar, start=1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast(enabled=(cfg.run.use_fp16 and device.type == "cuda"), device_type=str(device)):
                outputs = metamodel(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                memory_states = outputs.memory_states
                loradict = metanetwork(memory_states)
                new_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, loradict=loradict)
                loss = new_outputs.loss / max(1, cfg.run.gradient_accumulation_steps)

            writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], global_step)
            
            scaler.scale(loss).backward()

            valid_tokens = (labels != -100).sum().item()
            epoch_loss += loss.item() * valid_tokens * max(1, cfg.run.gradient_accumulation_steps)
            epoch_tokens += valid_tokens

            if step % max(1, cfg.run.gradient_accumulation_steps) == 0:
                if cfg.optim.grad_clip_norm and cfg.optim.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    for group in optimizer.param_groups:
                        torch.nn.utils.clip_grad_norm_(
                            group["params"], cfg.optim.grad_clip_norm
                        )

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                global_step += 1

                if cfg.logging.logging_steps and global_step % cfg.logging.logging_steps == 0:
                    avg_loss = (epoch_loss / max(epoch_tokens, 1))
                    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
                    writer.add_scalar("train/loss", avg_loss, global_step)
                    writer.add_scalar("train/ppl", ppl, global_step)
                    pbar.set_postfix({"lr": lr_scheduler.get_last_lr()[0], "loss": f"{avg_loss:.4f}", "ppl": f"{ppl:.2f}"})

                if cfg.eval.eval_steps and global_step % cfg.eval.eval_steps == 0:
                    eval_metrics = evaluate(model, metamodel, metanetwork, val_loader, device, use_amp=cfg.run.use_fp16)
                    writer.add_scalar("eval/loss", eval_metrics["eval_loss"], global_step)
                    writer.add_scalar("eval/ppl", eval_metrics["perplexity"], global_step)
                    logger.info(f"\n[Eval @ step {global_step}] loss={eval_metrics['eval_loss']:.4f} ppl={eval_metrics['perplexity']:.2f}")
                                        
                    if cfg.save.save_best and eval_metrics["eval_loss"] < best_eval_loss:
                        best_eval_loss = eval_metrics["eval_loss"]
                        best_dir = os.path.join(ckpt_root, "best")
                        logger.info(f"New best model! Saving to {best_dir}")
                        save_checkpoint(
                            model, metanetwork, tokenizer, best_dir, global_step,
                            extra_state={"global_step": global_step, "best_eval_loss": best_eval_loss}
                        )

                if cfg.save.save_steps and global_step % cfg.save.save_steps == 0:
                    ckpt_dir = os.path.join(ckpt_root, f"checkpoint-{global_step}")
                    logger.info(f"\nSaving checkpoint to {ckpt_dir}")
                    save_checkpoint(
                        model, metanetwork, tokenizer, ckpt_dir, global_step,
                        extra_state={"global_step": global_step}
                    )

        epoch_avg = (epoch_loss / max(epoch_tokens, 1))
        epoch_ppl = math.exp(epoch_avg) if epoch_avg < 20 else float("inf")
        logger.info(f"Epoch {epoch} done. train_loss={epoch_avg:.4f} train_ppl={epoch_ppl:.2f}")

        eval_metrics = evaluate(model, metamodel, metanetwork, val_loader, device, use_amp=cfg.run.use_fp16)
        writer.add_scalar("eval/loss", eval_metrics["eval_loss"], global_step)
        writer.add_scalar("eval/ppl", eval_metrics["perplexity"], global_step)
        logger.info(f"[Epoch {epoch} Eval] loss={eval_metrics['eval_loss']:.4f} ppl={eval_metrics['perplexity']:.2f}")
        if cfg.save.save_best and eval_metrics["eval_loss"] < best_eval_loss:
            best_eval_loss = eval_metrics["eval_loss"]
            best_dir = os.path.join(ckpt_root, "best")
            logger.info(f"New best model! Saving to {best_dir}")
            save_checkpoint(
                model, metanetwork, tokenizer, best_dir, global_step,
                extra_state={"global_step": global_step, "best_eval_loss": best_eval_loss}
            )
    
    eval_metrics = evaluate(model, metamodel, metanetwork, val_loader, device, use_amp=cfg.run.use_fp16)
    writer.add_scalar("eval/loss", eval_metrics["eval_loss"], global_step)
    writer.add_scalar("eval/ppl", eval_metrics["perplexity"], global_step)
    logger.info(f"\n[Eval @ step {global_step}] loss={eval_metrics['eval_loss']:.4f} ppl={eval_metrics['perplexity']:.2f}")
    for epoch in range(1, cfg.optim.num_epochs + 1):
        one_train_epoch(epoch)

    # Final save (both to Hydra run dir and an optional stable output_dir)
    logger.info("Saving final model...")
    final_dir = os.path.join(ckpt_root, "final")
    save_checkpoint(model, metanetwork, tokenizer, final_dir, global_step, extra_state={"global_step": global_step})

    if cfg.paths.output_dir:
        stable_out = cfg.paths.output_dir
        os.makedirs(stable_out, exist_ok=True)
        save_checkpoint(model, metanetwork, tokenizer, stable_out, global_step, extra_state={"global_step": global_step})
        logger.info(f"Model saved to {stable_out}")

    logger.info(f"All artifacts in Hydra run dir: {hydra_run_dir}")


if __name__ == "__main__":
    main()
