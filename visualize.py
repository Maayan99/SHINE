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
import math

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.myvisualize import visualize_loradict_to_files

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

from utils.mydataset import TextDataset, create_mock_dataset, SquadDataset, SquadCollator, PretrainCollator, GroupedSquadDataset, GroupTextDataset, GroupPretrainCollator, IFTCollator, IFTDataset, IFTC1QADataset
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
from utils.myinit import _resolve_device, _import_class
from collections import OrderedDict
from typing import Optional, Union, Mapping, Sequence
from utils.myvisualize import visualize_loradict_to_files

logger = get_logger("metalora")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@hydra.main(version_base=None, config_path="configs")
@torch.no_grad()
def main(cfg: DictConfig):
    amp_dtype = torch.bfloat16
    
    assert cfg.mode == 'visualize', "Only visualize mode is supported in this script."
    torch.set_float32_matmul_precision('high')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    logger.info("Resolved config:")
    logger.info(f"\n\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    # Seed & device
    # Make seed rank-dependent to vary shuffles but keep reproducibility per rank
    set_seed(int(cfg.run.seed))
    device = _resolve_device(cfg.run.device)
    torch.backends.cudnn.benchmark = True
    
    # Load model/tokenizer (supports your local LoRA-wrapped Qwen class)
    logger.info("Loading model & tokenizer...")
    MetaModelCls = _import_class(cfg.model.metamodel_class_path)
    ConfigCls = _import_class(cfg.model.config_class_path)
    config = ConfigCls.from_pretrained(cfg.model.model_from)
    config.num_mem_token = -1
    cfg.hidden_size = config.hidden_size
    cfg.num_layers = config.num_hidden_layers

    if cfg.metanetwork.type in ["transformer", "linear", "lineargate"]:
        tmp_model = MetaModelCls.from_pretrained(cfg.model.model_from, config=config)
        lora_numel = tmp_model.lora_params_numel(cfg.model.lora_r)
        assert lora_numel % (cfg.hidden_size * cfg.num_layers) == 0, \
            "For transformer metanetwork, num_mem_token must be set to model.lora_params_numel(lora_r) * mean_pool_size / (hidden_size * num_layers)"
        config.num_mem_token = tmp_model.lora_params_numel(cfg.model.lora_r) * cfg.metanetwork.transformer_cfg.mean_pool_size // (cfg.hidden_size * cfg.num_layers)
        cfg.num_mem_token = config.num_mem_token
        del tmp_model
        logger.info(f"Using {cfg.metanetwork.type} metanetwork, automatically set num_mem_token to {config.num_mem_token}")
    elif cfg.metanetwork.type in []:
        config.num_mem_token = cfg.num_mem_token
        logger.info(f"Using {cfg.metanetwork.type} metanetwork, set num_mem_token to {config.num_mem_token} as configured")
    else:
        raise ValueError(f"Unknown metanetwork type: {cfg.metanetwork.type}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_from, padding_side="left", use_fast=True)
    tokenizer.add_tokens(['<RECON>', '<COMP>', '<NOTHING>'])
    tokenizer.chat_template = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if (loop.last or (not loop.last and reasoning_content)) and (enable_thinking is not defined or enable_thinking != false) %}\n                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n            {%- else %}\n                {{- '<|im_start|>' + message.role + '\\n' + content }}\n            {%- endif %}\n        {%- else %}\n            {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is not defined or enable_thinking != false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"
    metamodel = MetaModelCls.from_pretrained(cfg.model.model_from, config=config)
    metamodel.reset_mem_tokens()
    metamodel.resize_token_embeddings(len(tokenizer))
    # nothing_id = tokenizer.convert_tokens_to_ids("<NOTHING>")
    # with torch.no_grad():
    #     metamodel.get_input_embeddings().weight[nothing_id].zero_()
    # if is_main_process():
    #     print("NOTHING:", metamodel.get_input_embeddings().weight[nothing_id])
    metanetwork = Metanetwork(metamodel, cfg, metamodel.lora_params_numel(cfg.model.lora_r))
    metanetwork.train()
    metanetwork.to(device)
    freeze(metamodel) 
    logger.info(f"Metanetwork type: {cfg.metanetwork.type}, Transform method: {cfg.metanetwork.method}")
        
    # Training loop scaffolding
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
    resume_state = load_training_state(resume_dir)
        
    metanetwork.metamodel.config.use_cache = False

    # ====== Wrap ONLY the trainable module in DDP when applicable ======
    metanetwork.to(device)
    
    # Data
    logger.info("Preparing data...")
    if cfg.data.source == "transmla":
        # raise ValueError(f"transmal not used")
        dataset = load_dataset(os.path.join("data", "transmla_pretrain_6B_tokens"), split="train")
        split_dataset = dataset.train_test_split(test_size=0.0001, seed=42)
        train_texts = split_dataset["train"]
        val_texts = split_dataset["test"]
        logger.info(f"Train len: {len(train_texts)}")
        logger.info(f"Val len: {len(val_texts)}")
        train_ds = TextDataset(train_texts["text"], tokenizer)
        val_ds = TextDataset(val_texts["text"], tokenizer)
        train_collator = PretrainCollator(tokenizer=tokenizer, metatrain=True, cfg=cfg, conversation_max_length=cfg.data.conversation_max_length, context_max_length=cfg.data.context_max_length)
        val_collator = PretrainCollator(tokenizer=tokenizer, metatrain=True, cfg=cfg, conversation_max_length=cfg.data.conversation_max_length, context_max_length=cfg.data.context_max_length)
    elif cfg.data.source == "grouptransmla":
        dataset = load_dataset(os.path.join("data", "transmla_pretrain_6B_tokens"), split="train")
        # dataset = dataset.select(range(10000))
        split_dataset = dataset.train_test_split(test_size=0.0001, seed=42)
        train_texts = split_dataset["train"]
        val_texts = split_dataset["test"]
        train_ds = GroupTextDataset(train_texts["text"], tokenizer, cfg.data.conversation_max_length, os.path.join("data", "transmla_pretrain_6B_tokens"), "train")
        val_ds = GroupTextDataset(val_texts["text"], tokenizer, cfg.data.conversation_max_length, os.path.join("data", "transmla_pretrain_6B_tokens"), "val")
        train_collator = GroupPretrainCollator(tokenizer, cfg, conversation_max_length=cfg.data.conversation_max_length, context_max_length=cfg.data.context_max_length, metatrain=True)
        val_collator = GroupPretrainCollator(tokenizer, cfg, conversation_max_length=cfg.data.conversation_max_length, context_max_length=cfg.data.context_max_length, metatrain=True)
    elif cfg.data.source == "squad":
        # features: ['id', 'title', 'context', 'question', 'answers'],
        # num_rows: 87599
        train_dataset = load_dataset(os.path.join("data", "squad"), split="train")
        val_dataset = load_dataset(os.path.join("data", "squad"), split="validation")
        val_dataset = val_dataset.shuffle(seed=42).select(range(1000))
        # train_ds = SquadDataset(train_dataset, tokenizer)
        # val_ds = SquadDataset(val_dataset, tokenizer)
        train_ds = GroupedSquadDataset(train_dataset, tokenizer, 512, name="Train", sep="\n\n")
        val_ds = GroupedSquadDataset(val_dataset, tokenizer, 512, name="Validation", sep="\n\n")
        train_collator = SquadCollator(tokenizer=tokenizer, conversation_max_length=cfg.data.conversation_max_length, context_max_length=cfg.data.context_max_length, metatrain=True, cfg=cfg)
        val_collator = SquadCollator(tokenizer=tokenizer, conversation_max_length=cfg.data.conversation_max_length, context_max_length=cfg.data.context_max_length, metatrain=True, cfg=cfg)
    elif cfg.data.source == "ift":
        data_path = os.path.join("data", "ift_cqa.json")
        group_idx_path = os.path.join("data", f"ift_cqa_group_idxs_context{cfg.data.context_max_length}_conversation{cfg.data.conversation_max_length}.json")        
        train_ds = IFTDataset(data_path, group_idx_path, use_exceed=True)
        val_dataset = load_dataset(os.path.join("data", "squad"), split="validation")
        val_dataset = val_dataset.shuffle(seed=42).select(range(1000))
        val_ds = GroupedSquadDataset(val_dataset, tokenizer, 512, name="Validation", sep="<|endoftext|>")
        train_collator = IFTCollator(tokenizer, cfg.data.context_max_length, cfg.data.conversation_max_length, cfg=cfg)
        val_collator = SquadCollator(tokenizer=tokenizer, conversation_max_length=cfg.data.conversation_max_length, context_max_length=cfg.data.context_max_length, metatrain=True, cfg=cfg)
    elif cfg.data.source == "ift-c1qa":
        data_path = os.path.join("data", "ift_c1qa.json")
        train_ds = IFTC1QADataset(data_path, use_exceed=False, max_context_len=cfg.data.context_max_length, max_conversation_len=cfg.data.conversation_max_length)
        val_dataset = load_dataset(os.path.join("data", "squad"), split="validation")
        val_dataset = val_dataset.shuffle(seed=42).select(range(1000))
        val_ds = GroupedSquadDataset(val_dataset, tokenizer, 512, name="Validation", sep="\n\n")
        train_collator = IFTCollator(tokenizer, cfg.data.context_max_length, cfg.data.conversation_max_length, cfg=cfg)
        val_collator = SquadCollator(tokenizer=tokenizer, conversation_max_length=cfg.data.conversation_max_length, context_max_length=cfg.data.context_max_length, metatrain=True, cfg=cfg)
    else:
        raise ValueError(f"Unknown data source: {cfg.data.source}")

    

    pin = (device.type == "cuda")

    # Use a few workers by default when on GPU
    num_workers_default = 2 if device.type == "cuda" else 0

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.train_batch_size,
        shuffle=True,
        collate_fn=train_collator,
        pin_memory=pin,
        num_workers=getattr(cfg.data, "num_workers", num_workers_default),
        persistent_workers=pin and getattr(cfg.data, "num_workers", num_workers_default) > 0,
    )
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

    global_step = 0
    best_eval_loss = float("inf")
    start_epoch = 0
    start_step_in_epoch = 0
    if resume_state is not None:
        global_step = resume_state["global_step"]
        best_eval_loss = resume_state["best_eval_loss"]
        start_epoch = resume_state["epoch"]
        start_step_in_epoch = resume_state["step_in_epoch"]
    
    for step, batch in enumerate(val_loader, start=1):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        input_attention_mask = batch["input_attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        evidence_ids = batch["evidence_ids"].to(device, non_blocking=True)
        evidence_attention_mask = batch["evidence_attention_mask"].to(device, non_blocking=True)
        
        loradict = metanetwork.generate_lora_dict(evidence_ids, evidence_attention_mask, metalora, use_gradient_checkpoint=False, return_plain=False)
        visualize_loradict_to_files(loradict, visualize_dir)
        break

if __name__ == "__main__":
    main()