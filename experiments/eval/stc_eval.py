"""
stc_eval.py — Steps-to-Convergence (STC) evaluation for SHINE LoRA initializations.

Measures how quickly loss decreases when fine-tuning a LoRA initialization with Adam
on held-out IFT examples.  Three conditions:
  1. sysprompt_shine  — LoRA from the system-prompt-finetuned hypernetwork (iftpwc/final)
  2. regular_shine    — LoRA from the original SHINE IFT hypernetwork (train/checkpoint-epoch-1)
  3. random_init      — Standard LoRA init (A ~ N(0, sqrt(scale)), B = zeros)

For each test system prompt, QA pairs are split 5 train / 5 eval.  Each condition runs
10 Adam steps on the train split, measuring eval loss at every step.  The Area Under the
eval Loss Curve (AUC) summarises convergence speed.

Usage:
    cd /workspace/SHINE
    conda activate shine
    python experiments/eval/stc_eval.py --lr_sweep          # pick LR first
    python experiments/eval/stc_eval.py --lr 5e-4           # full run
"""

import os
import sys
import gc
import json
import logging
import argparse
from datetime import datetime
from collections import defaultdict
from statistics import median

# ── path fix ────────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_LORA_FT_DIR = os.path.join(_PROJECT_ROOT, "experiments", "lora_finetune")
if _LORA_FT_DIR not in sys.path:
    sys.path.insert(0, _LORA_FT_DIR)

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from omegaconf import OmegaConf

from metanetwork_family import Metanetwork
from utils.myseed import set_seed
from utils.mysaveload import load_checkpoint
from utils.myfreeze import freeze
from utils.myinit import _import_class
from utils.myloradict import iter_learnable_tensors, merge_loradicts
from finetune_lora import clone_loradict_to_params, random_init_loradict

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("stc_eval")

# ── Chat template (from generate_results.py) ───────────────────────────────
CHAT_TEMPLATE = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if (loop.last or (not loop.last and reasoning_content)) and (enable_thinking is not defined or enable_thinking != false) %}\n                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n            {%- else %}\n                {{- '<|im_start|>' + message.role + '\\n' + content }}\n            {%- endif %}\n        {%- else %}\n            {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is not defined or enable_thinking != false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"


# ---------------------------------------------------------------------------
# Configuration (mirrors generate_results.py)
# ---------------------------------------------------------------------------

def build_config():
    conf_dict = {
        "name": "8gpu_8lora_128metalora_lr5e-5_grouppretrain_1150",
        "mode": "train",
        "resume_global_step": -1,
        "test_global_step": "final",
        "run": {
            "seed": 42,
            "use_amp": False,
            "gradient_accumulation_steps": 4,
            "device": "cuda",
            "use_gradient_checkpoint": False,
        },
        "paths": {
            "model_path": "./models/Qwen3-8B",
        },
        "data": {
            "context_max_length": 1024,
            "conversation_max_length": 1024,
            "train_batch_size": 1,
            "eval_batch_size": 1,
            "num_workers": 4,
            "source": "squad",
        },
        "model": {
            "lora_r": 8,
            "metalora_r": 128,
            "ift_additional_metalora_r": -1,
            "num_mem_token": 4,
            "metamodel_class_path": "LoraQwen.LoraQwen3ForCausalLM",
            "config_class_path": "LoraQwen.Qwen3Config",
            "tokenizer_from": "./models/Qwen3-8B",
            "model_from": "./models/Qwen3-8B",
        },
        "metanetwork": {
            "type": "transformer",
            "method": "rl",
            "transformer_cfg": {
                "encoder_cfg": {
                    "d_model": 4096,
                    "nhead": 32,
                    "dim_feedforward": 8192,
                    "dropout": 0,
                    "activation": "gelu",
                    "layer_norm_eps": 0.00001,
                    "batch_first": True,
                    "norm_first": False,
                    "bias": True,
                },
                "couple_encoder_cfg": {
                    "d_model": 4096,
                    "nhead": 32,
                    "dim_feedforward": 8192,
                    "dropout": 0,
                    "activation": "gelu",
                    "layer_norm_eps": 0.00001,
                    "batch_first": True,
                    "norm_first": False,
                    "bias": True,
                },
                "layer_transformer_first": True,
                "mean_pool_size": 1,
                "num_layers": 4,
                "couple_num_layers": 0,
                "scale": 0.001,
            },
        },
        "optim": {
            "adapter_reg": 0.0,
        },
        "test": {
            "context_max_length": 1550,
            "conversation_max_length": 5000,
            "max_new_tokens": 500,
        },
        "hidden_size": -1,
        "num_layers": -1,
        "num_mem_token": 4,
    }
    return OmegaConf.create(conf_dict)


# ---------------------------------------------------------------------------
# Model initialisation (mirrors generate_results.py)
# ---------------------------------------------------------------------------

def init_model(cfg, device):
    logger.info("Loading model classes...")
    MetaModelCls = _import_class(cfg.model.metamodel_class_path)
    ConfigCls = _import_class(cfg.model.config_class_path)

    config = ConfigCls.from_pretrained(cfg.model.model_from)
    config.num_mem_token = -1
    cfg.hidden_size = config.hidden_size
    cfg.num_layers = config.num_hidden_layers

    logger.info("Computing num_mem_token via tmp model...")
    tmp_model = MetaModelCls.from_pretrained(cfg.model.model_from, config=config)
    lora_params = tmp_model.lora_params_numel(cfg.model.lora_r)
    base_params = cfg.hidden_size * cfg.num_layers
    assert lora_params % base_params == 0
    config.num_mem_token = lora_params // base_params
    cfg.num_mem_token = config.num_mem_token
    del tmp_model
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"num_mem_token = {config.num_mem_token}")

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.tokenizer_from, padding_side="left", use_fast=True
    )
    tokenizer.add_tokens(["<RECON>", "<COMP>", "<NOTHING>"])
    tokenizer.chat_template = CHAT_TEMPLATE

    logger.info("Loading metamodel...")
    metamodel = MetaModelCls.from_pretrained(cfg.model.model_from, config=config)
    metamodel.reset_mem_tokens()
    metamodel.resize_token_embeddings(len(tokenizer))

    logger.info("Initialising metanetwork...")
    metanetwork = Metanetwork(metamodel, cfg, metamodel.lora_params_numel(cfg.model.lora_r))
    metanetwork.to(device)
    freeze(metamodel)

    return metanetwork, tokenizer


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_data(path):
    """Load test.jsonl and group by system_prompt_id."""
    groups = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            groups[entry["system_prompt_id"]].append(entry)
    return dict(groups)


# ---------------------------------------------------------------------------
# LoRA dict utilities
# ---------------------------------------------------------------------------

def loradict_to_cpu(loradict):
    """Recursively move all tensors in a nested loradict to CPU."""
    if loradict is None:
        return None
    if isinstance(loradict, dict):
        return {k: loradict_to_cpu(v) for k, v in loradict.items()}
    if torch.is_tensor(loradict):
        return loradict.detach().cpu().clone()
    return loradict


def loradict_to_device(loradict, device):
    """Recursively move all tensors in a nested loradict to device."""
    if loradict is None:
        return None
    if isinstance(loradict, dict):
        return {k: loradict_to_device(v, device) for k, v in loradict.items()}
    if torch.is_tensor(loradict):
        return loradict.to(device)
    return loradict


# ---------------------------------------------------------------------------
# IFT input construction (adapted from finetune_lora.build_recon_inputs)
# ---------------------------------------------------------------------------

def build_ift_inputs(tokenizer, user_message, response, device, max_length=1024):
    """
    Build (input_ids, labels, attention_mask) for IFT loss on a single QA pair.

    Labels mask everything except the assistant response tokens with -100.
    Uses the same backward-scan masking as IFTCollator / build_recon_inputs.

    Returns:
        input_ids      : [1, seq_len]
        labels         : [1, seq_len]  (-100 outside response tokens)
        attention_mask  : [1, seq_len]
    """
    messages = [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": response},
    ]
    enc = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=True,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        return_dict=True,
        padding=False,
        enable_thinking=False,
    )
    input_ids = enc["input_ids"]              # [1, seq_len]
    attention_mask = enc["attention_mask"]     # [1, seq_len]
    labels = input_ids.clone()

    # Token IDs for masking
    assistant_token_id = tokenizer.convert_tokens_to_ids("assistant")
    imstart_token_id   = tokenizer.convert_tokens_to_ids("<|im_start|>")
    imend_token_id     = tokenizer.convert_tokens_to_ids("<|im_end|>")

    mask = torch.zeros_like(labels)   # 1 = supervise, 0 = ignore
    ids = labels[0]
    last_imend = labels.shape[1]

    for j in range(len(ids) - 1, 0, -1):
        if ids[j].item() == imend_token_id:
            last_imend = j
        elif ids[j].item() == assistant_token_id and ids[j - 1].item() == imstart_token_id:
            # Sanity-check: position j+1 must be exactly one \n token
            nl_ids = tokenizer.encode('\n', add_special_tokens=False)
            assert (
                len(nl_ids) == 1 and ids[j + 1].item() == nl_ids[0]
            ), (
                f"Expected '\\n' token at j+1={j+1}, got token_id={ids[j+1].item()}. "
                f"Label masking offset may be wrong for this tokenizer/template."
            )
            # j+2 skips "<|im_start|>assistant\n"
            mask[0, j + 2 : last_imend + 2] = 1
            break

    labels = labels.masked_fill(mask == 0, -100)

    supervised = (labels[0] != -100).sum().item()
    if supervised == 0:
        logger.warning(
            f"build_ift_inputs: 0 supervised tokens! "
            f"user_message={user_message[:80]!r}, response={response[:80]!r}"
        )

    return input_ids.to(device), labels.to(device), attention_mask.to(device)


# ---------------------------------------------------------------------------
# Eval loss computation
# ---------------------------------------------------------------------------

def compute_eval_loss(model, tokenizer, lora_dict, eval_qa_pairs, device):
    """Compute average cross-entropy loss on eval QA pairs with given LoRA."""
    total_loss = 0.0
    with torch.no_grad():
        for user_msg, response in eval_qa_pairs:
            input_ids, labels, attn_mask = build_ift_inputs(
                tokenizer, user_msg, response, device
            )
            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                labels=labels,
                loradict=lora_dict,
                ignore_mem_token=True,
            )
            if outputs.loss is not None and not outputs.loss.isnan():
                total_loss += outputs.loss.item()
            else:
                logger.warning("compute_eval_loss: got None/NaN loss, treating as 0")
    return total_loss / len(eval_qa_pairs)


# ---------------------------------------------------------------------------
# Fine-tuning loop
# ---------------------------------------------------------------------------

def finetune_lora(model, tokenizer, lora_dict_init, train_qa_pairs, eval_qa_pairs,
                  device, num_steps=10, lr=5e-4):
    """
    Fine-tune a LoRA for num_steps Adam steps, measuring eval loss at every step.

    Returns:
        list of (step, eval_loss, train_loss) where train_loss is None at step 0
    """
    # Clone + detach + requires_grad
    lora_params = clone_loradict_to_params(lora_dict_init)
    flat_params = list(iter_learnable_tensors(lora_params))
    optimizer = torch.optim.Adam(flat_params, lr=lr)

    model.eval()

    # Step 0: eval before training
    loss_curve = []
    eval_loss = compute_eval_loss(model, tokenizer, lora_params, eval_qa_pairs, device)
    loss_curve.append((0, eval_loss, None))

    for step in range(1, num_steps + 1):
        idx = (step - 1) % len(train_qa_pairs)
        user_msg, response = train_qa_pairs[idx]

        optimizer.zero_grad()
        input_ids, labels, attn_mask = build_ift_inputs(
            tokenizer, user_msg, response, device
        )

        with torch.enable_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                labels=labels,
                loradict=lora_params,
                ignore_mem_token=True,
            )
            loss = outputs.loss
            if loss is None:
                raise RuntimeError(
                    f"step={step}: model returned None loss. "
                    f"Check that labels have non-(-100) tokens."
                )
            loss.backward()

        # Gradient flow sanity check on first step
        if step == 1 and flat_params[0].grad is None:
            raise RuntimeError(
                "Gradient flow broken: flat_params[0].grad is None after backward(). "
                "The model may be detaching LoRA tensors internally."
            )

        train_loss = loss.item()
        torch.nn.utils.clip_grad_norm_(flat_params, max_norm=1.0)
        optimizer.step()

        eval_loss = compute_eval_loss(model, tokenizer, lora_params, eval_qa_pairs, device)
        loss_curve.append((step, eval_loss, train_loss))

    return loss_curve


# ---------------------------------------------------------------------------
# LoRA generation for all prompts under one checkpoint
# ---------------------------------------------------------------------------

def generate_all_loras(metanetwork, metalora, tokenizer, prompt_groups, device,
                       context_max_length, condition_name):
    """Generate LoRA dicts for all system prompts, return as CPU dict."""
    metanetwork.eval()
    loras = {}
    sp_ids = list(prompt_groups.keys())

    for sp_id in tqdm(sp_ids, desc=f"Generating {condition_name} LoRAs"):
        entries = prompt_groups[sp_id]
        system_prompt = entries[0]["system_prompt"]

        evidence_enc = tokenizer(
            [system_prompt],
            max_length=context_max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        evidence_ids = evidence_enc["input_ids"].to(device)
        evidence_mask = evidence_enc["attention_mask"].to(device)

        with torch.no_grad():
            lora = metanetwork.generate_lora_dict(evidence_ids, evidence_mask, metalora)

        loras[sp_id] = loradict_to_cpu(lora)

    return loras


# ---------------------------------------------------------------------------
# Intermediate save
# ---------------------------------------------------------------------------

def save_results(results, skipped, metadata, output_path):
    """Save results to JSON."""
    # Aggregate statistics
    agg = {"mean_auc": {}, "median_auc": {}, "mean_initial_loss": {}, "mean_final_loss": {}}
    for condition in ["sysprompt_shine", "regular_shine", "random_init"]:
        aucs = [r["auc"][condition] for r in results if condition in r["auc"]]
        initial = [r["curves"][condition][0][1] for r in results if condition in r["curves"]]
        final = [r["curves"][condition][-1][1] for r in results if condition in r["curves"]]
        if aucs:
            agg["mean_auc"][condition] = sum(aucs) / len(aucs)
            agg["median_auc"][condition] = median(aucs)
        if initial:
            agg["mean_initial_loss"][condition] = sum(initial) / len(initial)
        if final:
            agg["mean_final_loss"][condition] = sum(final) / len(final)

    output = {
        "metadata": metadata,
        "per_prompt": results,
        "aggregate": agg,
        "skipped_prompts": skipped,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved results to {output_path}")


# ---------------------------------------------------------------------------
# LR sweep
# ---------------------------------------------------------------------------

def pick_sweep_prompts(prompt_groups):
    """Pick 3 diverse prompts for LR sweep (one each from different categories)."""
    by_category = defaultdict(list)
    for sp_id, entries in prompt_groups.items():
        cat = entries[0]["category"]
        by_category[cat].append(sp_id)

    # Prefer these categories for diversity
    preferred = ["persona", "simulation", "composite", "educational", "professional"]
    selected = []
    used_cats = set()
    for cat in preferred:
        if cat in by_category and cat not in used_cats:
            selected.append(by_category[cat][0])
            used_cats.add(cat)
            if len(selected) == 3:
                break

    # Fill remaining from any category
    if len(selected) < 3:
        for cat, sps in by_category.items():
            if cat not in used_cats:
                selected.append(sps[0])
                used_cats.add(cat)
                if len(selected) == 3:
                    break

    return selected


def run_lr_sweep(model, tokenizer, prompt_groups, all_loras, device, args):
    """Run LR sweep on 3 prompts with 3 learning rates."""
    sweep_sps = pick_sweep_prompts(prompt_groups)
    lrs = [1e-4, 5e-4, 1e-3]
    conditions = ["sysprompt_shine", "regular_shine", "random_init"]

    logger.info(f"LR Sweep: prompts={sweep_sps}, lrs={lrs}")

    sweep_results = {}  # lr -> condition -> list of AUCs

    for lr in lrs:
        sweep_results[lr] = {c: [] for c in conditions}

        for sp_id in sweep_sps:
            entries = prompt_groups[sp_id]
            qa_pairs = [(e["user_message"], e["response"]) for e in entries]
            train_qa = qa_pairs[:5]
            eval_qa = qa_pairs[5:10]

            for condition in conditions:
                lora_init = loradict_to_device(all_loras[(sp_id, condition)], device)
                curve = finetune_lora(
                    model, tokenizer, lora_init, train_qa, eval_qa,
                    device, num_steps=args.num_steps, lr=lr,
                )
                eval_losses = [el for _, el, _ in curve]
                auc = sum(
                    (eval_losses[i] + eval_losses[i + 1]) / 2.0
                    for i in range(len(eval_losses) - 1)
                )
                sweep_results[lr][condition].append(auc)

                logger.info(
                    f"  lr={lr}, {sp_id}, {condition}: "
                    f"AUC={auc:.2f}, init_loss={eval_losses[0]:.3f}, "
                    f"final_loss={eval_losses[-1]:.3f}"
                )

            torch.cuda.empty_cache()

    # Print summary
    print("\n" + "=" * 70)
    print("LR Sweep Results (3 prompts x 3 LRs):")
    print("=" * 70)
    for lr in lrs:
        parts = []
        for c in conditions:
            vals = sweep_results[lr][c]
            mean_auc = sum(vals) / len(vals) if vals else float("nan")
            parts.append(f"{c}={mean_auc:.2f}")
        print(f"  lr={lr}: mean AUC {', '.join(parts)}")
    print("=" * 70)

    # Save sweep results
    sweep_path = os.path.join(os.path.dirname(args.output), "stc_lr_sweep.json")
    os.makedirs(os.path.dirname(sweep_path) or ".", exist_ok=True)
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump(
            {str(lr): {c: vals for c, vals in conds.items()}
             for lr, conds in sweep_results.items()},
            f, indent=2,
        )
    logger.info(f"Sweep results saved to {sweep_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ckpt_base = "checkpoints/8gpu_8lora_128metalora_lr5e-5_grouppretrain_1150"
    p = argparse.ArgumentParser(description="STC evaluation for SHINE LoRA initializations")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--lr-sweep", action="store_true")
    p.add_argument("--output", default="experiments/eval/stc_results.json")
    p.add_argument("--regular-ckpt", default=os.path.join(ckpt_base, "train", "checkpoint-epoch-1"))
    p.add_argument("--sysprompt-ckpt", default=os.path.join(ckpt_base, "iftpwc", "final"))
    p.add_argument("--test-data", default="data/system_prompts/test.jsonl")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = build_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    # Validate checkpoint paths
    for label, path in [("Regular SHINE", args.regular_ckpt),
                        ("SysPrompt SHINE", args.sysprompt_ckpt)]:
        if not os.path.isdir(path):
            logger.error(f"{label} checkpoint not found: {path}")
            sys.exit(1)

    # Load test data
    prompt_groups = load_test_data(args.test_data)
    total_entries = sum(len(v) for v in prompt_groups.values())
    logger.info(f"Loaded {total_entries} entries across {len(prompt_groups)} system prompts")

    # Initialise model
    metanetwork, tokenizer = init_model(cfg, device)

    # ================================================================
    # Phase 1: Generate all LoRAs (2 checkpoint loads total)
    # ================================================================

    all_loras = {}   # (sp_id, condition) -> CPU loradict

    # --- Regular SHINE ---
    logger.info("=" * 60)
    logger.info(f"Loading Regular SHINE checkpoint: {args.regular_ckpt}")
    logger.info("=" * 60)
    metanetwork, metalora_regular, _ = load_checkpoint(
        metanetwork, args.regular_ckpt, device
    )
    regular_loras = generate_all_loras(
        metanetwork, metalora_regular, tokenizer, prompt_groups, device,
        cfg.test.context_max_length, "Regular SHINE",
    )
    for sp_id, lora in regular_loras.items():
        all_loras[(sp_id, "regular_shine")] = lora
    del metalora_regular, regular_loras
    gc.collect()
    torch.cuda.empty_cache()

    # --- SysPrompt SHINE ---
    logger.info("=" * 60)
    logger.info(f"Loading SysPrompt SHINE checkpoint: {args.sysprompt_ckpt}")
    logger.info("=" * 60)
    has_ift_file = os.path.isfile(
        os.path.join(args.sysprompt_ckpt, "ift_additional_metalora.pth")
    )
    metanetwork, metalora_sysprompt, ift_meta = load_checkpoint(
        metanetwork, args.sysprompt_ckpt, device,
        load_ift_additional_metalora=has_ift_file,
    )
    if ift_meta is not None:
        metalora_sysprompt = merge_loradicts(metalora_sysprompt, ift_meta)
        logger.info("Merged ift_additional_metalora into metalora.")
        del ift_meta
    sysprompt_loras = generate_all_loras(
        metanetwork, metalora_sysprompt, tokenizer, prompt_groups, device,
        cfg.test.context_max_length, "SysPrompt SHINE",
    )
    for sp_id, lora in sysprompt_loras.items():
        all_loras[(sp_id, "sysprompt_shine")] = lora
    del metalora_sysprompt, sysprompt_loras
    gc.collect()
    torch.cuda.empty_cache()

    # --- Random Init ---
    logger.info("Generating random-init LoRAs...")
    for sp_id in prompt_groups:
        template = all_loras[(sp_id, "regular_shine")]
        all_loras[(sp_id, "random_init")] = random_init_loradict(template, scale=0.001)

    logger.info(f"Generated {len(all_loras)} LoRAs total "
                f"({len(prompt_groups)} prompts x 3 conditions)")

    # Get the base model for forward passes
    model = metanetwork.metamodel
    model.eval()

    # ================================================================
    # LR Sweep mode
    # ================================================================

    if args.lr_sweep:
        run_lr_sweep(model, tokenizer, prompt_groups, all_loras, device, args)
        return

    # ================================================================
    # Phase 2: Fine-tuning loops (full evaluation)
    # ================================================================

    metadata = {
        "num_steps": args.num_steps,
        "optimizer": "Adam",
        "lr": args.lr,
        "grad_clip_norm": 1.0,
        "train_split_size": "first 5 of 10",
        "eval_split_size": "next 5 of 10",
        "timestamp": datetime.now().isoformat(),
        "sysprompt_checkpoint": args.sysprompt_ckpt,
        "regular_checkpoint": args.regular_ckpt,
    }

    results = []
    skipped = []
    conditions = ["sysprompt_shine", "regular_shine", "random_init"]

    sp_ids = list(prompt_groups.keys())
    for sp_idx, sp_id in enumerate(tqdm(sp_ids, desc="STC Eval")):
        entries = prompt_groups[sp_id]
        qa_pairs = [(e["user_message"], e["response"]) for e in entries]
        n = len(qa_pairs)

        if n < 4:
            logger.warning(f"Skipping {sp_id}: only {n} QA pairs")
            skipped.append(sp_id)
            continue

        if n >= 10:
            n_train, n_eval = 5, 5
        else:
            n_train = n // 2
            n_eval = n - n_train

        train_qa = qa_pairs[:n_train]
        eval_qa = qa_pairs[n_train:n_train + n_eval]

        prompt_result = {
            "system_prompt_id": sp_id,
            "system_prompt": entries[0]["system_prompt"],
            "category": entries[0]["category"],
            "num_train": n_train,
            "num_eval": n_eval,
            "curves": {},
            "auc": {},
        }

        try:
            for condition in conditions:
                lora_init = loradict_to_device(
                    all_loras[(sp_id, condition)], device
                )
                curve = finetune_lora(
                    model, tokenizer, lora_init, train_qa, eval_qa,
                    device, num_steps=args.num_steps, lr=args.lr,
                )

                # AUC via trapezoidal rule on eval losses
                eval_losses = [el for _, el, _ in curve]
                auc = sum(
                    (eval_losses[i] + eval_losses[i + 1]) / 2.0
                    for i in range(len(eval_losses) - 1)
                )

                prompt_result["curves"][condition] = curve
                prompt_result["auc"][condition] = auc

            logger.info(
                f"[{sp_id} / {entries[0]['category']}] AUC: "
                f"sysprompt={prompt_result['auc']['sysprompt_shine']:.2f}, "
                f"regular={prompt_result['auc']['regular_shine']:.2f}, "
                f"random={prompt_result['auc']['random_init']:.2f}"
            )
            results.append(prompt_result)

        except Exception as e:
            logger.error(f"Error on {sp_id}: {e}", exc_info=True)
            skipped.append(sp_id)

        # Intermediate save every 5 prompts
        if len(results) > 0 and len(results) % 5 == 0:
            save_results(results, skipped, metadata, args.output)

        torch.cuda.empty_cache()

    # Final save
    save_results(results, skipped, metadata, args.output)

    # Print aggregate summary
    print("\n" + "=" * 70)
    print("STC Evaluation Complete")
    print("=" * 70)
    print(f"  Prompts evaluated: {len(results)}")
    print(f"  Prompts skipped:   {len(skipped)}")
    for condition in conditions:
        aucs = [r["auc"][condition] for r in results]
        init_losses = [r["curves"][condition][0][1] for r in results]
        final_losses = [r["curves"][condition][-1][1] for r in results]
        print(f"\n  {condition}:")
        print(f"    Mean AUC:        {sum(aucs)/len(aucs):.2f}")
        print(f"    Median AUC:      {median(aucs):.2f}")
        print(f"    Mean init loss:  {sum(init_losses)/len(init_losses):.3f}")
        print(f"    Mean final loss: {sum(final_losses)/len(final_losses):.3f}")
    print("=" * 70)
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
