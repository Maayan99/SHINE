"""
Generate evaluation results for System Prompt SHINE.

Runs inference on data/system_prompts/test.jsonl across 4 conditions:
  1. regular_shine   — Original SHINE checkpoint (train/checkpoint-epoch-1)
  2. sysprompt_shine  — Our system-prompt-finetuned checkpoint (iftpwc/final)
  3. in_context       — System prompt as role:system message, no LoRA
  4. only_question    — No system prompt, no LoRA

Usage:
    cd /workspace/SHINE
    conda activate shine
    python experiments/eval/generate_results.py
"""

import os
import sys
import gc
import json
import re
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from omegaconf import OmegaConf

from metanetwork_family import Metanetwork
from utils.myseed import set_seed
from utils.mysaveload import load_checkpoint
from utils.myfreeze import freeze
from utils.myinit import _import_class

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("eval")

# ---------------------------------------------------------------------------
# Chat template (from inference.ipynb — handles enable_thinking=False)
# ---------------------------------------------------------------------------
CHAT_TEMPLATE = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if (loop.last or (not loop.last and reasoning_content)) and (enable_thinking is not defined or enable_thinking != false) %}\n                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n            {%- else %}\n                {{- '<|im_start|>' + message.role + '\\n' + content }}\n            {%- endif %}\n        {%- else %}\n            {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is not defined or enable_thinking != false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_think_and_answer(text: str) -> Tuple[str, str]:
    """Splits model output into (think_part, answer_part).
    Copied from test_pwc.py:87."""
    lower = text.lower()
    has_start = "<think>" in lower
    has_end = "</think>" in lower

    if has_start != has_end:
        if text.startswith("<think>\n") or text.startswith("<think>\n\n"):
            return "", text[len("<think>\n"):].strip()
        return "[error]", text

    if not has_start and not has_end:
        answer = text.strip()
        answer = re.sub(r"^(final answer|answer)\s*:\s*", "", answer, flags=re.IGNORECASE).strip()
        return "", answer

    start = lower.find("<think>")
    end = lower.find("</think>")

    if end < start:
        return "[error]", text

    think = text[start + len("<think>"): end].strip()
    answer = text[end + len("</think>"):].strip()
    answer = re.sub(r"^(final answer|answer)\s*:\s*", "", answer, flags=re.IGNORECASE).strip()

    return think, answer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def build_config(args) -> "DictConfig":
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
            "max_new_tokens": args.max_new_tokens,
        },
        "hidden_size": -1,
        "num_layers": -1,
        "num_mem_token": 4,
    }
    return OmegaConf.create(conf_dict)


# ---------------------------------------------------------------------------
# Model initialisation (follows inference.ipynb exactly)
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

def load_test_data(path: str) -> Dict[str, List[Dict]]:
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
# Single-turn generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_response(
    metamodel,
    tokenizer,
    messages: List[Dict[str, str]],
    device: torch.device,
    lora_dict=None,
    max_new_tokens: int = 500,
    max_length: int = 5000,
) -> str:
    """Generate a single response given a list of chat messages."""
    input_enc = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        return_dict=True,
        padding=False,
        enable_thinking=False,
    )
    input_ids = input_enc["input_ids"].to(device)
    attention_mask = input_enc["attention_mask"].to(device)

    outputs = metamodel.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
        ignore_mem_token=True,
        loradict=lora_dict,
    )

    new_tokens = outputs[0, input_ids.shape[1]:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    _, answer = extract_think_and_answer(raw_text)
    return answer


# ---------------------------------------------------------------------------
# Per-condition runners
# ---------------------------------------------------------------------------

def run_only_question(metanetwork, tokenizer, grouped_data, device, cfg):
    """Condition 4: just the user question, no system prompt, no LoRA."""
    metanetwork.eval()
    results = {}  # sp_id -> {entry_id: answer}
    failures = 0

    for sp_id, entries in tqdm(grouped_data.items(), desc="Only-Question"):
        results[sp_id] = {}
        for entry in entries:
            messages = [{"role": "user", "content": entry["user_message"]}]
            try:
                answer = generate_response(
                    metanetwork.metamodel, tokenizer, messages, device,
                    max_new_tokens=cfg.test.max_new_tokens,
                    max_length=cfg.test.conversation_max_length,
                )
                results[sp_id][entry["id"]] = answer
            except Exception as e:
                logger.warning(f"[Only-Question] Error on {entry['id']}: {e}")
                torch.cuda.empty_cache()
                results[sp_id][entry["id"]] = f"[ERROR: {e}]"
                failures += 1

    logger.info(f"Only-Question done. Failures: {failures}")
    return results


def run_in_context(metanetwork, tokenizer, grouped_data, device, cfg):
    """Condition 3: system prompt as role:system message, no LoRA."""
    metanetwork.eval()
    results = {}
    failures = 0

    for sp_id, entries in tqdm(grouped_data.items(), desc="In-Context"):
        system_prompt = entries[0]["system_prompt"]
        results[sp_id] = {}
        for entry in entries:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": entry["user_message"]},
            ]
            try:
                answer = generate_response(
                    metanetwork.metamodel, tokenizer, messages, device,
                    max_new_tokens=cfg.test.max_new_tokens,
                    max_length=cfg.test.conversation_max_length,
                )
                results[sp_id][entry["id"]] = answer
            except Exception as e:
                logger.warning(f"[In-Context] Error on {entry['id']}: {e}")
                torch.cuda.empty_cache()
                results[sp_id][entry["id"]] = f"[ERROR: {e}]"
                failures += 1

    logger.info(f"In-Context done. Failures: {failures}")
    return results


def run_shine(metanetwork, metalora, tokenizer, grouped_data, device, cfg,
              condition_name: str):
    """SHINE condition: generate LoRA from system_prompt, answer with LoRA."""
    metanetwork.eval()
    results = {}
    failures = 0

    sp_ids = list(grouped_data.keys())
    for idx, sp_id in enumerate(tqdm(sp_ids, desc=condition_name)):
        entries = grouped_data[sp_id]
        system_prompt = entries[0]["system_prompt"]

        # Generate LoRA from the system prompt (once per group)
        evidence_enc = tokenizer(
            [system_prompt],
            max_length=cfg.test.context_max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        evidence_ids = evidence_enc["input_ids"].to(device)
        evidence_mask = evidence_enc["attention_mask"].to(device)
        lora_dict = metanetwork.generate_lora_dict(evidence_ids, evidence_mask, metalora)

        results[sp_id] = {}
        for entry in entries:
            messages = [{"role": "user", "content": entry["user_message"]}]
            try:
                answer = generate_response(
                    metanetwork.metamodel, tokenizer, messages, device,
                    lora_dict=lora_dict,
                    max_new_tokens=cfg.test.max_new_tokens,
                    max_length=cfg.test.conversation_max_length,
                )
                results[sp_id][entry["id"]] = answer
            except Exception as e:
                logger.warning(f"[{condition_name}] Error on {entry['id']}: {e}")
                torch.cuda.empty_cache()
                results[sp_id][entry["id"]] = f"[ERROR: {e}]"
                failures += 1

    logger.info(f"{condition_name} done. Failures: {failures}")
    return results


# ---------------------------------------------------------------------------
# Assemble output
# ---------------------------------------------------------------------------

def assemble_output(
    grouped_data: Dict[str, List[Dict]],
    regular_shine: Dict,
    sysprompt_shine: Dict,
    in_context: Dict,
    only_question: Dict,
    args,
) -> Dict:
    results_list = []
    for sp_id, entries in grouped_data.items():
        system_prompt = entries[0]["system_prompt"]
        category = entries[0]["category"]
        questions = []
        for entry in entries:
            eid = entry["id"]
            questions.append({
                "id": eid,
                "question": entry["user_message"],
                "ground_truth": entry["response"],
                "responses": {
                    "regular_shine": regular_shine.get(sp_id, {}).get(eid, ""),
                    "sysprompt_shine": sysprompt_shine.get(sp_id, {}).get(eid, ""),
                    "in_context": in_context.get(sp_id, {}).get(eid, ""),
                    "only_question": only_question.get(sp_id, {}).get(eid, ""),
                },
            })
        results_list.append({
            "system_prompt_id": sp_id,
            "system_prompt": system_prompt,
            "category": category,
            "questions": questions,
        })

    return {
        "metadata": {
            "regular_shine_checkpoint": args.regular_ckpt,
            "sysprompt_shine_checkpoint": args.sysprompt_ckpt,
            "model": "Qwen3-8B",
            "max_new_tokens": args.max_new_tokens,
            "timestamp": datetime.now().isoformat(),
        },
        "results": results_list,
    }


# ---------------------------------------------------------------------------
# Intermediate save
# ---------------------------------------------------------------------------

def save_partial(output_dir: str, condition_name: str, results: Dict):
    """Save per-condition intermediate results."""
    path = os.path.join(output_dir, f"partial_{condition_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved partial results to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate evaluation results for System Prompt SHINE")
    ckpt_base = "checkpoints/8gpu_8lora_128metalora_lr5e-5_grouppretrain_1150"
    p.add_argument("--output", default="experiments/eval/results.json")
    p.add_argument("--regular-ckpt", default=os.path.join(ckpt_base, "train", "checkpoint-epoch-1"))
    p.add_argument("--sysprompt-ckpt", default=os.path.join(ckpt_base, "iftpwc", "final"))
    p.add_argument("--max-new-tokens", type=int, default=500)
    p.add_argument("--test-data", default="data/system_prompts/test.jsonl")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = build_config(args)
    device = torch.device("cuda")
    set_seed(42)

    # Validate checkpoint paths
    for label, path in [("Regular SHINE", args.regular_ckpt), ("SysPrompt SHINE", args.sysprompt_ckpt)]:
        if not os.path.isdir(path):
            logger.error(f"{label} checkpoint not found: {path}")
            sys.exit(1)

    # Load test data
    grouped_data = load_test_data(args.test_data)
    total_entries = sum(len(v) for v in grouped_data.values())
    logger.info(f"Loaded {total_entries} entries across {len(grouped_data)} system prompts")

    # Initialise model
    metanetwork, tokenizer = init_model(cfg, device)

    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)

    # --- Condition 1: Only-Question (no checkpoint needed) ---
    logger.info("=" * 60)
    logger.info("Condition: Only-Question")
    logger.info("=" * 60)
    only_question_results = run_only_question(metanetwork, tokenizer, grouped_data, device, cfg)
    save_partial(output_dir, "only_question", only_question_results)

    # --- Condition 2: In-Context (no checkpoint needed) ---
    logger.info("=" * 60)
    logger.info("Condition: In-Context")
    logger.info("=" * 60)
    in_context_results = run_in_context(metanetwork, tokenizer, grouped_data, device, cfg)
    save_partial(output_dir, "in_context", in_context_results)

    # --- Condition 3: Regular SHINE ---
    logger.info("=" * 60)
    logger.info(f"Condition: Regular SHINE — loading {args.regular_ckpt}")
    logger.info("=" * 60)
    metanetwork, metalora_regular, _ = load_checkpoint(metanetwork, args.regular_ckpt, device)
    gc.collect()
    torch.cuda.empty_cache()
    regular_shine_results = run_shine(
        metanetwork, metalora_regular, tokenizer, grouped_data, device, cfg,
        condition_name="Regular SHINE",
    )
    save_partial(output_dir, "regular_shine", regular_shine_results)

    # --- Condition 4: SysPrompt SHINE ---
    logger.info("=" * 60)
    logger.info(f"Condition: SysPrompt SHINE — loading {args.sysprompt_ckpt}")
    logger.info("=" * 60)
    metanetwork, metalora_sysprompt, _ = load_checkpoint(metanetwork, args.sysprompt_ckpt, device)
    gc.collect()
    torch.cuda.empty_cache()
    sysprompt_shine_results = run_shine(
        metanetwork, metalora_sysprompt, tokenizer, grouped_data, device, cfg,
        condition_name="SysPrompt SHINE",
    )
    save_partial(output_dir, "sysprompt_shine", sysprompt_shine_results)

    # --- Assemble and save final output ---
    output = assemble_output(
        grouped_data, regular_shine_results, sysprompt_shine_results,
        in_context_results, only_question_results, args,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Summary
    total_questions = sum(len(r["questions"]) for r in output["results"])
    logger.info("=" * 60)
    logger.info("DONE")
    logger.info(f"  System prompts: {len(output['results'])}")
    logger.info(f"  Total questions: {total_questions}")
    logger.info(f"  Output: {args.output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
