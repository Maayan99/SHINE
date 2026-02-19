#!/usr/bin/env python3
"""Sanity check: verify SystemPromptIFTDataset + IFTCollator pipeline.

Loads a few samples, passes them through the collator, and checks that:
  1. evidence_ids is non-empty (system prompt is tokenized)
  2. Some labels are supervised (not all -100)
  3. Supervised tokens decode to the assistant response text

Usage (from SHINE root):
  python experiments/system_prompt_ift/sanity_check.py
  python experiments/system_prompt_ift/sanity_check.py --data-path data/system_prompts/val.jsonl
"""

import argparse
import sys
import os

# Ensure SHINE root is on the path so `utils.*` imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dataclasses import dataclass
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from experiments.system_prompt_ift.system_prompt_dataset import SystemPromptIFTDataset
from utils.mydataset import IFTCollator


# Minimal cfg stub that BaseCollator expects
def make_stub_cfg():
    return OmegaConf.create({})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/system_prompts/train.jsonl")
    parser.add_argument("--tokenizer-path", default="models/Qwen3-8B")
    parser.add_argument("--context-max-len", type=int, default=1150)
    parser.add_argument("--conversation-max-len", type=int, default=1150)
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()

    # Set up tokenizer exactly like meta_train_parallel.py (lines 468-470)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, padding_side="left", use_fast=True)
    tokenizer.add_tokens(["<RECON>", "<COMP>", "<NOTHING>"])
    tokenizer.chat_template = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if (loop.last or (not loop.last and reasoning_content)) and (enable_thinking is not defined or enable_thinking != false) %}\n                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n            {%- else %}\n                {{- '<|im_start|>' + message.role + '\\n' + content }}\n            {%- endif %}\n        {%- else %}\n            {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is not defined or enable_thinking != false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"

    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # Load dataset
    ds = SystemPromptIFTDataset(
        args.data_path,
        max_context_len=args.context_max_len,
        max_conversation_len=args.conversation_max_len,
        use_exceed=True,
    )
    assert len(ds) > 0, f"Dataset is empty! Check {args.data_path}"

    # Build collator
    cfg = make_stub_cfg()
    collator = IFTCollator(
        tokenizer=tokenizer,
        context_max_length=args.context_max_len,
        conversation_max_length=args.conversation_max_len,
        cfg=cfg,
    )

    n = min(args.num_samples, len(ds))
    print(f"\n{'='*80}")
    print(f"Checking {n} samples from {args.data_path}")
    print(f"{'='*80}\n")

    for i in range(n):
        sample = ds[i]
        batch = collator([sample])

        evidence_ids = batch["evidence_ids"][0]
        input_ids = batch["input_ids"][0]
        labels = batch["labels"][0]
        evidence_mask = batch["evidence_attention_mask"][0]

        # Decode evidence (skip padding)
        evidence_nonpad = evidence_ids[evidence_mask.bool()]
        evidence_text = tokenizer.decode(evidence_nonpad, skip_special_tokens=True)

        # Find supervised positions
        supervised_mask = labels != -100
        supervised_ids = input_ids[supervised_mask]
        supervised_text = tokenizer.decode(supervised_ids, skip_special_tokens=True)

        # Original response
        original_response = sample["conversations"][1]["content"]

        print(f"--- Sample {i} ---")
        print(f"Evidence (first 200 chars): {evidence_text[:200]}...")
        print(f"Input decoded (first 300 chars): {tokenizer.decode(input_ids[input_ids != tokenizer.pad_token_id], skip_special_tokens=False)[:300]}...")
        print(f"Supervised tokens (first 200 chars): {supervised_text[:200]}...")
        print(f"Original response (first 200 chars): {original_response[:200]}...")
        print(f"Evidence non-pad tokens: {evidence_nonpad.shape[0]}")
        print(f"Supervised positions: {supervised_mask.sum().item()} / {labels.shape[0]}")
        print()

        # Assertions
        assert evidence_nonpad.shape[0] > 0, f"Sample {i}: evidence_ids is empty!"
        assert supervised_mask.sum().item() > 0, f"Sample {i}: no supervised labels!"
        # Check that supervised text contains the response (allowing for tokenization differences)
        response_words = original_response.split()[:5]
        for word in response_words:
            assert word in supervised_text, (
                f"Sample {i}: response word '{word}' not found in supervised text. "
                f"Supervised: {supervised_text[:200]}"
            )

    print(f"{'='*80}")
    print(f"All {n} samples passed sanity checks!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
