#!/usr/bin/env python3
"""
Step 1.3: Generate ground-truth responses using base Qwen3-8B.

For each (system_prompt, user_message) pair, runs Qwen3-8B with the system
prompt in-context using greedy decoding (batched).

Input:  data/with_messages.jsonl   (from Step 1.2)
Output: data/system_prompt_dataset.jsonl

Usage:
  python generate_responses.py
  python generate_responses.py --batch-size 8 --max-new-tokens 256
  python generate_responses.py --resume
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: str, device: str = "cuda"):
    """Load base Qwen3-8B model and tokenizer."""
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    return model, tokenizer


def generate_responses_batched(
    model,
    tokenizer,
    pairs: list[dict],
    max_new_tokens: int = 512,
    max_input_length: int = 2048,
) -> list[str]:
    """Generate responses for a batch of (system_prompt, user_message) pairs.

    Uses left-padding so all sequences align on the right for batched generation.
    Returns a list of response strings, one per pair.
    """
    # Build chat-templated input texts
    input_texts = []
    for p in pairs:
        messages = [
            {"role": "system", "content": p["system_prompt"]},
            {"role": "user", "content": p["user_message"]},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_texts.append(text)

    # Tokenize with left-padding
    inputs = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    ).to(model.device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode: generated tokens start after the padded input
    input_len = inputs["input_ids"].shape[1]
    responses = []
    for i in range(len(input_texts)):
        gen_tokens = output_ids[i, input_len:]
        response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        responses.append(response)

    return responses


def main():
    parser = argparse.ArgumentParser(description="Step 1.3: Generate ground-truth responses with Qwen3-8B")
    parser.add_argument("--input", default="data/with_messages.jsonl")
    parser.add_argument("--output", default="data/system_prompt_dataset.jsonl")
    parser.add_argument("--model-path", default="../../models/Qwen3-8B")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-input-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true", help="Continue from checkpoint")
    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(args.model_path, args.device)

    # Load input data
    with open(args.input) as f:
        items = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(items)} system prompts with messages from {args.input}")

    # Resume support: load already-processed pair IDs
    done_pair_ids: set[str] = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                if line.strip():
                    done_pair_ids.add(json.loads(line)["id"])
        print(f"Resuming: {len(done_pair_ids)} pairs already processed")

    # Flatten all pairs
    all_pairs = []
    for item in items:
        sp_id = item["id"]
        for q_idx, user_msg in enumerate(item["user_messages"]):
            pair_id = f"{sp_id}_q{q_idx:02d}"
            if pair_id in done_pair_ids:
                continue
            all_pairs.append({
                "pair_id": pair_id,
                "system_prompt_id": sp_id,
                "system_prompt": item["prompt"],
                "user_message": user_msg,
                "category": item.get("category", "general"),
            })

    print(f"Pairs to process: {len(all_pairs)} (skipped {len(done_pair_ids)} already done)")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    mode = "a" if args.resume else "w"
    stats = {"total": 0, "kept": 0, "filtered_empty": 0}

    with open(args.output, mode) as fout:
        for batch_start in tqdm(
            range(0, len(all_pairs), args.batch_size),
            desc="Batches",
            total=(len(all_pairs) + args.batch_size - 1) // args.batch_size,
        ):
            batch = all_pairs[batch_start : batch_start + args.batch_size]
            responses = generate_responses_batched(
                model, tokenizer, batch, args.max_new_tokens, args.max_input_length
            )

            for pair, response in zip(batch, responses):
                stats["total"] += 1

                if not response.strip():
                    stats["filtered_empty"] += 1
                    continue

                entry = {
                    "id": pair["pair_id"],
                    "system_prompt_id": pair["system_prompt_id"],
                    "system_prompt": pair["system_prompt"],
                    "user_message": pair["user_message"],
                    "response": response,
                    "category": pair["category"],
                }
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                stats["kept"] += 1

            fout.flush()

    # Report
    print(f"\n=== Generation complete ===")
    print(f"Total pairs processed: {stats['total']}")
    print(f"Kept: {stats['kept']}")
    print(f"Filtered (empty): {stats['filtered_empty']}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
