#!/usr/bin/env python3
"""
Step 1.4: Split dataset into train/val/test and convert to SHINE IFT format.

Splits by system prompt (NOT by individual QA pairs) so all pairs from one
system prompt go to the same split. Outputs in the format expected by SHINE's
IFTC1QADataset: JSON array of items with {context, conversations, contextlen,
conversationlen}.

Input:  data/system_prompt_dataset.jsonl   (from Step 1.3)
Output: data/train.json, data/val.json, data/test.json

Usage:
  python split_dataset.py
  python split_dataset.py --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
"""

import argparse
import json
import os
import random
from collections import defaultdict

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Step 1.4: Split dataset by system prompt")
    parser.add_argument("--input", default="data/system_prompt_dataset.jsonl")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--tokenizer-path", default="../../models/Qwen3-8B")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6

    # Load all pairs
    with open(args.input) as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(pairs)} pairs from {args.input}")

    # Group by system prompt ID
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_prompt[p["system_prompt_id"]].append(p)
    prompt_ids = sorted(by_prompt.keys())
    print(f"Unique system prompts: {len(prompt_ids)}")

    # Shuffle and split prompt IDs
    random.seed(args.seed)
    random.shuffle(prompt_ids)

    n = len(prompt_ids)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    train_ids = set(prompt_ids[:n_train])
    val_ids = set(prompt_ids[n_train : n_train + n_val])
    test_ids = set(prompt_ids[n_train + n_val :])

    print(f"Split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    # Load tokenizer for computing token lengths
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)

    def convert_to_shine_format(prompt_groups: dict[str, list[dict]]) -> list[dict]:
        """Convert grouped pairs into SHINE's IFTC1QADataset format.

        Each system prompt becomes one item with:
          - context: the system prompt text (fed to hypernetwork)
          - conversations: all (user, assistant) turns concatenated
          - contextlen: token count of context
          - conversationlen: token count of conversations
        """
        items = []
        for sp_id, group_pairs in sorted(prompt_groups.items()):
            system_prompt = group_pairs[0]["system_prompt"]
            conversations = []
            for p in group_pairs:
                conversations.append({"role": "user", "content": p["user_message"]})
                conversations.append({"role": "assistant", "content": p["response"]})

            # Compute token lengths (matching the notebook's approach)
            context_tokens = tokenizer.encode(system_prompt, add_special_tokens=False)
            conv_tokens = tokenizer.apply_chat_template(
                conversations,
                add_generation_prompt=False,
                tokenize=True,
                enable_thinking=False,
            )

            items.append(
                {
                    "contextlen": len(context_tokens),
                    "conversationlen": len(conv_tokens),
                    "context": system_prompt,
                    "conversations": conversations,
                }
            )
        return items

    # Build splits
    splits = {
        "train": {sid: by_prompt[sid] for sid in train_ids},
        "val": {sid: by_prompt[sid] for sid in val_ids},
        "test": {sid: by_prompt[sid] for sid in test_ids},
    }

    os.makedirs(args.output_dir, exist_ok=True)

    for split_name, prompt_groups in splits.items():
        items = convert_to_shine_format(prompt_groups)
        out_path = os.path.join(args.output_dir, f"{split_name}.json")
        with open(out_path, "w") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)

        total_pairs = sum(len(item["conversations"]) // 2 for item in items)
        avg_ctx = sum(item["contextlen"] for item in items) / max(len(items), 1)
        avg_conv = sum(item["conversationlen"] for item in items) / max(len(items), 1)
        print(
            f"  {split_name}: {len(items)} system prompts, {total_pairs} QA pairs, "
            f"avg context={avg_ctx:.0f} tokens, avg conv={avg_conv:.0f} tokens → {out_path}"
        )

    # Also save the raw test split in a format convenient for evaluation
    test_pairs = []
    for sid in test_ids:
        test_pairs.extend(by_prompt[sid])
    test_pairs_path = os.path.join(args.output_dir, "test_pairs.jsonl")
    with open(test_pairs_path, "w") as f:
        for p in test_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\n  Test pairs (flat): {len(test_pairs)} → {test_pairs_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
