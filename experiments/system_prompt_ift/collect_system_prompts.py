#!/usr/bin/env python3
"""
Step 1.1: Collect and filter system prompts from Awesome ChatGPT Prompts.

Uses Claude Haiku to judge each prompt on two axes:
  1. Is this a valid, reusable system prompt for a text-based LLM?
  2. What category does it belong to?

Filtering criteria (enforced by Haiku):
  - Must be in English
  - Must be a behavioral specification (not a one-shot task)
  - Must be for a text-based LLM (no image/video/audio generation)
  - Must not be raw code, HTML, or JSON that isn't a natural language prompt
  - Must define a persistent behavioral mode across multiple turns
  - Template variables like ${topic} are acceptable (will be filled later)

Output: data/raw/system_prompts.jsonl
  Each line: {"id": "sp_00001", "source": "...", "category": "...", "prompt": "...", "token_count": N}

Usage:
  python collect_system_prompts.py --input prompts.csv --output data/raw/system_prompts.jsonl
  python collect_system_prompts.py --input prompts.csv --dry-run  # preview without API calls
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import anthropic
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

CATEGORIES = [
    "persona",          # roleplay, characters, defined personality
    "professional",     # domain expert (doctor, lawyer, engineer)
    "coding",           # programming style, language, practices
    "formatting",       # output format rules (JSON, markdown, bullet lists, length)
    "behavioral",       # constraints (never do X, always do Y, ask before answering)
    "language",         # respond in French, formal English, simple language
    "tool_usage",       # tool/API calling patterns
    "composite",        # combines 3+ of the above
    "educational",      # tutoring, teaching, explaining
    "simulation",       # act as a terminal, database, interpreter (not a person)
]

CATEGORY_LIST_STR = ", ".join(CATEGORIES)

# ---------------------------------------------------------------------------
# Haiku judge prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """You are a dataset quality filter. You will be given a candidate system prompt and must decide:
1. Whether it is a VALID reusable system prompt for a text-based conversational LLM.
2. If valid, which category it belongs to.

A prompt is INVALID if ANY of these apply:
- It is an image, video, or audio generation instruction
- It is not in English (a few domain-specific non-English terms are fine, but the prompt itself must be primarily English)
- It is raw code, HTML, or structured data that is NOT a natural language behavioral instruction
- It is PURELY a one-shot task with no reusable behavioral definition (e.g., "write me an essay about X")
- It is too vague or generic to meaningfully affect model behavior (e.g., "be helpful")
- It describes a specific project to build rather than defining how an assistant should behave

IMPORTANT: Many prompts end with "my first request is..." or "my first sentence is..." or similar example inputs. This does NOT make them one-shot tasks. If the prompt defines a clear behavioral mode or persona that could handle many different user inputs, it IS valid — ignore the trailing example input.

A prompt IS VALID if:
- It defines how an assistant should behave across multiple conversation turns
- It sets a persona, style, constraints, domain expertise, or output format
- It could be used as a system prompt in an API call where many different user messages would follow
- Template variables like ${topic} are acceptable

Respond with EXACTLY one line of JSON, no markdown fences:
{"valid": true/false, "category": "one of the categories or null if invalid", "reason": "brief 5-10 word reason"}"""

JUDGE_USER_TEMPLATE = """Candidate system prompt:

{prompt}

Categories: {categories}"""


# ---------------------------------------------------------------------------
# Load prompts from CSV
# ---------------------------------------------------------------------------

def load_awesome_csv(path: str) -> list[dict]:
    """Load prompts from the Awesome ChatGPT Prompts CSV."""
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("prompt", "").strip()
            act = row.get("act", "").strip()
            ptype = row.get("type", "").strip()
            if text:
                prompts.append({
                    "source": "awesome-chatgpt-prompts",
                    "act": act,
                    "original_type": ptype,
                    "prompt": text,
                })
    return prompts


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def count_tokens(prompts: list[dict], tokenizer_path: str) -> list[dict]:
    """Add token_count field to each prompt."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    for p in prompts:
        p["token_count"] = len(tokenizer.encode(p["prompt"], add_special_tokens=False))
    return prompts


# ---------------------------------------------------------------------------
# Pre-filter (cheap, no API calls)
# ---------------------------------------------------------------------------

def pre_filter(prompts: list[dict], min_tokens: int, max_tokens: int) -> list[dict]:
    """Remove obviously bad prompts before spending API calls."""
    filtered = []
    seen = set()

    for p in prompts:
        text = p["prompt"].strip()
        if not text:
            continue

        # Deduplicate
        h = hashlib.md5(text.lower().strip().encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)

        # Token length filter
        tc = p.get("token_count", 0)
        if tc < min_tokens or tc > max_tokens:
            continue

        # Skip IMAGE type from the CSV (obvious non-text prompts)
        if p.get("original_type", "").upper() == "IMAGE":
            continue

        filtered.append(p)

    return filtered


# ---------------------------------------------------------------------------
# Haiku judging
# ---------------------------------------------------------------------------

def judge_prompt(client: anthropic.Anthropic, prompt_text: str, model: str) -> dict:
    """Ask Haiku to judge a single prompt. Returns {"valid": bool, "category": str, "reason": str}."""
    resp = client.messages.create(
        model=model,
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": JUDGE_USER_TEMPLATE.format(
                    prompt=prompt_text[:3000],  # truncate very long prompts for the judge
                    categories=CATEGORY_LIST_STR,
                ),
            }
        ],
        system=JUDGE_SYSTEM,
    )
    text = resp.content[0].text.strip()

    # Parse JSON response
    # Handle potential markdown fences
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    if text.endswith("```"):
        text = text[:text.rfind("```")]
    text = text.strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    result = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        else:
            return {"valid": False, "category": None, "reason": "Failed to parse judge response"}

    # Validate category
    cat = result.get("category")
    if cat and cat not in CATEGORIES:
        # Try to map to closest category
        cat_lower = cat.lower().strip()
        for c in CATEGORIES:
            if c in cat_lower or cat_lower in c:
                result["category"] = c
                break
        else:
            result["category"] = "persona"  # fallback

    return result


def judge_all_prompts(
    client: anthropic.Anthropic,
    prompts: list[dict],
    model: str,
    cache_path: str = None,
) -> list[dict]:
    """Judge all prompts, with optional caching to avoid re-judging on restart."""

    # Load cache if it exists
    cache = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                cache[entry["prompt_hash"]] = entry

    results = []
    cache_f = open(cache_path, "a") if cache_path else None

    try:
        for p in tqdm(prompts, desc="Judging prompts"):
            h = hashlib.md5(p["prompt"].encode()).hexdigest()

            if h in cache:
                judgment = cache[h]
            else:
                try:
                    judgment = judge_prompt(client, p["prompt"], model)
                    judgment["prompt_hash"] = h
                    # Cache the result
                    if cache_f:
                        cache_f.write(json.dumps(judgment, ensure_ascii=False) + "\n")
                        cache_f.flush()
                except Exception as e:
                    print(f"\n  Error judging prompt: {e}")
                    judgment = {"valid": False, "category": None, "reason": f"API error: {e}"}
                    time.sleep(2)  # back off on errors

            if judgment.get("valid"):
                p["category"] = judgment.get("category", "persona")
                p["judge_reason"] = judgment.get("reason", "")
                results.append(p)
            else:
                tqdm.write(
                    f"  REJECTED [{p.get('act', '?')[:40]}]: {judgment.get('reason', '?')}"
                )

            time.sleep(0.05)  # minimal rate-limit courtesy
    finally:
        if cache_f:
            cache_f.close()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 1.1: Collect and filter system prompts")
    parser.add_argument("--input", required=True, help="Path to prompts.csv")
    parser.add_argument("--output", default="data/raw/system_prompts.jsonl")
    parser.add_argument("--tokenizer-path", default="../../models/Qwen3-8B")
    parser.add_argument("--min-tokens", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--judge-model", default="claude-haiku-4-5-20251001",
                        help="Model to use for judging")
    parser.add_argument("--judge-cache", default="data/raw/.judge_cache.jsonl",
                        help="Cache file for judge results (avoids re-judging on restart)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only pre-filter, skip Haiku judging")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if args.judge_cache:
        os.makedirs(os.path.dirname(args.judge_cache) or ".", exist_ok=True)

    # --- Load ---
    print("=== Loading prompts from CSV ===")
    raw = load_awesome_csv(args.input)
    print(f"  Loaded {len(raw)} prompts")

    # --- Token counting ---
    print("\n=== Counting tokens ===")
    raw = count_tokens(raw, args.tokenizer_path)

    # --- Pre-filter ---
    print("\n=== Pre-filtering ===")
    pre = pre_filter(raw, args.min_tokens, args.max_tokens)
    print(f"  {len(raw)} → {len(pre)} after pre-filter")

    if args.dry_run:
        print("\n=== Dry run: skipping Haiku judging ===")
        for i, p in enumerate(pre):
            p["id"] = f"sp_{i:05d}"
            p["category"] = "unclassified"
        filtered = pre
    else:
        # --- Haiku judging ---
        print(f"\n=== Judging {len(pre)} prompts with {args.judge_model} ===")
        client = anthropic.Anthropic()
        filtered = judge_all_prompts(client, pre, args.judge_model, args.judge_cache)
        print(f"  {len(pre)} → {len(filtered)} after Haiku judging")

        # Assign sequential IDs
        for i, p in enumerate(filtered):
            p["id"] = f"sp_{i:05d}"

    # --- Clean up fields before saving ---
    save_fields = ["id", "source", "category", "prompt", "token_count"]
    cleaned = []
    for p in filtered:
        cleaned.append({k: p[k] for k in save_fields if k in p})

    # --- Save ---
    with open(args.output, "w", encoding="utf-8") as f:
        for p in cleaned:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(cleaned)} prompts to {args.output}")

    # --- Report ---
    cats = Counter(p["category"] for p in cleaned)
    print("\nBy category:")
    for cat, cnt in cats.most_common():
        print(f"  {cat}: {cnt}")

    token_counts = [p["token_count"] for p in cleaned]
    if token_counts:
        token_counts.sort()
        n = len(token_counts)
        print(f"\nToken stats: min={token_counts[0]}, "
              f"p25={token_counts[n//4]}, median={token_counts[n//2]}, "
              f"p75={token_counts[3*n//4]}, max={token_counts[-1]}")


if __name__ == "__main__":
    main()