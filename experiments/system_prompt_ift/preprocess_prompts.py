#!/usr/bin/env python3
"""
Preprocess system prompts:
  1. Strip "My first ..." suffixes (baked-in first-turn examples)
  2. Fill ${Var:Default} template variables with their default values
  3. Use Claude Haiku to fill remaining ${var} placeholders with sensible values
  4. Remove prompts that are still broken after all fixes

Input:  data/raw/system_prompts.jsonl  (from Step 1.1)
Output: data/raw/system_prompts_clean.jsonl

Usage:
  python preprocess_prompts.py
  python preprocess_prompts.py --dry-run          # preview changes, no API calls
  python preprocess_prompts.py --skip-haiku       # only do regex fixes, no API
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

import anthropic
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 1. Strip "My first ..." suffixes
# ---------------------------------------------------------------------------

# Matches: "My first sentence is ...", "my first request is ...",
#          "My first command is ...", "The first sentence is ...", etc.
# Captures everything from that phrase to the end of the string.
MY_FIRST_PATTERN = re.compile(
    r"""
    [\.\;\:\!\?\n]\s*                     # sentence boundary before "my first"
    (?:my|the|your)\s+first\s+            # "my first" / "the first" / "your first"
    (?:sentence|request|command|question|  # common nouns
       statement|message|task|prompt|
       input|query|assignment|text|
       word|phrase|problem|topic|
       suggestion|order|instruction|
       search\s+query|thing)
    \s+(?:is|will\s+be|:|—|-)             # "is" / "will be" / colon / dash
    .+$                                    # everything after
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)

# Simpler fallback: catches "My first ... is: <quote>" at end of string
MY_FIRST_SIMPLE = re.compile(
    r'[\.\;\:\!\?\n]\s*[Mm]y first\b.{0,50}(?:is|:).+$',
    re.DOTALL,
)


def strip_my_first(text):
    """Remove trailing 'My first X is ...' from a prompt."""
    original = text

    # Try the detailed pattern first
    match = MY_FIRST_PATTERN.search(text)
    if match:
        # Keep the sentence boundary character, strip the rest
        cut_pos = match.start() + 1  # keep the period/semicolon/etc
        text = text[:cut_pos].rstrip()
        return text, True

    # Fallback to simpler pattern
    match = MY_FIRST_SIMPLE.search(text)
    if match:
        cut_pos = match.start() + 1
        text = text[:cut_pos].rstrip()
        return text, True

    return original, False


# ---------------------------------------------------------------------------
# 2. Fill ${Var:Default} template variables
# ---------------------------------------------------------------------------

# Matches ${VarName:DefaultValue} — captures the default value
TEMPLATE_WITH_DEFAULT = re.compile(r'\$\{[^:}]+:([^}]+)\}')


def fill_template_defaults(text):
    """Replace ${Var:Default} with just Default."""
    new_text = TEMPLATE_WITH_DEFAULT.sub(r'\1', text)
    changed = new_text != text
    return new_text, changed


# ---------------------------------------------------------------------------
# 3. Detect remaining ${var} placeholders
# ---------------------------------------------------------------------------

# Matches ${anything} that was NOT already handled (no default value)
UNFILLED_VAR = re.compile(r'\$\{([^}]+)\}')

# Also catch [INSERT ...], [YOUR ...], {PLACEHOLDER} style
BRACKET_PLACEHOLDER = re.compile(
    r'\[(?:INSERT|YOUR|ENTER|ADD|PASTE|PROVIDE|SPECIFY|INCLUDE)\s+[^\]]+\]',
    re.IGNORECASE,
)
CURLY_PLACEHOLDER = re.compile(r'\{[A-Z_]{2,}(?:_[A-Z_]+)*\}')  # {OBJECT_NAME}, {TOPIC}


def find_placeholders(text):
    """Find all unfilled placeholders in text. Returns list of match strings."""
    found = []
    found.extend(UNFILLED_VAR.findall(text))
    found.extend(BRACKET_PLACEHOLDER.findall(text))
    found.extend(CURLY_PLACEHOLDER.findall(text))
    return found


def has_unfilled_placeholders(text):
    """Check if text still has unfilled template variables."""
    return bool(find_placeholders(text))


# ---------------------------------------------------------------------------
# 4. Haiku-based placeholder filling
# ---------------------------------------------------------------------------

FILL_SYSTEM = """You are a template filler. You will receive a system prompt that contains placeholder variables like ${variableName}, [INSERT X], or {PLACEHOLDER}.

Your job: replace each placeholder with a specific, realistic, concrete value that makes the prompt coherent and usable.

Rules:
- Replace EVERY placeholder with a concrete value
- Choose values that are realistic and specific (not generic)
- Keep the rest of the text EXACTLY as-is — do not rephrase, reformat, or add anything
- Output ONLY the filled prompt text, nothing else — no commentary, no markdown fences
- If a placeholder is clearly meant to be filled by the user at runtime (like "paste your code here"), replace it with a realistic example"""


def fill_with_haiku(client, text, model="claude-haiku-4-5-20251001"):
    """Use Haiku to fill remaining placeholders."""
    placeholders = find_placeholders(text)
    if not placeholders:
        return text, False

    resp = client.messages.create(
        model=model,
        max_tokens=2000,
        system=FILL_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Fill the placeholders in this system prompt:\n\n{text}",
            }
        ],
    )
    filled = resp.content[0].text.strip()

    # Sanity check: Haiku shouldn't have drastically changed the length
    # (allow 50% growth for filled values, but not 3x — that means it rewrote)
    if len(filled) > len(text) * 2.5 or len(filled) < len(text) * 0.3:
        return text, False  # reject, keep original

    # Check it actually reduced placeholders
    remaining = find_placeholders(filled)
    if len(remaining) >= len(placeholders):
        return text, False  # didn't help

    return filled, True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess system prompts")
    parser.add_argument("--input", default="data/raw/system_prompts.jsonl")
    parser.add_argument("--output", default="data/raw/system_prompts_clean.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing or calling API")
    parser.add_argument("--skip-haiku", action="store_true", help="Skip Haiku placeholder filling")
    parser.add_argument("--haiku-model", default="claude-haiku-4-5-20251001")
    args = parser.parse_args()

    # Load
    with open(args.input) as f:
        prompts = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(prompts)} prompts from {args.input}")

    # Stats
    stats = {
        "total_input": len(prompts),
        "my_first_stripped": 0,
        "defaults_filled": 0,
        "haiku_filled": 0,
        "removed_still_broken": 0,
        "total_output": 0,
    }

    # --- Pass 1: Strip "My first ..." suffixes ---
    print("\n=== Pass 1: Stripping 'My first ...' suffixes ===")
    for p in prompts:
        new_text, changed = strip_my_first(p["prompt"])
        if changed:
            stats["my_first_stripped"] += 1
            if args.dry_run:
                old_end = p["prompt"][-80:].replace("\n", "\\n")
                new_end = new_text[-80:].replace("\n", "\\n")
                tqdm.write(f"  [{p['id']}] ...{old_end}")
                tqdm.write(f"       → ...{new_end}\n")
            p["prompt"] = new_text
    print(f"  Stripped {stats['my_first_stripped']} suffixes")

    # --- Pass 2: Fill ${Var:Default} templates ---
    print("\n=== Pass 2: Filling ${Var:Default} templates ===")
    for p in prompts:
        new_text, changed = fill_template_defaults(p["prompt"])
        if changed:
            stats["defaults_filled"] += 1
            if args.dry_run:
                # Show first replacement
                old_match = TEMPLATE_WITH_DEFAULT.search(p["prompt"])
                if old_match:
                    tqdm.write(f"  [{p['id']}] {old_match.group(0)} → {old_match.group(1)}")
            p["prompt"] = new_text
    print(f"  Filled defaults in {stats['defaults_filled']} prompts")

    # --- Pass 3: Haiku fills remaining ${var} placeholders ---
    if not args.skip_haiku and not args.dry_run:
        print("\n=== Pass 3: Haiku filling remaining placeholders ===")
        client = anthropic.Anthropic()

        needs_filling = [(i, p) for i, p in enumerate(prompts) if has_unfilled_placeholders(p["prompt"])]
        print(f"  {len(needs_filling)} prompts still have placeholders")

        for idx, p in tqdm(needs_filling, desc="Haiku filling"):
            try:
                filled_text, changed = fill_with_haiku(client, p["prompt"], args.haiku_model)
                if changed:
                    stats["haiku_filled"] += 1
                    p["prompt"] = filled_text
            except Exception as e:
                tqdm.write(f"  [{p['id']}] Haiku error: {e}")
            time.sleep(0.05)

        print(f"  Haiku filled {stats['haiku_filled']} prompts")
    elif args.dry_run:
        needs_filling = [p for p in prompts if has_unfilled_placeholders(p["prompt"])]
        print(f"\n=== Pass 3 (dry run): {len(needs_filling)} prompts still have placeholders ===")
        for p in needs_filling[:10]:
            placeholders = find_placeholders(p["prompt"])
            tqdm.write(f"  [{p['id']}] {placeholders[:5]}")
        if len(needs_filling) > 10:
            print(f"  ... and {len(needs_filling) - 10} more")

    # --- Pass 4: Remove prompts that are STILL broken ---
    print("\n=== Pass 4: Removing still-broken prompts ===")
    clean = []
    for p in prompts:
        text = p["prompt"].strip()

        # Skip if still has unfilled placeholders
        if has_unfilled_placeholders(text):
            stats["removed_still_broken"] += 1
            tqdm.write(f"  REMOVED [{p['id']}]: still has placeholders: {find_placeholders(text)[:3]}")
            continue

        # Skip if prompt became too short after stripping
        if len(text.split()) < 15:
            stats["removed_still_broken"] += 1
            tqdm.write(f"  REMOVED [{p['id']}]: too short after processing ({len(text.split())} words)")
            continue

        # Skip empty
        if not text:
            stats["removed_still_broken"] += 1
            continue

        p["prompt"] = text
        clean.append(p)

    stats["total_output"] = len(clean)

    # --- Reassign IDs ---
    for i, p in enumerate(clean):
        p["id"] = f"sp_{i:05d}"

    # --- Save ---
    if not args.dry_run:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for p in clean:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(clean)} prompts to {args.output}")

    # --- Report ---
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Input prompts:           {stats['total_input']}")
    print(f"'My first' stripped:     {stats['my_first_stripped']}")
    print(f"Template defaults filled:{stats['defaults_filled']}")
    print(f"Haiku filled:            {stats['haiku_filled']}")
    print(f"Removed (still broken):  {stats['removed_still_broken']}")
    print(f"Output prompts:          {stats['total_output']}")

    cats = Counter(p["category"] for p in clean)
    print(f"\nBy category:")
    for cat, cnt in cats.most_common():
        print(f"  {cat}: {cnt}")


if __name__ == "__main__":
    main()