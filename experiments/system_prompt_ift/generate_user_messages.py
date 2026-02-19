#!/usr/bin/env python3
"""
Step 1.2: Generate diverse user messages for each system prompt.

For each system prompt, calls Claude API to generate 10 user messages that
exercise different aspects of the prompt's instructions, including 2-3
adversarial/edge-case messages.

Input:  data/raw/system_prompts.jsonl   (from Step 1.1)
Output: data/with_messages.jsonl

Usage:
  python generate_user_messages.py
  python generate_user_messages.py --messages-per-prompt 15
  python generate_user_messages.py --resume  # continue from where we left off
"""

import argparse
import json
import os
import time
from pathlib import Path

import anthropic
from tqdm import tqdm

META_PROMPT = """\
You are generating test user messages for evaluating an AI assistant that has \
been given a specific system prompt. Your goal is to generate diverse user \
messages that would exercise different aspects of the system prompt's instructions.

IMPORTANT RULES:
- Generate exactly {n} user messages
- Each message should test a DIFFERENT aspect or instruction within the system prompt
- Include 2-3 adversarial/edge-case messages that would behave VERY DIFFERENTLY \
with vs without the system prompt (these are the most important for training)
- Include some straightforward messages that naturally fit the system prompt's domain
- Messages should vary in length (some short 5-10 words, some longer 30-50 words)
- Do NOT include the system prompt text in your messages
- Messages should be realistic things a real user would type

System prompt:
{system_prompt}

Output ONLY a JSON array of strings. No markdown fences, no commentary.\
"""


def generate_messages(client: anthropic.Anthropic, system_prompt: str, n: int = 10) -> list[str]:
    """Call Claude to generate n user messages for the given system prompt."""
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": META_PROMPT.format(n=n, system_prompt=system_prompt),
            }
        ],
    )
    text = resp.content[0].text.strip()
    # Handle markdown code fences
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    if text.endswith("```"):
        text = text[: text.rfind("```")]
    messages = json.loads(text)
    assert isinstance(messages, list), f"Expected list, got {type(messages)}"
    return [m.strip() for m in messages if isinstance(m, str) and m.strip()]


def main():
    parser = argparse.ArgumentParser(description="Step 1.2: Generate user messages per system prompt")
    parser.add_argument("--input", default="data/raw/system_prompts.jsonl")
    parser.add_argument("--output", default="data/with_messages.jsonl")
    parser.add_argument("--messages-per-prompt", type=int, default=10)
    parser.add_argument("--resume", action="store_true", help="Skip prompts already in output file")
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    # Load input prompts
    with open(args.input) as f:
        prompts = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(prompts)} system prompts from {args.input}")

    # Load already-processed IDs if resuming
    done_ids: set[str] = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["id"])
        print(f"Resuming: {len(done_ids)} already processed")

    client = anthropic.Anthropic()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Open in append mode for resume support
    mode = "a" if args.resume else "w"
    failed = []
    with open(args.output, mode) as fout:
        for prompt_item in tqdm(prompts, desc="Generating user messages"):
            pid = prompt_item["id"]
            if pid in done_ids:
                continue

            # Retry logic
            messages = None
            for attempt in range(1, args.max_retries + 1):
                try:
                    messages = generate_messages(
                        client, prompt_item["prompt"], n=args.messages_per_prompt
                    )
                    break
                except Exception as e:
                    if attempt < args.max_retries:
                        wait = 2 ** attempt
                        tqdm.write(f"  [{pid}] Attempt {attempt} failed: {e}. Retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        tqdm.write(f"  [{pid}] All {args.max_retries} attempts failed: {e}")
                        failed.append(pid)

            if messages is None:
                continue

            out = {**prompt_item, "user_messages": messages}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()

            # Rate-limit courtesy
            time.sleep(0.3)

    # Report
    total_done = len(prompts) - len(failed) - len(done_ids)
    print(f"\nGenerated messages for {total_done} prompts")
    if failed:
        print(f"Failed prompts ({len(failed)}): {failed}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
