import hydra
from omegaconf import DictConfig
import asyncio
import os
import json
from openai import AsyncOpenAI
from openai import APIError, RateLimitError, APITimeoutError
from typing import Any, Dict, List
from tqdm import tqdm

PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "groundtruth = {reference}\n"
    "predict_answer = {pred}"
)

SYS_MSG = (
    "Given one question, there is a groundtruth and a predict_answer. "
    "Please decide whether they are the same or not in semantic. "
    "Please only output 'True' or 'False' ."
)

async def safe_chat_complete(
    client: AsyncOpenAI,
    model: str,
    messages: List[Dict[str, str]],
    sem: asyncio.Semaphore,
    max_retries: int = 6,
) -> str:
    """
    Call chat.completions.create with concurrency control and retries.
    """
    backoff = 0.5
    attempt = 0
    async with sem:
        while True:
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                return resp.choices[0].message.content or ""
            except (RateLimitError, APITimeoutError, APIError) as e:
                attempt += 1
                # For non-retryable 4xx (except 429), bubble up quickly
                status = getattr(e, "status_code", None)
                if isinstance(e, APIError) and status and 400 <= status < 500 and status != 429:
                    raise
                if attempt > max_retries:
                    raise
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 8.0)  # cap the backoff
            except Exception:
                attempt += 1
                if attempt > max_retries:
                    raise
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 8.0)

def parse_bool(text: str) -> bool:
    return "true" in (text or "").strip().lower()

async def score_one_qa(
    client: AsyncOpenAI,
    model: str,
    qa: Dict[str, Any],
    sem: asyncio.Semaphore,
    max_retries: int,
) -> bool:
    messages = [
        {"role": "system", "content": SYS_MSG},
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(
                question=qa["Q"],
                reference=qa["A"],
                pred=qa["P"],
            ),
        },
    ]
    reply = await safe_chat_complete(
        client=client,
        model=model,
        messages=messages,
        sem=sem,
        max_retries=max_retries,
    )
    return parse_bool(reply)

#???????????????????????????????????????????????????????????????????????????
async def process_sample(
    client: AsyncOpenAI,
    model: str,
    sample: Dict[str, Any],
    sem: asyncio.Semaphore,
    max_retries: int,
) -> Dict[str, Any]:
    # Normalize potential 'qa' key
    if "qa" in sample:  # Some mistakes in upstream data
        sample["qa_pairs"] = sample["qa"]
        del sample["qa"]

    qa_pairs = sample.get("qa_pairs", [])
    tasks = []
    for qa in qa_pairs:
        tasks.append(
            score_one_qa(
                client=client,
                model=model,
                qa=qa,
                sem=sem,
                max_retries=max_retries,
            )
        )
    scores: List[bool] = await asyncio.gather(*tasks)
    for qa, s in zip(qa_pairs, scores):
        qa["score"] = s
    return sample

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig):  
    asyncio.run(amain(cfg))

async def amain(cfg: DictConfig):
    if cfg.test.source == "loogle":
        names = ["shortdep_qa", "shortdep_cloze", "longdep_qa", "summarization"]
        load_dir = os.path.join(cfg.test.save_path, cfg.test.source)
    else:
        raise ValueError(f"Unknown data source: {cfg.test.source}")
    
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(max(1, cfg.test.max_concurrency))
    
    for name in names:
        json_path = os.path.join(load_dir, f"{name}.json")
        json_no_metanet_path = json_path.replace(".json", "_no_metanet.json")

        with open(json_path, "r", encoding="utf-8") as f:
            res = json.load(f)
        with open(json_no_metanet_path, "r", encoding="utf-8") as f:
            res_no_metanet = json.load(f)
            
        #?????????????????????????????????????????????????????????????
        all_scores: List[bool] = []

        # Process each sample (article) sequentially so we can append per-line safely,
        # but do each sample's QAs concurrently.
        pbar = tqdm.tqdm(res, desc="Evaluating with metanetwork")
        for sample in pbar:
            try:
                updated = await process_sample(
                    client=client,
                    model=cfg.test.model,
                    sample=sample,
                    sem=sem,
                    max_retries=cfg.test.max_retries,
                )
            except Exception as e:
                # If a sample fails entirely, keep a trace and continue
                updated = sample
                updated["_error"] = f"{type(e).__name__}: {e}"

            # Collect scores for average in this run
            for qa in updated.get("qa_pairs", []):
                if "score" in qa and isinstance(qa["score"], bool):
                    all_scores.append(qa["score"])

            # Write out
            if args.result_file.endswith(".jsonl"):
                with open(args.result_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(updated, ensure_ascii=True) + "\n")

        # If JSON output requested, dump the whole (possibly partially resumed) dataset
        if args.result_file.endswith(".json"):
            # Merge back the processed tail with the already-finished head (if any)
            merged = eval_data[:offset] + eval_data_to_run
            with open(args.result_file, "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=4, ensure_ascii=True)

        # Summary
        if all_scores:
            avg = sum(all_scores) / len(all_scores)
            print(f"Average = {avg:.6f}")
        else:
            print("Average = NA (no new samples evaluated)")
        
    
if __name__ == "__main__":
    main()