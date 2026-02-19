"""
finetune_lora.py
────────────────
Helpers for test-time LoRA fine-tuning experiment.

Key functions:
  clone_loradict_to_params  – deep-clone a (possibly non-leaf) loradict into
                              independent leaf tensors that can be optimized.
  zero_init_loradict        – same structure, all tensors zeroed.
  build_recon_inputs        – build RECON-prompt input_ids / labels from raw text.
  finetune_lora_on_evidence – run K Adam steps on the RECON LM loss, returning
                              loradict snapshots at requested step counts.
"""

import sys
import os
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ── path fix: allow imports from the repo root ───────────────────────────────
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from utils.myloradict import iter_learnable_tensors

logger = logging.getLogger("finetune_lora")


# ─────────────────────────────────────────────────────────────────────────────
# LoRA dict manipulation helpers
# ─────────────────────────────────────────────────────────────────────────────

def clone_loradict_to_params(loradict: Any) -> Any:
    """
    Deep-clone every tensor in a nested loradict into independent leaf
    parameters (detached from any computation graph, requires_grad=True).

    Works on arbitrarily nested dicts of tensors / None values.
    """
    if loradict is None:
        return None
    if isinstance(loradict, dict):
        return {k: clone_loradict_to_params(v) for k, v in loradict.items()}
    if torch.is_tensor(loradict):
        t = loradict.detach().clone()
        t.requires_grad_(True)
        return t
    # Anything else (int, float, …) – pass through untouched
    return loradict


def random_init_loradict(loradict: Any, scale: float = 0.001) -> Any:
    """
    Return a same-structure loradict with standard LoRA initialisation:
      A matrices  → N(0, sqrt(scale))   (non-zero so gradients can flow)
      B matrices  → zeros
      C tensors   → zeros

    This is the correct "scratch" baseline: identical initialisation to what
    the model uses before any SHINE/training, and avoids the dead-gradient
    problem of all-zeros init (when A=B=0, grad_A ∝ B = 0 and vice versa).

    The dict structure is walked recursively; leaf dicts with key "A"/"B"/"C"
    are detected and handled per-key.
    """
    if loradict is None:
        return None
    if isinstance(loradict, dict):
        # Leaf-level LoRA dict: contains "A", "B", (optionally "C") as keys
        if "A" in loradict or "B" in loradict:
            result = {}
            for k, v in loradict.items():
                if v is None:
                    result[k] = None
                elif torch.is_tensor(v):
                    if k == "A":
                        t = torch.randn_like(v).detach() * (scale ** 0.5)
                    else:  # "B" or "C"
                        t = torch.zeros_like(v).detach()
                    t.requires_grad_(True)
                    result[k] = t
                else:
                    result[k] = v
            return result
        # Non-leaf dict: recurse
        return {k: random_init_loradict(v, scale) for k, v in loradict.items()}
    if torch.is_tensor(loradict):
        # Bare tensor not inside an A/B/C dict — zero it
        t = torch.zeros_like(loradict).detach()
        t.requires_grad_(True)
        return t
    return loradict


# Keep old name as alias so nothing else breaks
zero_init_loradict = random_init_loradict


def _count_loradict_params(loradict: Any) -> int:
    """Return total number of elements across all tensors in a loradict."""
    total = 0
    if loradict is None:
        return 0
    if isinstance(loradict, dict):
        for v in loradict.values():
            total += _count_loradict_params(v)
    elif torch.is_tensor(loradict):
        total += loradict.numel()
    return total


# ─────────────────────────────────────────────────────────────────────────────
# RECON input / label construction
# ─────────────────────────────────────────────────────────────────────────────

def build_recon_inputs(
    tokenizer,
    evidence_text: str,
    recon_context_max_length: int,
    recon_conversation_max_length: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build the RECON-formatted (input_ids, labels, attention_mask) tensors for
    a single evidence string, matching exactly what TestPretrainCollator does
    in mode="recon".

    Layout:
        input  : <|im_start|>user\\n<RECON><|im_end|>\\n<|im_start|>assistant\\n
        labels : the full chat sequence, with -100 on everything except the
                 assistant turn (i.e. only the evidence reconstruction is supervised)

    Returns:
        full_input_ids  : [1, seq_len]
        labels          : [1, seq_len]  (-100 outside supervised tokens)
        attention_mask  : [1, seq_len]
    """
    logger.debug(f"build_recon_inputs: evidence length={len(evidence_text)} chars")

    messages = [{"role": "user", "content": "<RECON>"}]
    label_messages = [
        {"role": "user",      "content": "<RECON>"},
        {"role": "assistant", "content": evidence_text},
    ]

    # ── full label sequence (prompt + evidence) ───────────────────────────────
    label_enc = tokenizer.apply_chat_template(
        label_messages,
        add_generation_prompt=False,
        tokenize=True,
        return_tensors="pt",
        max_length=recon_conversation_max_length,
        truncation=True,
        return_dict=True,
        padding="max_length",
        enable_thinking=False,
    )
    full_input_ids = label_enc["input_ids"]          # [1, T]
    attention_mask = label_enc["attention_mask"]     # [1, T]

    # ── build labels: mask everything except the assistant answer ─────────────
    labels = full_input_ids.clone()
    assistant_token_id = tokenizer.convert_tokens_to_ids("assistant")
    imstart_token_id   = tokenizer.convert_tokens_to_ids("<|im_start|>")
    imend_token_id     = tokenizer.convert_tokens_to_ids("<|im_end|>")

    mask = torch.zeros_like(labels)  # 1 = supervise, 0 = ignore
    ids = labels[0]
    last_imend = labels.shape[1]
    for j in range(len(ids) - 1, 0, -1):
        if ids[j].item() == imend_token_id:
            last_imend = j
        elif ids[j].item() == assistant_token_id and ids[j - 1].item() == imstart_token_id:
            # j+2 skips "<|im_start|>assistant\n", last_imend+2 includes the <|im_end|>
            mask[0, j + 2 : last_imend + 2] = 1
            break  # only the last assistant turn

    labels = labels.masked_fill(mask == 0, -100)

    supervised_tokens = (labels[0] != -100).sum().item()
    logger.debug(
        f"build_recon_inputs: seq_len={full_input_ids.shape[1]}, "
        f"supervised_tokens={supervised_tokens}"
    )

    return (
        full_input_ids.to(device),
        labels.to(device),
        attention_mask.to(device),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core fine-tuning loop
# ─────────────────────────────────────────────────────────────────────────────

def finetune_lora_on_evidence(
    model,
    lora_params: dict,
    recon_input_ids: torch.Tensor,
    recon_labels: torch.Tensor,
    recon_mask: torch.Tensor,
    lr: float,
    eval_at: List[int],
    sample_id: int,
    condition_name: str = "shine",
) -> Dict[int, dict]:
    """
    Optimise `lora_params` in-place using the RECON LM loss for up to
    max(eval_at) gradient steps.

    Args:
        model          : LoraQwen3ForCausalLM (all parameters frozen).
        lora_params    : trainable loradict (leaf tensors, requires_grad=True).
        recon_input_ids: [1, T] — RECON-formatted full sequence.
        recon_labels   : [1, T] — -100 on non-supervised tokens.
        recon_mask     : [1, T] — attention mask.
        lr             : Adam learning rate.
        eval_at        : sorted list of step counts to snapshot.
        sample_id      : used only for log messages.
        condition_name : "shine" or "zero" — for log messages.

    Returns:
        Dict[step -> cloned loradict snapshot]
    """
    assert eval_at == sorted(eval_at), "eval_at must be sorted"

    flat_params = list(iter_learnable_tensors(lora_params))
    n_params = sum(p.numel() for p in flat_params)
    logger.info(
        f"[sample {sample_id}][{condition_name}] Starting fine-tuning: "
        f"n_leaf_tensors={len(flat_params)}, total_params={n_params:,}, "
        f"lr={lr}, max_steps={max(eval_at)}, eval_at={eval_at}"
    )

    optimizer = torch.optim.Adam(flat_params, lr=lr)

    snapshots: Dict[int, dict] = {}

    # Step-0 snapshot: before any gradient update (= pure SHINE output or zero init)
    if 0 in eval_at:
        snapshots[0] = clone_loradict_to_params(lora_params)
        logger.info(f"[sample {sample_id}][{condition_name}] step=0 snapshot saved (no grad yet)")

    model.eval()  # safety: ensure model is in eval mode

    for step in range(1, max(eval_at) + 1):
        optimizer.zero_grad()

        outputs = model(
            input_ids=recon_input_ids,
            attention_mask=recon_mask,
            labels=recon_labels,
            loradict=lora_params,
            ignore_mem_token=True,
        )

        loss = outputs.loss
        if loss is None:
            raise RuntimeError(
                f"[sample {sample_id}][{condition_name}] step={step}: "
                f"model returned None loss — check that labels contain non-(-100) tokens"
            )

        loss.backward()
        optimizer.step()

        logger.info(
            f"[sample {sample_id}][{condition_name}] "
            f"step={step}/{max(eval_at)}: recon_loss={loss.item():.6f}"
        )

        if step in eval_at:
            snapshots[step] = clone_loradict_to_params(lora_params)
            logger.info(
                f"[sample {sample_id}][{condition_name}] "
                f"step={step}: snapshot saved"
            )

    logger.info(
        f"[sample {sample_id}][{condition_name}] Fine-tuning complete. "
        f"Snapshots at steps: {sorted(snapshots.keys())}"
    )
    return snapshots
