import os
import json
import torch
from typing import Any, Dict
import numpy as np  # add at top
import random
from omegaconf import DictConfig, OmegaConf
from utils.mylogging import get_logger
from utils.myfreeze import freeze
from torch.utils.data import DataLoader
import time

logger = get_logger("save & load")

def save_checkpoint(metanetwork, tokenizer, out_dir: str, extra_state: Dict[str, Any] = None):
    os.makedirs(out_dir, exist_ok=True)
    metanetwork.metamodel.save_pretrained(os.path.join(out_dir, "metamodel"))
    torch.save(metanetwork.metanetwork.state_dict(), os.path.join(out_dir, "metanetwork.pth"))
    tokenizer.save_pretrained(os.path.join(out_dir, "tokenizer"))
    if extra_state is not None:
        with open(os.path.join(out_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
            json.dump(extra_state, f, ensure_ascii=False, indent=2)

def load_checkpoint(metanetwork, tokenizer, in_dir, device: str):
    metanetwork.to("cpu")
    metanetwork.metamodel = metanetwork.metamodel.__class__.from_pretrained(os.path.join(in_dir, "metamodel"))
    metanetwork.metanetwork.load_state_dict(torch.load(os.path.join(in_dir, "metanetwork.pth"), weights_only=False, map_location="cpu"))
    metanetwork.to(device)
    tokenizer = tokenizer.__class__.from_pretrained(os.path.join(in_dir, "tokenizer"))
    freeze(metanetwork.metamodel)
    return metanetwork, tokenizer

def _rng_state_dict():
    state = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    return state

def _set_rng_state(state: Dict[str, Any]):
    if state is None:
        return
    try:
        random.setstate(state["python_random"])
        np.random.set_state(state["numpy_random"])
        torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and state.get("torch_cuda_all") is not None:
            torch.cuda.set_rng_state_all(state["torch_cuda_all"])
    except Exception as e:
        logger.warning(f"Could not fully restore RNG states: {e}")

def save_training_state(
    out_dir: str,
    global_step: int,
    epoch: int,
    step_in_epoch: int,
    best_eval_loss: float,
):
    os.makedirs(os.path.join(out_dir, "trainer_state"), exist_ok=True)
    payload = {
        "global_step": global_step,
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "best_eval_loss": best_eval_loss,
        "rng_state": _rng_state_dict(),
    }
    torch.save(payload, os.path.join(out_dir, "trainer_state", "trainer_state.pt"))

def load_training_state(
    in_dir: str,
):
    path = os.path.join(in_dir, "trainer_state", "trainer_state.pt")
    if not os.path.isfile(path):
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _set_rng_state(payload.get("rng_state"))

    return {
        "global_step": payload.get("global_step", 0),
        "epoch": payload.get("epoch", 1),
        "step_in_epoch": payload.get("step_in_epoch", 0),
        "best_eval_loss": payload.get("best_eval_loss", float("inf")),
    }


def get_latest_checkpoint(root_dir: str) -> str:
    if not os.path.isdir(root_dir):
        return None
    cands = [d for d in os.listdir(root_dir) if d.startswith("checkpoint-")]
    if not cands:
        return None
    steps = []
    for d in cands:
        try:
            steps.append((int(d.split("-")[-1]), d))
        except Exception:
            pass
    if not steps:
        return None
    steps.sort()
    return os.path.join(root_dir, steps[-1][1])
