from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import torch
from torch.utils.data import Sampler
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

from torch.utils.tensorboard import SummaryWriter
from metanetwork_family import Metanetwork


# ---------------------------
# Mock dataset for demo
# ---------------------------
def create_mock_dataset() -> Tuple[List[str], List[str]]:
    texts = [
        "1231",
        "2342",
        "3453",
        "4564",
        "5675",
        "6786",
        "7897",
        "8908",
        "9019",
        "0120",
    ] * 5
    df = pd.DataFrame({'text': texts})
    train_texts, val_texts = train_test_split(df['text'], test_size=0.1, random_state=42)
    return train_texts.tolist(), val_texts.tolist()


# ---------------------------
# Dataset
# ---------------------------
class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx) -> Dict[str, Any]:
        return {"text": str(self.texts[idx])}

class LoogleDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Any]:
        return {"question": str(self.data[idx]['question']), "evidence": str(self.data[idx]['evidence']), "answer": str(self.data[idx]['answer'])}

# ---------------------------
# Collator with dynamic padding and label masking
# ---------------------------
@dataclass
class CausalLMDataCollator:
    tokenizer: Any
    max_length: int = 512

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [ex["text"] for ex in batch]
        enc = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        labels = input_ids.clone()

        # Ensure a pad token exists
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        pad_id = self.tokenizer.pad_token_id
        labels[labels == pad_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

@dataclass
class LoogleCollator:
    tokenizer: Any
    max_length: int = 512

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        questions = [ex["question"] for ex in batch]
        evidences = [ex["evidence"] for ex in batch]
        answers = [ex["answer"] for ex in batch]
        inputs = [f"Please read the following text and answer the question.\nText:{e}\nQuestion:{q}\nAnswer:" for q, e in zip(questions, evidences)]
           
        enc = self.tokenizer(
            inputs,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "answers": answers,
        }
