"""Dataset class for system prompt IFT training.

Reads flat JSONL files (one {system_prompt, user_message, response} per line)
and returns dicts compatible with IFTCollator: {evidence, conversations}.
"""

import json
from typing import Any, Dict

from torch.utils.data import Dataset

from utils.myddp import is_main_process


class SystemPromptIFTDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        max_context_len: int = 3000,
        max_conversation_len: int = 256,
        use_exceed: bool = False,
    ):
        with open(data_path) as f:
            all_items = [json.loads(line) for line in f if line.strip()]

        if use_exceed:
            self.item_list = all_items
        else:
            # Approximate token count: chars / 3.5
            self.item_list = [
                item for item in all_items
                if len(item["system_prompt"]) / 3.5 <= max_context_len
                and (len(item["user_message"]) + len(item["response"])) / 3.5 <= max_conversation_len
            ]

        if is_main_process():
            print(
                f"[SystemPromptIFTDataset] Loaded {len(self.item_list)} items "
                f"(from {len(all_items)} total) from {data_path}, use_exceed={use_exceed}"
            )

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, idx) -> Dict[str, Any]:
        item = self.item_list[idx]
        return {
            "evidence": item["system_prompt"],
            "conversations": [
                {"role": "user", "content": item["user_message"]},
                {"role": "assistant", "content": item["response"]},
            ],
        }
