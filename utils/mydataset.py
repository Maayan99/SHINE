from dataclasses import dataclass
from pyexpat.errors import messages
import re
from typing import Dict, List, Tuple, Any

import torch
from torch.utils.data import Sampler
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from metanetwork_family import Metanetwork

import random
from collections import defaultdict
from typing import Optional
import numpy as np
from copy import deepcopy
from datasets import Column
import json

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
    ] * 50
    df = pd.DataFrame({'text': texts})
    train_texts, val_texts = train_test_split(df['text'], test_size=0.1, random_state=42)
    return train_texts.tolist(), val_texts.tolist()


# ---------------------------
# Dataset
# ---------------------------
class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx) -> Dict[str, Any]:
        return {"text": str(self.texts[idx])}
    
class GroupTextDataset(Dataset):
    def __init__(self, texts, tokenizer, conversation_max_len, cache_dir):
        self.texts = texts
        self.tokenizer = tokenizer
        self.conversation_max_len = conversation_max_len
        self.cache_dir = cache_dir
        
        # Try find cache
        pass
    
        # If not find, get group
        test_q = "who is adam ?"
        test_a = "I don't know"
        message_1 = [{"role": "user", "content": f"{test_q}"}, {"role": "assistant", "content": f"{test_a}"}]
        input_enc_1 = self.tokenizer.apply_chat_template(
                message_1,
                add_generation_prompt=False,   # adds the assistant turn start
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=False,
            )
        len1 = len(input_enc_1["input_ids"])
        message_2 = message_1 * 2
        input_enc_2 = self.tokenizer.apply_chat_template(
                message_2,
                add_generation_prompt=False,   # adds the assistant turn start
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=False,
            )
        len2 = len(input_enc_2["input_ids"])
        len3 = len(self.tokenizer.tokenize(test_q)) + len(self.tokenizer.tokenize(test_a))
        self.base_len = len1 * 2 - len2
        self.chat_len = len1 - len3 - self.base_len
        
        
        
    def __len__(self):
        pass
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        pass

class SquadDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Any]:
        answer = [str(ans).strip() for ans in self.data[idx]['answers']['text']]
        for i in range(len(answer)):
            if answer[i][0].islower():
                answer[i] = answer[i][0].upper() + answer[i][1:]
        return {"evidence": str(self.data[idx]['context']).strip(), "question": str(self.data[idx]['question']).strip(), "answer": answer}

class GroupedSquadDataset(Dataset):
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        context_len: Optional[int] = None,
        sep: str = '\n\n',
        name: str = "Test",
        seed: int = 42,
    ):
        self.name = f"[GroupedSquadDataset: {name}]"
        self.tokenizer = tokenizer
        self.sep = sep
        self.context_len = context_len
        self.data = data
        self.seed = seed
        
        self.shuffle()
    
    def shuffle(self):
        # Apply seed if provided (for determinism)
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        text_to_idx = defaultdict(list)
        for i, ex in enumerate(self.data):
            ctx = str(ex["context"]).strip()
            text_to_idx[ctx].append(i)

        all_context_list = deepcopy(list(text_to_idx.keys()))
        self.text_to_idx = text_to_idx
        
        if self.context_len is None:
            self.groups = [[ctx] for ctx in all_context_list]
        else:
            num_tokens = len(self.tokenizer(self.sep.join(all_context_list))["input_ids"])
            num_groups = (num_tokens + self.context_len - 1) // self.context_len

            random.shuffle(all_context_list)
            context_list_per_group = np.array_split(all_context_list, num_groups)
            self.groups = [[str(s) for s in arr] for arr in context_list_per_group]
            
        self.idx_to_groupidx = {}
        for group_idx, ctx_list in enumerate(self.groups):
            for ctx in ctx_list:
                for ex_idx in text_to_idx[ctx]:
                    self.idx_to_groupidx[ex_idx] = group_idx
        
        self.group_token_num = []
        for group in self.groups:            
            token_num = len(self.tokenizer(self.sep.join(group))["input_ids"])
            self.group_token_num.append(token_num)
            
        print(f"{self.name}: {len(self.groups)} groups created from {len(self.data)} examples.")
        print(f"{self.name}: Average group token length: {np.mean(self.group_token_num):.2f}, "
              f"Max group token length: {np.max(self.group_token_num)}, "
              f"Min group token length: {np.min(self.group_token_num)}")
        print(f"{self.name}: Top 20 largest groups token lengths: {sorted(self.group_token_num, reverse=True)[:20]}")
        print(f"{self.name}: Average contexts per group: {len(self.data) / len(self.groups):.2f}")
        
        for group in self.groups:
            for ctx in group:
                assert not ctx.startswith(" "), f"Context has leading space: '{ctx}'"
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Any]:
        group = self.groups[self.idx_to_groupidx[idx]]
        evidence = self.sep.join(list(random.sample(group, len(group))))
        answer = [str(ans).strip() for ans in self.data[idx]['answers']['text']]
        for i in range(len(answer)):
            if answer[i][0].islower():
                answer[i] = answer[i][0].upper() + answer[i][1:]
        return {"evidence": str(evidence).strip(), "question": str(self.data[idx]['question']).strip(), "answer": answer}

class SFTDataset(Dataset):
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
    ):
        self.tokenizer = tokenizer
        self.data = data  
         
        context_token_lengths = []
        conversations_token_lengths = []
        tmp_data = data.shuffle().select(range(100))
        for item in tmp_data:
            context = item["context"]
            conversations = item["conversations"]

            context_encoded = tokenizer(
                context,
                padding=False,
                return_tensors=None,
            )
            context_token_lengths.append(len(context_encoded["input_ids"]))
            conversations_encoded = self.tokenizer.apply_chat_template(
                conversations,
                add_generation_prompt=False,   # adds the assistant turn start
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=False,
            )
            conversations_token_lengths.append(len(conversations_encoded["input_ids"][0]))

        print(f"[SFTDataset] Average context token length: {np.mean(context_token_lengths):.2f}({np.std(context_token_lengths):.2f})")
        print(f"[SFTDataset] Average conversation token length: {np.mean(conversations_token_lengths):.2f}({np.std(conversations_token_lengths):.2f})")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Any]:
        return {"evidence": self.data[idx]['context'], "conversation": self.data[idx]['conversations']}

# ---------------------------
# Collator with dynamic padding and label masking
# ---------------------------
@dataclass
class PretrainCollator:
    tokenizer: Any
    cfg: Any
    context_max_length: int = 1024
    conversation_max_length: int = 1024
    metatrain: bool = False
    thinkend_token_id: int = None
    
    def __post_init__(self):
        self.thinkend_token_id = self.tokenizer.convert_tokens_to_ids("</think>")
        self.completion_freq = self.cfg.pretrain.completion_freq
        self.max_completion_ratio = self.cfg.pretrain.max_completion_ratio
        self.min_completion_ratio = self.cfg.pretrain.min_completion_ratio
    
    def split_text(self, text):
        t = text.split()
        if len(t) < 2:
            return text, "Nothing to complete."

        ratio = 1.0 - random.uniform(self.min_completion_ratio, self.max_completion_ratio)
        split_index = round(len(t) * ratio)

        left = t[:split_index]
        right = t[split_index:]

        if not right:  # ensure right is not empty
            left, right = t[:-1], t[-1:]
        elif not left:
            left, right = t[:1], t[1:]
        
        return ' '.join(left), ' '.join(right)
    

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [ex["text"] for ex in batch]
        
        if self.metatrain:            
            t = random.random()
            if t < self.completion_freq:
                splits = [self.split_text(text) for text in texts]
                evidence_texts = [split[0] for split in splits]
                answer_texts = texts
                # answer_texts = [split[1] for split in splits]
                messages = [[
                    {"role": "user", "content": f"<COMP>"},
                    {"role": "assistant", "content": f"{answer}"}
                ] for answer in answer_texts]
            else:
                evidence_texts = texts
                answer_texts = texts
                messages = [[
                    {"role": "user", "content": f"<RECON>"},
                    {"role": "assistant", "content": f"{answer}"}
                ] for answer in answer_texts]
        else:
            raise NotImplementedError("metatrain=False mode is not implemented in PretrainCollator.")

        evidence_enc = self.tokenizer(
            evidence_texts,
            max_length=self.context_max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        evidence_ids = evidence_enc["input_ids"]
        evidence_attention_mask = evidence_enc["attention_mask"]
        answer_enc = self.tokenizer(
            answer_texts,
            max_length=self.conversation_max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        answer_ids = answer_enc["input_ids"]
        answer_attention_mask = answer_enc["attention_mask"]


        input_enc = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,   # adds the assistant turn start
                tokenize=True,
                return_tensors="pt",
                max_length=self.conversation_max_length,
                truncation=True,
                return_dict=True,
                padding="max_length",
                enable_thinking=False,
            )
        input_ids = input_enc["input_ids"]
        input_attention_mask = input_enc["attention_mask"]
        labels = None
        if self.metatrain:
            labels = input_ids.clone()
            for i, id in enumerate(input_ids):
                for j in range(len(id) - 1, -1, -1):
                    if id[j].item() == self.thinkend_token_id:
                        labels[i, :j+2] = -100
                        break
            # print("evidence #############\n", self.tokenizer.decode(evidence_ids[i], skip_special_tokens=False))
            # print("origin###########\n", texts[i])
            # print("answer #############\n", self.tokenizer.decode(answer_ids[i], skip_special_tokens=False))
            # exit()
            assert labels[i].sum().item() != -100 * labels.size(1), "All labels are masked!"
            # if evidence_ids[i][-1].item() == self.tokenizer.pad_token_id:
            #     tokens = self.tokenizer.convert_ids_to_tokens(evidence_ids[i])
            #     print("evidence", evidence_texts[i])
            #     print("answer", answer_texts[i])
            #     for j, t in enumerate(tokens):
            #         print(f"{j}: token_ids: {t} attention_mask: {evidence_attention_mask[i][j]}")
            #     exit()
            assert evidence_ids[i][-1].item() != self.tokenizer.pad_token_id, "Evidence evidence is all padding!"
        
        # res = "input"
        # tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        # for i, t in enumerate(tokens):
        #     res = f"{res}\n{i}: token_ids: {t} attention_mask: {input_attention_mask[0][i]} label: {labels[0][i] if labels is not None else 'N/A'}"
        # res = f"{res}\nevidence"
        # tokens = self.tokenizer.convert_ids_to_tokens(evidence_ids[0])
        # for i, t in enumerate(tokens):
        #     res = f"{res}\n{i}: token_ids: {t} attention_mask: {evidence_attention_mask[0][i]}"
        # print(res)
        # exit()
        
        # # Debug print for the first item
        # first_input_ids = input_ids[0]
        # first_labels = labels[0]
        # first_evidence_ids = evidence_ids[0]
        # first_input_text = self.tokenizer.decode(first_input_ids, skip_special_tokens=False)
        # first_evidence_ids = [i for i in first_evidence_ids if i != self.tokenizer.pad_token_id]
        # first_evidence_text = self.tokenizer.decode(first_evidence_ids, skip_special_tokens=False)
        # print("\n=== First input sentence (meta-train mode) ===")
        # print(first_input_text)
        # print("\n=== First evidence sentence (meta-train mode) ===")
        # print(first_evidence_text)
        # # tokens = self.tokenizer.convert_ids_to_tokens(first_input_ids)
        # # print("\n=== Tokens, labels, and corresponding words ===")
        # # for i, (tok, lab) in enumerate(zip(tokens, first_labels.tolist())):
        # # # for i, tok in enumerate(tokens):
        # #     # decode the token alone to see its text segment
        # #     word_piece = self.tokenizer.decode(
        # #         [self.tokenizer.convert_tokens_to_ids(tok)],
        # #         skip_special_tokens=True,
        # #         clean_up_tokenization_spaces=False,
        # #     )
        # #     # show both raw token, decoded string, and label
        # #     # print(f"{tok:<20} | {word_piece:<15} | mask={input_attention_mask[0][i]}")
        # #     print(f"{tok:<20} | {word_piece:<15} | label={lab} | mask={input_attention_mask[0][i]}")
        # exit()
                
        return {
            "evidence": texts,
            "evidence_ids": evidence_ids,
            "evidence_attention_mask": evidence_attention_mask,
            "input_ids": input_ids,
            "labels": labels,
            "input_attention_mask": input_attention_mask,
            "answers": texts,
            "answer_ids": answer_ids,
            "answer_attention_mask": answer_attention_mask,
            "questions": ["Please repeat what you have read."] * len(texts),
        }

@dataclass
class SquadCollator:
    tokenizer: Any
    context_max_length: int = 1024
    conversation_max_length: int = 1024
    use_reference: bool = False
    metatrain: bool = False
    only_question: bool= False
    thinkend_token_id: int = None
    
    def __post_init__(self):
        self.assistant_token_id = self.tokenizer.convert_tokens_to_ids("assistant")
        self.imstart_token_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.imend_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        questions = [ex["question"] for ex in batch]
        evidences = [ex["evidence"] for ex in batch]
        assert isinstance(batch[0]["answer"], list), "Answers should be a list of possible answers."
        answers = [str(random.choice(ex["answer"])) for ex in batch]
        full_answers = [ex["answer"] for ex in batch]
           
        evidence_enc = self.tokenizer(
            evidences,
            max_length=self.context_max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        evidence_ids = evidence_enc["input_ids"]
        evidence_attention_mask = evidence_enc["attention_mask"]
        
        answer_enc = self.tokenizer(
            answers,
            max_length=self.conversation_max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        answer_ids = answer_enc["input_ids"]
        answer_attention_mask = answer_enc["attention_mask"]


        # if self.metatrain:            
        #     messages = [[
        #         {"role": "system", "content": "You are a concise assistant. Only output the final answer with no extra words."},
        #         {"role": "user", "content": f"Please answer the following question: {question}"},
        #         {"role": "assistant", "content": f"<think>I know the answer because I have read something about this.</think>\n{answer}"}
        #     ] for question, answer in zip(questions, answers)]
        # elif self.use_reference:
        #     messages = [[
        #         {"role": "system", "content": "You are a concise assistant. Only output the final answer with no extra words."},
        #         {"role": "user", "content": f"Please review the following reference materials.\n\nReference:\n{evidence}\n\nBased on the reference, answer this question:\n{question}"},
        #     ] for evidence, question in zip(evidences, questions)]
        # elif self.only_question:
        #     messages = [[
        #         {"role": "system", "content": "You are a concise assistant. Only output the final answer with no extra words."},
        #         {"role": "user", "content": f"Please answer the following question: {question}"},
        #     ] for question in questions]
        # else:
        #     messages = [[
        #         {"role": "system", "content": "You are a concise assistant. Only output the final answer with no extra words."},
        #         {"role": "user", "content": f"Please answer the following question: {question}"},
        #         {"role": "assistant", "content": f"<think>I know the answer because I have read something about this.</think>\n"}
        #     ] for question in questions]
        
        if self.metatrain:            
            messages = [[
                {"role": "user", "content": f"{question}"},
                {"role": "assistant", "content": f"{answer}"}
            ] for question, answer in zip(questions, answers)]
        elif self.use_reference:
            messages = [[
                {"role": "user", "content": f"Reference:\n{evidence}\n\nBased on the reference, answer this question:\n{question}"},
            ] for evidence, question in zip(evidences, questions)]
        elif self.only_question:
            messages = [[
                {"role": "user", "content": f"{question}"},
            ] for question in questions]
        else:
            ##################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!check prompt!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!####################
            messages = [[
                {"role": "user", "content": f"{question}"},
                {"role": "assistant", "content": f"<think></think>\n"}
            ] for question in questions]


        input_enc = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True if (not self.metatrain and (self.use_reference or self.only_question)) else False,   # adds the assistant turn start
                tokenize=True,
                return_tensors="pt",
                max_length=self.conversation_max_length + 2 if not self.metatrain and not self.use_reference and not self.only_question else self.conversation_max_length,
                truncation=True,
                return_dict=True,
                padding="max_length",
                enable_thinking=False,
            )
        input_ids = input_enc["input_ids"]
        input_attention_mask = input_enc["attention_mask"]
        labels = None
        if self.metatrain:
            labels = input_ids.clone()
            masks = torch.zeros_like(labels)
            for i, id in enumerate(input_ids):
                last_imend = self.conversation_max_length
                for j in range(len(id) - 1, 0, -1):
                    if id[j].item() == self.imend_token_id:
                        last_imend = j
                    elif id[j].item() == self.assistant_token_id and id[j - 1] == self.imstart_token_id:
                        masks[i, j+2: last_imend+2] = 1
            labels = labels.masked_fill(masks == 0, -100)
        elif not (self.use_reference or self.only_question):
            input_ids = input_ids[:, :-2]
            input_attention_mask = input_attention_mask[:, :-2]
        
        # res = "input"
        # tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        # for i, t in enumerate(tokens):
        #     res = f"{res}\n{i}: token_ids: {t} attention_mask: {input_attention_mask[0][i]} label: {labels[0][i] if labels is not None else 'N/A'}"
        # res = f"{res}\nevidence"
        # tokens = self.tokenizer.convert_ids_to_tokens(evidence_ids[0])
        # for i, t in enumerate(tokens):
        #     res = f"{res}\n{i}: token_ids: {t} attention_mask: {evidence_attention_mask[0][i]}"
        # res = f"{res}\n\n"
        # print(res)
        # exit()
        
        # # Debug print for the first item
        # first_input_ids = input_ids[0]
        # first_labels = labels[0]
        # first_input_text = self.tokenizer.decode(first_input_ids, skip_special_tokens=False)
        # print("\n=== First input sentence (meta-train mode) ===")
        # print(first_input_text)
        # tokens = self.tokenizer.convert_ids_to_tokens(first_input_ids)
        # print("\n=== Tokens, labels, and corresponding words ===")
        # for i, (tok, lab) in enumerate(zip(tokens, first_labels.tolist())):
        # # for i, tok in enumerate(tokens):
        #     # decode the token alone to see its text segment
        #     word_piece = self.tokenizer.decode(
        #         [self.tokenizer.convert_tokens_to_ids(tok)],
        #         skip_special_tokens=True,
        #         clean_up_tokenization_spaces=False,
        #     )
        #     # show both raw token, decoded string, and label
        #     # print(f"{tok:<20} | {word_piece:<15} | mask={input_attention_mask[0][i]}")
        #     print(f"{tok:<20} | {word_piece:<15} | label={lab} | mask={input_attention_mask[0][i]}")
        # exit()
                
        return {
            "evidence": evidences,
            "evidence_ids": evidence_ids,
            "evidence_attention_mask": evidence_attention_mask,
            "messages": messages,
            "input_ids": input_ids,
            "labels": labels,
            "input_attention_mask": input_attention_mask,
            "answers": answers,
            "full_answers": full_answers,
            "answer_ids": answer_ids,
            "answer_attention_mask": answer_attention_mask,
            "questions": questions,
        }


@dataclass
class SFTCollator:
    tokenizer: Any
    context_max_length: int = 1024
    conversation_max_length: int = 1024
    
    def __post_init__(self):
        self.assistant_token_id = self.tokenizer.convert_tokens_to_ids("assistant")
        self.imstart_token_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.imend_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        evidence_texts = [t['evidence'] for t in batch]
        conversation_texts = [t['conversation'] for t in batch]
        if isinstance(conversation_texts[0], Column):
            conversation_texts = list(conversation_texts[0])  # or conversation_texts[0][:]

        evidence_enc = self.tokenizer(
            evidence_texts,
            max_length=self.context_max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        evidence_ids = evidence_enc["input_ids"]
        evidence_attention_mask = evidence_enc["attention_mask"]

        input_enc = self.tokenizer.apply_chat_template(
                conversation_texts,
                add_generation_prompt=False,   # adds the assistant turn start
                tokenize=True,
                return_tensors="pt",
                max_length=self.conversation_max_length,
                truncation=True,
                return_dict=True,
                padding="max_length",
                enable_thinking=False,
            )
        input_ids = input_enc["input_ids"]
        input_attention_mask = input_enc["attention_mask"]

        labels = input_ids.clone()
        masks = torch.zeros_like(labels)
        for i, id in enumerate(input_ids):
            last_imend = self.conversation_max_length
            for j in range(len(id) - 1, 0, -1):
                if id[j].item() == self.imend_token_id:
                    last_imend = j
                elif id[j].item() == self.assistant_token_id and id[j - 1] == self.imstart_token_id:
                    masks[i, j+2: last_imend+2] = 1
        labels = labels.masked_fill(masks == 0, -100)
        
           
        # res = "input"
        # tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        # for i, t in enumerate(tokens):
        #     res = f"{res}\n{i}: token_ids: {t} attention_mask: {input_attention_mask[0][i]} label: {labels[0][i] if labels is not None else 'N/A'}"
        # res = f"{res}\nevidence"
        # tokens = self.tokenizer.convert_ids_to_tokens(evidence_ids[0])
        # for i, t in enumerate(tokens):
        #     res = f"{res}\n{i}: token_ids: {t} attention_mask: {evidence_attention_mask[0][i]}"
        # res = f"{res}\n\n"
        # print(res)
        # exit()
                
        return {
            "evidence": evidence_texts,
            "evidence_ids": evidence_ids,
            "evidence_attention_mask": evidence_attention_mask,
            "input_ids": input_ids,
            "labels": labels,
            "input_attention_mask": input_attention_mask,
        }