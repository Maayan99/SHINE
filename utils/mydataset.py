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

import random

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
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx) -> Dict[str, Any]:
        return {"text": str(self.texts[idx])}

# class LoogleDataset(Dataset):
#     def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
#         self.data = data
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx) -> Dict[str, Any]:
#         return {"question": str(self.data[idx]['question']), "evidence": str(self.data[idx]['evidence']), "answer": str(self.data[idx]['answer'])}

class SquadDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Any]:
        return {"evidence": str(self.data[idx]['context']), "question": str(self.data[idx]['question']), "answer": self.data[idx]['answers']['text']}

# ---------------------------
# Collator with dynamic padding and label masking
# ---------------------------
@dataclass
class PretrainCollator:
    tokenizer: Any
    max_length: int = 1024
    use_reference: bool = True
    metatrain: bool = False
    thinkend_token_id: int = None
    
    def __post_init__(self):
        self.thinkend_token_id = self.tokenizer.convert_tokens_to_ids("</think>")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [ex["text"] for ex in batch]

        evidence_enc = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        evidence_ids = evidence_enc["input_ids"]
        evidence_attention_mask = evidence_enc["attention_mask"]
        
        answer_enc = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        answer_ids = answer_enc["input_ids"]
        answer_attention_mask = answer_enc["attention_mask"]

        if self.metatrain:            
            messages = [[
                {"role": "system", "content": "You are a concise assistant. Only output the final answer with no extra words."},
                {"role": "user", "content": f"Please repeat what you have read."},
                {"role": "assistant", "content": f"<think>I know the answer because I have read something about this.</think>\n{answer}"}
            ] for answer in texts]
        # Need update to add thinking token
        elif self.use_reference:
            raise NotImplementedError("use_reference=True is not implemented for PretrainCollator.")
            # messages = [[
            #     {"role": "system", "content": "You are a concise assistant. Only output the final answer with no extra words."},
            #     {"role": "user", "content": "Please repeat what you read."},
            #     {"role": "user", "content": f"{text}"},
            #     {"role": "user", "content": f"Please start to repeat."}
            # ] for text in texts]
        else:
            messages = [[
                {"role": "system", "content": "You are a concise assistant. Only output the final answer with no extra words."},
                {"role": "user", "content": f"Please repeat what you have read."},
                {"role": "assistant", "content": f"<think>I know the answer because I have read something about this.</think>\n"}
            ] for _ in texts]

        input_enc = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True if (not self.metatrain and self.use_reference) else False,   # adds the assistant turn start
                tokenize=True,
                return_tensors="pt",
                max_length=self.max_length + 2 if not self.metatrain and not self.use_reference else self.max_length,
                truncation=True,
                return_dict=True,
                padding="max_length",
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
        elif not self.use_reference:
            input_ids = input_ids[:, :-2]
            input_attention_mask = input_attention_mask[:, :-2]
        
        # tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        # for i, t in enumerate(tokens):
        #     print(f"{i}: token_ids: {t} attention_mask: {input_attention_mask[0][i]} label: {labels[0][i] if labels is not None else 'N/A'}")
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
        
# class CausalLMDataCollator:
#     tokenizer: Any
#     max_length: int = 512

#     def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
#         texts = [ex["text"] for ex in batch]
#         enc = self.tokenizer(
#             texts,
#             truncation=True,
#             padding=True,
#             max_length=self.max_length,
#             return_tensors="pt",
#         )
#         input_ids = enc["input_ids"]
#         attention_mask = enc["attention_mask"]
#         labels = input_ids.clone()

#         # Ensure a pad token exists
#         if self.tokenizer.pad_token_id is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#         pad_id = self.tokenizer.pad_token_id
#         labels[labels == pad_id] = -100

#         return {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "labels": labels,
#         }

# @dataclass
# class LoogleCollator:
#     tokenizer: Any
#     max_length: int = 1024
#     use_reference: bool = True

#     def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
#         questions = [ex["question"] for ex in batch]
#         evidences = [ex["evidence"] for ex in batch]
#         answers = [ex["answer"] for ex in batch]
           
#         evidence_enc = self.tokenizer(
#             evidences,
#             padding=True,
#             max_length=self.max_length,
#             truncation=True,
#             return_tensors="pt",
#         )
#         evidence_ids = evidence_enc["input_ids"]
#         evidence_attention_mask = evidence_enc["attention_mask"]
        
#         if self.use_reference:
#             messages = [[
#                 {"role": "system", "content": "You are a concise assistant. Only output the final answer with no extra words."},
#                 {"role": "user", "content": "Please review the following reference materials."},
#                 {"role": "user", "content": f"{evidence}"},
#                 {"role": "user", "content": f"Based on the above, answer this question: {question}"}
#             ] for evidence, question in zip(evidences, questions)]
#         else:
#             messages = [[
#                 {"role": "system", "content": "You are a concise assistant. Only output the final answer with no extra words."},
#                 {"role": "user", "content": f"Please answer the following question: {question}"}
#             ] for question in questions]

#         input_enc = self.tokenizer.apply_chat_template(
#                 messages,
#                 add_generation_prompt=True,   # adds the assistant turn start
#                 tokenize=True,
#                 return_tensors="pt",
#                 padding=True,
#                 max_length=self.max_length,
#                 truncation=True,
#                 return_dict=True,
#             )
#         input_ids = input_enc["input_ids"]
#         input_attention_mask = input_enc["attention_mask"]
#         return {
#             "evidence": evidences,
#             "evidence_ids": evidence_ids,
#             "evidence_attention_mask": evidence_attention_mask,
#             "input_ids": input_ids,
#             "input_attention_mask": input_attention_mask,
#             "answers": answers,
#         }


@dataclass
class SquadCollator:
    tokenizer: Any
    max_length: int = 1024
    use_reference: bool = False
    metatrain: bool = False
    only_question: bool= False
    thinkend_token_id: int = None
    
    def __post_init__(self):
        self.thinkend_token_id = self.tokenizer.convert_tokens_to_ids("</think>")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        questions = [ex["question"] for ex in batch]
        evidences = [ex["evidence"] for ex in batch]
        answers = [str(random.choice(ex["answer"])) for ex in batch]
        full_answers = [ex["answer"] for ex in batch]
           
        evidence_enc = self.tokenizer(
            evidences,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        evidence_ids = evidence_enc["input_ids"]
        evidence_attention_mask = evidence_enc["attention_mask"]
        
        answer_enc = self.tokenizer(
            answers,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        answer_ids = answer_enc["input_ids"]
        answer_attention_mask = answer_enc["attention_mask"]

        if self.metatrain:            
            messages = [[
                {"role": "system", "content": "You are a concise assistant. Only output the final answer with no extra words."},
                {"role": "user", "content": f"Please answer the following question: {question}"},
                {"role": "assistant", "content": f"<think>I know the answer because I have read something about this.</think>\n{answer}"}
            ] for question, answer in zip(questions, answers)]
        elif self.use_reference:
            messages = [[
                {"role": "system", "content": "You are a concise assistant. Only output the final answer with no extra words."},
                {"role": "user", "content": f"Please review the following reference materials.\n\nReference:\n{evidence}\n\nBased on the reference, answer this question:\n{question}"},
            ] for evidence, question in zip(evidences, questions)]
        elif self.only_question:
            messages = [[
                {"role": "system", "content": "You are a concise assistant. Only output the final answer with no extra words."},
                {"role": "user", "content": f"Please answer the following question: {question}"},
            ] for question in questions]
        else:
            messages = [[
                {"role": "system", "content": "You are a concise assistant. Only output the final answer with no extra words."},
                {"role": "user", "content": f"Please answer the following question: {question}"},
                {"role": "assistant", "content": f"<think>I know the answer because I have read something about this.</think>\n"}
            ] for question in questions]

        input_enc = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True if (not self.metatrain and (self.use_reference or self.only_question)) else False,   # adds the assistant turn start
                tokenize=True,
                return_tensors="pt",
                max_length=self.max_length + 2 if not self.metatrain and not self.use_reference and not self.only_question else self.max_length,
                truncation=True,
                return_dict=True,
                padding="max_length",
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
        elif not (self.use_reference or self.only_question):
            input_ids = input_ids[:, :-2]
            input_attention_mask = input_attention_mask[:, :-2]
        
        # tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        # for i, t in enumerate(tokens):
        #     print(f"{i}: {t}")
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
            "input_ids": input_ids,
            "labels": labels,
            "input_attention_mask": input_attention_mask,
            "answers": answers,
            "full_answers": full_answers,
            "answer_ids": answer_ids,
            "answer_attention_mask": answer_attention_mask,
            "questions": questions,
        }