from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from mlcore import runs
from mlcore.data.samplers import TokenSampler
from torch.utils.data import DataLoader

from datasets import DatasetDict


def collate_batch(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:

    output: dict[str, Any] = {}
    for type in ["src", "tgt"]:
        lang = "en" if type == "src" else "cy"
        input_tokens = [item[f"text_{lang}_tokenized"] for item in batch]
        max_len = max(len(ids) for ids in input_tokens)
        input_ids = torch.tensor(
            [ids + [pad_token_id] * (max_len - len(ids)) for ids in input_tokens], dtype=torch.long
        )
        padding_mask = (input_ids != pad_token_id).bool()
        output[f"{type}_input_ids"] = input_ids
        output[f"{type}_padding_mask"] = padding_mask

    output["tgt_output_ids"] = output["tgt_input_ids"][:, 1:].contiguous()
    output["tgt_input_ids"] = output["tgt_input_ids"][:, :-1].contiguous()
    output["tgt_padding_mask"] = output["tgt_padding_mask"][:, :-1].contiguous()

    output["src_text"] = [item["text_en"] for item in batch]
    output["tgt_text"] = [item["text_cy"] for item in batch]

    return output


def make_collate_fn(pad_token_id: int) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    def collate_fn(x: list[dict[str, Any]]) -> dict[str, Any]:
        return collate_batch(x, pad_token_id)

    return collate_fn


def create_dataloaders(
    ds_tokenized: DatasetDict, config: runs.Config, stage: str
) -> dict[str, DataLoader]:

    train_config = config["train"][stage]
    dataloaders: dict[str, DataLoader] = {}
    for split in ["train", "test"]:
        is_train = split == "train"
        dataloaders[split] = DataLoader(
            ds_tokenized[split],
            batch_sampler=TokenSampler(
                ds_tokenized[split], train_config["minibatch_token_size"], config["seed"]
            ),
            collate_fn=partial(collate_batch, pad_token_id=config["tokenizer"]["pad_token_id"]),
            pin_memory=True,
            num_workers=train_config.get("num_workers", 0) if is_train else 0,
            persistent_workers=is_train and train_config.get("num_workers", 0) > 0,
        )
    return dataloaders
