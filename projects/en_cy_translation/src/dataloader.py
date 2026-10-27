import random
from collections.abc import Callable, Iterator
from functools import partial
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

from datasets import Dataset, DatasetDict


class TokenSampler(BatchSampler):
    def __init__(self, ds: Dataset, token_batch_size: int, seed: int) -> None:
        self.token_batch_size = token_batch_size
        self.seed = seed
        self.batches = self.generate_batches(ds)

    def generate_batches(self, ds: Dataset) -> list[list[int]]:

        # Create batches based on token counts
        lengths = list(zip(ds["idx"], ds["en_token_length"], ds["cy_token_length"]))
        lengths_sorted = sorted(lengths, key=lambda x: max(x[1], x[2]))
        batches: list[list[int]] = []
        batch: list[int] = []
        batch_token_count = 0
        for idx, en_len, cy_len in lengths_sorted:
            token_count = max(en_len, cy_len)
            if batch_token_count + token_count > self.token_batch_size and batch:
                batches.append(batch)
                batch = []
                batch_token_count = 0
            batch.append(idx)
            batch_token_count += token_count
        if batch:
            batches.append(batch)

        return batches

    def __iter__(self) -> Iterator[list[int]]:
        # Shuffle batches to introduce randomness
        rng = random.Random(self.seed)
        while True:
            batches = rng.sample(self.batches, len(self.batches))
            yield from batches

    def __len__(self) -> int:
        return len(self.batches)


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
    ds_tokenized: DatasetDict,
    config: dict[str, Any],
) -> dict[str, DataLoader]:

    dataloaders: dict[str, DataLoader] = {}
    for split in ["train", "test"]:
        is_train = split == "train"
        dataloaders[split] = DataLoader(
            ds_tokenized[split],
            batch_sampler=TokenSampler(
                ds_tokenized[split], config["train"]["minibatch_token_size"], config["seed"]
            ),
            collate_fn=partial(collate_batch, pad_token_id=config["tokenizer"]["pad_token_id"]),
            pin_memory=True,
            num_workers=config["train"].get("num_workers", 0) if is_train else 0,
            persistent_workers=is_train and config["train"].get("num_workers", 0) > 0,
        )
    return dataloaders
