from functools import partial
from typing import Any

from lab import runs
from lab.data import pad, token_budget_loaders
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from datasets import DatasetDict


def collate_batch(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:

    output: dict[str, Any] = {}
    for type in ["src", "tgt"]:
        lang = "en" if type == "src" else "cy"
        rows = [item[f"text_{lang}_tokenized"] for item in batch]
        output[f"{type}_input_ids"], output[f"{type}_padding_mask"] = pad(rows, pad_token_id)

    output["tgt_output_ids"] = output["tgt_input_ids"][:, 1:].contiguous()
    output["tgt_input_ids"] = output["tgt_input_ids"][:, :-1].contiguous()
    output["tgt_padding_mask"] = output["tgt_padding_mask"][:, :-1].contiguous()

    output["src_text"] = [item["text_en"] for item in batch]
    output["tgt_text"] = [item["text_cy"] for item in batch]

    return output


def create_dataloaders(
    ds_tokenized: DatasetDict, tokenizer: PreTrainedTokenizerFast, config: runs.Config, stage: str
) -> dict[str, DataLoader]:
    return token_budget_loaders(
        ds_tokenized,
        partial(collate_batch, pad_token_id=tokenizer.pad_token_id),
        config["stages"][stage]["train"],
        config["seed"],
    )
