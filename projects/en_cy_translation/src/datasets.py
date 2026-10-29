from collections.abc import Callable
from typing import Any

import datasets as hf_datasets


def _load_cardiff(ds_config: dict[str, Any], seed: int) -> hf_datasets.DatasetDict:
    ds = hf_datasets.load_dataset("techiaith/cardiff-university-tm-en-cy")
    train = ds["train"]
    if "sample_size" in ds_config:
        train = train.select(range(ds_config["sample_size"]))
    return train.train_test_split(test_size=ds_config["test_split_ratio"], seed=seed)


def _load_flores(
    ds_config: dict[str, Any], _seed: int
) -> hf_datasets.DatasetDict | dict[str, hf_datasets.Dataset]:
    ds = hf_datasets.load_dataset("openlanguagedata/flores_plus")
    if "sample_size" in ds_config:
        ds = {split: ds[split].select(range(ds_config["sample_size"])) for split in ds}
    return ds


LOADERS: dict[str, Callable[[dict[str, Any], int], Any]] = {
    "techiaith/cardiff-university-tm-en-cy": _load_cardiff,
    "openlanguagedata/flores_plus": _load_flores,
}


def load_dataset(ds_config: dict[str, Any], seed: int) -> Any:
    name = ds_config["ds"]
    loader = LOADERS.get(name)
    if loader is None:
        raise ValueError(f"Unknown dataset: {name}")
    return loader(ds_config, seed)


def get_dataset(stage: str, config: dict[str, Any]) -> Any:
    """The dataset a stage trains on, from `stages.<stage>.data`."""
    return load_dataset(config["stages"][stage]["data"], config["seed"])
