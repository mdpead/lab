import pytest

from datasets import Dataset, DatasetDict
from src import datasets


def test_unknown_dataset_raises(config):
    config["data"]["train"]["ds"] = "nope/unknown"
    with pytest.raises(ValueError, match="Unknown dataset"):
        datasets.get_dataset("train", config)


def test_cardiff_loader_samples_and_splits(config, monkeypatch):
    full = DatasetDict(
        train=Dataset.from_dict({"text_en": [str(i) for i in range(100)], "text_cy": ["x"] * 100})
    )
    monkeypatch.setattr(datasets.hf_datasets, "load_dataset", lambda name: full)
    config["data"]["train"]["sample_size"] = 40

    ds = datasets.get_dataset("train", config)

    assert len(ds["train"]) == 30
    assert len(ds["test"]) == 10
    assert set(list(ds["train"]["text_en"]) + list(ds["test"]["text_en"])) == {
        str(i) for i in range(40)
    }
    assert list(datasets.get_dataset("train", config)["test"]["text_en"]) == list(
        ds["test"]["text_en"]
    )


def test_flores_loader_samples_each_split(config, monkeypatch):
    full = DatasetDict(
        dev=Dataset.from_dict({"text": list("abcdef")}),
        devtest=Dataset.from_dict({"text": list("ghijkl")}),
    )
    monkeypatch.setattr(datasets.hf_datasets, "load_dataset", lambda name: full)
    config["data"]["eval"] = {"ds": "openlanguagedata/flores_plus", "sample_size": 2}

    ds = datasets.get_dataset("eval", config)

    assert list(ds["dev"]["text"]) == ["a", "b"]
    assert list(ds["devtest"]["text"]) == ["g", "h"]
