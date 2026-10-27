import random
from collections.abc import Iterator, Sized
from typing import Any

import datasets as hf_datasets
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

# Rows a TokenSampler batches: an arrow dataset, or plain dicts for RL prompts
type Rows = hf_datasets.Dataset | list[dict[str, Any]]


class InfiniteRandomSampler(Sampler[int]):
    """Endless reshuffled row indices, resumable to an exact position.

    `batch_size` is only needed to resume, to convert the loop's minibatch count to rows.
    """

    def __init__(
        self, data_source: Sized, seed: int, start_index: int = 0, batch_size: int | None = None
    ) -> None:
        self.n = len(data_source)
        self.seed = seed
        self.start_index = start_index
        self.batch_size = batch_size

    @property
    def start_minibatch(self) -> int:
        if self.batch_size is None:
            raise ValueError("sampler built without batch_size; cannot resume by minibatch")
        return self.start_index // self.batch_size

    @start_minibatch.setter
    def start_minibatch(self, n: int) -> None:
        if self.batch_size is None:
            raise ValueError("sampler built without batch_size; cannot resume by minibatch")
        self.start_index = n * self.batch_size

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed)
        # Fast-forward whole epochs without yielding, keeping the RNG in sync
        skip = self.start_index
        while skip >= self.n:
            rng.permutation(self.n)
            skip -= self.n
        first = True
        while True:
            order = rng.permutation(self.n)
            if first:
                order = order[skip:]
                first = False
            yield from (int(i) for i in order)


class TokenSampler(Sampler[list[int]]):
    """Group rows of similar length into batches that fit a token budget.

    A padded batch costs len(batch) * its longest row, so that is what the budget checks.

    No __len__, because iteration never ends. A plain Sampler rather than a BatchSampler,
    whose inherited __len__ would fail confusingly.
    """

    def __init__(
        self, ds: Rows, token_batch_size: int | None, seed: int, start_minibatch: int = 0
    ) -> None:
        self.token_batch_size = token_batch_size
        self.seed = seed
        self.start_minibatch = start_minibatch
        self.batches = self.generate_batches(ds)

    def generate_batches(self, ds: Rows) -> list[list[int]]:
        # Read `length` as a whole column when the dataset has one; row access on arrow is slow.
        # Plain lists of dicts fall back to rows.
        if hasattr(ds, "column_names"):
            lengths = list(ds["length"])
        else:
            lengths = [row["length"] for row in ds]

        order = sorted(range(len(lengths)), key=lengths.__getitem__)

        batches: list[list[int]] = []
        batch: list[int] = []
        longest = 0
        for idx in order:
            length = lengths[idx]
            # A row over the whole budget still gets its own batch
            if max(longest, length) * (len(batch) + 1) > self.token_batch_size and batch:
                batches.append(batch)
                batch, longest = [], 0
            batch.append(idx)
            longest = max(longest, length)
        if batch:
            batches.append(batch)

        return batches

    def __iter__(self) -> Iterator[list[int]]:
        return _reshuffled_forever(self.batches, self.seed, self.start_minibatch)


def _reshuffled_forever(batches: list[list[int]], seed: int, skip: int) -> Iterator[list[int]]:
    """Yield batches forever, reshuffling each pass, resuming `skip` batches in."""
    rng = random.Random(seed)
    while True:
        order = rng.sample(batches, len(batches))
        # Fast-forward whole passes, still drawing from the RNG so the shuffle continues
        if skip >= len(order):
            skip -= len(order)
            continue
        yield from order[skip:]
        skip = 0


def resume_at_minibatch(dataloader: DataLoader, n_minibatches: int) -> None:
    """Fast-forward a train loader past n_minibatches without reading any data.

    With a batch_sampler set, DataLoader still builds a plain `.sampler` it never reads,
    so the resume state must go on whichever one is in use. Call on the train split only.
    """
    sampler = dataloader.batch_sampler if dataloader.batch_size is None else dataloader.sampler
    if not hasattr(sampler, "start_minibatch"):
        raise TypeError(f"cannot resume a {type(sampler).__name__}: it has no start_minibatch")
    sampler.start_minibatch = n_minibatches
