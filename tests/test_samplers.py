"""The resumable samplers: endless reshuffled rows, and token-budget batches of similar length.

Built on synthetic rows. Moved from python_reasoning's test_dataloader.py.
"""

import pytest

from lab.data.samplers import InfiniteRandomSampler, TokenSampler

NUM_BLOCKS = 20


def _rows_of_length(lengths):
    """Rows that only the sampler will look at: it reads `length` and nothing else."""
    return [{"length": length} for length in lengths]


# --- infinite sampler ---


def _take(sampler, n):
    it = iter(sampler)
    return [next(it) for _ in range(n)]


def test_sampler_is_deterministic_for_a_seed():
    a = _take(InfiniteRandomSampler(range(NUM_BLOCKS), seed=7), 3 * NUM_BLOCKS)
    b = _take(InfiniteRandomSampler(range(NUM_BLOCKS), seed=7), 3 * NUM_BLOCKS)
    assert a == b
    assert a != _take(InfiniteRandomSampler(range(NUM_BLOCKS), seed=8), 3 * NUM_BLOCKS)


def test_sampler_visits_every_index_once_per_epoch():
    """Reshuffled permutations, not sampling with replacement."""
    drawn = _take(InfiniteRandomSampler(range(NUM_BLOCKS), seed=0), 2 * NUM_BLOCKS)
    assert sorted(drawn[:NUM_BLOCKS]) == list(range(NUM_BLOCKS))
    assert sorted(drawn[NUM_BLOCKS:]) == list(range(NUM_BLOCKS))
    assert all(isinstance(i, int) for i in drawn)


@pytest.mark.parametrize("start_index", [0, 1, NUM_BLOCKS - 1])
def test_sampler_resumes_mid_epoch(start_index):
    full = _take(InfiniteRandomSampler(range(NUM_BLOCKS), seed=0), 2 * NUM_BLOCKS)
    resumed = _take(InfiniteRandomSampler(range(NUM_BLOCKS), seed=0, start_index=start_index), 5)
    assert resumed == full[start_index : start_index + 5]


@pytest.mark.parametrize("epochs,offset", [(1, 0), (1, 3), (2, 5)])
def test_sampler_resumes_after_whole_epochs(epochs, offset):
    """Fast-forwarding past an epoch boundary consumes the RNG, so the order continues."""
    start_index = epochs * NUM_BLOCKS + offset
    full = _take(InfiniteRandomSampler(range(NUM_BLOCKS), seed=0), start_index + 5)
    resumed = _take(InfiniteRandomSampler(range(NUM_BLOCKS), seed=0, start_index=start_index), 5)
    assert resumed == full[start_index:]


# --- token sampler ---


def test_token_sampler_covers_every_row_exactly_once_per_pass():
    lengths = [7, 31, 12, 45, 3, 28, 19, 22, 9, 50]
    sampler = TokenSampler(_rows_of_length(lengths), token_batch_size=200, seed=0)

    drawn = [i for batch in sampler.batches for i in batch]
    assert sorted(drawn) == list(range(len(lengths)))


@pytest.mark.parametrize("budget", [64, 200, 1000])
def test_token_sampler_respects_the_padded_token_budget(budget):
    """The budget bounds len(batch) * the longest row, not the sum of the rows."""
    lengths = [7, 31, 12, 45, 3, 28, 19, 22, 9, 50]
    sampler = TokenSampler(_rows_of_length(lengths), budget, seed=0)

    for batch in sampler.batches:
        cost = max(lengths[i] for i in batch) * len(batch)
        assert cost <= budget or len(batch) == 1


def test_token_sampler_groups_rows_of_similar_length():
    """Batches are contiguous runs of the length-ordered rows, which keeps padding low."""
    lengths = [17, 3, 9, 40, 12, 3, 25, 8]
    sampler = TokenSampler(_rows_of_length(lengths), token_batch_size=80, seed=0)

    flat = [i for batch in sampler.batches for i in batch]
    assert flat == sorted(range(len(lengths)), key=lambda i: lengths[i])


def test_token_sampler_keeps_a_row_larger_than_the_whole_budget():
    """A row over the whole budget is kept, not silently dropped."""
    sampler = TokenSampler(_rows_of_length([5, 500, 5]), token_batch_size=50, seed=0)

    assert sorted(i for batch in sampler.batches for i in batch) == [0, 1, 2]
    assert [1] in sampler.batches


def test_token_sampler_never_exhausts():
    """The train loop runs to num_steps, not to the end of the data."""
    sampler = TokenSampler(_rows_of_length([10] * 6), token_batch_size=40, seed=0)
    wanted = 3 * len(sampler.batches) + 1

    it = iter(sampler)
    assert len([next(it) for _ in range(wanted)]) == wanted


def test_token_sampler_reshuffles_order_but_keeps_batch_membership():
    """Order is reshuffled each pass but batch membership is fixed, to keep the length grouping."""
    sampler = TokenSampler(_rows_of_length(list(range(5, 45))), token_batch_size=100, seed=0)
    n = len(sampler.batches)
    assert n >= 4, "need several batches for the ordering assertion to mean anything"

    it = iter(sampler)
    first = [tuple(next(it)) for _ in range(n)]
    second = [tuple(next(it)) for _ in range(n)]

    assert sorted(first) == sorted(second)
    assert first != second


def test_token_sampler_reports_no_length():
    """len() raises rather than describe a single pass of an endless sampler."""
    sampler = TokenSampler(_rows_of_length([10] * 6), token_batch_size=40, seed=0)
    with pytest.raises(TypeError):
        len(sampler)


def test_token_sampler_is_deterministic_for_a_seed():
    rows = _rows_of_length(list(range(5, 45)))

    def take(seed):
        return [tuple(batch) for batch in _take(TokenSampler(rows, 100, seed=seed), 12)]

    assert take(7) == take(7)
    assert take(7) != take(8)


# --- resume ---


def _batches(sampler, n):
    return [tuple(b) for b in _take(sampler, n)]


@pytest.mark.parametrize("skip", [0, 1, 3, 7])
def test_token_sampler_resumes_mid_pass(skip):
    rows = _rows_of_length(list(range(5, 25)))
    full = _batches(TokenSampler(rows, 64, seed=0), skip + 3)
    resumed = _batches(TokenSampler(rows, 64, seed=0, start_minibatch=skip), 3)
    assert resumed == full[skip:]


def test_token_sampler_resumes_after_whole_passes():
    """Fast-forwarding past a pass boundary draws from the RNG, so the order continues."""
    rows = _rows_of_length(list(range(5, 25)))
    per_pass = len(TokenSampler(rows, 64, seed=0).batches)
    assert per_pass > 1, "need more than one batch for the boundary to exist"

    full = _batches(TokenSampler(rows, 64, seed=0), per_pass + 3)
    resumed = _batches(TokenSampler(rows, 64, seed=0, start_minibatch=per_pass), 3)

    assert resumed == full[per_pass:]
    assert resumed != full[:3]


def test_infinite_sampler_refuses_minibatch_resume_without_a_batch_size():
    """Without batch_size there is no rows-per-minibatch, so any offset would be invented."""
    sampler = InfiniteRandomSampler(range(NUM_BLOCKS), seed=0)
    with pytest.raises(ValueError, match="batch_size"):
        sampler.start_minibatch = 3
    with pytest.raises(ValueError, match="batch_size"):
        sampler.start_minibatch
