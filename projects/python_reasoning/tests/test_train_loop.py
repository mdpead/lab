"""The pretrain/SFT train loop: checkpointing on exit, with the results for its steps."""

from lab import runs, training

from src import train

DEV = "cuda"


def _loop_config(tmp_path, num_steps, checkpoint_steps):
    return {
        "name": "t",
        "seed": 0,
        "locations": {"models_dir": str(tmp_path)},
        "model": {"max_length": 64},
        "stages": {
            "sft": {
                "train": {
                    "device": DEV,
                    "learning_rate": 1e-4,
                    "adam_betas": [0.9, 0.95],
                    "adam_eps": 1e-8,
                    "label_smoothing": 0.0,
                    "warm_up_steps": 2,
                    "effective_batch_token_size": 256,
                    "minibatch_token_size": 128,
                    "num_steps": num_steps,
                    "checkpoint_steps": checkpoint_steps,
                    "keep_checkpoints": 10,
                    "validation_steps": 10_000,
                    "validation_batches": 1,
                    "router_aux_loss_coef": 0.01,
                    "num_workers": 0,
                }
            }
        },
    }


def _run_loop(cfg, make_model, tokenizer):
    from src import dataloader as dl

    rows = [
        {"ids": [1] + [5 + (i % 7)] * (10 + i % 5) + [2], "prompt_len": 3, "length": 11 + i % 5}
        for i in range(24)
    ]
    model = make_model(len(tokenizer)).float()
    tc = cfg["stages"]["sft"]["train"]
    loaders = dl.create_dataloaders_sft({"train": rows, "test": rows[:6]}, tokenizer, cfg)
    run = training.create_run(model, tc, tokenizer)
    run["run_path"] = runs.get_stage_path(cfg, "sft")
    train.train_loop("sft", model, loaders, tokenizer, run, cfg)
    return run["run_path"]


def test_final_step_is_checkpointed_when_it_is_not_a_multiple(tmp_path, make_model, tokenizer):
    """The steps after the last checkpoint are saved on exit, with their results."""
    cfg = _loop_config(tmp_path, num_steps=3, checkpoint_steps=2)
    run_path = _run_loop(cfg, make_model, tokenizer)

    assert training.checkpoint_steps_on_disk(f"{run_path}/checkpoints") == [2, 3]


def test_a_run_ending_on_a_boundary_is_not_checkpointed_twice(
    tmp_path, make_model, tokenizer, monkeypatch
):
    """A run ending on a checkpoint boundary saves that step once, counted by calls."""
    saved_steps = []
    real_save = training.save_checkpoint

    def spy(*args, **kwargs):
        saved_steps.append(args[5])
        return real_save(*args, **kwargs)

    monkeypatch.setattr(training, "save_checkpoint", spy)

    cfg = _loop_config(tmp_path, num_steps=4, checkpoint_steps=2)
    run_path = _run_loop(cfg, make_model, tokenizer)

    assert saved_steps == [2, 4]
    assert training.checkpoint_steps_on_disk(f"{run_path}/checkpoints") == [2, 4]


def test_final_checkpoint_carries_the_results_for_its_steps(tmp_path, make_model, tokenizer):
    """results.json is written by save_checkpoint, so an unsaved tail loses its metrics."""
    import json

    cfg = _loop_config(tmp_path, num_steps=3, checkpoint_steps=2)
    run_path = _run_loop(cfg, make_model, tokenizer)

    results = json.load(open(f"{run_path}/results.json"))
    assert [e["step_no"] for e in results if e["type"] == "train"] == [1, 2, 3]
