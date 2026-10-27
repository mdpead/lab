import copy
import os

from src import utils


def test_get_run_path(config):
    assert utils.get_run_path(config) == f"{config['locations']['runs_dir']}/pytest"


def test_save_and_load_config_roundtrip(tmp_path, config):
    utils.save_config(tmp_path, config)
    assert utils.load_config(tmp_path) == config


def test_resumable_key_change_does_not_require_restart(config):
    changed = copy.deepcopy(config)
    changed["train"]["num_steps"] = 1000
    changed["train"]["checkpoint_steps"] = 50
    assert not utils._config_requires_restart(config, changed)


def test_non_resumable_change_requires_restart(config):
    changed = copy.deepcopy(config)
    changed["train"]["learning_rate"] = 5e-4
    assert utils._config_requires_restart(config, changed)

    changed = copy.deepcopy(config)
    changed["model"]["d_model"] = 64
    assert utils._config_requires_restart(config, changed)


def test_restart_check_does_not_mutate_configs(config):
    original = copy.deepcopy(config)
    changed = copy.deepcopy(config)
    changed["train"]["num_steps"] = 1000
    utils._config_requires_restart(config, changed)
    assert config == original
    assert changed["train"]["num_steps"] == 1000


def test_init_run_keeps_run_when_only_resumable_keys_change(config):
    run_path = utils.init_run(config)
    marker = f"{run_path}/marker"
    open(marker, "w").close()

    config["train"]["num_steps"] = 1000
    utils.init_run(config)

    assert os.path.exists(marker)
    assert utils.load_config(run_path)["train"]["num_steps"] == 1000


def test_init_run_clears_run_when_config_changes(config):
    run_path = utils.init_run(config)
    marker = f"{run_path}/marker"
    open(marker, "w").close()

    config["model"]["d_model"] = 64
    utils.init_run(config)

    assert not os.path.exists(marker)
    assert utils.load_config(run_path)["model"]["d_model"] == 64
