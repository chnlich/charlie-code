"""Env resolution precedence and missing-variable reporting for evals/run.py."""

import pytest

from evals_helper import load_script

run = load_script("evals_run", "run.py")


def _models_cfg():
    return {
        "api_key_env": "CC_EVAL_API_KEY",
        "api_key_default": "EMPTY",
        "models": {
            "glm52": {"model_env": "CC_EVAL_GLM52_MODEL", "base_env": "CC_EVAL_GLM52_BASE"},
            "kimi-k3": {"model_env": "CC_EVAL_KIMI_K3_MODEL", "base_env": "CC_EVAL_KIMI_K3_BASE"},
        },
    }


def test_resolve_env_process_env_beats_file(tmp_path):
    env_file = tmp_path / "evals.env"
    env_file.write_text("CC_EVAL_GLM52_MODEL=from-file\nCC_EVAL_GLM52_BASE=file-base\n")
    resolved = run.resolve_env(
        ["CC_EVAL_GLM52_MODEL", "CC_EVAL_GLM52_BASE"],
        env_file=env_file, env={"CC_EVAL_GLM52_MODEL": "from-process"},
    )
    assert resolved["CC_EVAL_GLM52_MODEL"] == "from-process"
    assert resolved["CC_EVAL_GLM52_BASE"] == "file-base"


def test_resolve_env_file_fills_missing(tmp_path):
    env_file = tmp_path / "evals.env"
    env_file.write_text("CC_EVAL_GLM52_MODEL=file-model\nCC_EVAL_GLM52_BASE=file-base\n")
    resolved = run.resolve_env(
        ["CC_EVAL_GLM52_MODEL", "CC_EVAL_GLM52_BASE"], env_file=env_file, env={},
    )
    assert resolved == {"CC_EVAL_GLM52_MODEL": "file-model", "CC_EVAL_GLM52_BASE": "file-base"}


def test_resolve_env_missing_var_names_it(tmp_path):
    env_file = tmp_path / "evals.env"
    env_file.write_text("CC_EVAL_GLM52_BASE=file-base\n")
    with pytest.raises(SystemExit, match=r"missing required environment variable: CC_EVAL_GLM52_MODEL"):
        run.resolve_env(["CC_EVAL_GLM52_MODEL", "CC_EVAL_GLM52_BASE"], env_file=env_file, env={})


def test_resolve_env_no_file_no_process_names_var(tmp_path):
    missing = tmp_path / "absent.env"
    with pytest.raises(SystemExit, match="CC_EVAL_GLM52_BASE"):
        run.resolve_env(["CC_EVAL_GLM52_BASE"], env_file=missing, env={})


def test_resolve_model_returns_name_base_and_default_key(tmp_path):
    env_file = tmp_path / "evals.env"
    cfg = _models_cfg()
    env = {"CC_EVAL_GLM52_MODEL": "nvidia/GLM-5.2", "CC_EVAL_GLM52_BASE": "http://h/v1"}
    resolved = run.resolve_model(cfg, "glm52", env_file=env_file, env=env)
    assert resolved["model_id"] == "glm52"
    assert resolved["model_name"] == "nvidia/GLM-5.2"
    assert resolved["api_base"] == "http://h/v1"
    assert resolved["api_key"] == "EMPTY"


def test_resolve_model_picks_up_shared_key(tmp_path):
    env_file = tmp_path / "evals.env"
    cfg = _models_cfg()
    env = {
        "CC_EVAL_KIMI_K3_MODEL": "kimi", "CC_EVAL_KIMI_K3_BASE": "http://k/v1",
        "CC_EVAL_API_KEY": "sk-secret",
    }
    resolved = run.resolve_model(cfg, "kimi-k3", env_file=env_file, env=env)
    assert resolved["api_key"] == "sk-secret"


def test_resolve_model_unknown_id_is_rejected(tmp_path):
    env_file = tmp_path / "evals.env"
    with pytest.raises(SystemExit, match="unknown model id"):
        run.resolve_model(_models_cfg(), "nope", env_file=env_file, env={})
