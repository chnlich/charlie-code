"""Env resolution precedence and missing-variable reporting for evals/run.py."""

from types import SimpleNamespace

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


def test_resolve_episode_interpreter_unset_falls_back_to_sys_executable(tmp_path):
    env_file = tmp_path / "absent.env"
    interp = run.resolve_episode_interpreter(env_file=env_file, env={})
    assert interp == run.sys.executable


def test_resolve_episode_interpreter_process_env_beats_file(tmp_path):
    env_file = tmp_path / "evals.env"
    env_file.write_text("CC_EVAL_PYTHON=from-file\n")
    fake = tmp_path / "from-process"
    fake.write_text("")
    interp = run.resolve_episode_interpreter(
        env_file=env_file, env={"CC_EVAL_PYTHON": str(fake)},
    )
    assert interp == str(fake)
    assert interp != run.sys.executable


def test_resolve_episode_interpreter_file_fills_when_process_unset(tmp_path):
    env_file = tmp_path / "evals.env"
    fake = tmp_path / "from-file"
    fake.write_text("")
    env_file.write_text(f"CC_EVAL_PYTHON={fake}\n")
    interp = run.resolve_episode_interpreter(env_file=env_file, env={})
    assert interp == str(fake)


def test_resolve_episode_interpreter_nonexistent_is_hard_error(tmp_path):
    env_file = tmp_path / "absent.env"
    bogus = str(tmp_path / "no-such-interpreter")
    with pytest.raises(SystemExit, match="CC_EVAL_PYTHON"):
        run.resolve_episode_interpreter(env_file=env_file, env={"CC_EVAL_PYTHON": bogus})


def test_run_episode_argv_uses_resolved_interpreter(tmp_path, monkeypatch):
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = list(argv)
        return SimpleNamespace(
            stdout='{"type": "result", "completed": true, "n_steps": 0, "usage": {}}',
            returncode=0,
        )

    monkeypatch.setattr(run.subprocess, "run", fake_run)
    task = {"id": "t", "prompt": "do it", "step_limit": 1}
    model_cfg = {"model_name": "m", "api_base": "http://h/v1", "api_key": "EMPTY"}
    fake_interp = str(tmp_path / "episode-python")
    run.run_episode(task, model_cfg, tmp_path, 60, fake_interp)
    assert captured["argv"][0] == fake_interp
    assert captured["argv"][1:4] == ["-m", "main", "--json"]
