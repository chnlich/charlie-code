"""--task-file is the only task input: a UTF-8 file path, always a real file.

Covers the byte round-trip into the rendered instance template (including task
text beyond the 128 KiB single-argv-element ceiling), the failure surface,
and the rejection of any positional argument. All offline: Model.query is
replaced by a one-reply fake that records its prompts.
"""

import typer
from typer.testing import CliRunner

import main as cli_main
from agent import load_config, render
from conftest import assistant, final_answer
from model import Model

#: A single argv element cannot carry more than 128 KiB (MAX_ARG_STRLEN), so a
#: task larger than this proves the file channel, not argv, delivered it.
ARGV_ELEMENT_CEILING = 128 * 1024


def _cli_app():
    app = typer.Typer()
    app.command()(cli_main.run)
    return app


def _capture_model(monkeypatch):
    """One-reply fake Model that records every prompt it is shown."""
    seen = []

    def query(self, messages, tools=None):
        seen.append([dict(message) for message in messages])
        return final_answer("task acknowledged")

    monkeypatch.setattr(Model, "query", query)
    monkeypatch.setattr(
        Model,
        "usage",
        lambda self: {"n_calls": 1, "input_tokens": 2, "output_tokens": 3},
    )
    return seen


def _big_task():
    """Multiline, quoted, shell-mettled, non-ASCII text past the argv ceiling."""
    stanza = (
        "quote 'single' and \"double\"; shell $VAR `tick` $(sub) | < > & ; * glob"
        "; utf-8 héllo 世界 🚀\n"
    )
    text = "First paragraph.\n\n" + stanza * 4000 + "final line, no trailing newline"
    assert len(text.encode("utf-8")) > ARGV_ELEMENT_CEILING
    return text


def _expected_instance_content(task_text):
    return render(load_config()["templates"]["instance"], task=task_text)


def _invoke(runner, tmp_path, task_file, **kwargs):
    return runner.invoke(
        _cli_app(),
        ["--task-file", task_file, "--json", "--cwd", str(tmp_path),
         "--session-dir", str(tmp_path / "sessions"), "--steps", "1"],
        **kwargs,
    )


def test_task_file_content_arrives_byte_identical_in_the_instance_template(
    tmp_path, monkeypatch
):
    seen = _capture_model(monkeypatch)
    task_text = _big_task()
    task_path = tmp_path / "task.md"
    task_path.write_text(task_text, encoding="utf-8")

    result = _invoke(CliRunner(), tmp_path, str(task_path))

    assert result.exit_code == 0, result.output
    user_message = seen[0][1]
    assert user_message["role"] == "user"
    assert task_text in user_message["content"]
    assert user_message["content"] == _expected_instance_content(task_text)


def test_task_file_dash_is_an_ordinary_path_and_fails_to_read(tmp_path):
    """'-' has no special meaning: it hits the ordinary cannot-read error."""
    result = _invoke(CliRunner(), tmp_path, "-")
    assert result.exit_code != 0
    assert "--task-file: cannot read '-'" in result.stderr


def test_missing_task_file_flag_is_rejected_naming_the_flag(tmp_path):
    result = CliRunner().invoke(
        _cli_app(),
        ["--json", "--cwd", str(tmp_path),
         "--session-dir", str(tmp_path / "sessions")],
    )
    assert result.exit_code != 0
    assert "--task-file" in result.stderr


def test_task_file_nonexistent_path_is_rejected_naming_the_flag(tmp_path):
    result = _invoke(CliRunner(), tmp_path, str(tmp_path / "no-such-task.md"))
    assert result.exit_code != 0
    assert "--task-file" in result.stderr


def test_task_file_whitespace_only_content_is_rejected_naming_the_flag(tmp_path):
    task_path = tmp_path / "task.md"
    task_path.write_text("  \n\t  \n")
    result = _invoke(CliRunner(), tmp_path, str(task_path))
    assert result.exit_code != 0
    assert "--task-file" in result.stderr


def test_task_file_invalid_utf8_is_rejected_naming_the_flag(tmp_path):
    task_path = tmp_path / "task.md"
    task_path.write_bytes(b"bad bytes \xff\xfe")
    result = _invoke(CliRunner(), tmp_path, str(task_path))
    assert result.exit_code != 0
    assert "--task-file" in result.stderr


def test_any_positional_argument_is_rejected(tmp_path):
    task_path = tmp_path / "task.md"
    task_path.write_text("real task text")
    result = CliRunner().invoke(
        _cli_app(),
        ["--task-file", str(task_path), "stray positional",
         "--json", "--cwd", str(tmp_path),
         "--session-dir", str(tmp_path / "sessions")],
    )
    assert result.exit_code != 0
    assert "unexpected" in result.stderr.lower()
