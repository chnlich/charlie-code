"""--image attaches pictures to the task message as base64 image parts.

Covers the zero-flag regression (string content), parts construction (text
part = rendered instance template; image parts = data URLs whose base64
round-trips to the file bytes, in flag order), the resume path, and the
validation matrix (missing file, directory-as-file, bad suffix, oversize,
five occurrences, inclusive 4 MiB boundary). All offline: Model.query is replaced by a one-reply fake
that records its prompts, as in test_task_file.py.
"""

import base64
import json
import re

import pytest
import typer
from typer.testing import CliRunner

import main as cli_main
from agent import IMAGE_MIME, load_config, render
from conftest import assistant, final_answer
from model import Model


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


def _invoke(runner, tmp_path, task_path, extra=()):
    return runner.invoke(
        _cli_app(),
        ["--task-file", str(task_path), "--json", "--cwd", str(tmp_path),
         "--session-dir", str(tmp_path / "sessions"), "--steps", "1", *extra],
    )


def _expected_instance_text(task_text):
    return render(load_config()["templates"]["instance"], task=task_text)


def _write_task(tmp_path, text):
    task_path = tmp_path / "task.md"
    task_path.write_text(text, encoding="utf-8")
    return task_path


def _write_image(tmp_path, name, payload):
    path = tmp_path / name
    path.write_bytes(payload)
    return str(path)


def _stderr_text(result):
    """Error-panel text with box borders and line wrapping normalized away,
    so cause fragments survive whichever way typer's rich panel wraps."""
    flattened = re.sub(r"[\u2500-\u257f]+", " ", result.stderr)
    return " ".join(flattened.split())


def _assert_image_rejected(result, *fragments):
    """Exit 2 (typer.BadParameter), stderr naming the flag and the cause."""
    assert result.exit_code == 2, result.output
    stderr = _stderr_text(result)
    assert "--image:" in stderr
    for fragment in fragments:
        assert fragment in stderr


def test_no_image_keeps_the_plain_string_content(tmp_path, monkeypatch):
    seen = _capture_model(monkeypatch)
    task_path = _write_task(tmp_path, "plain task")

    result = _invoke(CliRunner(), tmp_path, task_path)

    assert result.exit_code == 0, result.output
    user_message = seen[0][1]
    assert user_message["role"] == "user"
    assert isinstance(user_message["content"], str)
    assert user_message["content"] == _expected_instance_text("plain task")


@pytest.mark.parametrize("count", [1, 2])
def test_images_become_parts_in_flag_order(tmp_path, monkeypatch, count):
    seen = _capture_model(monkeypatch)
    specs = [("first.PNG", b"png-payload-one"), ("second.jpg", b"jpeg-payload-two")]
    flags = []
    for name, payload in specs[:count]:
        flags += ["--image", _write_image(tmp_path, name, payload)]
    task_path = _write_task(tmp_path, "look at these")

    result = _invoke(CliRunner(), tmp_path, task_path, flags)

    assert result.exit_code == 0, result.output
    content = seen[0][1]["content"]
    assert isinstance(content, list)
    assert [part["type"] for part in content] == ["text"] + ["image_url"] * count
    assert content[0] == {
        "type": "text",
        "text": _expected_instance_text("look at these"),
    }
    for part, (name, payload) in zip(content[1:], specs[:count]):
        mime = IMAGE_MIME[name.rsplit(".", 1)[1].lower()]
        prefix = f"data:{mime};base64,"
        url = part["image_url"]["url"]
        assert url.startswith(prefix)
        assert base64.b64decode(url[len(prefix):]) == payload


def test_resume_turn_carries_image_parts(tmp_path, monkeypatch):
    seen = _capture_model(monkeypatch)
    runner = CliRunner()
    task_path = _write_task(tmp_path, "turn one")

    first = _invoke(runner, tmp_path, task_path)
    assert first.exit_code == 0, first.output
    session_id = json.loads(first.stdout.splitlines()[0])["session_id"]

    payload = b"resume-shot-bytes"
    image = _write_image(tmp_path, "shot.png", payload)
    task_path.write_text("turn two, now with a picture", encoding="utf-8")
    second = _invoke(runner, tmp_path, task_path,
                     ["--resume", session_id, "--image", image])
    assert second.exit_code == 0, second.output

    resumed_history = seen[1]
    task_message = resumed_history[-1]
    assert task_message["role"] == "user"
    assert isinstance(task_message["content"], list)
    assert task_message["content"][0] == {
        "type": "text",
        "text": _expected_instance_text("turn two, now with a picture"),
    }
    url = task_message["content"][1]["image_url"]["url"]
    assert url == "data:image/png;base64," + base64.b64encode(payload).decode("ascii")
    # The prior turn's own task message is untouched and stays string-shaped.
    assert resumed_history[1]["content"] == _expected_instance_text("turn one")


def test_missing_image_file_is_rejected(tmp_path):
    result = _invoke(CliRunner(), tmp_path, _write_task(tmp_path, "look"),
                     ["--image", str(tmp_path / "ghost.png")])
    _assert_image_rejected(result, "does not exist")


def test_directory_as_image_file_is_rejected(tmp_path):
    directory = tmp_path / "folder.png"
    directory.mkdir()
    result = _invoke(CliRunner(), tmp_path, _write_task(tmp_path, "look"),
                     ["--image", str(directory)])
    _assert_image_rejected(result, "is not a file")


def test_bad_suffix_is_rejected(tmp_path):
    drawing = _write_image(tmp_path, "drawing.bmp", b"bmp bytes")
    result = _invoke(CliRunner(), tmp_path, _write_task(tmp_path, "look"),
                     ["--image", drawing])
    _assert_image_rejected(result, "unsupported suffix", "'.bmp'")


def test_oversize_image_is_rejected(tmp_path):
    payload = b"x" * (4 * 1024 * 1024 + 1)
    huge = _write_image(tmp_path, "huge.png", payload)
    result = _invoke(CliRunner(), tmp_path, _write_task(tmp_path, "look"),
                     ["--image", huge])
    _assert_image_rejected(result, str(len(payload)), str(cli_main.IMAGE_SIZE_LIMIT))


def test_exactly_four_mib_is_accepted(tmp_path, monkeypatch):
    """The size limit is inclusive: 4 MiB on the nose is a legal image."""
    seen = _capture_model(monkeypatch)
    payload = b"y" * (4 * 1024 * 1024)
    image = _write_image(tmp_path, "exact.png", payload)

    result = _invoke(CliRunner(), tmp_path, _write_task(tmp_path, "look"),
                     ["--image", image])

    assert result.exit_code == 0, result.output
    url = seen[0][1]["content"][1]["image_url"]["url"]
    prefix = "data:image/png;base64,"
    assert url.startswith(prefix)
    assert base64.b64decode(url[len(prefix):]) == payload


def test_five_occurrences_exceed_the_count_cap(tmp_path):
    flags = []
    for index in range(5):
        flags += ["--image", _write_image(tmp_path, f"pic{index}.png", b"tiny")]
    result = _invoke(CliRunner(), tmp_path, _write_task(tmp_path, "look"), flags)
    _assert_image_rejected(result, "5th", "at most 4")
