"""--image attaches pictures to the task message as base64 image parts.

Covers the zero-flag regression (string content), parts construction (text
part = rendered instance template; image parts = data URLs whose base64
round-trips to the payload, in flag order), resume, image fitting, and the
validation matrix. All images are generated at runtime, and Model.query is
replaced by a one-reply fake that records its prompts.
"""

import base64
import io
import json
import re

import pytest
import typer
from typer.testing import CliRunner
from PIL import Image as PILImage

import main as cli_main
from agent import IMAGE_MAX_EDGE, IMAGE_MIME, IMAGE_SIZE_LIMIT, load_config, render
from conftest import assistant
from model import Model


def _cli_app():
    app = typer.Typer(add_completion=False, rich_markup_mode=None)
    app.command()(cli_main.run)
    return app


def _capture_model(monkeypatch):
    """One-reply fake Model that records every prompt it is shown."""
    seen = []

    def query(self, messages, tools=None):
        seen.append([dict(message) for message in messages])
        return assistant("task acknowledged")

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


def _image_bytes(image_format="PNG", size=(2, 2), color=(20, 40, 60)):
    output = io.BytesIO()
    PILImage.new("RGB", size, color).save(output, format=image_format)
    return output.getvalue()


def _write_oversize_jpeg(tmp_path, name="noise.jpg", orientation=None):
    size = (4032, 3024)
    image = PILImage.merge(
        "RGB", tuple(PILImage.effect_noise(size, 100) for _ in range(3))
    )
    exif = PILImage.Exif()
    if orientation is not None:
        exif[274] = orientation
    path = tmp_path / name
    image.save(path, format="JPEG", quality=97, exif=exif)
    assert path.stat().st_size > IMAGE_SIZE_LIMIT
    return str(path)


def _sent_image(seen):
    url = seen[0][1]["content"][1]["image_url"]["url"]
    prefix, encoded = url.split(",", 1)
    mime = prefix.removeprefix("data:").removesuffix(";base64")
    return mime, base64.b64decode(encoded)


def _stderr_text(result):
    """Normalize output whitespace so cause fragments stay easy to assert."""
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
    specs = [
        ("first.PNG", _image_bytes("PNG", color=(255, 0, 0))),
        ("second.jpg", _image_bytes("JPEG", color=(0, 255, 0))),
    ]
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

    payload = _image_bytes("PNG", color=(0, 0, 255))
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


def test_oversize_jpeg_is_fitted_to_a_small_jpeg_part(tmp_path, monkeypatch):
    seen = _capture_model(monkeypatch)
    image = _write_oversize_jpeg(tmp_path)

    result = _invoke(CliRunner(), tmp_path, _write_task(tmp_path, "look"),
                     ["--image", image])

    assert result.exit_code == 0, result.output
    mime, payload = _sent_image(seen)
    assert mime == "image/jpeg"
    assert len(payload) <= IMAGE_SIZE_LIMIT
    with PILImage.open(io.BytesIO(payload)) as fitted:
        assert max(fitted.size) <= IMAGE_MAX_EDGE


def test_oversize_jpeg_exif_orientation_is_applied(tmp_path, monkeypatch):
    seen = _capture_model(monkeypatch)
    image = _write_oversize_jpeg(tmp_path, orientation=6)

    result = _invoke(CliRunner(), tmp_path, _write_task(tmp_path, "look"),
                     ["--image", image])

    assert result.exit_code == 0, result.output
    mime, payload = _sent_image(seen)
    assert mime == "image/jpeg"
    with PILImage.open(io.BytesIO(payload)) as fitted:
        assert fitted.height > fitted.width


def test_oversize_jpeg_fits_when_pillow_pixel_limit_is_low(
    tmp_path, monkeypatch
):
    seen = _capture_model(monkeypatch)
    monkeypatch.setattr(PILImage, "MAX_IMAGE_PIXELS", 1000)
    image = _write_oversize_jpeg(tmp_path)

    result = _invoke(CliRunner(), tmp_path, _write_task(tmp_path, "look"),
                     ["--image", image])

    assert result.exit_code == 0, result.output
    mime, payload = _sent_image(seen)
    assert mime == "image/jpeg"
    assert len(payload) <= IMAGE_SIZE_LIMIT


def test_oversize_non_image_is_rejected_as_plain_text(tmp_path):
    payload = b"not an image" + b"x" * IMAGE_SIZE_LIMIT
    image = _write_image(tmp_path, "broken.jpg", payload)
    result = _invoke(CliRunner(), tmp_path, _write_task(tmp_path, "look"),
                     ["--image", image])

    _assert_image_rejected(result, "cannot be decoded as an image")
    output = result.output
    lines = output.splitlines()
    assert lines[0].startswith("Usage:")
    assert lines[1].startswith("Try ")
    assert lines[2] == ""
    assert lines[3].startswith("Error: ")
    assert "cannot be decoded as an image" in output
    assert "╭" not in output


def test_exactly_four_mib_is_sent_byte_identical(tmp_path, monkeypatch):
    """The size limit is inclusive: 4 MiB on the nose is sent unchanged."""
    seen = _capture_model(monkeypatch)
    payload = _image_bytes("PNG")
    payload += b"\0" * (IMAGE_SIZE_LIMIT - len(payload))
    image = _write_image(tmp_path, "exact.png", payload)

    result = _invoke(CliRunner(), tmp_path, _write_task(tmp_path, "look"),
                     ["--image", image])

    assert result.exit_code == 0, result.output
    mime, sent = _sent_image(seen)
    assert mime == "image/png"
    assert sent == payload


def test_five_occurrences_exceed_the_count_cap(tmp_path):
    flags = []
    for index in range(5):
        flags += [
            "--image",
            _write_image(tmp_path, f"pic{index}.png", _image_bytes("PNG")),
        ]
    result = _invoke(CliRunner(), tmp_path, _write_task(tmp_path, "look"), flags)
    _assert_image_rejected(result, "5th", "at most 4")
