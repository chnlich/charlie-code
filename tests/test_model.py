"""Unit tests for client-side reasoning-leak stripping. No network is touched."""

from model import strip_leaked_reasoning


def test_orphan_closing_tag_is_dropped():
    assert strip_leaked_reasoning("</think>\nreal answer") == "real answer"


def test_full_think_block_is_dropped():
    assert strip_leaked_reasoning("<think>reasoning</think>real") == "real"


def test_plain_content_without_tag_is_unchanged():
    assert strip_leaked_reasoning("just a normal answer") == "just a normal answer"


def test_bash_block_is_preserved_after_leaked_prefix():
    content = "<think>leaked thought</think>Let me do it.\n```bash\necho hi\n```"
    assert strip_leaked_reasoning(content) == "Let me do it.\n```bash\necho hi\n```"
