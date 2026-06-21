"""Core linear-history agent loop, modeled on mini-swe-agent's ~100-line core.

The conversation is a flat message list. Each step the model produces text, we parse
exactly one fenced ```bash block from it, run that command, and feed the result back
as the next observation. The model finishes by emitting the completion sentinel.
"""

import re
from pathlib import Path

import yaml

COMPLETION_SENTINEL = "CHARLIE_CODE_COMPLETE"
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "default.yaml"

_BASH_BLOCK_RE = re.compile(r"```bash\s*\n(.*?)```", re.DOTALL)


def load_config(path=DEFAULT_CONFIG_PATH):
    return yaml.safe_load(Path(path).read_text())


def render(template, **values):
    """Fill {{name}} placeholders. Double braces never collide with shell syntax."""
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", str(value))
    return template


def parse_bash_blocks(text):
    return [block.strip() for block in _BASH_BLOCK_RE.findall(text)]


def strip_bash_blocks(text):
    return _BASH_BLOCK_RE.sub("", text).strip()


def is_completion(output):
    """True iff the last non-empty line of output equals the completion sentinel."""
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    return bool(lines) and lines[-1] == COMPLETION_SENTINEL


class Agent:
    def __init__(
        self,
        model,
        environment,
        templates,
        step_limit,
        skills_catalog="",
        emit=None,
    ):
        self.model = model
        self.environment = environment
        self.templates = templates
        self.step_limit = step_limit
        self.skills_catalog = skills_catalog
        self.emit = emit
        self.messages = []

    def run(self, task):
        self.messages = [
            {"role": "system", "content": render(
                self.templates["system"],
                cwd=self.environment.cwd,
                completion_sentinel=COMPLETION_SENTINEL,
                skills=self.skills_catalog,
            )},
            {"role": "user", "content": render(self.templates["instance"], task=task)},
        ]
        steps = []
        for step_idx in range(1, self.step_limit + 1):
            content = self.model.query(self.messages)
            self.messages.append({"role": "assistant", "content": content})
            thought = strip_bash_blocks(content)
            blocks = parse_bash_blocks(content)
            if self.emit and thought:
                self.emit({"type": "thought", "step": step_idx, "text": thought})

            if not blocks:
                observation = render(
                    self.templates["format_reminder"],
                    completion_sentinel=COMPLETION_SENTINEL,
                )
                self.messages.append({"role": "user", "content": observation})
                steps.append({"thought": thought, "command": None,
                              "observation": observation, "note": "no bash block"})
                continue

            command = blocks[0]
            event_id = f"s-{step_idx}"
            if self.emit:
                self.emit({"type": "command", "step": step_idx,
                           "id": event_id, "command": command})
            note = None
            if len(blocks) > 1:
                note = f"{len(blocks)} bash blocks found; executed only the first."

            result = self.environment.execute(command)
            if self.emit:
                self.emit({"type": "observation", "step": step_idx,
                           "id": event_id, "returncode": result["returncode"],
                           "output": result["output"]})
            observation = render(
                self.templates["observation"],
                returncode=result["returncode"],
                output=result["output"] or "<no output>",
            )
            if note:
                observation = f"[{note}]\n{observation}"
            self.messages.append({"role": "user", "content": observation})
            steps.append({"thought": thought, "command": command,
                          "observation": observation, "returncode": result["returncode"],
                          "note": note})

            if is_completion(result["output"]):
                return {"task": task, "steps": steps, "completed": True,
                        "n_steps": step_idx, "usage": self.model.usage()}

        raise RuntimeError(
            f"Step limit ({self.step_limit}) exceeded without task completion."
        )
