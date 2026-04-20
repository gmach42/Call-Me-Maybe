import json
import sys
from .pydantic_models import FunctionDefinition, TestPrompt


def load_functions(path: str) -> list[FunctionDefinition]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return [FunctionDefinition(**entry) for entry in data]
    except FileNotFoundError:
        print(f"Error: functions file not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {path}: {e}", file=sys.stderr)
        sys.exit(1)

def load_prompts(path: str) -> list[TestPrompt]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return [TestPrompt(**entry) for entry in data]
    except FileNotFoundError:
        print(f"Error: functions file not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {path}: {e}", file=sys.stderr)
        sys.exit(1)


