import argparse
import sys
from .pydantic_models import PipelineConfig
from .parser import load_functions, load_prompts
from .pipeline import run_pipeline
from llm_sdk import Small_LLM_Model
from pydantic import ValidationError


def main() -> None:
    model = Small_LLM_Model()

    parser = argparse.ArgumentParser(description="Function calling with constrained decoding")
    parser.add_argument("--functions_definition", default="data/input/functions_definition.json")
    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calls.json")
    args = parser.parse_args()

    vocab = model.get_path_to_vocab_file()
    print(f"Using model with vocab file at: {vocab}")

    try:
        functions = load_functions(args.functions_definition)
        prompts = load_prompts(args.input)
        config = PipelineConfig(
            functions=functions,
            prompts=prompts,
            output_path=args.output,
        )
        run_pipeline(config)
    except ValidationError as e:
        try:
            print("Configuration validation error:", file=sys.stderr)
            for error in e.errors():
                loc = " -> ".join(str(l) for l in error["loc"])
                msg = error["msg"]
                print(f"  {loc}: {msg}", file=sys.stderr)
        except Exception:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
