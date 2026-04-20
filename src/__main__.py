import argparse
import sys
from .pydantic_models import PipelineConfig
from .parser import load_functions, load_prompts
from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Function calling with constrained decoding")
    parser.add_argument("--functions_definition", default="data/input/functions_definition.json")
    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calls.json")
    args = parser.parse_args()

    try:
        functions = load_functions(args.functions_definition)
        prompts = load_prompts(args.input)
        config = PipelineConfig(
            functions=functions,
            prompts=prompts,
            output_path=args.output,
        )
        run_pipeline(config)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
