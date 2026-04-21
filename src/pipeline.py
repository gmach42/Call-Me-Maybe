import json
import os
from .pydantic_models import PipelineConfig, OutputModel
from .constrained_decoding import generate_function_call


def run_pipeline(config: PipelineConfig) -> None:
    results: list[dict] = []

    for test in config.prompts:
        entry: OutputModel = generate_function_call(test.prompt, config)
        results.append(entry.model_dump())

    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
    with open(config.output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Done. {len(results)} results written to {config.output_path}")
