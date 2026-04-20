from .pydantic_models import PipelineConfig, OutputEntry


def generate_function_call(prompt: str, config: PipelineConfig) -> OutputEntry:
    # TODO: build full prompt string
    # TODO: tokenize it via llm_sdk
    # TODO: run constrained decoding loop
    #   - phase 1: constrain to known function names
    #   - phase 2: constrain parameters based on chosen function
    # TODO: parse result into OutputEntry
    raise NotImplementedError("constrained decoding not yet implemented")
