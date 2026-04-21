from .pydantic_models import PipelineConfig, OutputModel


def generate_function_call(prompt: str, config: PipelineConfig) -> OutputModel:
    # TODO: build full prompt string
    # TODO: tokenize it via llm_sdk
    # TODO: run constrained decoding loop
    #   - phase 1: constrain to known function names
    #   - phase 2: constrain parameters based on chosen function
    # TODO: parse result into OutputModel

    name: str = ""
    parameters: dict[str, int | float | str | bool] = {}
    output = OutputModel(
        prompt=prompt,
        name=name,
        parameters=parameters,
    )
    for fname in [f.name for f in config.functions]:
        if fname in prompt:
            output.name = fname
            break


    raise NotImplementedError("constrained decoding not yet implemented")
