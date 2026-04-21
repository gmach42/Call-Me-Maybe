from pydantic import ValidationError, BaseModel, model_validator, field_validator
from typing import Literal, Self
from pathlib import Path


class FunctionParameter(BaseModel):
    type: Literal["number", "string", "boolean"]
    optional_description: str | None = None

    @model_validator(mode='after')
    def type_must_be_known(cls, v: Literal["number", "string", "boolean"]) -> Literal["number", "string", "boolean"]:
        pass

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, FunctionParameter]
    returns: FunctionParameter


class PromptModel(BaseModel):
    prompt: str

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("prompt must not be empty or whitespace only")
        return v


class OutputModel(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, int | float | str | bool]

    @model_validator(mode='after')
    def check_name_in_known_functions(cls, v: str) -> str:
        pass



class PipelineConfig(BaseModel):
    functions: list[FunctionDefinition]
    prompts: list[PromptModel]
    output_path: Path

    @model_validator(mode="after")
    def validate_not_empty(self) -> Self:
        if not self.functions:
            raise ValueError("functions list must not be empty")
        if not self.prompts:
            raise ValueError("prompts list must not be empty")
        return self

    @model_validator(mode="after")
    def validate_unique_function_names(self) -> Self:
        names = [f.name for f in self.functions]
        if len(names) != len(set(names)):
            raise ValueError("function names must be unique")
        return self

    def get_function(self, name: str) -> FunctionDefinition | None:
        """Retrieve a function definition by name."""
        return next((f for f in self.functions if f.name == name), None)

    def function_names(self) -> list[str]:
        """Return all known function names."""
        return [f.name for f in self.functions]
