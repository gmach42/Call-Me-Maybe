import argparse
import sys
from .pydantic_models import PipelineConfig
from .parser import load_functions, load_prompts
from .pipeline import run_pipeline
from llm_sdk import Small_LLM_Model
from pydantic import ValidationError


def main() -> None:
    model = Small_LLM_Model()




if __name__ == "__main__":
    main()
