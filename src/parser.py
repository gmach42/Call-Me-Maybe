from pydantic import ValidationError, BaseModel, model_validator, Field
import json
import numpy as np


class ConfigVar(BaseModel):
    