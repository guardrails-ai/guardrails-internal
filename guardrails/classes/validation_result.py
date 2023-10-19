from typing import Any, Dict, Literal, Optional

import pydantic
from pydantic import Field


class ValidationResult(pydantic.BaseModel):
    outcome: str
    metadata: Optional[Dict[str, Any]] = None
    tokens_consumed: Optional[int] = None


class PassResult(ValidationResult):
    outcome: Literal["pass"] = "pass"

    class ValueOverrideSentinel:
        pass

    # should only be used if Validator.override_value_on_pass is True
    value_override: Optional[Any] = Field(default=ValueOverrideSentinel)


class FailResult(ValidationResult):
    outcome: Literal["fail"] = "fail"

    error_message: str
    fix_value: Optional[Any] = None


class ValidatorError(Exception):
    """Base class for all validator errors."""


class Filter:
    pass


class Refrain:
    pass
