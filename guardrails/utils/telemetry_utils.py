import os
from typing import Any
from guardrails.utils.logs_utils import FieldValidationLogs
from guardrails.validators import ValidationResult

def get_result_type(result: Any):
    try:
        # FIXME: Do we need to calculate whether or not a value was fixed? Probably
        name = result.__class__.__name__.lower()
        return name
    except Exception:
        return type(result)

def trace_validator_run(
    current_span,
    validator_name: str,
    result: ValidationResult,
    value: Any,
    attempt_number: int,
    **kwargs
):
        current_span.add_event(f"{validator_name}_run", {
            "validator_name": validator_name,
            "attempt_number": attempt_number,
            "result": result.outcome if hasattr(result, "outcome") else "unknown",
            "result_type": get_result_type(value),
            **kwargs
        })

def trace_validation_run(
        validation_logs: FieldValidationLogs,
        attempt_number: int,
        current_span = None
):
    # TODO: Switch the tracer to dependency injection eventually
    if os.environ.get("GUARDRAILS_API_KEY") is not None:
        from opentelemetry import trace
        current_span = current_span if current_span is not None else trace.get_current_span()
        for log in validation_logs.validator_logs:
            trace_validator_run(
                current_span,
                log.validator_name,
                log.validation_result,
                log.value_after_validation,
                attempt_number
            )
        if validation_logs.children:
            trace_validation_run(
                validation_logs.children.values(),
                attempt_number,
                current_span
            )