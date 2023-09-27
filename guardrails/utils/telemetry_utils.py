import inspect
import logging
import os
from functools import wraps
from operator import attrgetter
from typing import Any
from guardrails.utils.logs_utils import FieldValidationLogs, ReAsk, ValidatorLogs
from guardrails.validators import ValidationResult, Filter, Refrain

def get_result_type(before_value: Any, after_value: Any, outcome: str):
    try:
        if isinstance(after_value, (Filter, Refrain, ReAsk)):
            name = after_value.__class__.__name__.lower()
        elif after_value != before_value:
            name = "fix"
        else:
            name = outcome
        return name
    except Exception:
        return type(after_value)

def trace_validator_result(
    current_span,
    validator_log: ValidatorLogs,
    attempt_number: int,
    **kwargs
):      
        (
            validator_name,
            value_before_validation,
            validation_result,
            value_after_validation
        ) = attrgetter(
            "validator_name",
            "value_before_validation",
            "validation_result",
            "value_after_validation",
        )(validator_log)
        result = validation_result.outcome if hasattr(validation_result, "outcome") else "unknown"
        result_type = get_result_type(value_before_validation, value_after_validation, result)
        current_span.add_event(f"{validator_name}_result", {
            "validator_name": validator_name,
            "attempt_number": attempt_number,
            "result": result,
            "result_type": result_type
            **kwargs
        })

def trace_validation_result(
        validation_logs: FieldValidationLogs,
        attempt_number: int,
        current_span = None
):
    # TODO: Switch to dependency injection
    # This won't work bc the GUARDRAILS_API_KEY doesn't exist on the server
    if os.environ.get("GUARDRAILS_API_KEY") is not None:
        from opentelemetry import trace
        current_span = current_span if current_span is not None else trace.get_current_span()
        for log in validation_logs.validator_logs:
            trace_validator_result(
                current_span,
                log,
                attempt_number
            )
        if validation_logs.children:
            trace_validation_result(
                validation_logs.children.values(),
                attempt_number,
                current_span
            )

def get_validator_name (fn, *args):
    try:
        arg_spec = inspect.getfullargspec(fn)
        is_method  = arg_spec[0][0] == 'self'
    except Exception as e:
        is_method = False

    if is_method:
        name = args[0].__class__.__name__
    else:
        name = fn.__name__
    return name

def get_error_code() -> int:
    try:
        from opentelemetry.trace import StatusCode
        return StatusCode.ERROR
    except Exception as e:
        logging.debug(f"Failed to import StatusCode from opentelemetry.trace: {str(e)}")
        return 2


def trace_validator(tracer = None, name = None, namespace = None):
    def trace_validator_wrapper(fn):
        @wraps(fn)
        def with_trace(*args, **kwargs):
            validator_name = name if name is not None else get_validator_name(fn)
            span_name = (
                f"{namespace}.{validator_name}.validate"
                if namespace is not None
                else f"{validator_name}.validate"
            )
            with tracer.start_as_current_span(span_name) as validator_span:
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    validator_span.set_status(status=get_error_code(), description=str(e))
                    raise e
        
        def without_a_trace(*args, **kwargs):
            return fn(*args, **kwargs)

        # TODO: Do we ever need to consider supporting non-otel tracers?
        if tracer is not None and hasattr(tracer, 'start_as_current_span'):
            return with_trace
        else:
            return without_a_trace
    return trace_validator_wrapper