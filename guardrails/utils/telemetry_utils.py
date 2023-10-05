import inspect
import logging
from functools import wraps
from operator import attrgetter
from typing import Any, Optional

from guardrails.classes.validation_result import Filter, Refrain
from guardrails.stores.context import ContextStore, Tracer
from guardrails.utils.logs_utils import FieldValidationLogs, ReAsk, ValidatorLogs


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


def get_validator_name(fn, *args):
    try:
        arg_spec = inspect.getfullargspec(fn)
        is_method = arg_spec[0][0] == "self"
    except Exception:
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


def get_tracer(tracer: Tracer = None) -> Tracer:
    # TODO: Do we ever need to consider supporting non-otel tracers?
    context_store = ContextStore()
    _tracer = tracer if tracer is not None else context_store.get_tracer()
    return _tracer


def get_span(span=None):
    if span is not None and hasattr(span, "add_event"):
        return span
    try:
        from opentelemetry import trace

        return trace.get_current_span()
    except Exception as e:
        import traceback

        traceback.print_exception(e)
        return None


def trace_validator_result(
    current_span, validator_log: ValidatorLogs, attempt_number: int, **kwargs
):
    (
        validator_name,
        value_before_validation,
        validation_result,
        value_after_validation,
    ) = attrgetter(
        "registered_name",
        "value_before_validation",
        "validation_result",
        "value_after_validation",
    )(
        validator_log
    )
    result = (
        validation_result.outcome
        if hasattr(validation_result, "outcome")
        else "unknown"
    )
    result_type = get_result_type(
        value_before_validation, value_after_validation, result
    )
    current_span.add_event(
        f"{validator_name}_result",
        {
            "validator_name": validator_name,
            "attempt_number": attempt_number,
            "result": result,
            "result_type": result_type,
            # TODO: to_string these
            "input": value_before_validation,
            "output": value_after_validation,
            **kwargs,
        },
    )


def trace_validation_result(
    validation_logs: FieldValidationLogs,
    attempt_number: int,
    current_span=None,
):
    _current_span = get_span(current_span)
    if _current_span is not None:
        for log in validation_logs.validator_logs:
            trace_validator_result(_current_span, log, attempt_number)
        if validation_logs.children:
            for child in validation_logs.children:
                trace_validation_result(
                    validation_logs.children.get(child), attempt_number, _current_span
                )


def trace_validator(
    name: str = None, namespace: str = None, tracer: Optional[Tracer] = None
):
    def trace_validator_wrapper(fn):
        _tracer = get_tracer(tracer)

        @wraps(fn)
        def with_trace(*args, **kwargs):
            validator_name = name if name is not None else get_validator_name(fn)
            span_name = (
                f"{namespace}.{validator_name}.validate"
                if namespace is not None
                else f"{validator_name}.validate"
            )
            with _tracer.start_as_current_span(span_name) as validator_span:
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    validator_span.set_status(
                        status=get_error_code(), description=str(e)
                    )
                    raise e

        @wraps(fn)
        def without_a_trace(*args, **kwargs):
            return fn(*args, **kwargs)

        if _tracer is not None and hasattr(_tracer, "start_as_current_span"):
            return with_trace
        else:
            return without_a_trace

    return trace_validator_wrapper


def trace(name: str, tracer: Optional[Tracer] = None):
    def trace_wrapper(fn):
        _tracer = get_tracer(tracer)

        @wraps(fn)
        def with_trace(*args, **kwargs):
            with _tracer.start_as_current_span(name) as trace_span:
                try:
                    # TODO: Capture args and kwargs as attributes?
                    return fn(*args, **kwargs)
                except Exception as e:
                    trace_span.set_status(status=get_error_code(), description=str(e))
                    raise e

        @wraps(fn)
        def without_a_trace(*args, **kwargs):
            return fn(*args, **kwargs)

        if _tracer is not None and hasattr(_tracer, "start_as_current_span"):
            return with_trace
        else:
            return without_a_trace

    return trace_wrapper
