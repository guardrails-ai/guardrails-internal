import builtins
import logging
import traceback
import pytest

from datetime import datetime
from guardrails.classes.validation_result import FailResult, Filter, PassResult, Refrain, ValidationResult
from guardrails.utils.logs_utils import FieldValidationLogs, ValidatorLogs
from guardrails.stores.context import ContextStore
from guardrails.utils.reask_utils import FieldReAsk, NonParseableReAsk, ReAsk, SkeletonReAsk
from guardrails.utils.telemetry_utils import (
    get_result_type,
    get_error_code,
    get_tracer,
    get_span,
    trace_validator_result,
    trace_validation_result,
    trace_validator,
    trace,
    async_trace
)
from tests.mocks.mock_trace import MockSpan, MockTracer, MockTrace

@pytest.mark.parametrize(
    "before_value,after_value,outcome,expected_value",
    [
        (
            "before",
            Filter(),
            "pass",
            "filter"
        ),
        (
            "before",
            Refrain(),
            "pass",
            "refrain"
        ),
        (
            "before",
            FieldReAsk(incorrect_value="incorrect", fail_results=[], path=[]),
            "fail",
            "fieldreask"
        ),
        (
            "before",
            SkeletonReAsk(incorrect_value="incorrect", fail_results=[]),
            "fail",
            "skeletonreask"
        ),
        (
            "before",
            NonParseableReAsk(incorrect_value="incorrect", fail_results=[]),
            "fail",
            "nonparseablereask"
        ),
        (
            "before",
            ReAsk(incorrect_value="incorrect", fail_results=[]),
            "fail",
            "reask"
        ),
        (
            "before",
            "after",
            "pass",
            "fix"
        ),
        (
            "value",
            "value",
            "pass",
            "pass"
        )
    ]
)
def test_get_result_type(before_value, after_value, outcome, expected_value):
    result_type = get_result_type(before_value, after_value, outcome)

    assert result_type == expected_value

def test_get_result_type_exception():
    class BadString(str):
        def lower(self):
            raise Exception('Error!')
    class BadInstance(ReAsk):
        pass
    bad_instance = BadInstance(incorrect_value="incorrect", fail_results=[])
    bad_instance.__class__.__name__ = BadString("BadInstance")

    result_type = get_result_type("before", bad_instance, "fail")

    assert result_type == type(bad_instance)

def test_get_error_code():
    from opentelemetry.trace import StatusCode
    error_code = get_error_code()

    assert error_code == StatusCode.ERROR

def test_get_error_code_exception(mocker):
    realimport = builtins.__import__

    def mock_import(name, globals, locals, fromlist, level):
        if name.startswith("opentelemetry"):
            raise ImportError("Error!")
        return realimport(name, globals, locals, fromlist, level)
    mocker.patch.object(builtins, '__import__', mock_import)
    debug_spy = mocker.spy(logging, "debug")

    error_code = get_error_code()

    assert error_code == 2
    assert debug_spy.call_count == 1
    debug_spy.assert_called_with("Failed to import StatusCode from opentelemetry.trace: Error!")

def test_get_tracer__injected ():
    mock_tracer = MockTracer()
    actual_tracer = get_tracer(mock_tracer)

    assert actual_tracer == mock_tracer

def test_get_tracer__context_store ():
    mock_tracer = MockTracer()
    context_store = ContextStore()
    context_store.set_tracer(mock_tracer)
    actual_tracer = get_tracer()

    assert actual_tracer == mock_tracer
    context_store.reset()

def test_get_tracer__none ():
    actual_tracer = get_tracer()

    assert actual_tracer == None


def test_get_span__injected ():
    mock_span = MockSpan()
    actual_tracer = get_span(mock_span)

    assert actual_tracer == mock_span

def test_get_span__current_span (mocker):
    mock_trace = mocker.patch("opentelemetry.trace", MockTrace)
    get_current_span_spy = mocker.spy(mock_trace, 'get_current_span')

    actual_tracer = get_span()

    assert isinstance(actual_tracer, MockSpan)
    assert get_current_span_spy.called_once

def test_get_span__none (mocker):
    realimport = builtins.__import__
    import_error = ImportError("Error!") 
    def mock_import(name, globals, locals, fromlist, level):
        if name.startswith("opentelemetry"):
            raise import_error
        return realimport(name, globals, locals, fromlist, level)
    import_mocker = mocker.patch.object(builtins, '__import__', mock_import)

    print_exception_spy = mocker.spy(traceback, 'print_exception')

    actual_tracer = get_span()

    assert actual_tracer == None
    print_exception_spy.assert_called_once_with(import_error)
    mocker.stop(import_mocker)

def test_trace_validator_result (mocker):
    span = MockSpan()
    add_event_spy = mocker.spy(span, 'add_event')

    start_time = datetime.fromtimestamp(1696872115.90162)
    end_time = datetime.fromtimestamp(1696872116)

    validator_logs = ValidatorLogs(
        validator_name="ValidLength",
        registered_name="length",
        value_before_validation="abc",
        validation_result=PassResult(),
        value_after_validation="abccc",
        start_time=start_time,
        end_time=end_time,
    )

    trace_validator_result(
        span,
        validator_logs,
        0
    )

    assert add_event_spy.call_count == 1
    add_event_spy.assert_called_with(
        "length_result",
        {
            "validator_name": "length",
            "attempt_number": 0,
            "result": "pass",
            "result_type": "fix",
            "input": "abc",
            "output": "abccc",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
        }
    )

def test_trace_validator_result__unknown_result (mocker):
    span = MockSpan()
    add_event_spy = mocker.spy(span, 'add_event')

    start_time = datetime.fromtimestamp(1696872115.90162)
    end_time = datetime.fromtimestamp(1696872116)

    val_result = ValidationResult(outcome="not standard")
    val_result.outcome = None

    validator_logs = ValidatorLogs(
        validator_name="ValidLength",
        registered_name="length",
        value_before_validation="abc",
        validation_result=val_result,
        value_after_validation="abccc",
        start_time=start_time,
        end_time=end_time,
    )

    trace_validator_result(
        span,
        validator_logs,
        0
    )

    assert add_event_spy.call_count == 1
    add_event_spy.assert_called_with(
        "length_result",
        {
            "validator_name": "length",
            "attempt_number": 0,
            "result": "unknown",
            "result_type": "fix",
            "input": "abc",
            "output": "abccc",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
        }
    )

def test_trace_validation_result(mocker):
    mock_span = MockSpan()
    mock_get_span = mocker.patch("guardrails.utils.telemetry_utils.get_span")
    mock_get_span.return_value = mock_span
    mock_trace_validator_result = mocker.patch("guardrails.utils.telemetry_utils.trace_validator_result")

    start_time = datetime.fromtimestamp(1696872115.90162)
    end_time = datetime.fromtimestamp(1696872116)

    validator_logs = ValidatorLogs(
        validator_name="ValidLength",
        registered_name="length",
        value_before_validation="abc",
        validation_result=PassResult(),
        value_after_validation="abccc",
        start_time=start_time,
        end_time=end_time,
    )
    child_validator_logs = ValidatorLogs(
        validator_name="TwoWords",
        registered_name="two-words",
        value_before_validation="oneword",
        validation_result=FailResult(
            error_message="must be exactly two words",
            fix_value="oneword "
        ),
        value_after_validation="oneword ",
        start_time=start_time,
        end_time=end_time,
    )
    child_validation_logs = FieldValidationLogs(
        validator_logs=[child_validator_logs]
    )

    validation_logs = FieldValidationLogs(
        validator_logs=[validator_logs],
        children={
            "child":  child_validation_logs
        }
    )

    trace_validation_result(validation_logs, 0)

    assert mock_get_span.call_count == 2
    assert mock_trace_validator_result.call_count == 2


def test_trace_validator__with_trace(mocker):
    mock_tracer = MockTracer()
    mock_get_tracer = mocker.patch("guardrails.utils.telemetry_utils.get_tracer")
    mock_get_tracer.return_value = mock_tracer

    start_as_current_span_spy = mocker.spy(mock_tracer, 'start_as_current_span')
    
    mock_fn = mocker.stub(name="mock_fn")
    mock_fn.__name__ = "mock_fn"
    mock_fn.__qualname__ = "mock_fn"
    mock_fn.__annotations__ = {}
    

    decorated_fn = trace_validator(validator_name="mock-validator")(mock_fn)

    decorated_fn("arg1", kwarg1="kwarg1")

    assert mock_get_tracer.call_count == 1
    assert start_as_current_span_spy.call_count == 1
    start_as_current_span_spy.assert_called_once_with("mock-validator.validate")
    mock_fn.assert_called_once_with("arg1", kwarg1="kwarg1")

def test_trace_validator__with_trace_exception(mocker):
    from opentelemetry.trace import StatusCode
    
    mock_span = MockSpan()
    mock_tracer = MockTracer()
    mock_tracer.span = mock_span
    mock_get_tracer = mocker.patch("guardrails.utils.telemetry_utils.get_tracer")
    mock_get_tracer.return_value = mock_tracer

    start_as_current_span_spy = mocker.spy(mock_tracer, 'start_as_current_span')
    
    set_status_spy = mocker.spy(mock_span, 'set_status')
    
    mock_fn = mocker.stub(name="mock_fn")
    mock_fn.__name__ = "mock_fn"
    mock_fn.__qualname__ = "mock_fn"
    mock_fn.__annotations__ = {}
    
    mock_fn_error = Exception("Error!")
    mock_fn.side_effect = mock_fn_error
    
    with pytest.raises(Exception) as error:
        decorated_fn = trace_validator(validator_name="mock-validator")(mock_fn)

        decorated_fn("arg1", kwarg1="kwarg1")

        assert mock_get_tracer.call_count == 1
        assert start_as_current_span_spy.call_count == 1
        start_as_current_span_spy.assert_called_once_with("mock-validator.validate")
        mock_fn.assert_called_once_with("arg1", kwarg1="kwarg1")
        set_status_spy.assert_called_once_with(
            status=StatusCode.ERROR,
            description="Error!"
        )
        assert error == mock_fn_error

def test_trace_validator__without_a_trace(mocker):
    mock_get_tracer = mocker.patch("guardrails.utils.telemetry_utils.get_tracer")
    mock_get_tracer.return_value = None
    
    mock_fn = mocker.stub(name="mock_fn")
    mock_fn.__name__ = "mock_fn"
    mock_fn.__qualname__ = "mock_fn"
    mock_fn.__annotations__ = {}
    

    decorated_fn = trace_validator(validator_name="mock-validator")(mock_fn)

    decorated_fn("arg1", kwarg1="kwarg1")

    assert mock_get_tracer.call_count == 1
    mock_fn.assert_called_once_with("arg1", kwarg1="kwarg1")


def test_trace__with_trace(mocker):
    mock_tracer = MockTracer()
    mock_get_tracer = mocker.patch("guardrails.utils.telemetry_utils.get_tracer")
    mock_get_tracer.return_value = mock_tracer

    start_as_current_span_spy = mocker.spy(mock_tracer, 'start_as_current_span')
    
    mock_fn = mocker.stub(name="mock_fn")
    mock_fn.__name__ = "mock_fn"
    mock_fn.__qualname__ = "mock_fn"
    mock_fn.__annotations__ = {}
    

    decorated_fn = trace(name="mock-method")(mock_fn)

    decorated_fn("arg1", kwarg1="kwarg1")

    assert mock_get_tracer.call_count == 1
    assert start_as_current_span_spy.call_count == 1
    start_as_current_span_spy.assert_called_once_with("mock-method")
    mock_fn.assert_called_once_with("arg1", kwarg1="kwarg1")

def test_trace__with_trace_exception(mocker):
    from opentelemetry.trace import StatusCode
    
    mock_span = MockSpan()
    mock_tracer = MockTracer()
    mock_tracer.span = mock_span
    mock_get_tracer = mocker.patch("guardrails.utils.telemetry_utils.get_tracer")
    mock_get_tracer.return_value = mock_tracer

    start_as_current_span_spy = mocker.spy(mock_tracer, 'start_as_current_span')
    
    set_status_spy = mocker.spy(mock_span, 'set_status')
    
    mock_fn = mocker.stub(name="mock_fn")
    mock_fn.__name__ = "mock_fn"
    mock_fn.__qualname__ = "mock_fn"
    mock_fn.__annotations__ = {}
    
    mock_fn_error = Exception("Error!")
    mock_fn.side_effect = mock_fn_error
    
    with pytest.raises(Exception) as error:
        decorated_fn = trace(name="mock-method")(mock_fn)

        decorated_fn("arg1", kwarg1="kwarg1")

        assert mock_get_tracer.call_count == 1
        assert start_as_current_span_spy.call_count == 1
        start_as_current_span_spy.assert_called_once_with("mock-method")
        mock_fn.assert_called_once_with("arg1", kwarg1="kwarg1")
        set_status_spy.assert_called_once_with(
            status=StatusCode.ERROR,
            description="Error!"
        )
        assert error == mock_fn_error

def test_trace__without_a_trace(mocker):
    mock_get_tracer = mocker.patch("guardrails.utils.telemetry_utils.get_tracer")
    mock_get_tracer.return_value = None
    
    mock_fn = mocker.stub(name="mock_fn")
    mock_fn.__name__ = "mock_fn"
    mock_fn.__qualname__ = "mock_fn"
    mock_fn.__annotations__ = {}
    

    decorated_fn = trace(name="mock-method")(mock_fn)

    decorated_fn("arg1", kwarg1="kwarg1")

    assert mock_get_tracer.call_count == 1
    mock_fn.assert_called_once_with("arg1", kwarg1="kwarg1")


@pytest.mark.asyncio
async def test_async_trace__with_trace(mocker):
    mock_tracer = MockTracer()
    mock_get_tracer = mocker.patch("guardrails.utils.telemetry_utils.get_tracer")
    mock_get_tracer.return_value = mock_tracer

    start_as_current_span_spy = mocker.spy(mock_tracer, 'start_as_current_span')
    
    mock_fn = mocker.async_stub(name="mock_fn")
    mock_fn.__name__ = "mock_fn"
    mock_fn.__qualname__ = "mock_fn"
    mock_fn.__annotations__ = {}
    

    decorated_fn = async_trace(name="mock-method")(mock_fn)

    await decorated_fn("arg1", kwarg1="kwarg1")

    assert mock_get_tracer.call_count == 1
    assert start_as_current_span_spy.call_count == 1
    start_as_current_span_spy.assert_called_once_with("mock-method")
    mock_fn.assert_called_once_with("arg1", kwarg1="kwarg1")

@pytest.mark.asyncio
async def test_async_trace__with_trace_exception(mocker):
    from opentelemetry.trace import StatusCode
    
    mock_span = MockSpan()
    mock_tracer = MockTracer()
    mock_tracer.span = mock_span
    mock_get_tracer = mocker.patch("guardrails.utils.telemetry_utils.get_tracer")
    mock_get_tracer.return_value = mock_tracer

    start_as_current_span_spy = mocker.spy(mock_tracer, 'start_as_current_span')
    
    set_status_spy = mocker.spy(mock_span, 'set_status')
    
    mock_fn = mocker.async_stub(name="mock_fn")
    mock_fn.__name__ = "mock_fn"
    mock_fn.__qualname__ = "mock_fn"
    mock_fn.__annotations__ = {}
    
    mock_fn_error = Exception("Error!")
    mock_fn.side_effect = mock_fn_error
    
    with pytest.raises(Exception) as error:
        decorated_fn = async_trace(name="mock-method")(mock_fn)

        await decorated_fn("arg1", kwarg1="kwarg1")

        assert mock_get_tracer.call_count == 1
        assert start_as_current_span_spy.call_count == 1
        start_as_current_span_spy.assert_called_once_with("mock-method")
        mock_fn.assert_called_once_with("arg1", kwarg1="kwarg1")
        set_status_spy.assert_called_once_with(
            status=StatusCode.ERROR,
            description="Error!"
        )
        assert error == mock_fn_error

@pytest.mark.asyncio
async def test_async_trace__without_a_trace(mocker):
    mock_get_tracer = mocker.patch("guardrails.utils.telemetry_utils.get_tracer")
    mock_get_tracer.return_value = None
    
    mock_fn = mocker.async_stub(name="mock_fn")
    mock_fn.__name__ = "mock_fn"
    mock_fn.__qualname__ = "mock_fn"
    mock_fn.__annotations__ = {}
    

    decorated_fn = async_trace(name="mock-method")(mock_fn)

    await decorated_fn("arg1", kwarg1="kwarg1")

    assert mock_get_tracer.call_count == 1
    mock_fn.assert_called_once_with("arg1", kwarg1="kwarg1")