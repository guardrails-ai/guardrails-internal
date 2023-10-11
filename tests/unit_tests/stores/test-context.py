import asyncio
from contextvars import Context
import pytest
from guardrails.stores.context import (
    set_document_store,
    get_document_store,
    set_tracer,
    set_context_var,
    set_call_kwargs,
    get_call_kwarg,
    get_call_kwargs,
    get_context_var,
    get_tracer
)
from tests.mocks.mock_trace import MockTracer
from tests.unit_tests.mocks import MockDocumentStore


a_doc_store = MockDocumentStore()
@pytest.mark.parametrize(
    "doc_store,expected_doc_store",
    [
        (a_doc_store, a_doc_store),
        (None, None)
    ]
)
def test_set_and_get_document_store(doc_store, expected_doc_store):
    set_document_store(doc_store)

    actual_doc_store = get_document_store()

    assert actual_doc_store == expected_doc_store


a_tracer = MockTracer()
@pytest.mark.parametrize(
    "tracer,expected_tracer",
    [
        (a_tracer, a_tracer),
        (None, None)
    ]
)
def test_set_and_get_tracer(tracer, expected_tracer):
    set_tracer(tracer)

    actual_tracer = get_tracer()

    assert actual_tracer == expected_tracer


some_kwargs = { "kwarg1": 1, "kwarg2": 2 }
@pytest.mark.parametrize(
    "kwargs,expected_kwargs",
    [
        (some_kwargs, some_kwargs),
        (None, {})
    ]
)
def test_set_and_get_call_kwargs(kwargs, expected_kwargs):
    set_call_kwargs(kwargs)

    actual_tracer = get_call_kwargs()

    assert actual_tracer == expected_kwargs


some_kwargs = { "exists": True }
@pytest.mark.parametrize(
    "kwargs,key,expected_kwarg_value",
    [
        (some_kwargs, "exists", True),
        (some_kwargs, "rando", None),
        (None, "exists", None)
    ]
)
def test_get_call_kwarg(kwargs, key, expected_kwarg_value):
    set_call_kwargs(kwargs)

    actual_kwarg_value = get_call_kwarg(key)

    assert actual_kwarg_value == expected_kwarg_value

@pytest.mark.asyncio
async def test_context_store_closure():
    # Assert the values set in the store are isolated to their own contexts

    async def task_one():
        task_one_context = Context()
        async def __task_one():
            set_context_var("task", "task_one")
            await asyncio.sleep(0.5)
            task_context_var = get_context_var("task")
            assert task_context_var == "task_one"
        return await task_one_context.run(__task_one)

    async def task_two():
        task_two_context = Context()
        async def __task_two():
            set_context_var("task", "task_two")
            await asyncio.sleep(0.1)
            task_context_var = get_context_var("task")
            assert task_context_var == "task_two"
        return await task_two_context.run(__task_two)

    # loop = asyncio.get_running_loop()
    await asyncio.gather(task_one(), task_two())