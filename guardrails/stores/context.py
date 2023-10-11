from contextvars import ContextVar, Token
from typing import Any, Dict, Literal, TypeVar, Union
from uuid import uuid4

try:
    from guardrails.stores.document.base import DocumentStoreBase
except Exception:

    class DocumentStoreBase:
        pass


try:
    from opentelemetry.trace import Tracer
except Exception:

    class Tracer:
        pass


_T = TypeVar("_T")

TRACER: Literal["tracer"] = "tracer"
DOCUMENT_STORE: Literal["document_store"] = "document_store"
CALL_KWARGS: Literal["call_kwargs"] = "call_kwargs"


class ContextStore:
    _instance = None
    _id: str = None
    _context: Dict[str, ContextVar] = None
    _context_tokens: Dict[str, Token] = None
    _document_store_key: str = None
    _tracer_key: str = None
    _call_kwargs_key: str = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ContextStore, cls).__new__(cls)
            cls._context = cls._context or {}
            cls._context_tokens = cls._context_tokens or {}
            cls._id = str(uuid4())
            cls._tracer_key = f"{cls._id}.{TRACER}"
            cls._document_store_key = f"{cls._id}.{DOCUMENT_STORE}"
            cls._call_kwargs_key = f"{cls._id}.{CALL_KWARGS}"
        return cls._instance

    def set_document_store(
        self, document_store: Union[DocumentStoreBase, None]
    ) -> None:
        self.set_context_var(self._document_store_key, document_store)

    def get_document_store(self) -> Union[DocumentStoreBase, None]:
        return self.get_context_var(self._document_store_key)

    def set_tracer(self, tracer: Union[Tracer, None]) -> None:
        self.set_context_var(self._tracer_key, tracer)

    def get_tracer(self) -> Union[Tracer, None]:
        return self.get_context_var(self._tracer_key)

    def set_call_kwargs(self, kwargs: Dict[str, Any]) -> None:
        self.set_context_var(self._call_kwargs_key, kwargs)

    def get_call_kwargs(self) -> Dict[str, Any]:
        return self.get_context_var(self._call_kwargs_key) or {}

    def get_call_kwarg(self, kwarg_key: str) -> Union[Any, None]:
        kwargs = self.get_call_kwargs()
        return kwargs.get(kwarg_key)

    def set_context_var(self, key: str, value: _T) -> None:
        context_var = ContextVar(key)
        context_var_token = context_var.set(value)
        self._context[key] = context_var
        self._context_tokens[key] = context_var_token

    def get_context_var(self, key: str) -> _T:
        try:
            context_var = self._context.get(key)
            if context_var:
                return context_var.get()
        except Exception:
            return None

    def reset(self):
        for context_var_key in self._context:
            context_var = self._context.get(context_var_key)
            context_var_token = self._context_tokens.get(context_var_key)
            context_var.reset(context_var_token)
        self._context = {}
        self._context_tokens = {}
