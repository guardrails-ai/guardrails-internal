from contextlib import AbstractContextManager
from types import TracebackType


class MockSpan(AbstractContextManager):
    def __exit__(
        self,
        __exc_type: type[BaseException] | None,
        __exc_value: BaseException | None,
        __traceback: TracebackType | None,
    ) -> bool | None:
        return super().__exit__(__exc_type, __exc_value, __traceback)

    def add_event(self, *args, **kwargs):
        pass

    def set_status(self, *args, **kwargs):
        pass


class MockTrace:
    @staticmethod
    def get_current_span(*args, **kwargs):
        return MockSpan()


class MockTracer:
    def start_as_current_span(self, *args, **kwargs):
        return MockSpan()
