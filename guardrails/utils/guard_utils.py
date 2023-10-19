from functools import reduce
from typing import List
from guardrails import Guard
from guardrails.utils.logs_utils import GuardState, GuardHistory, GuardLogs

def get_guard_logs_token_consumption (guard_logs: GuardLogs) -> int:
    return (guard_logs.llm_response.prompt_token_count or 0) + (guard_logs.llm_response.response_token_count or 0)

def get_history_token_consumption (guard_history: GuardHistory) -> int:
    history: List[GuardLogs] = guard_history.history
    return reduce(
        lambda currentLogs, nextLogs: get_guard_logs_token_consumption(currentLogs) + get_guard_logs_token_consumption(nextLogs),
        history
    )

def get_state_token_consumption (guard_state: GuardState) -> int:
    all_histories: List[GuardHistory] = guard_state.all_histories
    return reduce(
        lambda currentHistory, nextHistory: get_history_token_consumption(currentHistory) + get_history_token_consumption(nextHistory),
        all_histories
    )

def get_guard_token_consumption (guard: Guard):
    return get_state_token_consumption(guard.guard_state)