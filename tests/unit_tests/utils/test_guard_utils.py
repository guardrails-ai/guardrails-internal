from typing import List
from guardrails import Guard
from guardrails.utils.logs_utils import GuardState, GuardHistory, GuardLogs, LLMResponse
from guardrails.utils.guard_utils import get_guard_token_consumption

def test_get_guard_token_consumption():
    rail_str = """
<rail version="0.1">

<output
    type="string"
    description="A word."
    format="length: 1 10"
    on-fail-length="fix"
/>


<prompt>
Generate a single word with a length betwen 1 and 10.
</prompt>

</rail>
"""
    log_one_one = GuardLogs(llm_response=LLMResponse(prompt_token_count=88, response_token_count=128, output="output"))
    log_one_two = GuardLogs(llm_response=LLMResponse(prompt_token_count=96, response_token_count=154, output="output"))
    guard_history_one = GuardHistory(history=[log_one_one, log_one_two])
    
    log_two_one = GuardLogs(llm_response=LLMResponse(prompt_token_count=76, response_token_count=113, output="output"))
    log_two_two = GuardLogs(llm_response=LLMResponse(prompt_token_count=102, response_token_count=196, output="output"))
    guard_history_two = GuardHistory(history=[log_two_one, log_two_two])
    guard_state = GuardState(all_histories=[guard_history_one, guard_history_two])
    guard = Guard.from_rail_string(rail_str)
    guard.guard_state = guard_state

    total_token_consumption = get_guard_token_consumption(guard)

    assert total_token_consumption == 953

