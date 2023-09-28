from guardrails.classes.validation_result import FailResult
from guardrails.utils.reask_utils import FieldReAsk

VALIDATED_OUTPUT_REASK = FieldReAsk(
    incorrect_value="Tomato Cheese Pizza",
    fail_results=[
        FailResult(
            error_message="must be exactly two words",
            fix_value="Tomato Cheese",
        )
    ],
)
