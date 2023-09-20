"""Instructions to the LLM, to be passed in the prompt."""
from string import Template
from warnings import warn

from guardrails.utils.parsing_utils import get_template_variables

from .base_prompt import BasePrompt


class Instructions(BasePrompt):
    """Instructions class.

    The instructions are passed to the LLM as secondary input. Different
    model may use these differently. For example, chat models may
    receive instructions in the system-prompt.
    """

    def __repr__(self) -> str:
        # Truncate the prompt to 50 characters and add ellipsis if it's longer.
        truncated_instructions = self.source[:50]
        if len(self.source) > 50:
            truncated_instructions += "..."
        return f"Instructions({truncated_instructions})"

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, Instructions) and self.source == __value.source

    def format(self, **kwargs):
        """Format the prompt using the given keyword arguments."""
        # Only use the keyword arguments that are present in the prompt.

        # FIXME: Is the super format call still necesary?
        # filtered_kwargs = (
        # {k: v for k, v in kwargs.items()
        # if k in self.variable_names}
        # )

        # # Return another instance of the class with the formatted prompt.
        # # If the convention of double escaping prompt params changes,
        #   send filtered_kwarfs to super.format instead
        # formatted_source = super().format()
        # return Instructions(
        # formatted_source.format(**filtered_kwargs),
        # format_instructions_start=self.format_instructions_start)

        vars = get_template_variables(self.source)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in vars}
        if len(filtered_kwargs) == 0:
            warn(
                "Instructions do not have any variables, "
                "if you are migrating follow the new variable convention "
                "documented here: https://docs.getguardrails.ai/0-2-migration/"
            )

        # Return another instance of the class with the formatted prompt.
        formatted_instructions = Template(self.source).safe_substitute(
            **filtered_kwargs
        )
        return Instructions(formatted_instructions)

    def _to_request(self) -> str:
        return self.source
