import json
import os

import openai
import pytest

from guardrails.applications.text2sql import Text2Sql
from guardrails.utils.openai_utils import OPENAI_VERSION

CURRENT_DIR_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA_PATH = os.path.join(CURRENT_DIR_PARENT, "test_assets/text2sql/schema.sql")
EXAMPLES_PATH = os.path.join(CURRENT_DIR_PARENT, "test_assets/text2sql/examples.json")
DB_PATH = os.path.join(
    CURRENT_DIR_PARENT, "test_assets/text2sql/department_management.sqlite"
)


@pytest.mark.parametrize(
    "conn_str, schema_path, examples",
    [
        ("sqlite://", SCHEMA_PATH, EXAMPLES_PATH),
        (f"sqlite:///{DB_PATH}", None, None),
    ],
)
def test_text2sql_with_examples(conn_str: str, schema_path: str, examples: str, mocker):
    """Test that Text2Sql can be initialized with examples."""

    # Mock the call to the OpenAI API.
    mocker.patch(
        "guardrails.embedding.OpenAIEmbedding._get_embedding",
        new=lambda *args, **kwargs: [[0.1] * 1536],
    )

    if examples is not None:
        with open(examples, "r") as f:
            examples = json.load(f)

    # This should not raise an exception.
    Text2Sql(conn_str, schema_file=schema_path, examples=examples)


@pytest.mark.skipif(not OPENAI_VERSION.startswith("0"), reason="Only for OpenAI v0")
def test_text2sql_with_coro():
    s = Text2Sql("sqlite://", llm_api=openai.Completion.acreate)
    with pytest.raises(ValueError):
        s("")
