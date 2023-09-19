import os
from unittest.mock import Mock, patch

import pytest

from guardrails.embedding import OpenAIEmbedding


class MockOpenAIEmbedding:
    def __init__(
        self,
        model=None,
        encoding_name=None,
        max_tokens=None,
        api_key=None,
        api_base=None,
    ):
        pass

    def _len_safe_get_embedding(self, text, embedder, average=True):
        return [1.0, 2.0, 3.0]


class MockResponse:
    def __init__(self, data=None):
        self.data = data or []

    def json(self):
        return {"data": self.data}

    def __getitem__(self, key: str):
        return getattr(self, key)


@pytest.fixture
def mock_openai_embedding(monkeypatch):
    monkeypatch.setattr("openai.Embedding.create", MockOpenAIEmbedding())
    return MockOpenAIEmbedding


@pytest.mark.skipif(
    os.environ.get("OPENAI_API_KEY") is None, reason="openai api key not set"
)
class TestOpenAIEmbedding:
    def test_embedding_texts(self):
        e = OpenAIEmbedding()
        result = e.embed(["foo", "bar"])
        assert len(result) == 2
        assert len(result[0]) == 1536

    def test_embedding_query(self):
        e = OpenAIEmbedding()
        result = e.embed_query("foo")
        assert len(result) == 1536

    def test_embed_query(self, mock_openai_embedding):
        instance = OpenAIEmbedding()
        instance._get_embedding = Mock(return_value=[[1.0, 2.0, 3.0]])
        result = instance.embed_query("test query")
        assert result == [1.0, 2.0, 3.0]

    @patch("os.environ.get", return_value="test_api_key")
    @patch("openai.Embedding.create", return_value=MockResponse(data=[{ "embedding": [1.0, 2.0, 3.0] }]))
    def test_get_embedding(self, mock_create, mock_get_env):
        instance = OpenAIEmbedding(api_key="test_api_key")
        result = instance._get_embedding(["test text"])
        assert result == [[1.0, 2.0, 3.0]]
        mock_create.assert_called_once_with(
            api_key="test_api_key",
            model="text-embedding-ada-002",
            input=["test text"],
            api_base=None,
        )


@pytest.fixture
def openai_embeddings_instance():
    # You can customize this fixture creation based on your actual class initialization
    return OpenAIEmbedding("text-embedding-ada-002")  # Initialize with a model name


def test_output_dim_for_text_embedding_ada_002(openai_embeddings_instance):
    assert openai_embeddings_instance.output_dim == 1536


def test_output_dim_for_ada_model(openai_embeddings_instance):
    openai_embeddings_instance._model = "some-ada-model"
    assert openai_embeddings_instance.output_dim == 1024


def test_output_dim_for_babbage_model(openai_embeddings_instance):
    openai_embeddings_instance._model = "some-babbage-model"
    assert openai_embeddings_instance.output_dim == 2048


def test_output_dim_for_curie_model(openai_embeddings_instance):
    openai_embeddings_instance._model = "some-curie-model"
    assert openai_embeddings_instance.output_dim == 4096


def test_output_dim_for_davinci_model(openai_embeddings_instance):
    openai_embeddings_instance._model = "some-davinci-model"
    assert openai_embeddings_instance.output_dim == 12288


def test_output_dim_for_unknown_model(openai_embeddings_instance):
    openai_embeddings_instance._model = "unknown-model"
    with pytest.raises(ValueError):
        openai_embeddings_instance.output_dim
