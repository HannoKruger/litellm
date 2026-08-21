from unittest.mock import MagicMock, patch

from litellm import embedding


def _mock_embedding_response():
    response = MagicMock()
    response.parse.return_value = MagicMock(
        model_dump=lambda: {
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "Qwen/Qwen3-Embedding-8B",
            "object": "list",
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }
    )
    response.headers = {}
    return response


def test_deepinfra_embedding_routes_to_deepinfra_not_openai(monkeypatch):
    monkeypatch.setenv("DEEPINFRA_API_KEY", "sk-deepinfra-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with patch("litellm.llms.openai.openai.OpenAIChatCompletion._get_openai_client") as mock_get_client:
        client = MagicMock()
        mock_get_client.return_value = client
        client.embeddings.with_raw_response.create.return_value = _mock_embedding_response()

        result = embedding(model="deepinfra/Qwen/Qwen3-Embedding-8B", input="hello world")

        client_kwargs = mock_get_client.call_args[1]
        assert client_kwargs["api_base"] == "https://api.deepinfra.com/v1/openai"
        assert client_kwargs["api_key"] == "sk-deepinfra-test"

    assert result.data[0]["embedding"] == [0.1, 0.2, 0.3]


def test_deepinfra_embedding_sends_bare_model_name_upstream(monkeypatch):
    monkeypatch.setenv("DEEPINFRA_API_KEY", "sk-deepinfra-test")

    with patch("litellm.llms.openai.openai.OpenAIChatCompletion._get_openai_client") as mock_get_client:
        client = MagicMock()
        mock_get_client.return_value = client
        client.embeddings.with_raw_response.create.return_value = _mock_embedding_response()

        embedding(model="deepinfra/Qwen/Qwen3-Embedding-8B", input="hello world")

        create_kwargs = client.embeddings.with_raw_response.create.call_args[1]
        assert create_kwargs["model"] == "Qwen/Qwen3-Embedding-8B"
