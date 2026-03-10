"""Tests for verbatim_core.llm_client."""

import json


class TestComplete:
    def test_basic_complete(self, mock_llm_client, mock_openai_response):
        client, mock_sync, _, make_response = mock_llm_client
        mock_sync.chat.completions.create.return_value = make_response("Hello!")

        result = client.complete("Say hello")
        assert result == "Hello!"

        call_kwargs = mock_sync.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["messages"] == [{"role": "user", "content": "Say hello"}]

    def test_json_mode(self, mock_llm_client, mock_openai_response):
        client, mock_sync, _, make_response = mock_llm_client
        mock_sync.chat.completions.create.return_value = make_response('{"key": "value"}')

        client.complete("Give JSON", json_mode=True)

        call_kwargs = mock_sync.chat.completions.create.call_args[1]
        assert call_kwargs["response_format"] == {"type": "json_object"}

    def test_temperature_override(self, mock_llm_client, mock_openai_response):
        client, mock_sync, _, make_response = mock_llm_client
        mock_sync.chat.completions.create.return_value = make_response("ok")

        client.complete("test", temperature=0.0)

        call_kwargs = mock_sync.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.0


class TestExtractSpans:
    def test_successful_extraction(self, mock_llm_client, mock_openai_response):
        client, mock_sync, _, make_response = mock_llm_client
        response_data = {"doc_0": ["span one", "span two"], "doc_1": []}
        mock_sync.chat.completions.create.return_value = make_response(json.dumps(response_data))

        result = client.extract_spans("What?", {"doc_0": "text one", "doc_1": "text two"})
        assert result == response_data

    def test_json_decode_error_returns_empty(self, mock_llm_client, mock_openai_response):
        client, mock_sync, _, make_response = mock_llm_client
        mock_sync.chat.completions.create.return_value = make_response("not valid json")

        result = client.extract_spans("What?", {"doc_0": "text"})
        assert result == {"doc_0": []}


class TestExtractRelevantSpans:
    def test_single_doc_convenience(self, mock_llm_client, mock_openai_response):
        client, mock_sync, _, make_response = mock_llm_client
        response_data = {"doc": ["found span"]}
        mock_sync.chat.completions.create.return_value = make_response(json.dumps(response_data))

        result = client.extract_relevant_spans("What?", "some document text")
        assert result == ["found span"]


class TestBuildExtractionPrompt:
    def test_prompt_contains_question_and_docs(self, mock_llm_client):
        client, _, _, _ = mock_llm_client
        prompt = client._build_extraction_prompt("What color?", {"doc_0": "The sky is blue."})
        assert "What color?" in prompt
        assert "The sky is blue." in prompt
        assert "doc_0" in prompt


class TestGenerateTemplate:
    def test_per_fact_template(self, mock_llm_client, mock_openai_response):
        client, mock_sync, _, make_response = mock_llm_client
        mock_sync.chat.completions.create.return_value = make_response(
            "Here are the findings:\n[FACT_1]\n[FACT_2]"
        )

        result = client.generate_template("What?", ["span1", "span2"], citation_count=0)
        assert "[FACT_1]" in result

    def test_fallback_on_error(self, mock_llm_client):
        client, mock_sync, _, _ = mock_llm_client
        mock_sync.chat.completions.create.side_effect = Exception("API error")

        result = client.generate_template("What?", ["span1"], citation_count=0)
        assert "[DISPLAY_SPANS]" in result


class TestFallbackTemplate:
    def test_without_citations(self, mock_llm_client):
        client, _, _, _ = mock_llm_client
        result = client._fallback_template(has_citations=False)
        assert "[DISPLAY_SPANS]" in result
        assert "[CITATION_REFS]" not in result

    def test_with_citations(self, mock_llm_client):
        client, _, _, _ = mock_llm_client
        result = client._fallback_template(has_citations=True)
        assert "[CITATION_REFS]" in result
