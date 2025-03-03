"""
Extractors for identifying relevant spans in documents.

This module provides interfaces for extracting relevant spans from documents,
allowing for easy implementation of different extraction methods.
"""

from abc import ABC, abstractmethod

import openai

from verbatim_rag.document import Document


class SpanExtractor(ABC):
    """Abstract base class for span extractors."""

    @abstractmethod
    def extract_spans(
        self, question: str, documents: list[Document]
    ) -> dict[str, list[str]]:
        """
        Extract relevant spans from documents based on a question.

        :param question: The query or question to extract spans for
        :param documents: List of documents to extract spans from
        :return: Dictionary mapping document content to list of relevant spans
        """
        pass


class LLMSpanExtractor(SpanExtractor):
    """Extract spans using an LLM with XML tagging approach."""

    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize the LLM span extractor.

        :param model: The LLM model to use for extraction
        """
        self.model = model
        self.system_prompt = """
You are a Q&A text extraction system. Your task is to identify and mark EXACT verbatim text spans from the provided document that is relevant to answer the user's question.

# Rules
1. Mark **only** text that explicitly addresses the question
2. Never paraphrase, modify, or add to the original text
3. Preserve original wording, capitalization, and punctuation
4. Mark all relevant segments - even if they're non-consecutive
5. If there is no relevant information, don't add any tags.

# Output Format
Wrap each relevant text span with <relevant> tags. 
Return ONLY the marked document text - no explanations or summaries.

# Example
Question: What causes climate change?
Document: "Scientists agree that carbon emissions (CO2) from burning fossil fuels are the primary driver of climate change. Deforestation also contributes significantly."
Marked: "Scientists agree that <relevant>carbon emissions (CO2) from burning fossil fuels</relevant> are the primary driver of climate change. <relevant>Deforestation also contributes significantly</relevant>."

# Your Task
Question: {QUESTION}
Document: {DOCUMENT}

Mark the relevant text:
"""

    def extract_spans(
        self, question: str, documents: list[Document]
    ) -> dict[str, list[str]]:
        """
        Extract relevant spans using an LLM with XML tagging.

        :param question: The query or question
        :param documents: List of documents to extract from
        :return: Dictionary mapping document content to list of relevant spans
        """
        relevant_spans = {}

        for doc in documents:
            prompt = self.system_prompt.replace("{QUESTION}", question).replace(
                "{DOCUMENT}", doc.content
            )

            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )

            marked_text = response.choices[0].message.content

            # Extract spans between <relevant> tags
            spans = []
            start_tag = "<relevant>"
            end_tag = "</relevant>"

            start_pos = 0
            while True:
                start_idx = marked_text.find(start_tag, start_pos)
                if start_idx == -1:
                    break

                end_idx = marked_text.find(end_tag, start_idx)
                if end_idx == -1:
                    break

                span = marked_text[start_idx + len(start_tag) : end_idx]
                spans.append(span)
                start_pos = end_idx + len(end_tag)

            relevant_spans[doc.content] = spans

        return relevant_spans
