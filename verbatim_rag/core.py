"""
Core implementation of the Verbatim RAG system.
"""

import re
from typing import Any

import openai

from verbatim_rag.document import Document
from verbatim_rag.index import VerbatimIndex
from verbatim_rag.template_manager import TemplateManager


MARKING_SYSTEM_PROMPT = """
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


class VerbatimRAG:
    """
    A RAG system that prevents hallucination by ensuring all generated content
    is explicitly derived from source documents.
    """

    def __init__(
        self,
        index: VerbatimIndex,
        model: str = "gpt-4o-mini",
        k: int = 5,
        template_manager: TemplateManager | None = None,
    ):
        """
        Initialize the VerbatimRAG system.

        Args:
            index: The document index
            model: The OpenAI model to use
            k: Number of documents to retrieve from the index
            template_manager: Optional TemplateManager for template matching/generation
        """
        self.index = index
        self.model = model
        self.k = k

        if template_manager is None:
            self.template_manager = TemplateManager(
                model=model,
            )
        else:
            self.template_manager = template_manager

        self.marking_system_prompt = MARKING_SYSTEM_PROMPT

    def _generate_template(self, question: str) -> str:
        """
        Generate or retrieve a response template based on the question.

        :param question: The user's question
        :return: A template string with placeholders for content
        """
        matched_template, score = self.template_manager.match_template(question)
        if matched_template and score >= self.template_manager.threshold:
            return matched_template

        return self.template_manager.create_template(question)

    def _mark_relevant_context(
        self, question: str, documents: list[Document]
    ) -> dict[str, list[str]]:
        """
        Identify and extract relevant passages from the retrieved documents using XML tags.

        :param question: The user's question
        :param documents: The retrieved documents
        :return: A dictionary mapping document contents to lists of extracted text spans
        """
        if not documents:
            return {}

        doc_to_spans = {}

        for doc in documents:
            prompt = self.marking_system_prompt.format(
                QUESTION=question, DOCUMENT=doc.content
            )

            messages = [
                {"role": "user", "content": prompt},
            ]

            response = openai.chat.completions.create(
                model=self.model, messages=messages, temperature=0
            )

            marked_text = response.choices[0].message.content

            # Extract the text between <relevant> tags
            spans = re.findall(r"<relevant>(.*?)</relevant>", marked_text, re.DOTALL)

            verified_spans = []
            if spans:
                for span in spans:
                    span = span.strip()

                    if span in doc.content:
                        verified_spans.append(span)

            doc_to_spans[doc.content] = verified_spans

        return doc_to_spans

    def _fill_template(self, template: str, facts: list[list[str]]) -> str:
        """
        Fill the template with the extracted facts.

        :param template: The response template with [RELEVANT_SENTENCES] placeholder
        :param facts: The list of extracted text spans

        :return: The completed response with facts inserted into the template
        """
        # Format the facts as a numbered list
        if facts:
            formatted_content = []
            for doc_facts in facts:
                for fact in doc_facts:
                    formatted_content.append(f"{len(formatted_content) + 1}. {fact}")

            formatted_content = "\n".join(formatted_content)
        else:
            formatted_content = (
                "No relevant information found in the provided documents."
            )

        filled_template = template.replace("[RELEVANT_SENTENCES]", formatted_content)

        return filled_template

    def query(self, question: str) -> tuple[str, dict[str, Any]]:
        """
        Process a query through the Verbatim RAG system.

        :param question: The user's question
        :param template: Optional template to use for this specific query

        :return: A tuple containing:
            - The final response
            - A dictionary with intermediate results for transparency
        """
        # Use the provided template if available, otherwise generate/match one
        template = self._generate_template(question)

        # Step 2: Retrieve relevant documents
        docs = self.index.search(question, k=self.k)

        # Step 3: Mark relevant context in the documents using XML tags
        relevant_spans = self._mark_relevant_context(question, docs)

        # Step 4: Fill the template with the marked context
        response = self._fill_template(template, relevant_spans.values())

        # Return the response and intermediate results for transparency
        return response, {
            "template": template,
            "retrieved_docs": [doc.content for doc in docs],
            "relevant_spans": relevant_spans,
        }
