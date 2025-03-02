"""
Core implementation of the Verbatim RAG system.
"""

import re
from typing import Any, Optional, Tuple, Dict

import openai

from verbatim_rag.document import Document
from verbatim_rag.index import VerbatimIndex
from verbatim_rag.template_manager import TemplateManager


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
        template_system_prompt: Optional[str] = None,
        marking_system_prompt: Optional[str] = None,
        simple_template: Optional[str] = None,
        template: Optional[str] = None,
        template_manager: Optional[TemplateManager] = None,
        template_match_threshold: float = 0.7,
    ):
        """
        Initialize the VerbatimRAG system.

        Args:
            index: The document index
            model: The OpenAI model to use
            k: Number of documents to retrieve from the index
            template_system_prompt: Custom system prompt for template generation
            marking_system_prompt: Custom system prompt for context marking
            simple_template: A simple template string with [CONTENT] placeholder
            template: A pre-generated template to use (overrides simple_template)
            template_manager: Optional TemplateManager for template matching
            template_match_threshold: Similarity threshold for template matching (0-1)
        """
        self.index = index
        self.model = model
        self.k = k
        self.simple_template = simple_template
        self.template = template
        self.template_manager = template_manager
        self.template_match_threshold = template_match_threshold

        # Default system prompts
        self.template_system_prompt = template_system_prompt or (
            "You are an assistant that creates response templates. "
            "Given a question, create a template for answering it with the following structure:\n"
            "1. A brief introduction acknowledging the question\n"
            "2. A placeholder [CONTENT] where all the relevant information from documents will be inserted as a numbered list\n"
            "3. A brief conclusion summarizing or wrapping up the response\n\n"
            "Do not include any specific facts or information that would need to be "
            "retrieved from documents. Only create a structure with introduction "
            "and conclusion that frames the extracted content appropriately."
        )

        self.marking_system_prompt = marking_system_prompt or (
            "You are an assistant that identifies relevant information in documents. "
            "Given a question and a document, mark the exact text spans that "
            "contain information relevant to answering the question using XML tags. "
            "Use <relevant>...</relevant> tags to mark the relevant text. "
            "Do not modify the text in any way, just add the XML tags around the relevant parts. "
            "Do not add any additional text or explanations, just return the document with XML tags."
            "If there is no relevant information, don't add any tags."
        )

    def _generate_template(self, question: str) -> str:
        """
        Generate a response template with a placeholder based on the question.

        Args:
            question: The user's question

        Returns:
            A template string with [CONTENT] placeholder
        """
        # If a pre-generated template is provided, use it
        if self.template:
            return self.template

        # If a simple template is provided, use it
        if self.simple_template:
            return self.simple_template

        # Try to match with existing templates if a template manager is available
        if self.template_manager:
            matched_template, score = self.template_manager.match_template(
                question, self.template_match_threshold
            )
            if matched_template and score >= self.template_match_threshold:
                return matched_template

        # If no match found or no template manager available, generate a new template
        messages = [
            {"role": "system", "content": self.template_system_prompt},
            {
                "role": "user",
                "content": f"Create a response template for the following question: {question}",
            },
        ]

        response = openai.chat.completions.create(
            model=self.model, messages=messages, temperature=0
        )

        template = response.choices[0].message.content

        # If we have a template manager, save the new template for future use
        if self.template_manager:
            self.template_manager.templates[question] = template

        return template

    def _mark_relevant_context(
        self, question: str, documents: list[Document]
    ) -> list[str]:
        """
        Identify and extract relevant passages from the retrieved documents using XML tags.

        :param question: The user's question
        :param documents: The retrieved documents
        :return: A list of extracted text spans from the documents
        """
        if not documents:
            return []

        # Combine all documents into a single prompt with document identifiers
        combined_docs = ""
        for i, doc in enumerate(documents, 1):
            combined_docs += f"\nDOCUMENT {i}:\n{doc.content}\n"

        prompt = f"""
Question: {question}

Documents:
{combined_docs}

Mark the exact text spans that contain information relevant to answering the question using <relevant>...</relevant> XML tags.
Do not modify the text in any way, just add the XML tags around the relevant parts. If there is no relevant information in a document, don't add any tags to it.
Make sure to preserve the exact text when adding tags.
"""

        messages = [
            {"role": "system", "content": self.marking_system_prompt},
            {"role": "user", "content": prompt},
        ]

        response = openai.chat.completions.create(
            model=self.model, messages=messages, temperature=0
        )

        marked_text = response.choices[0].message.content

        # Extract the text between <relevant> tags
        spans = re.findall(r"<relevant>(.*?)</relevant>", marked_text, re.DOTALL)

        # If no tags were found, try to extract using other patterns
        if not spans:
            # Look for any text that might be marked differently
            spans = re.findall(
                r"<[^>]*?relevant[^>]*?>(.*?)</[^>]*?relevant[^>]*?>",
                marked_text,
                re.DOTALL,
            )

        # If still no spans, check if the model returned a list format
        if not spans:
            # Try to extract numbered list items
            list_spans = re.findall(
                r"\d+\.\s+(.*?)(?=\n\d+\.|\Z)", marked_text, re.DOTALL
            )
            if list_spans:
                spans = list_spans

        # Verify extracted spans against the original documents to ensure verbatim extraction
        verified_spans = []
        for span in spans:
            # Clean up any whitespace
            span = span.strip()

            # Check against all documents
            found_match = False
            for doc in documents:
                # Check for exact match
                if span in doc.content:
                    verified_spans.append(span)
                    found_match = True
                    break

            # If no exact match in any document, try to find closest match
            if not found_match:
                for doc in documents:
                    closest_match = self._find_closest_match(span, doc.content)
                    if closest_match:
                        verified_spans.append(closest_match)
                        found_match = True
                        break

        return verified_spans

    def _find_closest_match(self, span: str, text: str) -> str | None:
        """
        Find the closest matching span in the original text.

        :param span: The span to find
        :param text: The original text to search in

        :return: The closest matching span from the original text, or None if no match is found
        """
        # Try with different whitespace patterns
        span_no_whitespace = re.sub(r"\s+", "", span)

        for i in range(len(text) - len(span_no_whitespace) + 1):
            chunk = text[i : i + len(span_no_whitespace)]
            chunk_no_whitespace = re.sub(r"\s+", "", chunk)

            if chunk_no_whitespace == span_no_whitespace:
                # Find the boundaries of the actual text with original whitespace
                start = i
                end = i + len(span_no_whitespace)

                # Expand to include full words
                while start > 0 and text[start - 1].isalnum():
                    start -= 1
                while end < len(text) and text[end].isalnum():
                    end += 1

                return text[start:end].strip()

        # If no match found, try with a more lenient approach using words
        words = re.findall(r"\b\w+\b", span)
        if len(words) >= 3:  # Only try if we have at least 3 words
            pattern = r"\b" + r"\b.*?\b".join(words) + r"\b"
            matches = re.findall(pattern, text)
            if matches:
                return matches[0].strip()

        return None

    def _fill_template(self, template: str, facts: list[str]) -> str:
        """
        Fill the template with the extracted facts.

        :param template: The response template with [CONTENT] placeholder
        :param facts: The list of extracted text spans
        :return: The completed response with facts inserted into the template
        """
        # Format the facts as a numbered list
        if facts:
            formatted_content = "\n".join(
                [f"{i}. {fact}" for i, fact in enumerate(facts, 1)]
            )
        else:
            formatted_content = (
                "No relevant information found in the provided documents."
            )

        # Replace [CONTENT] placeholder with the formatted content
        if "[CONTENT]" in template:
            filled_template = template.replace("[CONTENT]", formatted_content)
        else:
            # If [CONTENT] is not in the template, append the content
            filled_template = template + "\n\n" + formatted_content

        return filled_template

    def query(
        self, question: str, template: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Process a query through the Verbatim RAG system.

        :param question: The user's question
        :param template: Optional template to use for this specific query

        :return: A tuple containing:
            - The final response
            - A dictionary with intermediate results for transparency
        """
        # Use the provided template if available, otherwise generate/match one
        current_template = template or self._generate_template(question)

        # Step 2: Retrieve relevant documents
        docs = self.index.search(question, k=self.k)

        # Step 3: Mark relevant context in the documents using XML tags
        relevant_spans = self._mark_relevant_context(question, docs)

        # Step 4: Fill the template with the marked context
        response = self._fill_template(current_template, relevant_spans)

        # Return the response and intermediate results for transparency
        return response, {
            "template": current_template,
            "retrieved_docs": [doc.content for doc in docs],
            "relevant_spans": relevant_spans,
        }
