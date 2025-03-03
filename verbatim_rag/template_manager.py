"""
Template management for the Verbatim RAG system.

This module provides functionality to create, save, load, and match templates
for use with the VerbatimRAG system.
"""

import json
import os
import re
from difflib import SequenceMatcher

import openai

DEFAULT_TEMPLATE_SYSTEM_PROMPT = """
You create simple answer templates for questions. Your templates should:
1. Start with an acknowledgment of the question.
2. Include a placeholder for relevant sentences from documents.
3. End with a brief conclusion.

# Rules
- Use ONLY these placeholders: [RELEVANT_SENTENCES].
- Never include specific facts, numbers, or names in the template.
- Keep language generic and neutral.
- Avoid markdown, bullet points, or formatting.

# Example
Question: "What are the benefits of exercise?"
Template:
"Thanks for your question! Based on the documents, here are the key points:
[RELEVANT_SENTENCES]
These factors highlight the importance of regular exercise."

# Your Task
Generate a template for this question:
Question: {QUESTION}

Respond ONLY with the template text. No extra commentary.
"""


class TemplateManager:
    """
    Manages templates for the Verbatim RAG system.

    Templates are stored with their associated questions to allow for matching
    new questions to existing templates.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        template_system_prompt: str | None = None,
        threshold: float = 0.7,
    ):
        """
        Initialize the TemplateManager.

        :param model: The OpenAI model to use for template generation
        :param template_system_prompt: Custom system prompt for template generation
        :param threshold: Similarity threshold for template matching (0-1)
        """
        self.model = model
        self.templates = {}
        self.threshold = threshold

        self.template_system_prompt = (
            template_system_prompt or DEFAULT_TEMPLATE_SYSTEM_PROMPT
        )

    def create_template(self, question: str) -> str:
        """
        Generate a response template with a placeholder based on the question.

        :param question: The user's question
        :return: A template string with [RELEVANT_SENTENCES] placeholder
        """
        messages = [
            {
                "role": "user",
                "content": self.template_system_prompt.format(QUESTION=question),
            },
        ]

        response = openai.chat.completions.create(
            model=self.model, messages=messages, temperature=0
        )

        template = response.choices[0].message.content

        # Store the template with its question
        self.templates[question] = template

        return template

    def create_templates_batch(self, questions: list[str]) -> dict[str, str]:
        """
        Generate templates for multiple questions.

        :param questions: List of questions to generate templates for
        :return: Dictionary mapping questions to their templates
        """
        results = {}
        for question in questions:
            template = self.create_template(question)
            results[question] = template

        return results

    def save_templates(self, filepath: str) -> None:
        """
        Save templates to a JSON file.

        :param filepath: Path to save the templates
        """
        with open(filepath, "w") as f:
            json.dump(self.templates, f, indent=2)

    def load_templates(self, filepath: str) -> None:
        """
        Load templates from a JSON file.

        :param filepath: Path to load the templates from
        """
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                self.templates = json.load(f)

    def get_template(self, question: str) -> str | None:
        """
        Get a template for a question.

        :param question: The question to get a template for
        :return: The template if found, None otherwise
        """
        return self.templates.get(question)

    def match_template(
        self, question: str, threshold: float | None = None
    ) -> tuple[str | None, float]:
        """
        Match a question to an existing template.

        :param question: The question to match
        :param threshold: Optional override for similarity threshold (0-1)
        :return: Tuple of (matched template, similarity score) or (None, 0) if no match
        """
        if not self.templates:
            return None, 0

        match_threshold = threshold if threshold is not None else self.threshold

        if question in self.templates:
            return self.templates[question], 1.0

        normalized_question = re.sub(r"[^\w\s]", "", question.lower())

        best_match = None
        best_score = 0

        for q, template in self.templates.items():
            normalized_q = re.sub(r"[^\w\s]", "", q.lower())

            similarity = SequenceMatcher(
                None, normalized_question, normalized_q
            ).ratio()

            if similarity > best_score:
                best_score = similarity
                best_match = template

        if best_score >= match_threshold:
            return best_match, best_score

        return None, best_score
